from collections import deque
from typing import Union, Iterable
import math

from game import *


class PlayerController:
    """
    oppav6 — phase-based architecture with regen-gating, beacon strategy,
    BFS-corrected MST routing, and opponent pattern tracking.

    Improvements over oppav5:
    1. Phase system: FARM / RUSH / DEFEND / CONTEST / ENDGAME
    2. Regen-gating: paint locally before committing to distant hills
    3. Smart dynamic bidding based on hill proximity
    4. Opponent pattern tracking (16-turn window, aggression classification)
    5. Full BFS for MST hill sequence (no Manhattan approximation)
    6. Beacon placement and teleportation
    7. Endgame strategy (tiebreak-aware play)
    8. Territory pressure when ahead
    9. Improved defense (lower urgency threshold, faster response)
    10. Powerup awareness (upcoming spawns)
    """

    # Phase constants
    PHASE_FARM = 'FARM'
    PHASE_RUSH = 'RUSH'
    PHASE_DEFEND = 'DEFEND'
    PHASE_CONTEST = 'CONTEST'
    PHASE_ENDGAME = 'ENDGAME'

    def __init__(self, player_parity: int, time_left):
        self.pp = player_parity
        self._committed_hill_id = None
        self._map_info = None

        # Position history for oscillation detection (last 8 turns)
        self._position_history = []

        # MST-inspired hill visit sequence
        self._hill_sequence = []
        self._hill_sig = None

        # Collision rush (triggered by oscillation only)
        self._rush_mode = False
        self._rush_counter = 0

        # Current phase
        self._phase = self.PHASE_RUSH

        # Opponent tracking — extended (16-turn window)
        self._opp_positions = []       # list of (r, c) for last 16 turns
        self._opp_stamina_history = [] # list of stamina values
        self._opp_monitor = {
            'prev_stamina': None,
            'est_regen': 5,
            'max_moves': 1,
            'threat_range': 2,
            'proximity': 999,
            'approaching': False,
            'prev_dist': None,
            'stamina': 100,
            'direction_r': 0,    # average row direction
            'direction_c': 0,    # average col direction
            'aggression': 'FARMING',  # FARMING, RUSHING, HUNTING
            'stamina_trend': 'STABLE',  # GAINING, STABLE, DEPLETING
        }

        # BFS cache for MST hill sequencing
        self._bfs_cache = {}
        self._bfs_cache_round = -1

        # Beacon tracking
        self._beacon_placed_hills = set()  # hill IDs where we placed beacons

    # ===============================================================
    # BIDDING
    # ===============================================================

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        pp = player_parity
        me = board.get_player(pp)
        opp = board.get_opponent(pp)

        if len(board.hills) == 0:
            return 5

        my_nearest_hill = 999
        opp_nearest_hill = 999
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                md_me = abs(me.loc.r - hloc.r) + abs(me.loc.c - hloc.c)
                md_opp = abs(opp.loc.r - hloc.r) + abs(opp.loc.c - hloc.c)
                my_nearest_hill = min(my_nearest_hill, md_me)
                opp_nearest_hill = min(opp_nearest_hill, md_opp)

        if my_nearest_hill <= 2 and opp_nearest_hill >= 5:
            return 15  # Almost there, bid high to capture first
        if my_nearest_hill <= opp_nearest_hill - 2:
            return 10  # We're closer, moderate bid
        if len(me.controlled_hills) > len(opp.controlled_hills):
            return 3   # Ahead, save stamina
        if board.current_round > 400:
            return 4   # Late game, conserve
        return 6  # Default

    # ===============================================================
    # PHASE DETERMINATION
    # ===============================================================

    def _determine_phase(self, board, pp, me, opp, my_dist, opp_dist):
        """Determine current game phase from board state."""
        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)
        pos = me.loc
        mi = self._map_info
        large_map = mi['large']
        mon = self._opp_monitor

        # Count local friendly cells in 5x5
        local_friendly = self._count_local_friendly(board, pos, pp)

        # Nearest uncaptured hill distance
        nearest_hill_dist = 999
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                d = my_dist.get(hloc, 999)
                if d < nearest_hill_dist:
                    nearest_hill_dist = d

        # ENDGAME: late turns or dominant position
        if board.current_round > 700:
            return self.PHASE_ENDGAME
        if our_hills == total_hills and total_hills > 0:
            territory_lead = board.get_territory_count(pp) - board.get_territory_count(-pp)
            if territory_lead > 20:
                return self.PHASE_ENDGAME

        # CONTEST: both players close to same uncaptured/enemy hill
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            my_hill_d = min((my_dist.get(hloc, 999) for hloc in hill.cells), default=999)
            opp_hill_d = min((opp_dist.get(hloc, 999) for hloc in hill.cells), default=999)
            if my_hill_d <= 4 and opp_hill_d <= 4:
                return self.PHASE_CONTEST

        # DEFEND: our hill under threat
        for hid in me.controlled_hills:
            hill = board.hills[hid]
            hs = len(hill.cells)
            opp_cells = sum(1 for h in hill.cells
                            if board.cells[h.r][h.c].owner_parity == -pp)
            urgency = opp_cells / hs if hs > 0 else 0

            # Check if opponent is approaching the hill
            opp_approaching_hill = False
            for hloc in hill.cells:
                if opp_dist.get(hloc, 999) <= 3:
                    opp_approaching_hill = True
                    break

            if urgency >= 0.10 and opp_approaching_hill:
                return self.PHASE_DEFEND

        # FARM: need regen before rushing on large maps
        if (local_friendly < 6 and nearest_hill_dist > 3
                and board.current_round < 100 and large_map):
            return self.PHASE_FARM

        # Default: RUSH
        return self.PHASE_RUSH

    def _count_local_friendly(self, board, pos, pp):
        """Count friendly cells in 5x5 (radius 2) around position."""
        count = 0
        radius = GameConstants.ADJACENCY_RADIUS
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = pos.r + dr, pos.c + dc
                if 0 <= r < board.board_size.r and 0 <= c < board.board_size.c:
                    if board.cells[r][c].owner_parity == pp:
                        count += 1
        return count

    # ===============================================================
    # MAIN PLAY
    # ===============================================================

    def play(
        self, board: Board, player_parity: int, time_left,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        pp = player_parity
        me = board.get_player(pp)
        opp = board.get_opponent(pp)
        pos = me.loc
        stamina = me.stamina
        actions = []
        painted = {}

        # ── 1. TRACKING ─────────────────────────────────────────────────────
        self._update_opp_monitor(board, pp, me, opp)
        mon = self._opp_monitor
        opp_reach = mon['threat_range']

        self._position_history.append((pos.r, pos.c))
        if len(self._position_history) > 8:
            self._position_history.pop(0)

        # ── 2. MAP / DISTANCES ──────────────────────────────────────────────
        if self._map_info is None:
            self._map_info = self._classify_map(board)
        mi = self._map_info
        large_map = mi['large']
        maze_map = mi['maze']

        my_dist = self._bfs(board, pos, pp)
        opp_dist = self._bfs_raw(board, opp.loc)

        # Invalidate BFS cache if new round
        if board.current_round != self._bfs_cache_round:
            self._bfs_cache = {}
            self._bfs_cache_round = board.current_round

        # ── 3. PHASE DETERMINATION ─────────────────────────────────────────
        self._phase = self._determine_phase(board, pp, me, opp, my_dist, opp_dist)

        # ── 4. ALWAYS-FIRST CHECKS ─────────────────────────────────────────
        kill = self._try_kill(board, me, opp, pp, stamina)
        if kill is not None:
            return self._validate_paint_actions(board, pp, kill)

        cell = board.cells[pos.r][pos.c]
        near_any_hill = self._near_hill(board, pos, pp, radius=4)

        if not near_any_hill and cell.owner_parity == -pp and mon['proximity'] <= 3:
            esc = self._escape_dir(board, pos, opp.loc, pp)
            if esc:
                return [Action.Move(esc)]

        if not near_any_hill and stamina < 35 and cell.owner_parity != pp:
            ret = self._retreat_dir(board, pos, opp.loc, pp)
            if ret:
                return [Action.Move(ret)]

        # ── 5. BEACON TELEPORT CHECK ───────────────────────────────────────
        # Check before phase dispatch — may shortcut movement
        beacon_travel = None  # will be set if we decide to teleport

        # ── 6. PHASE DISPATCH ──────────────────────────────────────────────
        if self._phase == self.PHASE_FARM:
            return self._play_farm(board, pp, me, opp, my_dist, opp_dist, stamina)
        elif self._phase == self.PHASE_DEFEND:
            result = self._play_defend(board, pp, me, opp, my_dist, opp_dist, stamina)
            if result is not None:
                return result
            # Fall through to RUSH if defend returns None
        elif self._phase == self.PHASE_ENDGAME:
            result = self._play_endgame(board, pp, me, opp, my_dist, opp_dist, stamina)
            if result is not None:
                return result
            # Fall through to RUSH if endgame returns None

        # CONTEST and RUSH both use the main hill-rushing logic below
        # (CONTEST just has more aggressive parameters set via flags)

        # ── 7. HILL STATE ──────────────────────────────────────────────────
        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)

        # ── 8. COLLISION RUSH (fast oscillation trigger) ───────────────────
        oscillating = self._is_oscillating()

        if self._rush_mode:
            self._rush_counter += 1
            if our_hills > their_hills or self._rush_counter > 20:
                self._rush_mode = False
                self._rush_counter = 0
            else:
                rush_result = self._rush_for_collision(board, pp, me, opp, my_dist, stamina)
                if rush_result is not None:
                    return self._validate_paint_actions(board, pp, rush_result)
                self._rush_mode = False
                self._rush_counter = 0

        if oscillating and not self._rush_mode and their_hills >= our_hills and stamina >= 40:
            self._rush_mode = True
            self._rush_counter = 0
            rush_result = self._rush_for_collision(board, pp, me, opp, my_dist, stamina)
            if rush_result is not None:
                return self._validate_paint_actions(board, pp, rush_result)
            self._rush_mode = False

        # ── 9. NO-HILL MAP ─────────────────────────────────────────────────
        if total_hills == 0:
            return self._validate_paint_actions(board, pp,
                self._play_territory(board, pp, me, opp, my_dist, stamina))

        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD)

        # ── 10. MST-INSPIRED HILL SEQUENCE (BFS-corrected) ────────────────
        self._refresh_hill_sequence(board, pp, me, opp, my_dist, opp_dist)

        # ── 11. TARGET SELECTION ───────────────────────────────────────────
        defend = self._defend_target(board, pp, me, opp, my_dist, dom_needed, their_hills)
        attack, attack_hid = self._pick_attack_target_sequenced(
            board, pp, me, opp, my_dist, opp_dist, dom_needed)

        # Strong commitment: keep targeting a committed hill until captured
        if self._committed_hill_id is not None:
            cid = self._committed_hill_id
            if cid in board.hills and board.hills[cid].controller_parity != pp:
                hill = board.hills[cid]
                best_loc = None
                best_d = 999
                for hloc in hill.cells:
                    if board.cells[hloc.r][hloc.c].owner_parity != pp:
                        d = my_dist.get(hloc, 999)
                        if d < best_d:
                            best_d = d
                            best_loc = hloc
                if best_loc:
                    attack = best_loc
                    attack_hid = cid
            else:
                self._committed_hill_id = None

        target = None
        chosen_hid = None
        if defend and attack:
            dd = my_dist.get(defend, 999)
            ad = my_dist.get(attack, 999)
            if their_hills + 1 >= dom_needed or dd <= ad + 2:
                target = defend
            else:
                target = attack
                chosen_hid = attack_hid
        elif attack:
            target = attack
            chosen_hid = attack_hid
        else:
            target = defend

        if chosen_hid is not None:
            self._committed_hill_id = chosen_hid
        elif target is None:
            self._committed_hill_id = None

        if target is None:
            return self._validate_paint_actions(board, pp,
                self._play_territory(board, pp, me, opp, my_dist, stamina))

        td = my_dist.get(target, 999)

        # ── 12. REGEN-GATING CHECK ─────────────────────────────────────────
        # On large maps, if we're far from the target and our regen is low,
        # paint locally first before committing to the rush
        if large_map and td > 5 and not near_any_hill:
            local_friendly = self._count_local_friendly(board, pos, pp)
            regen = self._estimate_regen(board, pos, pp)
            if local_friendly < 6 and regen < 15 and stamina < 60:
                return self._play_farm(board, pp, me, opp, my_dist, opp_dist, stamina)

        # ── 13. BEACON TRAVEL CHECK ────────────────────────────────────────
        beacon_action = self._try_beacon_travel(board, pp, target)
        if beacon_action is not None and td > 5:
            # Teleport closer to target
            actions.append(beacon_action)
            # After teleport, we need to recalculate position
            # The beacon_target is where we end up
            new_pos = beacon_action.beacon_target
            my_dist = self._bfs(board, new_pos, pp)
            pos = new_pos
            td = my_dist.get(target, 999)
            # Paint at new location
            actions, stamina = self._paint_hill_cells(
                board, pp, pos, actions, stamina, painted, 10, False, 2)
            actions, stamina = self._paint_trail(
                board, pp, pos, actions, stamina, painted, 10, target, 1)
            # Continue with rush logic from new position (but first move is used)
            final = self._simpos(board, me.loc, actions)
            actions = self._validate_paint_actions(board, pp, actions)
            if not any(isinstance(a, Action.Move) for a in actions):
                fb = self._fallback_move(board, pp, me, opp)
                if fb:
                    actions.append(fb)
                else:
                    for d in Direction.cardinals():
                        nxt = me.loc + d
                        if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                            actions.append(Action.Move(d))
                            break
            return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

        # ── 14. POWERUP DETOUR ─────────────────────────────────────────────
        pu = self._powerup_nearby(board, my_dist, 1)
        if pu:
            target = pu
            td = 1
        elif (stamina < 60 or board.current_round < 50) and td > 3:
            pu = self._powerup_nearby(board, my_dist, 2)
            if pu:
                target = pu
                td = my_dist.get(pu, 999)

        # ── 14b. SCHEDULED POWERUP AWARENESS ──────────────────────────────
        if td > 3 and hasattr(board, 'scheduled_powerups'):
            for sp in board.scheduled_powerups:
                if sp.round_num <= board.current_round + 3:
                    sp_dist = my_dist.get(sp.location, 999)
                    if sp_dist <= 3 and sp_dist < td:
                        target = sp.location
                        td = sp_dist
                        break

        # ── 15. CONTEXT FLAGS ──────────────────────────────────────────────
        target_cell = board.cells[target.r][target.c] if not board.oob(target) else None
        target_hill_obj = None
        opp_hill_dist = 999

        if target_cell and target_cell.hill_id != 0:
            target_hill_obj = board.hills.get(target_cell.hill_id)
            if target_hill_obj:
                opp_hill_dist = min(
                    (self._md(hloc, opp.loc) for hloc in target_hill_obj.cells),
                    default=999)

        dist_to_opp = mon['proximity']
        contested = opp_hill_dist <= 4 or self._phase == self.PHASE_CONTEST
        approaching = (4 < opp_hill_dist <= 10) or mon['approaching']
        danger_zone = dist_to_opp <= opp_reach + 1
        reinforce = contested or approaching

        neutral_hills_exist = any(h.controller_parity == 0 for h in board.hills.values())
        all_hills_claimed = not neutral_hills_exist and total_hills > 0
        max_layers = 2 if (contested and all_hills_claimed) else 1

        targeting_hill = not board.oob(target) and board.cells[target.r][target.c].hill_id != 0
        paint_territory_en_route = all_hills_claimed or danger_zone
        rushing_to_hill = targeting_hill and td > 2
        trail_layers = 2 if danger_zone else 1
        edge_attack = approaching and targeting_hill and mon['proximity'] <= 8

        # Opponent behavior adjustments
        if mon['aggression'] == 'FARMING' and our_hills < total_hills:
            # Opponent is farming — rush more aggressively
            rushing_to_hill = targeting_hill and td > 3  # less painting en route
        elif mon['aggression'] == 'HUNTING':
            # Opponent coming for us — stay safe
            danger_zone = dist_to_opp <= opp_reach + 2

        # ── 16. ON-HILL RETARGET ──────────────────────────────────────────
        cur_hill_id = board.cells[pos.r][pos.c].hill_id
        on_target_hill = (target_hill_obj and cur_hill_id == target_cell.hill_id) if target_cell else False

        if cur_hill_id != 0:
            hill_obj = board.hills.get(cur_hill_id)
            if hill_obj and hill_obj.controller_parity != pp:
                unpainted = self._pick_hill_cell_to_claim(
                    board, pp, hill_obj, my_dist, contested, large_map, opp, stamina)
                if unpainted:
                    target = unpainted
                    td = my_dist.get(target, 999)
                    on_target_hill = True
                    targeting_hill = True
                    if target_cell and target_cell.hill_id != 0:
                        target_hill_obj = hill_obj

        # ── 17. ANTI-OSCILLATION ESCAPE ────────────────────────────────────
        if oscillating and td > 1 and not on_target_hill:
            escape_loc = self._oscillation_escape(board, pp, pos, opp, my_dist, target)
            if escape_loc is not None:
                target = escape_loc
                td = my_dist.get(target, 999)

        # ── 18. CHOKEPOINT ─────────────────────────────────────────────────
        if td > 2 and not on_target_hill and target_cell and target_cell.hill_id != 0:
            choke, choke_dist = self._find_chokepoint_on_path(board, pos, target)
            if choke and choke_dist <= td:
                choke_cell = board.cells[choke.r][choke.c]
                opp_to_choke = self._md(opp.loc, choke)
                if opp_to_choke <= choke_dist + 4 and choke_cell.owner_parity != pp:
                    target = choke
                    td = my_dist.get(target, 999)

        # ── 19. STAMINA BUDGET ─────────────────────────────────────────────
        regen = self._estimate_regen(board, pos, pp)
        if regen >= 20:
            buf = 5
        elif regen >= 12:
            buf = 10
        elif regen >= 8:
            buf = 15
        else:
            buf = 20
        if on_target_hill:
            buf = max(5, buf - 5)
        if maze_map and td > 3:
            buf = max(buf, 15)
        hill_buf = buf

        # ── 20. STEP SELECTOR ──────────────────────────────────────────────
        def _pick_step(from_pos):
            if edge_attack and on_target_hill:
                return self._edge_hill_step(board, from_pos, target, pp, opp.loc, target_hill_obj)
            elif on_target_hill and contested:
                return self._safe_hill_step(board, from_pos, target, pp, opp.loc)
            elif danger_zone:
                return self._collision_aware_step(board, from_pos, target, pp, opp.loc)
            else:
                return self._step_toward(board, from_pos, target, pp, opp.loc,
                                         hill_target=targeting_hill)

        # ── 21. PRE-MOVE PAINTING ──────────────────────────────────────────
        actions, stamina = self._paint_hill_cells(
            board, pp, pos, actions, stamina, painted, hill_buf, reinforce, max_layers)

        if not rushing_to_hill:
            if paint_territory_en_route or maze_map:
                actions, stamina = self._paint_trail(
                    board, pp, pos, actions, stamina, painted, buf, target, trail_layers)
            elif large_map and td > 2:
                actions, stamina = self._paint_trail(
                    board, pp, pos, actions, stamina, painted, buf, target)
            elif stamina > buf + GameConstants.PAINT_STAMINA_COST and td > 2:
                actions, stamina = self._paint_trail(
                    board, pp, pos, actions, stamina, painted, buf, target)

        # ── 22. MOVEMENT (up to 4 steps; 4th only on hills) ───────────────

        def _make_move(from_pos, move_num, cur_stamina):
            """Attempt one move from from_pos. Returns (direction, new_stamina)
            or (None, cur_stamina) if not taken."""
            d = _pick_step(from_pos)
            if d is None:
                return None, cur_stamina
            nxt = from_pos + d
            if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall:
                return None, cur_stamina
            # Safety: skip dangerous 2nd/3rd moves unless on hill
            if move_num >= 2 and not targeting_hill:
                if self._is_dangerous_monitor(board, nxt, opp, pp):
                    return None, cur_stamina
            # Pre-paint neutral destination if in threat range
            nxt_c = board.cells[nxt.r][nxt.c]
            if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                    and not nxt_c.is_wall
                    and self._md(nxt, opp.loc) <= opp_reach
                    and cur_stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                actions.append(Action.Paint(nxt))
                painted[nxt] = painted.get(nxt, 0) + 1
                cur_stamina -= GameConstants.PAINT_STAMINA_COST
            use_erase = self._should_erase_step(
                board, pp, nxt, large_map, cur_stamina, buf, opp, target_hill_obj)
            # Check for beacon placement after capturing hill cells
            place_beacon = self._should_place_beacon(board, pp, nxt)
            if use_erase:
                actions.append(Action.Move(d, move_type=MoveType.ERASE, place_beacon=place_beacon))
            else:
                actions.append(Action.Move(d, place_beacon=place_beacon))
            return d, cur_stamina

        # Move 1 (always free)
        d1, stamina = _make_move(pos, 1, stamina)
        if d1 is not None:
            p1 = pos + d1

            actions, stamina = self._paint_hill_cells(
                board, pp, p1, actions, stamina, painted, hill_buf, reinforce, max_layers)

            if not rushing_to_hill and (paint_territory_en_route or maze_map or (large_map and td > 3)):
                actions, stamina = self._paint_trail(
                    board, pp, p1, actions, stamina, painted, buf, target, trail_layers)

            # Retarget sweep cell from p1 when already on hill
            if on_target_hill:
                p1_hill_id = board.cells[p1.r][p1.c].hill_id
                if p1_hill_id != 0:
                    h1 = board.hills.get(p1_hill_id)
                    if h1 and h1.controller_parity != pp:
                        nc = self._pick_hill_cell_to_claim(
                            board, pp, h1, my_dist, contested, large_map, opp, stamina)
                        if nc and nc != p1:
                            target = nc

            # Move 2 (costs EXTRA_MOVE_COST = 10)
            allow_extra_moves = True
            # When opponent is farming and we should rush, allow more moves
            if mon['aggression'] == 'FARMING' and targeting_hill:
                allow_extra_moves = True

            if allow_extra_moves and (td > 1 or on_target_hill) and stamina >= GameConstants.EXTRA_MOVE_COST + buf:
                d2, stamina = _make_move(p1, 2, stamina)
                if d2 is not None:
                    stamina -= GameConstants.EXTRA_MOVE_COST
                    p2 = p1 + d2

                    actions, stamina = self._paint_hill_cells(
                        board, pp, p2, actions, stamina, painted, hill_buf, reinforce, max_layers)

                    if not rushing_to_hill and (paint_territory_en_route or (large_map and td > 3)):
                        actions, stamina = self._paint_trail(
                            board, pp, p2, actions, stamina, painted, buf, target, trail_layers)

                    # Move 3 (costs 20)
                    if not maze_map and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + buf:
                        d3, stamina = _make_move(p2, 3, stamina)
                        if d3 is not None:
                            stamina -= GameConstants.EXTRA_MOVE_COST * 2
                            p3 = p2 + d3
                            actions, stamina = self._paint_hill_cells(
                                board, pp, p3, actions, stamina, painted,
                                max(hill_buf, 5), reinforce, max_layers)

                            # Move 4 (costs 30) — hills only
                            if (on_target_hill and not maze_map
                                    and stamina >= GameConstants.EXTRA_MOVE_COST * 3 + hill_buf):
                                p3_hid = board.cells[p3.r][p3.c].hill_id
                                if p3_hid != 0:
                                    h4 = board.hills.get(p3_hid)
                                    if h4 and h4.controller_parity != pp:
                                        nc4 = self._pick_hill_cell_to_claim(
                                            board, pp, h4, my_dist, contested, large_map, opp, stamina)
                                        if nc4 and nc4 != p3:
                                            target = nc4
                                d4, stamina = _make_move(p3, 4, stamina)
                                if d4 is not None:
                                    stamina -= GameConstants.EXTRA_MOVE_COST * 3
                                    p4 = p3 + d4
                                    actions, stamina = self._paint_hill_cells(
                                        board, pp, p4, actions, stamina, painted,
                                        max(hill_buf, 5), reinforce, max_layers)

        # ── 23. POST-MOVE PAINTING ─────────────────────────────────────────
        final = self._simpos(board, pos, actions)
        actions, stamina = self._paint_hill_cells(
            board, pp, final, actions, stamina, painted, max(hill_buf, 5), reinforce, max_layers)

        if board.cells[final.r][final.c].hill_id != 0 and stamina > buf + GameConstants.PAINT_STAMINA_COST:
            actions, stamina = self._paint_trail(
                board, pp, final, actions, stamina, painted, buf, target, trail_layers)
        elif paint_territory_en_route or (large_map and stamina > buf + GameConstants.PAINT_STAMINA_COST):
            actions, stamina = self._paint_trail(
                board, pp, final, actions, stamina, painted, buf, target, trail_layers)

        # ── 24. BEACON PLACEMENT CHECK ─────────────────────────────────────
        # After hill capture, consider placing beacon on the hill
        final_cell = board.cells[final.r][final.c]
        if final_cell.hill_id != 0:
            hill_obj = board.hills.get(final_cell.hill_id)
            if hill_obj and hill_obj.controller_parity == pp:
                if final_cell.hill_id not in self._beacon_placed_hills:
                    if self._can_place_beacon_at(board, pp, final):
                        # The next move we make from here should place a beacon
                        self._beacon_placed_hills.add(final_cell.hill_id)

        # ── 25. GUARANTEE MOVE ─────────────────────────────────────────────
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        # ── 26. VALIDATE + SAFE ENDING ─────────────────────────────────────
        actions = self._validate_paint_actions(board, pp, actions)
        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

    # ===============================================================
    # PHASE: FARM
    # ===============================================================

    def _play_farm(self, board, pp, me, opp, my_dist, opp_dist, stamina):
        """Farm phase: paint locally to build regen before committing to distant hills.
        This is the SINGLE MOST IMPORTANT improvement — prevents stamina death on large maps."""
        pos = me.loc
        actions = []
        painted = {}
        buf = 15
        mon = self._opp_monitor

        # Paint as many adjacent cells as possible
        actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, buf)

        # Move toward nearest unpainted cell that's within 3 steps
        best_target = None
        best_score = -9999

        for loc, d in my_dist.items():
            if d > 4 or d == 0:
                continue
            c = board.cells[loc.r][loc.c]
            if c.is_wall or c.owner_parity == -pp:
                continue

            score = -d * 5
            if c.owner_parity == 0:
                score += 30
            elif c.owner_parity == pp and abs(c.paint_value) < GameConstants.MAX_PAINT_VALUE:
                score += 5
            else:
                continue

            # Prefer cells toward hills (so we farm in the right direction)
            for hid, hill in board.hills.items():
                if hill.controller_parity == pp:
                    continue
                for hloc in hill.cells:
                    if self._md(loc, hloc) < self._md(pos, hloc):
                        score += 8
                        break
                break

            # Prefer cells that build 5x5 territory
            if d <= 2:
                score += 15

            if score > best_score:
                best_score = score
                best_target = loc

        if best_target is None:
            # No good farm target — switch to rush
            return self._validate_paint_actions(board, pp,
                self._play_territory(board, pp, me, opp, my_dist, stamina))

        d1 = self._step_toward(board, pos, best_target, pp, opp.loc)
        if d1:
            actions.append(Action.Move(d1))
            p1 = pos + d1
            actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, buf)

            # Second move if we have stamina
            if stamina >= GameConstants.EXTRA_MOVE_COST + buf + GameConstants.PAINT_STAMINA_COST:
                # Pick new target from p1
                nt = None
                ns = -9999
                for d in Direction.cardinals():
                    nxt = p1 + d
                    if board.oob(nxt):
                        continue
                    c = board.cells[nxt.r][nxt.c]
                    if c.is_wall:
                        continue
                    if c.owner_parity == 0:
                        sc = 20
                    elif c.owner_parity == pp and abs(c.paint_value) < GameConstants.MAX_PAINT_VALUE:
                        sc = 5
                    else:
                        continue
                    if c.owner_parity == pp:
                        sc += 10  # safe
                    if sc > ns:
                        ns = sc
                        nt = d
                if nt:
                    nxt = p1 + nt
                    if not self._is_dangerous_monitor(board, nxt, opp, pp):
                        actions.append(Action.Move(nt))
                        stamina -= GameConstants.EXTRA_MOVE_COST
                        p2 = nxt
                        actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, buf)
        else:
            # Can't move toward target, just move somewhere safe
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        actions = self._validate_paint_actions(board, pp, actions)
        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

    # ===============================================================
    # PHASE: DEFEND
    # ===============================================================

    def _play_defend(self, board, pp, me, opp, my_dist, opp_dist, stamina):
        """Defend phase: rush to threatened hill and reinforce paint."""
        actions = []
        painted = {}
        pos = me.loc
        mon = self._opp_monitor
        buf = 10

        # Find most threatened hill
        worst_hid = None
        worst_urgency = 0
        worst_target = None

        for hid in me.controlled_hills:
            hill = board.hills[hid]
            hs = len(hill.cells)
            opp_cells = sum(1 for h in hill.cells
                            if board.cells[h.r][h.c].owner_parity == -pp)
            if opp_cells == 0:
                continue

            needed = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD) + 1
            urgency = opp_cells / hs
            if opp_cells >= needed - 1:
                urgency += 0.5

            total_hills = len(board.hills)
            dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD)
            their_hills = len(opp.controlled_hills)
            if their_hills + 1 >= dom_needed:
                urgency += 0.3

            if urgency > worst_urgency:
                worst_urgency = urgency
                worst_hid = hid
                # Target the opponent's painted cell closest to us
                nd = 999
                nl = None
                for hloc in hill.cells:
                    hc = board.cells[hloc.r][hloc.c]
                    if hc.owner_parity == -pp:
                        d = my_dist.get(hloc, 999)
                        if d < nd:
                            nd = d
                            nl = hloc
                    elif hc.owner_parity == 0:
                        d = my_dist.get(hloc, 999)
                        if d < nd:
                            nd = d
                            nl = hloc
                worst_target = nl

        if worst_target is None:
            return None  # No defense needed, fall through to RUSH

        td = my_dist.get(worst_target, 999)

        # Check beacon teleport for defense
        beacon_action = self._try_beacon_travel(board, pp, worst_target)
        if beacon_action is not None and td > 4:
            actions.append(beacon_action)
            new_pos = beacon_action.beacon_target
            my_dist_new = self._bfs(board, new_pos, pp)
            # Repaint at new location
            actions, stamina = self._paint_hill_cells(
                board, pp, new_pos, actions, stamina, painted, buf, True, 3)
            final = self._simpos(board, pos, actions)
            actions = self._validate_paint_actions(board, pp, actions)
            actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)
            if not any(isinstance(a, Action.Move) for a in actions):
                fb = self._fallback_move(board, pp, me, opp)
                if fb:
                    actions.append(fb)
                else:
                    for d in Direction.cardinals():
                        nxt = pos + d
                        if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                            actions.append(Action.Move(d))
                            break
            return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

        # Rush to threatened hill with up to 3 moves
        # Reinforce paint to 3+ layers on cells near hill edge
        hill_obj = board.hills.get(worst_hid)

        # Paint hill cells we're adjacent to
        actions, stamina = self._paint_hill_cells(
            board, pp, pos, actions, stamina, painted, buf, True, 3)

        d1 = self._step_toward(board, pos, worst_target, pp, opp.loc, hill_target=True)
        if d1:
            nxt = pos + d1
            if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                # Use erase on thick opponent paint within hill
                use_erase = False
                nxt_c = board.cells[nxt.r][nxt.c]
                if (nxt_c.owner_parity == -pp and nxt_c.hill_id != 0
                        and abs(nxt_c.paint_value) >= 2
                        and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + buf + 20):
                    use_erase = True

                if use_erase:
                    actions.append(Action.Move(d1, move_type=MoveType.ERASE))
                else:
                    actions.append(Action.Move(d1))
                p1 = nxt

                actions, stamina = self._paint_hill_cells(
                    board, pp, p1, actions, stamina, painted, buf, True, 3)

                # Move 2
                if td > 1 and stamina >= GameConstants.EXTRA_MOVE_COST + buf:
                    d2 = self._step_toward(board, p1, worst_target, pp, opp.loc, hill_target=True)
                    if d2:
                        nxt2 = p1 + d2
                        if not board.oob(nxt2) and not board.cells[nxt2.r][nxt2.c].is_wall:
                            nxt2_c = board.cells[nxt2.r][nxt2.c]
                            use_erase2 = False
                            if (nxt2_c.owner_parity == -pp and nxt2_c.hill_id != 0
                                    and abs(nxt2_c.paint_value) >= 2
                                    and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + GameConstants.EXTRA_MOVE_COST + buf):
                                use_erase2 = True

                            if use_erase2:
                                actions.append(Action.Move(d2, move_type=MoveType.ERASE))
                            else:
                                actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                            p2 = nxt2

                            actions, stamina = self._paint_hill_cells(
                                board, pp, p2, actions, stamina, painted, buf, True, 3)

                            # Move 3
                            if td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + buf:
                                d3 = self._step_toward(board, p2, worst_target, pp, opp.loc, hill_target=True)
                                if d3:
                                    nxt3 = p2 + d3
                                    if not board.oob(nxt3) and not board.cells[nxt3.r][nxt3.c].is_wall:
                                        actions.append(Action.Move(d3))
                                        stamina -= GameConstants.EXTRA_MOVE_COST * 2
                                        p3 = nxt3
                                        actions, stamina = self._paint_hill_cells(
                                            board, pp, p3, actions, stamina, painted, buf, True, 3)
        else:
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        actions = self._validate_paint_actions(board, pp, actions)
        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

    # ===============================================================
    # PHASE: ENDGAME
    # ===============================================================

    def _play_endgame(self, board, pp, me, opp, my_dist, opp_dist, stamina):
        """Endgame phase: play based on tiebreak position."""
        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)
        our_territory = board.get_territory_count(pp)
        their_territory = board.get_territory_count(-pp)
        pos = me.loc
        actions = []
        painted = {}
        mon = self._opp_monitor

        # Determine if we're winning the tiebreak
        winning_hills = our_hills > their_hills
        winning_territory = our_territory > their_territory
        winning = winning_hills or (our_hills == their_hills and winning_territory)

        if winning:
            # DEFENSIVE ENDGAME: reinforce hills, avoid risks
            buf = 15

            # First priority: reinforce our hill cells to max layers
            for hid in me.controlled_hills:
                hill = board.hills[hid]
                for hloc in hill.cells:
                    if self._md(pos, hloc) <= 1:
                        hc = board.cells[hloc.r][hloc.c]
                        if hc.owner_parity == pp and abs(hc.paint_value) < GameConstants.MAX_PAINT_VALUE:
                            if stamina >= GameConstants.PAINT_STAMINA_COST + buf:
                                cur_val = abs(hc.paint_value) + painted.get(hloc, 0)
                                if cur_val < GameConstants.MAX_PAINT_VALUE:
                                    actions.append(Action.Paint(hloc))
                                    painted[hloc] = painted.get(hloc, 0) + 1
                                    stamina -= GameConstants.PAINT_STAMINA_COST

            # Check for threatened hills
            defend_target = None
            for hid in me.controlled_hills:
                hill = board.hills[hid]
                hs = len(hill.cells)
                opp_cells = sum(1 for h in hill.cells
                                if board.cells[h.r][h.c].owner_parity == -pp)
                if opp_cells > 0:
                    nd = 999
                    nl = None
                    for hloc in hill.cells:
                        if board.cells[hloc.r][hloc.c].owner_parity == -pp:
                            d = my_dist.get(hloc, 999)
                            if d < nd:
                                nd = d
                                nl = hloc
                    if nl and (defend_target is None or nd < my_dist.get(defend_target, 999)):
                        defend_target = nl

            if defend_target:
                target = defend_target
            else:
                # Expand territory defensively (paint near our owned cells)
                target = self._frontier_target(board, pp, my_dist, pos)

            if target:
                d1 = self._step_toward(board, pos, target, pp, opp.loc)
                if d1:
                    actions.append(Action.Move(d1))
                    p1 = pos + d1

                    # Reinforce paint around new position
                    actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, buf)

                    if stamina >= GameConstants.EXTRA_MOVE_COST + buf:
                        d2 = self._step_toward(board, p1, target, pp, opp.loc)
                        if d2 and not self._is_dangerous_monitor(board, p1 + d2, opp, pp):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                            p2 = p1 + d2
                            actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, buf)

            if not any(isinstance(a, Action.Move) for a in actions):
                fb = self._fallback_move(board, pp, me, opp)
                if fb:
                    actions.append(fb)
                else:
                    for d in Direction.cardinals():
                        nxt = pos + d
                        if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                            actions.append(Action.Move(d))
                            break

            actions = self._validate_paint_actions(board, pp, actions)
            actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

            if not any(isinstance(a, Action.Move) for a in actions):
                fb = self._fallback_move(board, pp, me, opp)
                if fb:
                    actions.append(fb)
                else:
                    for d in Direction.cardinals():
                        nxt = pos + d
                        if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                            actions.append(Action.Move(d))
                            break

            return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

        else:
            # AGGRESSIVE ENDGAME: territory expansion, push into opponent's zone
            # Fall through to normal RUSH logic but with territory pressure
            return None  # Let main play() handle as RUSH

    # ===============================================================
    # TERRITORY PRESSURE (when ahead)
    # ===============================================================

    def _territory_pressure_target(self, board, pp, me, opp, my_dist, opp_dist):
        """When we control more hills, find cells to expand toward opponent."""
        opp_pos = opp.loc
        best = None
        best_score = -9999

        for loc, d in my_dist.items():
            if d > 8 or d == 0:
                continue
            c = board.cells[loc.r][loc.c]
            if c.is_wall or c.owner_parity == pp:
                continue

            score = -d * 3

            # Prefer cells near opponent's territory
            adj_opp = 0
            for dr in Direction.cardinals():
                adj = loc + dr
                if not board.oob(adj) and board.cells[adj.r][adj.c].owner_parity == -pp:
                    adj_opp += 1
            if adj_opp > 0:
                score += adj_opp * 15

            # Prefer cells toward opponent
            if self._md(loc, opp_pos) < self._md(me.loc, opp_pos):
                score += 10

            if c.owner_parity == 0:
                score += 15
            elif c.owner_parity == -pp:
                score += 5  # Can't paint these but can erase

            if score > best_score:
                best_score = score
                best = loc

        return best

    # ===============================================================
    # OSCILLATION DETECTION & ESCAPE
    # ===============================================================

    def _is_oscillating(self):
        """True when recent positions show a 2-cell (or smaller) cycle."""
        if len(self._position_history) < 5:
            return False
        recent = self._position_history[-5:]
        return len(set(recent)) <= 2

    def _oscillation_escape(self, board, pp, pos, opp, my_dist, current_target):
        """When oscillating, return a nearby cell we haven't visited recently."""
        recent = set(self._position_history[-5:])
        best = None
        best_score = -9999

        target_d = my_dist.get(current_target, 999) if current_target else 999

        for d in Direction.cardinals():
            nxt = pos + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.beacon_parity == -pp:
                continue
            if nxt == opp.loc and c.owner_parity != pp:
                continue
            if (nxt.r, nxt.c) in recent:
                continue

            nxt_d = my_dist.get(nxt, 999)
            progress = target_d - nxt_d

            score = progress * 10
            if c.owner_parity == pp:
                score += 3
            if c.hill_id != 0:
                h = board.hills.get(c.hill_id)
                if h and h.controller_parity != pp:
                    score += 15

            if score > best_score:
                best_score = score
                best = nxt

        return best

    # ===============================================================
    # MST-INSPIRED HILL VISIT SEQUENCE (BFS-corrected)
    # ===============================================================

    def _refresh_hill_sequence(self, board, pp, me, opp, my_dist, opp_dist):
        """Compute/refresh the hill visit order using greedy nearest-neighbor.

        Improvement over oppav5: uses full BFS for subsequent hops instead of
        Manhattan distance from centroid. This fixes approximation weakness
        on maze maps.
        """
        sig = frozenset((hid, h.controller_parity) for hid, h in board.hills.items())
        if sig == self._hill_sig:
            return
        self._hill_sig = sig

        uncaptured = [(hid, h) for hid, h in board.hills.items()
                      if h.controller_parity != pp]
        if not uncaptured:
            self._hill_sequence = []
            return

        order = []
        remaining = list(uncaptured)
        cur_dist = my_dist  # BFS from current position (accurate for first hop)

        while remaining:
            best_hid = None
            best_score = 9999.0

            for hid, hill in remaining:
                my_d = min((cur_dist.get(loc, 999) for loc in hill.cells), default=999)
                if my_d == 999:
                    continue
                opp_d = min((opp_dist.get(loc, 999) for loc in hill.cells), default=999)
                race_adv = max(0, opp_d - my_d)
                score = float(my_d) - race_adv * 0.5
                if hill.controller_parity == -pp:
                    score -= 20.0
                if score < best_score:
                    best_score = score
                    best_hid = hid

            if best_hid is None:
                order.extend(hid for hid, _ in remaining)
                break

            order.append(best_hid)
            picked = next(h for hid, h in remaining if hid == best_hid)
            remaining = [(hid, h) for hid, h in remaining if hid != best_hid]

            # For subsequent hops: use BFS from picked hill's nearest cell
            # This is the key improvement over oppav5's Manhattan approximation
            if remaining:
                cells = list(picked.cells)
                # Find the cell of the picked hill that's closest to us
                best_cell = min(cells, key=lambda l: cur_dist.get(l, 999))
                # BFS from that cell for next hop distance calculation
                cache_key = (best_cell.r, best_cell.c)
                if cache_key in self._bfs_cache:
                    cur_dist = self._bfs_cache[cache_key]
                else:
                    cur_dist = self._bfs_raw(board, best_cell)
                    self._bfs_cache[cache_key] = cur_dist

        self._hill_sequence = order

    def _pick_attack_target_sequenced(self, board, pp, me, opp, my_dist, opp_dist, dom_needed):
        """Pick attack target following the MST-computed hill visit sequence."""
        for hid in self._hill_sequence:
            if hid not in board.hills:
                continue
            hill = board.hills[hid]
            if hill.controller_parity == pp:
                continue
            best_loc = None
            best_d = 999
            for hloc in hill.cells:
                if board.cells[hloc.r][hloc.c].owner_parity != pp:
                    d = my_dist.get(hloc, 999)
                    if d < best_d:
                        best_d = d
                        best_loc = hloc
            if best_loc:
                return best_loc, hid

        return self._pick_attack_target(board, pp, me, opp, my_dist, opp_dist, dom_needed)

    # ===============================================================
    # BEACON STRATEGY
    # ===============================================================

    def _can_place_beacon_at(self, board, pp, pos):
        """Check if 7/9 cells in 3x3 around pos are owned by us."""
        count = 0
        total = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = pos.r + dr, pos.c + dc
                if 0 <= r < board.board_size.r and 0 <= c < board.board_size.c:
                    total += 1
                    cell = board.cells[r][c]
                    if cell.owner_parity == pp and cell.beacon_parity == 0:
                        count += 1
        return count >= GameConstants.BEACON_REQUIREMENT_Q

    def _should_place_beacon(self, board, pp, pos):
        """Decide if we should place a beacon when moving to this cell."""
        if board.oob(pos):
            return False
        cell = board.cells[pos.r][pos.c]

        # Only place on hill cells we own (for strategic value)
        if cell.hill_id == 0:
            return False

        # Don't place if there's already a beacon here
        if cell.beacon_parity != 0:
            return False

        # Check if we meet the 7/9 requirement
        if not self._can_place_beacon_at(board, pp, pos):
            return False

        # Place beacon on hills we've captured
        hill = board.hills.get(cell.hill_id)
        if hill and hill.controller_parity == pp:
            if cell.hill_id not in self._beacon_placed_hills:
                self._beacon_placed_hills.add(cell.hill_id)
                return True

        return False

    def _try_beacon_travel(self, board, pp, target):
        """If we're on a beacon and another beacon is closer to target, teleport."""
        me = board.get_player(pp)
        pos = me.loc
        cell = board.cells[pos.r][pos.c]
        if cell.beacon_parity != pp:
            return None  # Not on our beacon

        # Find all our other beacons
        our_beacons = []
        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                loc = Location(r, c)
                if loc != pos and board.cells[r][c].beacon_parity == pp:
                    our_beacons.append(loc)

        if not our_beacons:
            return None

        # Find beacon closest to target
        my_dist_to_target = abs(pos.r - target.r) + abs(pos.c - target.c)
        best_beacon = None
        best_dist = my_dist_to_target
        for bloc in our_beacons:
            d = abs(bloc.r - target.r) + abs(bloc.c - target.c)
            if d < best_dist - 3:  # Only teleport if significantly closer
                best_dist = d
                best_beacon = bloc

        if best_beacon:
            return Action.Move(Direction.UP, move_type=MoveType.BEACON_TRAVEL,
                               beacon_target=best_beacon)
        return None

    # ===============================================================
    # ACTION VALIDATION
    # ===============================================================

    def _validate_paint_actions(self, board, pp, actions):
        """Remove any Paint actions that would be illegal given the current board."""
        valid = []
        MV = GameConstants.MAX_PAINT_VALUE
        for a in actions:
            if isinstance(a, Action.Paint):
                loc = a.location
                if board.oob(loc):
                    continue
                c = board.cells[loc.r][loc.c]
                if c.is_wall or c.beacon_parity != 0:
                    continue
                if c.owner_parity == -pp:
                    continue
                if c.owner_parity == pp and abs(c.paint_value) >= MV:
                    continue
                valid.append(a)
            else:
                valid.append(a)
        return valid

    # ===============================================================
    # OPPONENT MONITOR (extended)
    # ===============================================================

    def _update_opp_monitor(self, board, pp, me, opp):
        """Update opponent tracking data every turn — extended with pattern detection."""
        mon = self._opp_monitor
        dist = self._md(me.loc, opp.loc)

        if mon['prev_dist'] is not None:
            mon['approaching'] = dist < mon['prev_dist']
        mon['prev_dist'] = dist
        mon['proximity'] = dist
        mon['stamina'] = opp.stamina

        # Track opponent position history (16 turns)
        self._opp_positions.append((opp.loc.r, opp.loc.c))
        if len(self._opp_positions) > 16:
            self._opp_positions.pop(0)

        # Track stamina history
        self._opp_stamina_history.append(opp.stamina)
        if len(self._opp_stamina_history) > 16:
            self._opp_stamina_history.pop(0)

        # Compute direction vector (average movement direction over last 8 turns)
        if len(self._opp_positions) >= 2:
            recent = self._opp_positions[-8:] if len(self._opp_positions) >= 8 else self._opp_positions
            dr_sum = 0
            dc_sum = 0
            count = 0
            for i in range(1, len(recent)):
                dr_sum += recent[i][0] - recent[i-1][0]
                dc_sum += recent[i][1] - recent[i-1][1]
                count += 1
            if count > 0:
                mon['direction_r'] = dr_sum / count
                mon['direction_c'] = dc_sum / count

        # Classify aggression
        mon['aggression'] = self._classify_aggression(board, pp, me, opp)

        # Classify stamina trend
        mon['stamina_trend'] = self._classify_stamina_trend()

        # Original monitor logic
        opp_regen = self._estimate_opp_regen(board, pp, opp)
        mon['est_regen'] = opp_regen
        mon['prev_stamina'] = opp.stamina

        effective_stamina = min(opp.max_stamina, opp.stamina + opp_regen)
        remaining = effective_stamina
        moves = 1
        extra = 1
        while True:
            cost = GameConstants.EXTRA_MOVE_COST * extra
            if remaining < cost:
                break
            remaining -= cost
            moves += 1
            extra += 1
            if moves >= 4:
                break
        mon['max_moves'] = moves
        mon['threat_range'] = moves

    def _classify_aggression(self, board, pp, me, opp):
        """Classify opponent behavior: FARMING, RUSHING, or HUNTING."""
        if len(self._opp_positions) < 4:
            return 'FARMING'

        recent = self._opp_positions[-8:] if len(self._opp_positions) >= 8 else self._opp_positions

        # Calculate total movement
        total_movement = 0
        for i in range(1, len(recent)):
            total_movement += abs(recent[i][0] - recent[i-1][0]) + abs(recent[i][1] - recent[i-1][1])

        avg_movement = total_movement / (len(recent) - 1) if len(recent) > 1 else 0

        # Check if moving toward us
        if len(recent) >= 4:
            start_dist = abs(recent[0][0] - me.loc.r) + abs(recent[0][1] - me.loc.c)
            end_dist = abs(recent[-1][0] - me.loc.r) + abs(recent[-1][1] - me.loc.c)
            approaching_us = end_dist < start_dist - 2

            if approaching_us and avg_movement >= 1.0:
                return 'HUNTING'

        # Check if moving toward any uncaptured hill
        for hid, hill in board.hills.items():
            if hill.controller_parity == -pp:
                continue
            for hloc in hill.cells:
                if len(recent) >= 4:
                    start_dist = abs(recent[0][0] - hloc.r) + abs(recent[0][1] - hloc.c)
                    end_dist = abs(recent[-1][0] - hloc.r) + abs(recent[-1][1] - hloc.c)
                    if end_dist < start_dist - 3 and avg_movement >= 1.0:
                        return 'RUSHING'
                break  # Only check first cell of each hill for speed

        # Low movement = farming
        if avg_movement < 0.8:
            return 'FARMING'

        return 'RUSHING'  # Default to RUSHING if moving but not toward us

    def _classify_stamina_trend(self):
        """Classify opponent stamina trend over recent history."""
        if len(self._opp_stamina_history) < 4:
            return 'STABLE'

        recent = self._opp_stamina_history[-8:] if len(self._opp_stamina_history) >= 8 else self._opp_stamina_history

        # Compare first half to second half
        mid = len(recent) // 2
        first_half_avg = sum(recent[:mid]) / mid
        second_half_avg = sum(recent[mid:]) / (len(recent) - mid)

        diff = second_half_avg - first_half_avg
        if diff > 10:
            return 'GAINING'
        elif diff < -10:
            return 'DEPLETING'
        return 'STABLE'

    def _estimate_opp_regen(self, board, pp, opp):
        """Estimate opponent's stamina regen from territory and adjacency."""
        regen = GameConstants.BASE_STAMINA_REGEN
        radius = GameConstants.ADJACENCY_RADIUS
        local_friendly = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = opp.loc.r + dr, opp.loc.c + dc
                if 0 <= r < board.board_size.r and 0 <= c < board.board_size.c:
                    if board.cells[r][c].owner_parity == -pp:
                        local_friendly += 1
        regen += local_friendly * GameConstants.ADJACENT_REGEN_BONUS
        opp_territory = board.get_territory_count(-pp)
        regen += min(opp_territory // GameConstants.GLOBAL_PAINT_REGEN_RATIO,
                     GameConstants.GLOBAL_PAINT_REGEN_CAP)
        if board.turn_count > GameConstants.GLOBAL_DECAY_TURN_THRESHOLD:
            delta = board.turn_count - GameConstants.GLOBAL_DECAY_TURN_THRESHOLD - 1
            intervals = (delta // GameConstants.GLOBAL_DECAY_INTERVAL) + 1
            regen -= intervals * GameConstants.GLOBAL_DECAY_REGEN_PENALTY
            regen = max(0, regen)
        return regen

    def _is_collision_safe(self, board, loc, pp, opp):
        """Check if ending at loc is safe from losing a collision."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]
        mon = self._opp_monitor
        if c.owner_parity == pp and c.beacon_parity != -pp:
            return True
        dist = self._md(loc, opp.loc)
        threat = mon['threat_range']
        if dist > threat + 1:
            return True
        if mon['approaching'] and mon['stamina'] > 60:
            threat += 1
        if dist > threat:
            return True
        return False

    def _is_dangerous_monitor(self, board, loc, opp, pp):
        """Enhanced danger check using opponent monitor."""
        if board.oob(loc):
            return True
        c = board.cells[loc.r][loc.c]
        if c.is_wall:
            return True
        mon = self._opp_monitor
        dist = self._md(loc, opp.loc)
        if c.owner_parity == -pp and dist <= mon['threat_range']:
            return True
        if c.owner_parity == 0 and dist <= 2 and mon['approaching']:
            return True
        return False

    # ===============================================================
    # ERASE HILL TRAVERSAL
    # ===============================================================

    def _should_erase_step(self, board, pp, loc, large_map, stamina, buf, opp, target_hill_obj):
        """Decide if we should use ERASE move type when stepping onto a cell."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]
        if c.owner_parity != -pp or c.hill_id == 0:
            return False
        layers = abs(c.paint_value)
        erase_cost = GameConstants.ERASE_STEP_EXTRA_COST
        if not large_map:
            return layers >= 3 and stamina >= erase_cost + buf + 30
        return layers >= 2 and stamina >= erase_cost + buf + 20

    def _pick_hill_cell_to_claim(self, board, pp, hill, my_dist, contested, large_map, opp, stamina):
        """Pick next hill cell to visit using sweep-aware scoring."""
        best = None
        best_score = -9999
        me_loc = board.get_player(pp).loc
        mon = self._opp_monitor

        for hloc in hill.cells:
            hc = board.cells[hloc.r][hloc.c]
            if hc.owner_parity == pp:
                continue
            if hloc == me_loc:
                continue

            d = my_dist.get(hloc, 999)
            if d > 50:
                continue

            layers = abs(hc.paint_value)
            opp_dist_to_cell = self._md(hloc, opp.loc)

            if hc.owner_parity == 0:
                score = 100 - d * 3
            elif hc.owner_parity == -pp:
                if large_map and layers >= 2 and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + 30:
                    score = 80 - d * 3 - layers * 2
                else:
                    score = 60 - d * 3 - layers * 5
            else:
                continue

            if d == 1:
                score += 50
            elif d == 2:
                score += 15

            if (contested or mon['approaching']) and opp_dist_to_cell <= 2:
                score -= 60
            elif mon['approaching'] and opp_dist_to_cell <= 3:
                score -= 30

            if self._is_hill_edge(board, hloc, hill):
                score += 15

            if score > best_score:
                best_score = score
                best = hloc

        return best

    def _is_hill_edge(self, board, loc, hill):
        """Check if a hill cell is on the edge (adjacent to non-hill cell)."""
        for d in Direction.cardinals():
            nxt = loc + d
            if board.oob(nxt):
                return True
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or c.hill_id != hill.id:
                return True
        return False

    # ===============================================================
    # EDGE-BASED HILL ATTACK
    # ===============================================================

    def _edge_hill_step(self, board, start, target, pp, opp_loc, hill_obj):
        """Step toward target on a hill while preferring edge cells."""
        if start == target:
            return None
        best = None
        best_score = -9999
        mon = self._opp_monitor

        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.beacon_parity == -pp:
                continue
            if nxt == opp_loc and c.owner_parity != pp:
                continue
            if c.owner_parity == -pp and self._md(nxt, opp_loc) <= 1:
                continue

            score = -self._md(nxt, target) * 10

            if c.owner_parity == pp and c.beacon_parity != -pp:
                score += 30
            elif c.owner_parity == 0:
                score += 10
            elif c.owner_parity == -pp:
                score -= 15

            if hill_obj and c.hill_id == hill_obj.id:
                if self._is_hill_edge(board, nxt, hill_obj):
                    score += 20

            opp_d = self._md(nxt, opp_loc)
            if opp_d <= 2:
                score -= 25
            elif opp_d >= 4:
                score += 10

            if nxt == target:
                score += 50

            if score > best_score:
                best_score = score
                best = d

        return best

    # ===============================================================
    # TARGET SELECTION
    # ===============================================================

    def _pick_attack_target(self, board, pp, me, opp, my_dist, opp_dist, dom_needed):
        """Race-aware hill scoring. Used as fallback when sequence is empty."""
        best = None
        best_hid = None
        best_score = -9999
        our_hills = len(me.controlled_hills)
        committed = self._committed_hill_id
        if committed is not None:
            if committed not in board.hills or board.hills[committed].controller_parity == pp:
                self._committed_hill_id = None
                committed = None

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            hill_dist = 999
            nearest_dist = 999
            nearest_loc = None
            our_cells = 0
            opp_cells = 0

            for hloc in hill.cells:
                hc = board.cells[hloc.r][hloc.c]
                d = my_dist.get(hloc, 999)
                if d < hill_dist:
                    hill_dist = d
                if hc.owner_parity == pp:
                    our_cells += 1
                else:
                    if hc.owner_parity == -pp:
                        opp_cells += 1
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_loc = hloc

            if nearest_loc is None:
                continue

            hs = len(hill.cells)
            needed = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD) + 1
            to_paint = max(0, needed - our_cells)
            our_cost = nearest_dist + to_paint * 1.5

            opp_nearest = 999
            opp_to_paint = max(0, needed - opp_cells)
            for hloc in hill.cells:
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == -pp:
                    continue
                d = opp_dist.get(hloc, 999)
                if d < opp_nearest:
                    opp_nearest = d
            opp_cost = opp_nearest + opp_to_paint * 1.5

            race_adv = min(opp_cost - our_cost, 30)
            size_bonus = max(0, 10 - hs) * 3
            close_to_dom = 80 if (our_hills + 1 >= dom_needed) else 0
            steal_bonus = 40 if hill.controller_parity == -pp else 0
            late_bonus = 30 if board.current_round > 400 else 0
            commit_bonus = 200 if (hid == committed) else 0
            proximity_bonus = max(0, 100 - hill_dist * 20)
            opp_on_hill = any(self._md(hloc, opp.loc) == 0 for hloc in hill.cells)
            camping_penalty = -80 if opp_on_hill else 0

            if our_hills == 0:
                score = (proximity_bonus * 5 + commit_bonus + size_bonus
                         - to_paint * 5 + race_adv * 2)
            elif our_hills == 1:
                contested_bonus = 0
                if hill.controller_parity is None and opp_nearest < 999:
                    dist_diff = abs(our_cost - opp_cost)
                    if dist_diff <= 5:
                        contested_bonus = max(0, 80 - dist_diff * 15)
                score = (race_adv * 4 + size_bonus + close_to_dom + steal_bonus
                         + late_bonus + commit_bonus + proximity_bonus * 2
                         + contested_bonus + camping_penalty - to_paint * 5)
            else:
                score = (race_adv * 8 + size_bonus + close_to_dom + steal_bonus
                         + late_bonus + commit_bonus + proximity_bonus
                         + camping_penalty - to_paint * 5)

            if score > best_score:
                best_score = score
                best = nearest_loc
                best_hid = hid

        return best, best_hid

    def _defend_target(self, board, pp, me, opp, my_dist, dom_needed, their_hills):
        """Find our most threatened controlled hill to defend.
        Improved: lower urgency threshold (0.10 vs 0.15)."""
        worst = None
        worst_urgency = 0

        for hid in me.controlled_hills:
            hill = board.hills[hid]
            hs = len(hill.cells)
            opp_cells = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            if opp_cells == 0:
                continue

            needed = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD) + 1
            urgency = opp_cells / hs
            if opp_cells >= needed - 1:
                urgency += 0.5
            if their_hills + 1 >= dom_needed:
                urgency += 0.3

            if urgency > worst_urgency:
                worst_urgency = urgency
                nd = 999
                nl = None
                for hloc in hill.cells:
                    if board.cells[hloc.r][hloc.c].owner_parity == -pp:
                        d = my_dist.get(hloc, 999)
                        if d < nd:
                            nd = d
                            nl = hloc
                    elif board.cells[hloc.r][hloc.c].owner_parity == 0:
                        d = my_dist.get(hloc, 999)
                        if d < nd:
                            nd = d
                            nl = hloc
                worst = nl

        # Lowered threshold from 0.15 to 0.10
        return worst if worst_urgency >= 0.10 else None

    def _powerup_nearby(self, board, my_dist, max_range):
        """Find closest powerup within range."""
        best = None
        best_d = max_range + 1
        for loc, d in my_dist.items():
            if d >= best_d:
                continue
            if board.cells[loc.r][loc.c].powerup:
                best_d = d
                best = loc
        return best

    # ===============================================================
    # PAINTING
    # ===============================================================

    def _paint_hill_cells(self, board, pp, pos, actions, stamina, painted, buf, contested=False, max_layers=None):
        """Paint adjacent unpainted hill cells."""
        COST = GameConstants.PAINT_STAMINA_COST
        MV = GameConstants.MAX_PAINT_VALUE
        cap = max_layers if max_layers is not None else MV

        while stamina - COST >= buf:
            best = None
            best_score = -1

            for d in Direction.cardinals():
                t = pos + d
                if board.oob(t):
                    continue
                c = board.cells[t.r][t.c]
                if c.is_wall or c.beacon_parity != 0 or c.owner_parity == -pp:
                    continue
                if c.hill_id == 0:
                    continue

                base = abs(c.paint_value) if c.owner_parity == pp else 0
                added = painted.get(t, 0)
                if base + added >= cap:
                    continue

                is_new = (c.owner_parity == 0 and added == 0)

                if is_new:
                    score = 200
                elif contested:
                    score = 80 + (cap - base - added) * 10
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best = t

            if best is None:
                break
            actions.append(Action.Paint(best))
            painted[best] = painted.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    def _paint_trail(self, board, pp, pos, actions, stamina, painted, buf, toward=None, reinforce_to=1):
        """Paint at most 1 adjacent cell."""
        COST = GameConstants.PAINT_STAMINA_COST
        if stamina - COST < buf:
            return actions, stamina

        best = None
        best_score = -1

        for d in Direction.cardinals():
            t = pos + d
            if board.oob(t):
                continue
            c = board.cells[t.r][t.c]
            if c.is_wall or c.beacon_parity != 0:
                continue

            base = abs(c.paint_value) if c.owner_parity == pp else 0
            added = painted.get(t, 0)
            eff = base + added

            if c.owner_parity == 0 and added == 0:
                score = 50
                if c.hill_id != 0:
                    score = 150
            elif c.owner_parity == pp and eff < reinforce_to:
                score = 20
                if c.hill_id != 0:
                    score = 80
            elif c.owner_parity == -pp:
                continue  # Can't paint opponent cells
            else:
                continue

            if toward and self._md(t, toward) < self._md(pos, toward):
                score += 8
            if score > best_score:
                best_score = score
                best = t

        if best:
            actions.append(Action.Paint(best))
            painted[best] = painted.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    def _paint_expand(self, board, pp, pos, actions, stamina, painted, buf, toward=None):
        """Paint best adjacent cell for territory expansion."""
        COST = GameConstants.PAINT_STAMINA_COST
        MV = GameConstants.MAX_PAINT_VALUE

        while stamina - COST >= buf:
            best = None
            best_score = -1

            for d in Direction.cardinals():
                t = pos + d
                if board.oob(t):
                    continue
                c = board.cells[t.r][t.c]
                if c.is_wall or c.beacon_parity != 0 or c.owner_parity == -pp:
                    continue

                base = abs(c.paint_value) if c.owner_parity == pp else 0
                added = painted.get(t, 0)
                if base + added >= MV:
                    continue

                is_new = (c.owner_parity == 0 and added == 0)
                is_hill = (c.hill_id != 0)

                if is_hill and is_new:
                    score = 200
                elif is_hill:
                    score = 100 + (MV - base - added) * 10
                elif is_new:
                    score = 50
                else:
                    score = 5 + (MV - base - added) * 2

                if toward and is_new and self._md(t, toward) < self._md(pos, toward):
                    score += 12

                if score > best_score:
                    best_score = score
                    best = t

            if best is None:
                break
            actions.append(Action.Paint(best))
            painted[best] = painted.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    # ===============================================================
    # TERRITORY PLAY
    # ===============================================================

    def _play_territory(self, board, pp, me, opp, my_dist, stamina):
        """Expand territory for tiebreak."""
        actions = []
        painted = {}
        pos = me.loc
        mon = self._opp_monitor

        if len(me.controlled_hills) > 0:
            for hid in me.controlled_hills:
                hill = board.hills[hid]
                oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
                if oc > 0:
                    nl = None
                    nd = 999
                    for hloc in hill.cells:
                        if board.cells[hloc.r][hloc.c].owner_parity != pp:
                            d = my_dist.get(hloc, 999)
                            if d < nd:
                                nd = d
                                nl = hloc
                    if nl:
                        return self._validate_paint_actions(board, pp,
                            self._rush_to(board, pp, me, opp, my_dist, stamina, nl))

        # Territory pressure when we control more hills
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)
        if our_hills > their_hills:
            pressure_target = self._territory_pressure_target(
                board, pp, me, opp, my_dist, self._bfs_raw(board, opp.loc))
            if pressure_target:
                actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, 15)
                d1 = self._step_toward(board, pos, pressure_target, pp, opp.loc)
                if d1:
                    actions.append(Action.Move(d1))
                    p1 = pos + d1
                    actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, 15)

                    td = my_dist.get(pressure_target, 999)
                    if td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + 20:
                        d2 = self._step_toward(board, p1, pressure_target, pp, opp.loc)
                        if d2 and not self._is_dangerous_monitor(board, p1 + d2, opp, pp):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                            p2 = p1 + d2
                            actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, 15)

                if any(isinstance(a, Action.Move) for a in actions):
                    actions = self._validate_paint_actions(board, pp, actions)
                    actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)
                    if not any(isinstance(a, Action.Move) for a in actions):
                        fb = self._fallback_move(board, pp, me, opp)
                        if fb:
                            actions.append(fb)
                        else:
                            for d in Direction.cardinals():
                                nxt = pos + d
                                if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                                    actions.append(Action.Move(d))
                                    break
                    return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

        actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, 15)

        target = self._frontier_target(board, pp, my_dist, pos)
        if target:
            d1 = self._step_toward(board, pos, target, pp, opp.loc)
            if d1:
                actions.append(Action.Move(d1))
                p1 = pos + d1
                actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, 15)

                td = my_dist.get(target, 999)
                if td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + 20:
                    d2 = self._step_toward(board, p1, target, pp, opp.loc)
                    if d2 and not self._is_dangerous_monitor(board, p1 + d2, opp, pp):
                        actions.append(Action.Move(d2))
                        stamina -= GameConstants.EXTRA_MOVE_COST
                        p2 = p1 + d2
                        actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, 15)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        return self._validate_paint_actions(board, pp, actions) if actions else [Action.Move(Direction.UP)]

    def _rush_to(self, board, pp, me, opp, my_dist, stamina, target):
        """Simple rush to a specific target with hill painting."""
        actions = []
        painted = {}
        pos = me.loc
        td = my_dist.get(target, 999)
        buf = 20

        actions, stamina = self._paint_hill_cells(board, pp, pos, actions, stamina, painted, buf)

        d1 = self._step_toward(board, pos, target, pp, opp.loc)
        if d1:
            actions.append(Action.Move(d1))
            p1 = pos + d1
            actions, stamina = self._paint_hill_cells(board, pp, p1, actions, stamina, painted, buf)

            if td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + buf + 5:
                d2 = self._step_toward(board, p1, target, pp, opp.loc)
                if d2 and not self._is_dangerous_monitor(board, p1 + d2, opp, pp):
                    actions.append(Action.Move(d2))
                    stamina -= GameConstants.EXTRA_MOVE_COST
                    p2 = p1 + d2
                    actions, stamina = self._paint_hill_cells(board, pp, p2, actions, stamina, painted, 15)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)
            else:
                for d in Direction.cardinals():
                    nxt = pos + d
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        actions.append(Action.Move(d))
                        break

        return actions if actions else [Action.Move(Direction.UP)]

    def _frontier_target(self, board, pp, my_dist, pos):
        """Find best unowned cell adjacent to our territory."""
        best = None
        best_score = -9999

        for loc, d in my_dist.items():
            c = board.cells[loc.r][loc.c]
            if c.is_wall or c.owner_parity == pp or loc == pos:
                continue

            adj_friendly = any(
                not board.oob(loc + dr) and board.cells[(loc + dr).r][(loc + dr).c].owner_parity == pp
                for dr in Direction.cardinals()
            )

            score = -d * 3
            if c.owner_parity == 0:
                score += 20
            if adj_friendly:
                score += 25
            if c.hill_id != 0:
                score += 40
            if c.powerup and d <= 2:
                score += 20

            if score > best_score:
                best_score = score
                best = loc

        return best

    # ===============================================================
    # KILLS
    # ===============================================================

    def _try_kill(self, board, me, opp, pp, stam):
        """Check kill opportunities — including hunting vulnerable opponents."""
        oc = board.cells[opp.loc.r][opp.loc.c]
        dist = self._md(me.loc, opp.loc)

        # Distance 1: direct collision kill
        if dist == 1:
            for d in Direction.cardinals():
                nxt = me.loc + d
                if nxt != opp.loc:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                if c.owner_parity == pp:
                    return [Action.Move(d)]
                if c.owner_parity == 0 and c.beacon_parity == 0 and stam >= GameConstants.PAINT_STAMINA_COST:
                    return [Action.Paint(nxt), Action.Move(d)]

        # Distance 2-3 on our painted cell
        if oc.owner_parity == pp:
            if dist == 2 and stam >= GameConstants.EXTRA_MOVE_COST + 20:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        if mid + d2 == opp.loc:
                            return [Action.Move(d1), Action.Move(d2)]
            if dist == 3 and stam >= GameConstants.EXTRA_MOVE_COST * 3 + 20:
                for d1 in Direction.cardinals():
                    p1 = me.loc + d1
                    if board.oob(p1) or board.cells[p1.r][p1.c].is_wall or p1 == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        p2 = p1 + d2
                        if board.oob(p2) or board.cells[p2.r][p2.c].is_wall or p2 == opp.loc:
                            continue
                        for d3 in Direction.cardinals():
                            if p2 + d3 == opp.loc:
                                return [Action.Move(d1), Action.Move(d2), Action.Move(d3)]

        # Paint-then-move kill on neutral cell
        if oc.owner_parity == 0 and oc.beacon_parity == 0 and oc.owner_parity != -pp:
            if dist == 2 and stam >= GameConstants.EXTRA_MOVE_COST + GameConstants.PAINT_STAMINA_COST + 15:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    if self._md(mid, opp.loc) == 1:
                        for d2 in Direction.cardinals():
                            if mid + d2 == opp.loc:
                                return [Action.Move(d1), Action.Paint(opp.loc), Action.Move(d2)]
            if dist == 3 and stam >= GameConstants.EXTRA_MOVE_COST * 3 + GameConstants.PAINT_STAMINA_COST + 10:
                for d1 in Direction.cardinals():
                    p1 = me.loc + d1
                    if board.oob(p1) or board.cells[p1.r][p1.c].is_wall or p1 == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        p2 = p1 + d2
                        if board.oob(p2) or board.cells[p2.r][p2.c].is_wall or p2 == opp.loc:
                            continue
                        if self._md(p2, opp.loc) == 1:
                            for d3 in Direction.cardinals():
                                if p2 + d3 == opp.loc:
                                    return [Action.Move(d1), Action.Move(d2),
                                            Action.Paint(opp.loc), Action.Move(d3)]

        return None

    # ===============================================================
    # COLLISION RUSH
    # ===============================================================

    def _rush_for_collision(self, board, pp, me, opp, my_dist, stamina):
        """Aggressively rush the opponent to win via collision."""
        pos = me.loc
        opp_loc = opp.loc
        dist = self._md(pos, opp_loc)
        actions = []
        painted = {}

        if dist == 1:
            kill = self._try_kill(board, me, opp, pp, stamina)
            if kill is not None:
                return kill

        if dist > 8:
            return self._rush_toward_opponent(board, pp, me, opp, my_dist, stamina)

        path = self._bfs_path_to(board, pos, opp_loc, pp)
        if not path:
            return None

        COST = GameConstants.PAINT_STAMINA_COST
        buf = 15

        for cell_loc in path[:4]:
            if self._md(pos, cell_loc) != 1:
                continue
            c = board.cells[cell_loc.r][cell_loc.c]
            if (c.owner_parity == 0 and c.beacon_parity == 0
                    and not c.is_wall and stamina >= COST + buf):
                actions.append(Action.Paint(cell_loc))
                stamina -= COST
                painted[cell_loc] = painted.get(cell_loc, 0) + 1

        if dist <= 3 and stamina >= COST + buf:
            opp_cell = board.cells[opp_loc.r][opp_loc.c]
            if (opp_cell.owner_parity != pp and opp_cell.owner_parity != -pp
                    and opp_cell.beacon_parity == 0 and not opp_cell.is_wall):
                if self._md(pos, opp_loc) == 1:
                    actions.append(Action.Paint(opp_loc))
                    stamina -= COST
                    painted[opp_loc] = painted.get(opp_loc, 0) + 1

        d1 = self._rush_step_toward(board, pos, opp_loc, pp, path)
        if not d1:
            return None

        nxt = pos + d1
        if board.oob(nxt):
            return None

        nxt_c = board.cells[nxt.r][nxt.c]

        if nxt == opp_loc:
            cell_owner = nxt_c.owner_parity
            if opp_loc in painted:
                actions.append(Action.Move(d1))
                return self._validate_paint_actions(board, pp, actions)
            elif cell_owner == -pp:
                if stamina >= COST and nxt_c.beacon_parity == 0:
                    actions.append(Action.Paint(opp_loc))
                    stamina -= COST
                    painted[opp_loc] = painted.get(opp_loc, 0) + 1
                    actions.append(Action.Move(d1))
                    return self._validate_paint_actions(board, pp, actions)
                return None
            else:
                actions.append(Action.Move(d1))
                return self._validate_paint_actions(board, pp, actions)

        actions.append(Action.Move(d1))
        p1 = nxt

        if stamina >= COST + buf:
            for d in Direction.cardinals():
                t = p1 + d
                if board.oob(t):
                    continue
                tc = board.cells[t.r][t.c]
                if tc.is_wall or tc.beacon_parity != 0:
                    continue
                if t == opp_loc and tc.owner_parity != -pp and stamina >= COST + buf:
                    actions.append(Action.Paint(t))
                    stamina -= COST
                    painted[t] = painted.get(t, 0) + 1
                    break
                elif (tc.owner_parity == 0 and self._md(t, opp_loc) < self._md(p1, opp_loc)
                      and stamina >= COST + buf):
                    actions.append(Action.Paint(t))
                    stamina -= COST
                    painted[t] = painted.get(t, 0) + 1
                    break

        new_dist = self._md(p1, opp_loc)
        if new_dist >= 1 and stamina >= GameConstants.EXTRA_MOVE_COST + buf:
            d2 = self._rush_step_toward(board, p1, opp_loc, pp)
            if d2:
                nxt2 = p1 + d2
                if not board.oob(nxt2):
                    if nxt2 == opp_loc:
                        nxt2_c = board.cells[nxt2.r][nxt2.c]
                        cell_owner = nxt2_c.owner_parity
                        if opp_loc in painted:
                            actions.append(Action.Move(d2))
                            return self._validate_paint_actions(board, pp, actions)
                        elif cell_owner == -pp:
                            if (self._md(p1, opp_loc) == 1
                                    and nxt2_c.beacon_parity == 0
                                    and stamina >= COST + GameConstants.EXTRA_MOVE_COST):
                                actions.append(Action.Paint(opp_loc))
                                stamina -= COST
                                painted[opp_loc] = 1
                                actions.append(Action.Move(d2))
                                return self._validate_paint_actions(board, pp, actions)
                        elif cell_owner == 0 or cell_owner == pp:
                            actions.append(Action.Move(d2))
                            return self._validate_paint_actions(board, pp, actions)
                    else:
                        actions.append(Action.Move(d2))
                        stamina -= GameConstants.EXTRA_MOVE_COST
                        p2 = nxt2

                        if self._md(p2, opp_loc) >= 1 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + buf:
                            d3 = self._rush_step_toward(board, p2, opp_loc, pp)
                            if d3:
                                nxt3 = p2 + d3
                                if not board.oob(nxt3):
                                    if nxt3 == opp_loc:
                                        nxt3_c = board.cells[nxt3.r][nxt3.c]
                                        if opp_loc in painted:
                                            actions.append(Action.Move(d3))
                                            return self._validate_paint_actions(board, pp, actions)
                                        elif nxt3_c.owner_parity == -pp:
                                            if (self._md(p2, opp_loc) == 1
                                                    and nxt3_c.beacon_parity == 0
                                                    and stamina >= COST + GameConstants.EXTRA_MOVE_COST * 2):
                                                actions.append(Action.Paint(opp_loc))
                                                stamina -= COST
                                                actions.append(Action.Move(d3))
                                                return self._validate_paint_actions(board, pp, actions)
                                        else:
                                            actions.append(Action.Move(d3))
                                            return self._validate_paint_actions(board, pp, actions)
                                    else:
                                        actions.append(Action.Move(d3))
                                        stamina -= GameConstants.EXTRA_MOVE_COST * 2

        final = self._simpos(board, pos, actions)
        if not self._is_collision_safe(board, final, pp, opp):
            fc = board.cells[final.r][final.c]
            if (fc.owner_parity == 0 and fc.beacon_parity == 0
                    and not fc.is_wall and stamina >= COST):
                for i in range(len(actions) - 1, -1, -1):
                    if isinstance(actions[i], Action.Move):
                        pre = self._simpos(board, pos, actions[:i])
                        if self._md(pre, final) == 1:
                            actions.insert(i, Action.Paint(final))
                            stamina -= COST
                            break
                        break

        return self._validate_paint_actions(board, pp, actions) if actions else None

    def _rush_toward_opponent(self, board, pp, me, opp, my_dist, stamina):
        """When opponent is far, close the distance while painting."""
        pos = me.loc
        opp_loc = opp.loc
        actions = []
        painted = {}
        buf = 20

        actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, opp_loc, 1)

        d1 = self._step_toward(board, pos, opp_loc, pp, opp_loc, hill_target=False)
        if d1:
            actions.append(Action.Move(d1))
            p1 = pos + d1
            actions, stamina = self._paint_trail(board, pp, p1, actions, stamina, painted, buf, opp_loc, 1)

            if stamina >= GameConstants.EXTRA_MOVE_COST + buf:
                d2 = self._step_toward(board, p1, opp_loc, pp, opp_loc, hill_target=False)
                if d2:
                    actions.append(Action.Move(d2))
                    stamina -= GameConstants.EXTRA_MOVE_COST
                    p2 = p1 + d2
                    actions, stamina = self._paint_trail(board, pp, p2, actions, stamina, painted, buf, opp_loc, 1)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)
        return self._validate_paint_actions(board, pp, actions) if actions else None

    def _rush_step_toward(self, board, start, target, pp, path=None):
        """Step toward target during rush — more aggressive than normal pathfinding."""
        if start == target:
            return None

        if path and len(path) >= 1:
            for cell in path:
                if self._md(start, cell) == 1:
                    for d in Direction.cardinals():
                        if start + d == cell:
                            return d

        best = None
        best_score = -9999
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if nxt == target:
                if c.owner_parity == -pp:
                    score = -50
                else:
                    score = 500
            else:
                score = -self._md(nxt, target) * 10
                if c.owner_parity == pp:
                    score += 5
                elif c.owner_parity == -pp:
                    score -= 10
            if score > best_score:
                best_score = score
                best = d

        return best

    def _bfs_path_to(self, board, start, target, pp):
        """BFS to find shortest path from start to target."""
        if start == target:
            return []
        visited = {start}
        parent = {}
        queue = deque([start])
        found = False
        while queue:
            loc = queue.popleft()
            if loc == target:
                found = True
                break
            for d in Direction.cardinals():
                nxt = loc + d
                if nxt in visited or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                visited.add(nxt)
                parent[nxt] = loc
                queue.append(nxt)
        if not found:
            return None
        path = []
        cur = target
        while cur != start:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # ===============================================================
    # SAFETY
    # ===============================================================

    def _ensure_safe_ending(self, board, pp, start_pos, opp, actions, stamina, painted):
        """Never end turn on a non-owned cell if opponent can reach us."""
        final = self._simpos(board, start_pos, actions)
        mon = self._opp_monitor

        def _is_safe(loc):
            return self._is_collision_safe(board, loc, pp, opp)

        if _is_safe(final):
            return actions, stamina

        last_move_idx = None
        for i in range(len(actions) - 1, -1, -1):
            if isinstance(actions[i], Action.Move):
                last_move_idx = i
                break

        if last_move_idx is None:
            return actions, stamina

        pre_pos = self._simpos(board, start_pos, actions[:last_move_idx])
        opp_reach = mon['threat_range']

        def _clean_and_repaint(acts, move_idx, new_final):
            cleaned = acts[:move_idx + 1]
            nonlocal stamina
            for d in Direction.cardinals():
                t = new_final + d
                if board.oob(t):
                    continue
                c = board.cells[t.r][t.c]
                if c.is_wall or c.beacon_parity != 0 or c.owner_parity == -pp:
                    continue
                if stamina - GameConstants.PAINT_STAMINA_COST >= 10:
                    if c.owner_parity == 0 or (c.owner_parity == pp and abs(c.paint_value) < GameConstants.MAX_PAINT_VALUE):
                        cleaned.append(Action.Paint(t))
                        stamina -= GameConstants.PAINT_STAMINA_COST
                        painted[t] = painted.get(t, 0) + 1
                        break
            return cleaned

        fc = board.cells[final.r][final.c]

        if (fc.owner_parity == 0 and fc.beacon_parity == 0
                and not fc.is_wall
                and stamina >= GameConstants.PAINT_STAMINA_COST
                and self._md(pre_pos, final) == 1):
            actions.insert(last_move_idx, Action.Paint(final))
            stamina -= GameConstants.PAINT_STAMINA_COST
            painted[final] = painted.get(final, 0) + 1
            return actions, stamina

        best_dir = None
        best_score = -9999
        for d in Direction.cardinals():
            nxt = pre_pos + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or c.beacon_parity == -pp:
                continue
            if nxt == opp.loc and c.owner_parity != pp:
                continue
            nd = self._md(nxt, opp.loc)
            if c.owner_parity == pp and c.beacon_parity != -pp:
                score = 200
            elif nd > opp_reach:
                score = 100
            elif c.owner_parity == 0:
                score = -50
            else:
                score = -200
            score += nd
            if score > best_score:
                best_score = score
                best_dir = d

        if best_dir and best_score > -50:
            new_final = pre_pos + best_dir
            actions[last_move_idx] = Action.Move(best_dir)
            actions = _clean_and_repaint(actions, last_move_idx, new_final)
            return actions, stamina

        if stamina >= GameConstants.PAINT_STAMINA_COST:
            for d in Direction.cardinals():
                nxt = pre_pos + d
                if board.oob(nxt) or nxt == final:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or c.beacon_parity != 0:
                    continue
                if c.owner_parity == 0:
                    actions = actions[:last_move_idx]
                    actions.append(Action.Paint(nxt))
                    actions.append(Action.Move(d))
                    stamina -= GameConstants.PAINT_STAMINA_COST
                    painted[nxt] = painted.get(nxt, 0) + 1
                    return actions, stamina

        if _is_safe(pre_pos):
            actions = actions[:last_move_idx]
            return actions, stamina

        for trim in range(len(actions) - 1, -1, -1):
            if isinstance(actions[trim], Action.Move):
                test_pos = self._simpos(board, start_pos, actions[:trim])
                if _is_safe(test_pos):
                    actions = actions[:trim]
                    return actions, stamina

        if best_dir:
            new_final = pre_pos + best_dir
            actions[last_move_idx] = Action.Move(best_dir)
            actions = _clean_and_repaint(actions, last_move_idx, new_final)
        return actions, stamina

    def _is_dangerous(self, board, loc, opp_loc, pp):
        if board.oob(loc):
            return True
        c = board.cells[loc.r][loc.c]
        if c.is_wall:
            return True
        return c.owner_parity == -pp and self._md(loc, opp_loc) <= 2

    def _escape_dir(self, board, pos, opp_loc, pp):
        """Best direction to escape opponent threat."""
        best = None
        best_score = -999
        for d in Direction.cardinals():
            nxt = pos + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or nxt == opp_loc:
                continue
            score = self._md(nxt, opp_loc) * 3
            if c.owner_parity == pp:
                score += 10
            elif c.owner_parity == -pp:
                score -= 5
            if score > best_score:
                best_score = score
                best = d
        return best

    def _retreat_dir(self, board, pos, opp_loc, pp):
        """BFS toward nearest friendly cell."""
        visited = {pos}
        queue = deque()
        for d in Direction.cardinals():
            nxt = pos + d
            if board.oob(nxt) or nxt in visited:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or nxt == opp_loc:
                continue
            visited.add(nxt)
            if c.owner_parity == pp:
                return d
            queue.append((nxt, d))
        while queue:
            loc, fd = queue.popleft()
            for d in Direction.cardinals():
                nxt = loc + d
                if board.oob(nxt) or nxt in visited:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or nxt == opp_loc:
                    continue
                visited.add(nxt)
                if c.owner_parity == pp:
                    return fd
                queue.append((nxt, fd))
        return None

    # ===============================================================
    # PATHFINDING
    # ===============================================================

    def _bfs(self, board, start, pp):
        """BFS from start, treating opponent on enemy cell as wall."""
        opp = board.get_opponent(pp)
        dist = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for dr in Direction.cardinals():
                nxt = loc + dr
                if nxt in dist or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                    continue
                dist[nxt] = d + 1
                queue.append((nxt, d + 1))
        return dist

    def _bfs_raw(self, board, start):
        """BFS from start, only walls blocked."""
        dist = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for dr in Direction.cardinals():
                nxt = loc + dr
                if nxt in dist or board.oob(nxt):
                    continue
                if board.cells[nxt.r][nxt.c].is_wall:
                    continue
                dist[nxt] = d + 1
                queue.append((nxt, d + 1))
        return dist

    def _step_toward(self, board, start, target, pp, opp_loc, hill_target=False):
        """BFS first-step toward target. Tries safe path first, relaxes if needed."""
        if start == target:
            return None

        danger_levels = [1, 0] if hill_target else [2, 1, 0]
        for danger_r in danger_levels:
            visited = {start}
            queue = deque()
            for d in Direction.cardinals():
                nxt = start + d
                if board.oob(nxt) or nxt in visited:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                if c.beacon_parity == -pp:
                    continue
                if nxt == opp_loc and c.owner_parity != pp:
                    continue
                if danger_r > 0 and c.owner_parity == -pp and self._md(nxt, opp_loc) <= danger_r:
                    continue
                visited.add(nxt)
                if nxt == target:
                    return d
                queue.append((nxt, d))

            while queue:
                loc, fd = queue.popleft()
                for d in Direction.cardinals():
                    nxt = loc + d
                    if board.oob(nxt) or nxt in visited:
                        continue
                    c = board.cells[nxt.r][nxt.c]
                    if c.is_wall:
                        continue
                    if c.beacon_parity == -pp:
                        continue
                    if nxt == opp_loc and c.owner_parity != pp:
                        continue
                    if danger_r > 0 and c.owner_parity == -pp and self._md(nxt, opp_loc) <= danger_r:
                        continue
                    visited.add(nxt)
                    if nxt == target:
                        return fd
                    queue.append((nxt, fd))

        return None

    def _collision_aware_step(self, board, start, target, pp, opp_loc):
        """Step toward target, preferring our own painted cells when near opponent."""
        if start == target:
            return None

        opp_reach = self._opp_monitor['threat_range']

        candidates = []
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.beacon_parity == -pp:
                continue
            if nxt == opp_loc and c.owner_parity != pp:
                continue

            nd = self._md(nxt, opp_loc)
            safety = 0
            if c.owner_parity == pp and c.beacon_parity != -pp:
                safety = 20
            elif c.owner_parity == -pp:
                safety = -60
            elif c.owner_parity == 0 and nd <= opp_reach:
                safety = -15
            else:
                safety = 5

            progress = self._md(start, target) - self._md(nxt, target)
            candidates.append((d, safety + progress * 10, nxt))

        candidates.sort(key=lambda x: -x[1])

        for d, _score, nxt in candidates:
            if nxt == target:
                return d
            if self._can_reach(board, nxt, target):
                return d

        return self._step_toward(board, start, target, pp, opp_loc, hill_target=True)

    def _can_reach(self, board, start, target):
        """Quick BFS check: can we reach target from start?"""
        visited = {start}
        queue = deque([start])
        while queue:
            loc = queue.popleft()
            for d in Direction.cardinals():
                nxt = loc + d
                if nxt in visited or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)
        return False

    def _safe_hill_step(self, board, start, target, pp, opp_loc):
        """Step toward target on a contested hill, avoiding collision death."""
        if start == target:
            return None
        best = None
        best_score = -9999
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.beacon_parity == -pp:
                continue
            if c.owner_parity == -pp and self._md(nxt, opp_loc) <= 1:
                continue
            score = -self._md(nxt, target) * 10
            if c.owner_parity == pp and c.beacon_parity != -pp:
                score += 15
            elif c.owner_parity == 0:
                score += 5
            elif c.owner_parity == -pp:
                score -= 10
            if nxt == target:
                score += 50
            if score > best_score:
                best_score = score
                best = d
        return best

    # ===============================================================
    # MAP CLASSIFICATION
    # ===============================================================

    def _classify_map(self, board):
        """Classify map characteristics once at start."""
        area = board.board_size.r * board.board_size.c
        wall_count = sum(
            1 for r in range(board.board_size.r)
            for c in range(board.board_size.c)
            if board.cells[r][c].is_wall
        )
        wall_ratio = wall_count / area if area > 0 else 0
        return {
            'area': area,
            'large': area >= 400,
            'small': area < 300,
            'maze': wall_ratio >= 0.30,
            'wall_ratio': wall_ratio,
            'num_hills': len(board.hills),
        }

    # ===============================================================
    # CHOKEPOINT DETECTION
    # ===============================================================

    def _is_chokepoint(self, board, loc):
        passable = 0
        for d in Direction.cardinals():
            nxt = loc + d
            if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                passable += 1
        return passable <= 2

    def _find_chokepoint_on_path(self, board, start, target):
        if start == target:
            return None, 999
        visited = {start}
        parent = {}
        queue = deque([start])
        found = False
        while queue:
            loc = queue.popleft()
            if loc == target:
                found = True
                break
            for d in Direction.cardinals():
                nxt = loc + d
                if nxt in visited or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                visited.add(nxt)
                parent[nxt] = loc
                queue.append(nxt)
        if not found:
            return None, 999
        path = []
        cur = target
        while cur != start:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        for i, cell in enumerate(path):
            if self._is_chokepoint(board, cell):
                return cell, i + 1
        return None, 999

    # ===============================================================
    # UTILITIES
    # ===============================================================

    def _near_hill(self, board, pos, pp, radius=4):
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                if self._md(pos, hloc) <= radius:
                    return True
        return False

    def _estimate_regen(self, board, pos, pp):
        regen = GameConstants.BASE_STAMINA_REGEN
        radius = GameConstants.ADJACENCY_RADIUS
        local_friendly = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = pos.r + dr, pos.c + dc
                if 0 <= r < board.board_size.r and 0 <= c < board.board_size.c:
                    if board.cells[r][c].owner_parity == pp:
                        local_friendly += 1
        regen += local_friendly * GameConstants.ADJACENT_REGEN_BONUS
        territory = board.get_territory_count(pp)
        regen += min(territory // GameConstants.GLOBAL_PAINT_REGEN_RATIO,
                     GameConstants.GLOBAL_PAINT_REGEN_CAP)
        if board.turn_count > GameConstants.GLOBAL_DECAY_TURN_THRESHOLD:
            delta = board.turn_count - GameConstants.GLOBAL_DECAY_TURN_THRESHOLD - 1
            intervals = (delta // GameConstants.GLOBAL_DECAY_INTERVAL) + 1
            regen -= intervals * GameConstants.GLOBAL_DECAY_REGEN_PENALTY
            regen = max(0, regen)
        return regen

    def _md(self, a, b):
        return abs(a.r - b.r) + abs(a.c - b.c)

    def _simpos(self, board, start, actions, opp_loc=None):
        loc = start
        for a in actions:
            if isinstance(a, Action.Move):
                if a.move_type == MoveType.BEACON_TRAVEL and a.beacon_target is not None:
                    loc = a.beacon_target
                elif a.direction is not None:
                    nxt = loc + a.direction
                    if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                        loc = nxt
                        if opp_loc and loc == opp_loc:
                            break
        return loc

    def _fallback_move(self, board, pp, me, opp):
        """Any valid move, preferring safe and friendly cells."""
        best = None
        best_p = -999
        mon = self._opp_monitor
        opp_reach = mon['threat_range']
        for d in Direction.cardinals():
            nxt = me.loc + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                continue
            if c.beacon_parity == -pp:
                continue
            nd = self._md(nxt, opp.loc)
            if c.owner_parity == pp and c.beacon_parity != -pp:
                p = 20
            elif c.owner_parity == 0 and nd > opp_reach:
                p = 10
            elif c.owner_parity == 0:
                p = -5
            elif c.owner_parity == -pp:
                p = -20
            else:
                p = 0
            if c.owner_parity == -pp and nd <= 2:
                p -= 20
            if p > best_p:
                best_p = p
                best = d
        return Action.Move(best) if best else None

    def commentate(self, board, player_parity, time_left):
        p = board.get_player(player_parity)
        o = board.get_opponent(player_parity)
        mon = self._opp_monitor
        rush = "RUSH" if self._rush_mode else ""
        osc = "OSC" if self._is_oscillating() else ""
        seq = str(self._hill_sequence[:3]) if self._hill_sequence else "[]"
        phase = self._phase
        aggr = mon['aggression']
        trend = mon['stamina_trend']
        bcn = len(self._beacon_placed_hills)
        return (
            f"oa6 {phase} h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} oTR={mon['threat_range']}"
            f" r={board.current_round} {rush}{osc} opp={aggr}/{trend}"
            f" bcn={bcn} seq={seq}"
        )
