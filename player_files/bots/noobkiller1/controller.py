from collections import deque
from typing import Union, Iterable, Dict, Optional
import math
import time

from game import *


class PlayerController:
    """
    noobkiller1 — upgraded from painter1 with data-driven improvements.
    Key upgrades: map-adaptive play modes, erase logic, aggressive multi-move,
    always-paint minimum, dynamic bidding, opponent style detection, time management.
    """

    def __init__(self, player_parity: int, time_left):
        self.pp = player_parity
        self._committed_hill_id = None
        self._map_info = None
        self._recent_positions = []
        self._stalemate_counter = 0
        # Opponent tracking
        self._opp_moves_per_turn = []  # track opponent moves/turn
        self._opp_prev_loc = None
        self._opp_style = None  # "camper", "rusher", "compute_heavy", None
        self._turn_count = 0
        # BFS caching
        self._bfs_cache_turn = -1
        self._bfs_cache_result = None
        self._opp_bfs_cache_turn = -1
        self._opp_bfs_cache_result = None

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        mi = self._classify_map(board)
        self._map_info = mi
        if mi['small'] and mi['num_hills'] <= 4:
            return 5  # first move critical on small maps
        elif mi['large']:
            return 2  # save stamina on big maps
        elif mi['maze']:
            return 3  # moderate for maze
        else:
            return 3  # moderate maps

    def play(
        self, board: Board, player_parity: int, time_left,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        t_start = time.time()
        pp = player_parity
        me = board.get_player(pp)
        opp = board.get_opponent(pp)
        pos = me.loc
        stamina = me.stamina
        actions = []
        painted = {}
        self._turn_count += 1
        _time_left = time_left() if callable(time_left) else time_left

        # Opponent reach based on stamina
        opp_kill_stam = opp.stamina - GameConstants.PAINT_STAMINA_COST
        if opp_kill_stam >= 60:
            opp_reach = 4
        elif opp_kill_stam >= 30:
            opp_reach = 3
        elif opp_kill_stam >= 10:
            opp_reach = 2
        else:
            opp_reach = 1

        # === OPPONENT TRACKING ===
        self._track_opponent(board, opp)

        # === KILL CHECK (always first) ===
        kill = self._try_kill(board, me, opp, pp, stamina)
        if kill is not None:
            return kill

        # === Check if near a hill (suppress flee behavior) ===
        cell = board.cells[pos.r][pos.c]
        near_any_hill = self._near_hill(board, pos, pp, radius=4)

        # === ESCAPE: on enemy territory near opponent — but NOT if near a hill ===
        if not near_any_hill and cell.owner_parity == -pp and self._md(pos, opp.loc) <= 3:
            esc = self._escape_dir(board, pos, opp.loc, pp)
            if esc:
                return [Action.Move(esc)]

        # === LOW STAMINA RETREAT — but NOT if near a hill ===
        if not near_any_hill and stamina < 35 and cell.owner_parity != pp:
            ret = self._retreat_dir(board, pos, opp.loc, pp)
            if ret:
                return [Action.Move(ret)]

        # === MAP CLASSIFICATION (cached) ===
        if self._map_info is None:
            self._map_info = self._classify_map(board)
        mi = self._map_info

        # === PLAY MODE ===
        mode = self._play_mode(board, pp, me, opp, mi, _time_left)

        # === DISTANCES (with caching + time budget) ===
        my_dist = self._bfs_cached(board, pos, pp)
        if _time_left > 20:
            opp_dist = self._bfs_raw_cached(board, opp.loc)
        else:
            opp_dist = {}  # skip opponent BFS when low on time

        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)

        # === STALEMATE DETECTION ===
        self._recent_positions.append((pos.r, pos.c))
        if len(self._recent_positions) > 12:
            self._recent_positions.pop(0)
        in_stalemate = False
        if len(self._recent_positions) >= 8:
            unique = set(self._recent_positions[-8:])
            if len(unique) <= 3:
                self._stalemate_counter += 1
                in_stalemate = True
            else:
                self._stalemate_counter = 0
        if in_stalemate and self._stalemate_counter >= 4 and their_hills > our_hills:
            flank = self._find_flank_target(board, pp, me, opp, my_dist, opp_dist)
            if flank:
                return self._rush_to(board, pp, me, opp, my_dist, stamina, flank)

        # === NO HILLS: pure territory game ===
        if total_hills == 0:
            return self._play_territory(board, pp, me, opp, my_dist, stamina)

        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD)

        # === TARGET SELECTION (with commitment) ===
        defend = self._defend_target(board, pp, me, opp, my_dist, dom_needed, their_hills)
        attack, attack_hid = self._pick_attack_target(board, pp, me, opp, my_dist, opp_dist, dom_needed)

        target = None
        chosen_hid = None
        if defend and attack:
            dd = my_dist.get(defend, 999)
            ad = my_dist.get(attack, 999)
            if their_hills + 1 >= dom_needed or dd <= ad + 2:
                target = defend
                chosen_hid = None
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

        # All hills ours — territory for tiebreak
        if target is None:
            if our_hills >= dom_needed:
                return self._play_territory(board, pp, me, opp, my_dist, stamina)
            return self._play_territory(board, pp, me, opp, my_dist, stamina)

        td = my_dist.get(target, 999)

        # === POWERUP DETOUR ===
        if stamina < 50:
            pu = self._powerup_nearby(board, my_dist, 1)
            if pu:
                target = pu
                td = my_dist.get(pu, 999)

        # === MAP CHARACTERISTICS ===
        large_map = mi['large']
        maze_map = mi['maze']

        # === OPPONENT ANALYSIS ===
        dist_to_opp = self._md(pos, opp.loc)
        target_cell = board.cells[target.r][target.c] if not board.oob(target) else None
        target_hill_obj = None
        opp_hill_dist = 999

        if target_cell and target_cell.hill_id != 0:
            target_hid = target_cell.hill_id
            target_hill_obj = board.hills.get(target_hid)
            if target_hill_obj:
                opp_hill_dist = min(
                    (self._md(hloc, opp.loc) for hloc in target_hill_obj.cells),
                    default=999
                )

        contested = opp_hill_dist <= 4
        approaching = 4 < opp_hill_dist <= 10
        danger_zone = dist_to_opp <= 3
        reinforce = contested or approaching

        neutral_hills_exist = any(
            h.controller_parity == 0 for h in board.hills.values()
        )
        all_hills_claimed = not neutral_hills_exist and total_hills > 0

        # === MAX LAYERS ===
        if neutral_hills_exist:
            max_layers = 2
        elif not large_map:
            max_layers = 2 if not contested else 3
        elif contested:
            max_layers = 3
        else:
            max_layers = 2

        # === WHEN ON/NEAR A HILL: retarget to walk across it ===
        cur_hill_id = board.cells[pos.r][pos.c].hill_id
        on_target_hill = (target_hill_obj and cur_hill_id == target_cell.hill_id) if target_cell else False
        if cur_hill_id != 0:
            hill_obj = board.hills.get(cur_hill_id)
            if hill_obj and hill_obj.controller_parity != pp:
                unpainted = self._nearest_unclaimed_hill_cell(board, pp, hill_obj, my_dist, contested)
                if unpainted:
                    target = unpainted
                    td = my_dist.get(target, 999)
                    on_target_hill = True

        # === CHOKEPOINT ===
        if td > 2 and not on_target_hill and target_cell and target_cell.hill_id != 0:
            choke, choke_dist = self._find_chokepoint_on_path(board, pos, target)
            if choke and choke_dist <= td:
                choke_cell = board.cells[choke.r][choke.c]
                opp_to_choke = self._md(opp.loc, choke)
                if opp_to_choke <= choke_dist + 4 and choke_cell.owner_parity != pp:
                    target = choke
                    td = my_dist.get(target, 999)

        # === STAMINA BUDGET ===
        if maze_map:
            buf = 25 if td > 3 else 15
        elif td > 5:
            buf = 55
        elif td > 2:
            buf = 30
        else:
            buf = 15

        if large_map and td > 8 and not maze_map:
            buf = 35

        if on_target_hill and contested:
            buf = 10

        hill_buf = buf
        if on_target_hill:
            regen = self._estimate_regen(board, pos, pp)
            hill_buf = max(5, buf - regen // 2)

        # === SHOULD WE PAINT TERRITORY EN ROUTE? ===
        # UPGRADE: Always paint at least 1 cell when stamina > 50 (brief 4e)
        paint_territory_en_route = all_hills_claimed or danger_zone or stamina > 50

        trail_layers = 2 if danger_zone else 1

        # === PRE-MOVE PAINTING ===
        actions, stamina = self._paint_hill_cells(board, pp, pos, actions, stamina, painted, hill_buf, reinforce, max_layers)

        # Trail painting: more aggressive than painter1
        if paint_territory_en_route or maze_map:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target, trail_layers)
        elif large_map and td > 2:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)
        elif stamina > 60 and td > 2:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)

        # === MOVEMENT ===
        targeting_hill = board.cells[target.r][target.c].hill_id != 0 if not board.oob(target) else False

        def _pick_step(from_pos):
            if on_target_hill and contested:
                return self._safe_hill_step(board, from_pos, target, pp, opp.loc)
            elif danger_zone:
                return self._collision_aware_step(board, from_pos, target, pp, opp.loc)
            else:
                return self._step_toward(board, from_pos, target, pp, opp.loc, hill_target=targeting_hill)

        d1 = _pick_step(pos)
        if d1:
            nxt_pos = pos + d1
            nxt_c = board.cells[nxt_pos.r][nxt_pos.c] if not board.oob(nxt_pos) else None

            # Pre-paint destination if neutral and opponent within reach
            if nxt_c and not nxt_c.is_wall:
                if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                        and self._md(nxt_pos, opp.loc) <= opp_reach
                        and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                    actions.append(Action.Paint(nxt_pos))
                    stamina -= GameConstants.PAINT_STAMINA_COST
                    painted[nxt_pos] = painted.get(nxt_pos, 0) + 1

            # === ERASE STEP on hill cells (brief 4b) ===
            erase_used = False
            if (nxt_c and nxt_c.hill_id != 0 and nxt_c.owner_parity == -pp
                    and abs(nxt_c.paint_value) >= 3
                    and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + buf
                    and nxt_pos != opp.loc):
                actions.append(Action.Move(d1, move_type=MoveType.ERASE))
                stamina -= GameConstants.ERASE_STEP_EXTRA_COST
                erase_used = True
            else:
                actions.append(Action.Move(d1))

            p1 = pos + d1

            actions, stamina = self._paint_hill_cells(board, pp, p1, actions, stamina, painted, hill_buf, reinforce, max_layers)

            if paint_territory_en_route or maze_map or (large_map and td > 3):
                actions, stamina = self._paint_trail(board, pp, p1, actions, stamina, painted, buf, target, trail_layers)

            # === 2nd MOVE — more aggressive (brief 4d) ===
            if large_map and not maze_map:
                move2_threshold = GameConstants.EXTRA_MOVE_COST + 15
            elif on_target_hill:
                move2_threshold = GameConstants.EXTRA_MOVE_COST + hill_buf
            elif mode == "territory_expand":
                move2_threshold = GameConstants.EXTRA_MOVE_COST + 15
            else:
                move2_threshold = GameConstants.EXTRA_MOVE_COST + buf + (15 if maze_map else 5)

            # On a target hill: re-pick target from new position
            if on_target_hill:
                p1_hill_id = board.cells[p1.r][p1.c].hill_id
                if p1_hill_id != 0:
                    hill_obj_now = board.hills.get(p1_hill_id)
                    if hill_obj_now and hill_obj_now.controller_parity != pp:
                        next_cell = self._nearest_unclaimed_hill_cell(
                            board, pp, hill_obj_now, my_dist, contested)
                        if next_cell and next_cell != p1:
                            target = next_cell

            # Take 2nd move more often — default on large maps and when far from target
            should_take_2nd = (td > 2 or on_target_hill or (large_map and td > 1)
                               or mode == "territory_expand")
            if should_take_2nd and stamina >= move2_threshold:
                d2 = _pick_step(p1)
                if d2 and (targeting_hill or not self._is_dangerous(board, p1 + d2, opp.loc, pp)):
                    nxt_pos2 = p1 + d2
                    if not board.oob(nxt_pos2):
                        nxt_c2 = board.cells[nxt_pos2.r][nxt_pos2.c]
                        if (nxt_c2.owner_parity == 0 and nxt_c2.beacon_parity == 0
                                and not nxt_c2.is_wall
                                and self._md(nxt_pos2, opp.loc) <= opp_reach
                                and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                            actions.append(Action.Paint(nxt_pos2))
                            stamina -= GameConstants.PAINT_STAMINA_COST
                            painted[nxt_pos2] = painted.get(nxt_pos2, 0) + 1

                        # Erase on 2nd move for hill cells
                        if (nxt_c2.hill_id != 0 and nxt_c2.owner_parity == -pp
                                and abs(nxt_c2.paint_value) >= 3
                                and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + GameConstants.EXTRA_MOVE_COST + 10
                                and not (nxt_pos2 == opp.loc)):
                            actions.append(Action.Move(d2, move_type=MoveType.ERASE))
                            stamina -= GameConstants.EXTRA_MOVE_COST + GameConstants.ERASE_STEP_EXTRA_COST
                        else:
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST

                    p2 = p1 + d2
                    actions, stamina = self._paint_hill_cells(board, pp, p2, actions, stamina, painted, max(hill_buf - 10, 5), reinforce, max_layers)

                    if paint_territory_en_route or (large_map and td > 5):
                        actions, stamina = self._paint_trail(board, pp, p2, actions, stamina, painted, max(buf - 10, 15), target, trail_layers)

                    # 3rd move
                    if not maze_map and td > 5 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + max((hill_buf if on_target_hill else buf) - 15, 15):
                        d3 = _pick_step(p2)
                        if d3 and (targeting_hill or not self._is_dangerous(board, p2 + d3, opp.loc, pp)):
                            nxt_pos3 = p2 + d3
                            if not board.oob(nxt_pos3):
                                nxt_c3 = board.cells[nxt_pos3.r][nxt_pos3.c]
                                if (nxt_c3.owner_parity == 0 and nxt_c3.beacon_parity == 0
                                        and not nxt_c3.is_wall
                                        and self._md(nxt_pos3, opp.loc) <= opp_reach
                                        and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                                    actions.append(Action.Paint(nxt_pos3))
                                    stamina -= GameConstants.PAINT_STAMINA_COST
                                    painted[nxt_pos3] = painted.get(nxt_pos3, 0) + 1
                            actions.append(Action.Move(d3))
                            stamina -= GameConstants.EXTRA_MOVE_COST * 2
                            p3 = p2 + d3
                            actions, stamina = self._paint_hill_cells(board, pp, p3, actions, stamina, painted, max(hill_buf, 5), reinforce, max_layers)

        # === POST-MOVE ===
        final = self._simpos(board, pos, actions)
        actions, stamina = self._paint_hill_cells(board, pp, final, actions, stamina, painted, max(hill_buf, 5), reinforce, max_layers)

        if board.cells[final.r][final.c].hill_id != 0 and stamina > 40:
            actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target, trail_layers)
        elif paint_territory_en_route or (large_map and stamina > 30):
            actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target, trail_layers)

        # === ALWAYS-PAINT MINIMUM (brief 4e) ===
        # If no paint action was taken and stamina > 50, paint an adjacent cell
        has_paint = any(isinstance(a, Action.Paint) for a in actions)
        if not has_paint and stamina > 50:
            actions, stamina = self._paint_one_cell(board, pp, final, actions, stamina, painted)

        # === STRATEGIC ERASE (brief 4b, 5g) ===
        # If we're near opponent territory and have plenty of stamina, erase strategically
        if not any(isinstance(a, Action.Move) for a in actions if hasattr(a, 'move_type') and a.move_type == MoveType.ERASE):
            erase_target = self._find_erase_target(board, pp, final, opp, stamina, buf)
            if erase_target:
                # Replace planned movement or add erase move
                pass  # Erase is already handled in movement above

        # === GUARANTEE MOVE ===
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        # === COLLISION SAFETY ===
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

        return actions if actions else [Action.Move(Direction.UP)]

    # ===============================================================
    # PLAY MODE (brief 5a)
    # ===============================================================

    def _play_mode(self, board, pp, me, opp, mi, time_left):
        """Top-level strategy selector based on game state."""
        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)
        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD) if total_hills > 0 else 0

        # Clock pressure against compute-heavy opponents
        if (self._opp_style == "compute_heavy" and mi['large']
                and time_left > 60):
            return "clock_pressure"

        # Close to domination — rush remaining hills
        if total_hills > 0 and our_hills + 1 >= dom_needed:
            return "hill_rush"

        # Losing on hills — rush to contest
        if total_hills > 0 and their_hills > our_hills:
            return "hill_rush"

        # Small map — always hill rush
        if mi['small'] and total_hills > 0:
            return "hill_rush"

        # Large open map — territory + speed
        if mi['large'] and not mi['maze']:
            neutral_hills = sum(1 for h in board.hills.values() if h.controller_parity == 0)
            if neutral_hills == 0 and our_hills >= their_hills:
                return "territory_expand"
            return "balanced"

        # Dense/maze — efficient pathing
        if mi['maze']:
            return "balanced"

        # Default
        if total_hills > 0:
            neutral = sum(1 for h in board.hills.values() if h.controller_parity == 0)
            if neutral > 0:
                return "hill_rush"
            if our_hills > their_hills:
                return "territory_expand"
        return "balanced"

    # ===============================================================
    # OPPONENT TRACKING (brief 5f)
    # ===============================================================

    def _track_opponent(self, board, opp):
        """Track opponent behavior to detect their style."""
        if self._opp_prev_loc is not None:
            dist_moved = self._md(self._opp_prev_loc, opp.loc)
            self._opp_moves_per_turn.append(dist_moved)
        self._opp_prev_loc = opp.loc

        # Classify after 20 turns of data
        if len(self._opp_moves_per_turn) >= 20 and self._opp_style is None:
            avg_moves = sum(self._opp_moves_per_turn[-20:]) / 20
            # Check for camping (barely moves)
            stationary = sum(1 for m in self._opp_moves_per_turn[-20:] if m == 0)
            if stationary >= 10:
                self._opp_style = "camper"
            elif avg_moves <= 1.05:
                self._opp_style = "compute_heavy"  # likely joebrr
            elif avg_moves >= 1.5:
                self._opp_style = "rusher"
            else:
                self._opp_style = "balanced"

    # ===============================================================
    # TARGET SELECTION
    # ===============================================================

    def _pick_attack_target(self, board, pp, me, opp, my_dist, opp_dist, dom_needed):
        """Pick best uncaptured hill to rush — race-aware scoring with commitment."""
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
            commit_bonus = 60 if (hid == committed) else 0
            proximity_bonus = max(0, 100 - hill_dist * 20)

            opp_on_hill = any(
                self._md(hloc, opp.loc) == 0 for hloc in hill.cells
            )
            camping_penalty = -80 if opp_on_hill else 0

            score = (race_adv * 8 + size_bonus + close_to_dom + steal_bonus
                     + late_bonus + commit_bonus + proximity_bonus
                     + camping_penalty - to_paint * 5)

            if score > best_score:
                best_score = score
                best = nearest_loc
                best_hid = hid

        return best, best_hid

    def _defend_target(self, board, pp, me, opp, my_dist, dom_needed, their_hills):
        """Find our most threatened hill to defend."""
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

        return worst if worst_urgency >= 0.15 else None

    def _find_flank_target(self, board, pp, me, opp, my_dist, opp_dist):
        """When stuck in a standoff, find a hill cell reachable via a path that
        stays far from the opponent."""
        best = None
        best_score = -9999
        opp_loc = opp.loc

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == pp:
                    continue
                d = my_dist.get(hloc, 999)
                if d > 50:
                    continue
                opp_d = self._md(hloc, opp_loc)
                score = opp_d * 10 - d * 3
                if hc.owner_parity == 0:
                    score += 20
                if score > best_score:
                    best_score = score
                    best = hloc

        if best is None:
            return None

        waypoint = None
        wp_score = -9999
        target_d = my_dist.get(best, 999)
        for loc, d in my_dist.items():
            if d > target_d or d < 2:
                continue
            c = board.cells[loc.r][loc.c]
            if c.is_wall:
                continue
            opp_d = self._md(loc, opp_loc)
            tgt_from_loc = self._md(loc, best)
            if tgt_from_loc > target_d:
                continue
            score = opp_d * 5 - d
            if c.owner_parity == pp:
                score += 5
            if score > wp_score:
                wp_score = score
                waypoint = loc

        if waypoint and self._md(waypoint, opp_loc) > 4:
            return waypoint
        return best

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
        """Paint adjacent unpainted hill cells.
        If uncontested: only paint NEW cells.
        If contested: also reinforce up to max_layers."""
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
        """Paint at most 1 adjacent cell — new cells first, then reinforce."""
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
        """Paint best adjacent cell — used in territory/expand phase."""
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

    def _paint_one_cell(self, board, pp, pos, actions, stamina, painted):
        """Paint exactly one adjacent cell — always-paint minimum guarantee (brief 4e)."""
        COST = GameConstants.PAINT_STAMINA_COST
        if stamina < COST + 15:
            return actions, stamina

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
            if base + added >= GameConstants.MAX_PAINT_VALUE:
                continue

            is_new = (c.owner_parity == 0 and added == 0)
            if is_new:
                score = 100
                if c.hill_id != 0:
                    score = 200
            else:
                score = 10
                if c.hill_id != 0:
                    score = 50

            if score > best_score:
                best_score = score
                best = t

        if best:
            actions.append(Action.Paint(best))
            painted[best] = painted.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    # ===============================================================
    # STRATEGIC ERASE (brief 4b, 5g)
    # ===============================================================

    def _find_erase_target(self, board, pp, pos, opp, stamina, buf):
        """Find a strategic erase target — opponent territory that hurts their regen."""
        if stamina < GameConstants.ERASE_STEP_EXTRA_COST + buf + 20:
            return None

        best = None
        best_score = -1

        for d in Direction.cardinals():
            t = pos + d
            if board.oob(t):
                continue
            c = board.cells[t.r][t.c]
            if c.is_wall or c.owner_parity != -pp:
                continue
            if t == opp.loc:
                continue

            layers = abs(c.paint_value)
            score = layers * 10  # more layers = more value to erase

            # Hill cell — very high value to erase
            if c.hill_id != 0:
                score += 100

            # Near opponent — disrupts their regen cluster
            if self._md(t, opp.loc) <= 3:
                score += 30

            # Far from opponent — they can't repaint soon
            if self._md(t, opp.loc) >= 6:
                score += 20

            if score > best_score:
                best_score = score
                best = t

        return best

    # ===============================================================
    # TERRITORY PLAY
    # ===============================================================

    def _play_territory(self, board, pp, me, opp, my_dist, stamina):
        """Expand territory for tiebreak."""
        actions = []
        painted = {}
        pos = me.loc

        # Defend contested hills
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
                        return self._rush_to(board, pp, me, opp, my_dist, stamina, nl)

        actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, 15)

        target = self._frontier_target(board, pp, my_dist, pos)
        if target:
            d1 = self._step_toward(board, pos, target, pp, opp.loc)
            if d1:
                actions.append(Action.Move(d1))
                p1 = pos + d1
                actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, 15)

                td = my_dist.get(target, 999)
                # More aggressive 2nd move in territory mode
                if td > 1 and stamina >= GameConstants.EXTRA_MOVE_COST + 15:
                    d2 = self._step_toward(board, p1, target, pp, opp.loc)
                    if d2 and not self._is_dangerous(board, p1 + d2, opp.loc, pp):
                        actions.append(Action.Move(d2))
                        stamina -= GameConstants.EXTRA_MOVE_COST
                        p2 = p1 + d2
                        actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, 15)

                        # 3rd move on large maps
                        if self._map_info and self._map_info['large'] and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + 15:
                            d3 = self._step_toward(board, p2, target, pp, opp.loc)
                            if d3 and not self._is_dangerous(board, p2 + d3, opp.loc, pp):
                                actions.append(Action.Move(d3))
                                stamina -= GameConstants.EXTRA_MOVE_COST * 2
                                p3 = p2 + d3
                                actions, stamina = self._paint_expand(board, pp, p3, actions, stamina, painted, 15)

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

    def _rush_to(self, board, pp, me, opp, my_dist, stamina, target):
        """Rush to a specific target with hill painting."""
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
                if d2 and not self._is_dangerous(board, p1 + d2, opp.loc, pp):
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

        # === ADJACENT KILLS ===
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

        # === MULTI-STEP KILLS: opponent on our cell ===
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

        # === HUNT: opponent on neutral cell — paint & rush ===
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
                                    return [Action.Move(d1), Action.Move(d2), Action.Paint(opp.loc), Action.Move(d3)]

        return None

    # ===============================================================
    # SAFETY
    # ===============================================================

    def _ensure_safe_ending(self, board, pp, start_pos, opp, actions, stamina, painted):
        """ABSOLUTE RULE: never end turn on a non-owned cell if opponent can reach us."""
        final = self._simpos(board, start_pos, actions)
        fc = board.cells[final.r][final.c]
        opp_reach = min(3, 1 + opp.stamina // GameConstants.EXTRA_MOVE_COST)

        def _is_safe(loc):
            if board.oob(loc):
                return False
            c = board.cells[loc.r][loc.c]
            if c.owner_parity == pp and c.beacon_parity != -pp:
                return True
            if self._md(loc, opp.loc) > opp_reach:
                return True
            return False

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

        # Strategy 1: pre-paint final cell
        if (fc.owner_parity == 0 and fc.beacon_parity == 0
                and not fc.is_wall
                and stamina >= GameConstants.PAINT_STAMINA_COST
                and self._md(pre_pos, final) == 1):
            actions.insert(last_move_idx, Action.Paint(final))
            stamina -= GameConstants.PAINT_STAMINA_COST
            painted[final] = painted.get(final, 0) + 1
            return actions, stamina

        # Strategy 2: swap last move to safe cell
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

        # Strategy 3: paint different adjacent cell and move there
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

        # Strategy 4: remove last move
        if _is_safe(pre_pos):
            actions = actions[:last_move_idx]
            return actions, stamina

        # Strategy 5: walk back through moves
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

    def _bfs_cached(self, board, start, pp):
        """BFS with turn-based caching."""
        turn = board.turn_count
        if self._bfs_cache_turn == turn and self._bfs_cache_result is not None:
            return self._bfs_cache_result
        result = self._bfs(board, start, pp)
        self._bfs_cache_turn = turn
        self._bfs_cache_result = result
        return result

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

    def _bfs_raw_cached(self, board, start):
        """Raw BFS with caching — cache for 2 turns since board changes slowly."""
        turn = board.turn_count
        if abs(self._opp_bfs_cache_turn - turn) <= 2 and self._opp_bfs_cache_result is not None:
            return self._opp_bfs_cache_result
        result = self._bfs_raw(board, start)
        self._opp_bfs_cache_turn = turn
        self._opp_bfs_cache_result = result
        return result

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

        opp_reach = 2

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

    # ===============================================================
    # MAP CLASSIFICATION
    # ===============================================================

    def _classify_map(self, board):
        area = board.board_size.r * board.board_size.c
        wall_count = sum(
            1 for r in range(board.board_size.r)
            for c in range(board.board_size.c)
            if board.cells[r][c].is_wall
        )
        wall_ratio = wall_count / area if area > 0 else 0
        passable = area - wall_count
        return {
            'area': area,
            'passable': passable,
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

    def _nearest_unclaimed_hill_cell(self, board, pp, hill, my_dist, contested):
        best = None
        best_score = -9999
        opp = board.get_opponent(pp)
        for hloc in hill.cells:
            hc = board.cells[hloc.r][hloc.c]
            if hc.owner_parity == pp:
                continue
            d = my_dist.get(hloc, 999)
            if d > 50:
                continue
            if hc.owner_parity == 0:
                score = 100 - d * 3
            else:
                layers = abs(hc.paint_value)
                score = 60 - d * 3 - layers * 5
            if contested and self._md(hloc, opp.loc) <= 1:
                score -= 50
            if score > best_score:
                best_score = score
                best = hloc
        return best

    def _safe_hill_step(self, board, start, target, pp, opp_loc):
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
            if isinstance(a, Action.Move) and a.direction is not None:
                nxt = loc + a.direction
                if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                    loc = nxt
                    if opp_loc and loc == opp_loc:
                        break
        return loc

    def _fallback_move(self, board, pp, me, opp):
        best = None
        best_p = -999
        opp_reach = min(3, 1 + opp.stamina // GameConstants.EXTRA_MOVE_COST)
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
        _tl = time_left() if callable(time_left) else time_left
        mode = self._play_mode(board, player_parity, p, o, self._map_info or {}, _tl)
        return (
            f"nk1[{mode}] h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} r={board.current_round}"
        )
