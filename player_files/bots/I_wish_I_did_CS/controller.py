from collections import deque
from typing import Union, Iterable, Dict, Optional
import math

from game import *


class PlayerController:
    """
    nosleep v3 — painter1 baseline + opponent monitor + erase hill traversal.

    Key additions over painter1:
    1. Opponent monitor: tracks stamina, regen, max moves, threat range every turn
    2. Collision safety uses monitor — determines if ending on neutral/enemy is safe
    3. Hill traversal with erase on large maps when opponent has thick layers
    4. Edge-based attacking when opponent approaches — end turns on own paint
    """

    def __init__(self, player_parity: int, time_left):
        self.pp = player_parity
        self._committed_hill_id = None
        self._map_info = None
        self._recent_positions = []
        self._stalemate_counter = 0

        # Beacon state
        self._opp_beacons = []     # list of Location of opponent beacons
        self._our_beacons = []     # list of Location of our beacons
        self._beacon_cooldown = 0  # turns until we can place again

        # Opponent monitor state — updated every turn
        self._opp_monitor = {
            'prev_stamina': None,      # opponent stamina last turn
            'est_regen': 5,            # estimated opponent regen per turn
            'max_moves': 1,            # max moves opponent can take
            'threat_range': 2,         # how far opponent can reach this turn
            'proximity': 999,          # distance to opponent
            'approaching': False,      # is opponent getting closer?
            'prev_dist': None,         # distance to opponent last turn
            'stamina': 100,            # current opponent stamina
        }

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        return 6

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

        # === UPDATE OPPONENT MONITOR ===
        self._update_opp_monitor(board, pp, me, opp)
        mon = self._opp_monitor
        opp_reach = mon['threat_range']

        # === KILL CHECK (always first) ===
        kill = self._try_kill(board, me, opp, pp, stamina)
        if kill is not None:
            return kill

        # === KILL APPROACH: opponent close on killable cell, rush to close gap ===
        # If _try_kill failed but we're close, stop painting and multi-move toward them.
        # Next turn we'll have more stamina + be closer = kill.
        oc_approach = board.cells[opp.loc.r][opp.loc.c]
        approach_dist = self._md(pos, opp.loc)
        if (oc_approach.owner_parity != -pp  # killable cell
                and 2 <= approach_dist <= 7
                and stamina >= 10):  # need at least 10 for a 2nd move
            # How many more moves do we need?
            max_moves_now = 1
            rem = stamina
            ex = 1
            while True:
                cost = GameConstants.EXTRA_MOVE_COST * ex
                if rem < cost:
                    break
                rem -= cost
                max_moves_now += 1
                ex += 1
                if max_moves_now >= 8:
                    break

            # If we're within 2 moves of being in range, rush instead of normal play
            gap = approach_dist - max_moves_now
            if 0 < gap <= 2:
                # Multi-move toward opponent, no painting, close the gap
                rush_actions = []
                cur = pos
                moves_left = max_moves_now
                for _ in range(moves_left):
                    d = self._step_toward(board, cur, opp.loc, pp, opp.loc, hill_target=False)
                    if not d:
                        break
                    nxt = cur + d
                    if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall:
                        break
                    rush_actions.append(Action.Move(d))
                    cur = nxt
                if rush_actions:
                    # Ensure we end safely
                    rush_final = self._simpos(board, pos, rush_actions)
                    if self._is_collision_safe(board, rush_final, pp, opp):
                        return rush_actions

        # === Check if near a hill (suppress flee behavior) ===
        cell = board.cells[pos.r][pos.c]
        near_any_hill = self._near_hill(board, pos, pp, radius=4)

        # === ESCAPE: on enemy territory near opponent — but NOT if near a hill ===
        if not near_any_hill and cell.owner_parity == -pp and mon['proximity'] <= 3:
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

        # === DISTANCES ===
        my_dist = self._bfs(board, pos, pp)
        opp_dist = self._bfs_raw(board, opp.loc)

        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)

        # === BEACON SCANNING ===
        self._scan_beacons(board, pp)
        if self._beacon_cooldown > 0:
            self._beacon_cooldown -= 1

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

        # === TARGET SELECTION (with strong commitment) ===
        defend = self._defend_target(board, pp, me, opp, my_dist, dom_needed, their_hills)
        attack, attack_hid = self._pick_attack_target(board, pp, me, opp, my_dist, opp_dist, dom_needed)

        # STRONG COMMITMENT: if committed to a hill, keep targeting it until captured
        if self._committed_hill_id is not None:
            cid = self._committed_hill_id
            if cid in board.hills and board.hills[cid].controller_parity != pp:
                # Committed hill still needs capturing — force it as attack target
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
                # Hill captured or gone — clear commitment
                self._committed_hill_id = None

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

        # === POWERUP DETOUR: always grab if 1 BFS step away ===
        pu = self._powerup_nearby(board, my_dist, 1)
        if pu:
            target = pu
            td = 1
        elif (stamina < 60 or board.current_round < 50) and td > 3:
            # Early game or low stamina: also grab powerups 2 steps away
            pu = self._powerup_nearby(board, my_dist, 2)
            if pu:
                target = pu
                td = my_dist.get(pu, 999)

        # === MAP CHARACTERISTICS ===
        large_map = mi['large']
        maze_map = mi['maze']

        # === OPPONENT ANALYSIS (using monitor) ===
        dist_to_opp = mon['proximity']
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
        approaching = 4 < opp_hill_dist <= 10 or mon['approaching']
        danger_zone = dist_to_opp <= opp_reach + 1
        reinforce = contested or approaching

        # === NEUTRAL HILLS CHECK ===
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
                # Use the enhanced hill cell picker that considers erase
                unpainted = self._pick_hill_cell_to_claim(
                    board, pp, hill_obj, my_dist, contested, large_map, opp, stamina)
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
        paint_territory_en_route = all_hills_claimed or danger_zone

        trail_layers = 2 if danger_zone else 1

        # === PRE-MOVE PAINTING ===
        actions, stamina = self._paint_hill_cells(board, pp, pos, actions, stamina, painted, hill_buf, reinforce, max_layers)

        if paint_territory_en_route or maze_map:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target, trail_layers)
        elif large_map and td > 2:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)
        elif stamina > 70 and td > 2:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)

        # === MOVEMENT ===
        targeting_hill = board.cells[target.r][target.c].hill_id != 0 if not board.oob(target) else False

        # When opponent is approaching our target hill, prefer edge-based movement
        edge_attack = approaching and targeting_hill and mon['proximity'] <= 8

        def _pick_step(from_pos):
            if edge_attack and on_target_hill:
                return self._edge_hill_step(board, from_pos, target, pp, opp.loc, target_hill_obj)
            elif on_target_hill and contested:
                return self._safe_hill_step(board, from_pos, target, pp, opp.loc)
            elif danger_zone:
                return self._collision_aware_step(board, from_pos, target, pp, opp.loc)
            else:
                return self._step_toward(board, from_pos, target, pp, opp.loc, hill_target=targeting_hill)

        d1 = _pick_step(pos)
        if d1:
            nxt_pos = pos + d1
            if not board.oob(nxt_pos):
                nxt_c = board.cells[nxt_pos.r][nxt_pos.c]
                # Pre-paint destination if neutral and within threat range
                if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                        and not nxt_c.is_wall
                        and self._md(nxt_pos, opp.loc) <= opp_reach
                        and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                    actions.append(Action.Paint(nxt_pos))
                    stamina -= GameConstants.PAINT_STAMINA_COST
                    painted[nxt_pos] = painted.get(nxt_pos, 0) + 1

                # === ERASE STEP: on large maps, if destination has thick opponent layers ===
                use_erase = self._should_erase_step(
                    board, pp, nxt_pos, large_map, stamina, buf, opp, target_hill_obj)
                if use_erase:
                    actions.append(Action.Move(d1, move_type=MoveType.ERASE))
                else:
                    actions.append(Action.Move(d1))
            else:
                actions.append(Action.Move(d1))
            p1 = pos + d1

            actions, stamina = self._paint_hill_cells(board, pp, p1, actions, stamina, painted, hill_buf, reinforce, max_layers)

            if paint_territory_en_route or maze_map or (large_map and td > 3):
                actions, stamina = self._paint_trail(board, pp, p1, actions, stamina, painted, buf, target, trail_layers)

            # 2nd move
            move2_threshold = GameConstants.EXTRA_MOVE_COST + (hill_buf if on_target_hill else buf) + (15 if maze_map else 5)
            if on_target_hill:
                p1_hill_id = board.cells[p1.r][p1.c].hill_id
                if p1_hill_id != 0:
                    hill_obj_now = board.hills.get(p1_hill_id)
                    if hill_obj_now and hill_obj_now.controller_parity != pp:
                        next_cell = self._pick_hill_cell_to_claim(
                            board, pp, hill_obj_now, my_dist, contested, large_map, opp, stamina)
                        if next_cell and next_cell != p1:
                            target = next_cell
            if (td > 2 or on_target_hill) and stamina >= move2_threshold:
                d2 = _pick_step(p1)
                if d2 and (targeting_hill or not self._is_dangerous_monitor(board, p1 + d2, opp, pp)):
                    nxt_pos = p1 + d2
                    if not board.oob(nxt_pos):
                        nxt_c = board.cells[nxt_pos.r][nxt_pos.c]
                        if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                                and not nxt_c.is_wall
                                and self._md(nxt_pos, opp.loc) <= opp_reach
                                and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                            actions.append(Action.Paint(nxt_pos))
                            stamina -= GameConstants.PAINT_STAMINA_COST
                            painted[nxt_pos] = painted.get(nxt_pos, 0) + 1

                        use_erase = self._should_erase_step(
                            board, pp, nxt_pos, large_map, stamina, buf, opp, target_hill_obj)
                        if use_erase:
                            actions.append(Action.Move(d2, move_type=MoveType.ERASE))
                        else:
                            actions.append(Action.Move(d2))
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
                        if d3 and (targeting_hill or not self._is_dangerous_monitor(board, p2 + d3, opp, pp)):
                            nxt_pos = p2 + d3
                            if not board.oob(nxt_pos):
                                nxt_c = board.cells[nxt_pos.r][nxt_pos.c]
                                if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                                        and not nxt_c.is_wall
                                        and self._md(nxt_pos, opp.loc) <= opp_reach
                                        and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                                    actions.append(Action.Paint(nxt_pos))
                                    stamina -= GameConstants.PAINT_STAMINA_COST
                                    painted[nxt_pos] = painted.get(nxt_pos, 0) + 1

                                use_erase = self._should_erase_step(
                                    board, pp, nxt_pos, large_map, stamina, buf, opp, target_hill_obj)
                                if use_erase:
                                    actions.append(Action.Move(d3, move_type=MoveType.ERASE))
                                else:
                                    actions.append(Action.Move(d3))
                            else:
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

        # === GUARANTEE MOVE ===
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        # === COLLISION SAFETY (using monitor) ===
        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        # === BEACON PLACEMENT: place forward bases near captured hills ===
        actions = self._try_place_beacon(board, pp, actions, pos, me, opp, our_hills, total_hills)

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
    # OPPONENT MONITOR
    # ===============================================================

    def _update_opp_monitor(self, board, pp, me, opp):
        """Update opponent tracking data every turn.
        Uses territory-based regen estimation for accurate threat range."""
        mon = self._opp_monitor
        dist = self._md(me.loc, opp.loc)

        # Track approaching behavior
        if mon['prev_dist'] is not None:
            mon['approaching'] = dist < mon['prev_dist']
        mon['prev_dist'] = dist
        mon['proximity'] = dist
        mon['stamina'] = opp.stamina

        # Estimate opponent regen from territory and adjacency (deterministic)
        opp_regen = self._estimate_opp_regen(board, pp, opp)
        mon['est_regen'] = opp_regen
        mon['prev_stamina'] = opp.stamina

        # Effective stamina: current + regen they'll receive after our turn ends
        # This is what they'll have when they move next
        effective_stamina = min(opp.max_stamina, opp.stamina + opp_regen)

        # Max moves calculation — CORRECT cost formula from engine:
        # moves_this_turn=0: free, =1: costs 10*1, =2: costs 10*2, =3: costs 10*3
        remaining = effective_stamina
        moves = 1  # first move is always free (moves_this_turn=0)
        extra = 1  # next extra move's moves_this_turn value
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

        # === BEACON-ADJUSTED THREAT RANGE ===
        # If opponent has beacons, their effective range from a beacon is:
        # travel(free, resets move counter) + walk from beacon (10, 20, 30...)
        if self._opp_beacons:
            closest_beacon_dist = min(
                self._md(b, me.loc) for b in self._opp_beacons)
            # How many walk steps can they take after beacon travel?
            # Teleport is free (resets moves_this_turn=0), then 10*1 + 10*2 + ... for walks
            beacon_remaining = effective_stamina
            beacon_walks = 0
            walk_extra = 1
            while beacon_remaining > 0:
                cost = GameConstants.EXTRA_MOVE_COST * walk_extra
                if beacon_remaining < cost:
                    break
                beacon_remaining -= cost
                beacon_walks += 1
                walk_extra += 1
                if beacon_walks >= 4:
                    break
            # If they can reach us from beacon: beacon_dist <= beacon_walks
            if closest_beacon_dist <= beacon_walks:
                # Their true threat range includes beacon teleport
                beacon_threat = closest_beacon_dist + 999  # effectively infinite direction
                mon['threat_range'] = max(mon['threat_range'], closest_beacon_dist + beacon_walks)
                mon['beacon_threat'] = True
                mon['beacon_dist'] = closest_beacon_dist
            else:
                mon['beacon_threat'] = False
                mon['beacon_dist'] = closest_beacon_dist
        else:
            mon['beacon_threat'] = False
            mon['beacon_dist'] = 999

    def _estimate_opp_regen(self, board, pp, opp):
        """Estimate opponent's stamina regen based on their position and territory."""
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
        if board.turn_count >= GameConstants.GLOBAL_DECAY_TURN_THRESHOLD + 200:
            delta = board.turn_count - (GameConstants.GLOBAL_DECAY_TURN_THRESHOLD + 200)
            intervals = (delta // 200) + 1
            regen -= intervals * GameConstants.GLOBAL_DECAY_REGEN_PENALTY
            regen = max(0, regen)
        return regen

    def _is_collision_safe(self, board, loc, pp, opp):
        """Check if ending a turn at loc is safe from collision.
        Uses opponent monitor for accurate threat assessment.
        Includes beacon-powered assassination detection."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]
        mon = self._opp_monitor

        # On our cell (not opponent beacon) = always safe — we win collisions
        if c.owner_parity == pp and c.beacon_parity != -pp:
            return True

        # === BEACON THREAT CHECK ===
        # If opponent has beacons, they can teleport from ANYWHERE to a beacon
        # then walk to us. Check distance from each opponent beacon.
        if mon.get('beacon_threat') and self._opp_beacons:
            for blc in self._opp_beacons:
                beacon_to_us = self._md(blc, loc)
                # After teleport (free, resets move counter), they walk (10,20,30 cost each)
                # Compute how many walks they can do post-teleport
                eff_stam = min(opp.max_stamina, opp.stamina + mon['est_regen'])
                bcn_rem = eff_stam  # teleport is free (resets moves_this_turn=0)
                bcn_walks = 0
                we = 1
                while bcn_rem > 0:
                    cost = GameConstants.EXTRA_MOVE_COST * we
                    if bcn_rem < cost:
                        break
                    bcn_rem -= cost
                    bcn_walks += 1
                    we += 1
                    if bcn_walks >= 4:
                        break
                if beacon_to_us <= bcn_walks:
                    return False  # opponent can beacon-kill us here

        # Normal distance-based check
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
        # On opponent cell near them = very dangerous
        if c.owner_parity == -pp and dist <= mon['threat_range']:
            return True
        # On neutral cell very close to approaching opponent = risky
        if c.owner_parity == 0 and dist <= 2 and mon['approaching']:
            return True
        return False

    # ===============================================================
    # ERASE HILL TRAVERSAL
    # ===============================================================

    def _should_erase_step(self, board, pp, loc, large_map, stamina, buf, opp, target_hill_obj):
        """Decide if we should use ERASE move type when stepping onto a cell.
        Returns True if the cell has thick opponent paint on a hill and erase is worthwhile."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]

        # Only erase opponent-owned cells on hills
        if c.owner_parity != -pp or c.hill_id == 0:
            return False

        # Check if this cell belongs to a hill we're targeting
        layers = abs(c.paint_value)

        # On large maps with thick layers, erase is worth the 40 stamina cost
        # because it saves multiple turns of weakening visits
        erase_cost = GameConstants.ERASE_STEP_EXTRA_COST  # 40

        if not large_map:
            # On small maps, only erase if layers >= 3 and we have plenty of stamina
            if layers >= 3 and stamina >= erase_cost + buf + 30:
                return True
            return False

        # Large map: erase if layers >= 2 and we have enough stamina
        if layers >= 2 and stamina >= erase_cost + buf + 20:
            return True

        return False

    def _pick_hill_cell_to_claim(self, board, pp, hill, my_dist, contested, large_map, opp, stamina):
        """Enhanced hill cell picker for sweep traversal.
        Key principle: skip the cell we're standing on (it gets painted as we leave),
        prefer adjacent cells (distance 1) to sweep across the hill cell-by-cell.
        When we step onto an opponent 1-layer cell, it becomes neutral; then from the
        next cell we paint it behind us."""
        best = None
        best_score = -9999
        me_loc = board.get_player(pp).loc

        mon = self._opp_monitor

        for hloc in hill.cells:
            hc = board.cells[hloc.r][hloc.c]
            if hc.owner_parity == pp:
                continue  # already ours

            # Skip the cell we're currently standing on — it will be painted
            # from the next cell after we move away (sweep pattern)
            if hloc == me_loc:
                continue

            d = my_dist.get(hloc, 999)
            if d > 50:
                continue

            layers = abs(hc.paint_value)
            opp_dist_to_cell = self._md(hloc, opp.loc)

            if hc.owner_parity == 0:
                # Neutral cell — easy to paint from adjacent
                score = 100 - d * 3
            elif hc.owner_parity == -pp:
                # Opponent cell — step weakens by 1, or erase clears it
                if large_map and layers >= 2 and stamina >= GameConstants.ERASE_STEP_EXTRA_COST + 30:
                    score = 80 - d * 3 - layers * 2
                else:
                    score = 60 - d * 3 - layers * 5
            else:
                continue

            # SWEEP BONUS: strongly prefer adjacent cells (d=1) so we move
            # across the hill instead of oscillating on one cell
            if d == 1:
                score += 50
            elif d == 2:
                score += 15

            # When opponent is approaching or close, prefer edge cells
            if (contested or mon['approaching']) and opp_dist_to_cell <= 2:
                score -= 60
            elif mon['approaching'] and opp_dist_to_cell <= 3:
                score -= 30

            # Prefer edge cells (safer to reach and retreat from)
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
        """Step toward target on a hill while preferring edge cells.
        When opponent is approaching, we attack from edges and avoid the interior
        where we'd be vulnerable. Always prefer stepping onto our own paint."""
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
            # Avoid opponent cells adjacent to opponent
            if c.owner_parity == -pp and self._md(nxt, opp_loc) <= 1:
                continue

            score = -self._md(nxt, target) * 10

            # Safety: strongly prefer our own cells
            if c.owner_parity == pp and c.beacon_parity != -pp:
                score += 30
            elif c.owner_parity == 0:
                score += 10
            elif c.owner_parity == -pp:
                score -= 15

            # Edge bonus: prefer cells on the edge of the hill
            if hill_obj and c.hill_id == hill_obj.id:
                if self._is_hill_edge(board, nxt, hill_obj):
                    score += 20

            # Distance from opponent bonus
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
            commit_bonus = 200 if (hid == committed) else 0
            proximity_bonus = max(0, 100 - hill_dist * 20)

            opp_on_hill = any(
                self._md(hloc, opp.loc) == 0 for hloc in hill.cells
            )
            camping_penalty = -80 if opp_on_hill else 0

            # When we have no hills, prioritize grabbing the nearest one ASAP.
            # Race advantage is less important than raw speed to first capture.
            if our_hills == 0:
                score = (proximity_bonus * 5 + commit_bonus
                         + size_bonus - to_paint * 5 + race_adv * 2)
            elif our_hills == 1:
                # After first capture, prioritize contested neutral hills that
                # both players can reach — don't cede the middle for a far hill.
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
        """When stuck in a standoff, find a hill cell reachable via a flank."""
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

        return actions if actions else [Action.Move(Direction.UP)]

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

        # === ADJACENT KILLS ===
        if dist == 1:
            for d in Direction.cardinals():
                nxt = me.loc + d
                if nxt != opp.loc:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                # On our paint = we win collision
                if c.owner_parity == pp:
                    return [Action.Move(d)]
                # On neutral = mover wins collision, just walk in
                if c.owner_parity == 0:
                    return [Action.Move(d)]
                # On their paint = THEY win collision. Can't paint their cell.
                # Only option: erase step (50 stamina) makes it neutral, then
                # we'd need another move. Skip — too expensive for adjacent.

        # === MULTI-STEP KILLS: opponent on our cell OR neutral ===
        if oc.owner_parity == pp or oc.owner_parity == 0:
            if dist == 2 and stam >= GameConstants.EXTRA_MOVE_COST:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        if mid + d2 == opp.loc:
                            return [Action.Move(d1), Action.Move(d2)]
            if dist == 3 and stam >= GameConstants.EXTRA_MOVE_COST * 3:
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

        # === STAMINA-BASED HUNT ===
        # If opponent is on neutral or our paint, calculate max moves from
        # current stamina and BFS to see if we can reach them in one turn.
        # Collision win = instant game over, so no stamina buffer needed.
        if oc.owner_parity != -pp:  # killable cell (neutral or ours)
            # Calculate max affordable moves from stamina
            # Move 1: free, Move 2: 10, Move 3: 20, Move 4: 30...
            max_moves = 1
            remaining = stam
            extra = 1
            while True:
                cost = GameConstants.EXTRA_MOVE_COST * extra
                if remaining < cost:
                    break
                remaining -= cost
                max_moves += 1
                extra += 1
                if max_moves >= 8:  # cap to avoid wasting time on BFS
                    break

            # Hunt if we can do 2+ moves (adjacent kills handled above)
            if max_moves >= 2 and dist <= max_moves:
                # BFS from our position to opponent, capped at max_moves depth
                path = self._find_short_path(board, me.loc, opp.loc, max_moves)
                if path is not None and len(path) <= max_moves:
                    # Verify we can actually afford this many moves
                    n = len(path)
                    total_cost = sum(GameConstants.EXTRA_MOVE_COST * i for i in range(1, n))
                    if stam >= total_cost:
                        actions = [Action.Move(d) for d in path]
                        return actions

        # === BEACON-POWERED KILL ===
        # Place new beacon + teleport to existing beacon near opp + walk to opp
        if self._our_beacons and oc.owner_parity != -pp:
            # Find our beacon closest to opponent
            for our_bcn in self._our_beacons:
                bcn_to_opp = self._md(our_bcn, opp.loc)
                if bcn_to_opp > 3:
                    continue  # too far to walk after teleport

                # Cost: place+travel(10) + walk from beacon
                # Walk costs after reset: 10, 20, 30...
                walk_cost = sum(GameConstants.EXTRA_MOVE_COST * i for i in range(1, bcn_to_opp + 1))
                total_cost = GameConstants.EXTRA_MOVE_COST + walk_cost  # 10 for travel + walk
                if stam < total_cost + 5:
                    continue

                # Can we place a beacon? Need to move to a valid spot
                for place_dir in Direction.cardinals():
                    place_dest = me.loc + place_dir
                    if board.oob(place_dest) or board.cells[place_dest.r][place_dest.c].is_wall:
                        continue
                    if place_dest == opp.loc:
                        continue
                    if not self._can_place_beacon_at(board, pp, place_dest):
                        continue

                    # Build path from beacon to opponent
                    path = self._find_short_path(board, our_bcn, opp.loc, bcn_to_opp)
                    if path is None:
                        continue

                    # Verify final cell is killable (we own it or it's neutral)
                    opp_cell = board.cells[opp.loc.r][opp.loc.c]
                    if opp_cell.owner_parity == -pp:
                        continue  # they own it, we'd die

                    # Build action sequence
                    actions = [Action.Move(place_dir, place_beacon=True)]
                    actions.append(Action.Move(
                        direction=None,
                        move_type=MoveType.BEACON_TRAVEL,
                        beacon_target=our_bcn
                    ))
                    for step_dir in path:
                        actions.append(Action.Move(step_dir))
                    return actions

        return None

    # ===============================================================
    # SAFETY (enhanced with opponent monitor)
    # ===============================================================

    def _ensure_safe_ending(self, board, pp, start_pos, opp, actions, stamina, painted):
        """ABSOLUTE RULE: never end turn on a non-owned cell if opponent can reach us.
        Uses opponent monitor for accurate threat assessment."""
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

        # Last resort
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
        if board.turn_count >= GameConstants.GLOBAL_DECAY_TURN_THRESHOLD + 200:
            delta = board.turn_count - (GameConstants.GLOBAL_DECAY_TURN_THRESHOLD + 200)
            intervals = (delta // 200) + 1
            regen -= intervals * GameConstants.GLOBAL_DECAY_REGEN_PENALTY
            regen = max(0, regen)
        return regen

    def _md(self, a, b):
        return abs(a.r - b.r) + abs(a.c - b.c)

    def _simpos(self, board, start, actions, opp_loc=None):
        loc = start
        for a in actions:
            if isinstance(a, Action.Move):
                if a.direction is None and hasattr(a, 'beacon_target') and a.beacon_target is not None:
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

    # ===============================================================
    # BEACON SYSTEM
    # ===============================================================

    def _scan_beacons(self, board, pp):
        """Scan entire board for beacon locations. Called once per turn."""
        self._our_beacons = []
        self._opp_beacons = []
        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                bp = board.cells[r][c].beacon_parity
                if bp == pp:
                    self._our_beacons.append(Location(r, c))
                elif bp == -pp:
                    self._opp_beacons.append(Location(r, c))

    def _can_place_beacon_at(self, board, pp, loc):
        """Check if beacon can be placed at loc using engine's proportional rule.
        Returns True/False. STRICT — invalid placement = instant loss."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]
        if c.is_wall:
            return False
        # Tile must be our color or neutral (engine rejects opponent-owned)
        if c.owner_parity != pp and c.owner_parity != 0:
            return False
        # Can place on opponent beacons (clears theirs) or empty cells, not on our own
        if c.beacon_parity == pp:
            return False

        non_wall = 0
        controlled = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r2, c2 = loc.r + dr, loc.c + dc
                if 0 <= r2 < board.board_size.r and 0 <= c2 < board.board_size.c:
                    cell = board.cells[r2][c2]
                    if not cell.is_wall:
                        non_wall += 1
                        if cell.owner_parity == pp:
                            controlled += 1
        # Engine check: friendly * P^2 >= Q * valid_cells
        return controlled * (GameConstants.BEACON_WINDOW_SIZE_P ** 2) >= GameConstants.BEACON_REQUIREMENT_Q * non_wall

    def _find_short_path(self, board, start, target, max_len):
        """BFS to find a path of cardinal directions from start to target.
        Returns list of Direction or None."""
        if start == target:
            return []
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            loc, path = queue.popleft()
            if len(path) >= max_len:
                continue
            for d in Direction.cardinals():
                nxt = loc + d
                if nxt in visited or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                new_path = path + [d]
                if nxt == target:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
        return None

    def _try_place_beacon(self, board, pp, actions, start_pos, me, opp, our_hills, total_hills):
        """Try to add place_beacon=True to the last REGULAR move.
        Strategic: place near captured hills as forward teleport bases.
        Returns modified actions list (never crashes on invalid)."""
        if self._beacon_cooldown > 0:
            return actions
        # Don't on tiny maps
        if self._map_info and self._map_info['small']:
            return actions
        # Need at least 1 hill to make beacons useful
        if our_hills == 0:
            return actions
        # Cap at 2 active beacons (more = wasted paint)
        if len(self._our_beacons) >= 2:
            return actions

        # Find last REGULAR move with a direction
        last_move_idx = None
        for i in range(len(actions) - 1, -1, -1):
            if isinstance(actions[i], Action.Move):
                last_move_idx = i
                break
        if last_move_idx is None:
            return actions

        move = actions[last_move_idx]
        if move.move_type != MoveType.REGULAR:
            return actions
        if move.direction is None:
            return actions
        if hasattr(move, 'place_beacon') and move.place_beacon:
            return actions

        # Where would the beacon be placed?
        final = self._simpos(board, start_pos, actions[:last_move_idx + 1])
        if board.oob(final):
            return actions
        if not self._can_place_beacon_at(board, pp, final):
            return actions

        # Strategic check: only place near a hill we control
        near_our_hill = False
        for hid in me.controlled_hills:
            hill = board.hills[hid]
            for hloc in hill.cells:
                if self._md(final, hloc) <= 2:
                    near_our_hill = True
                    break
            if near_our_hill:
                break
        if not near_our_hill:
            return actions

        # Check there's a distant target worth teleporting to later
        has_distant = False
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                if self._md(final, hloc) >= 6:
                    has_distant = True
                    break
            if has_distant:
                break
        if not has_distant and total_hills < 3:
            return actions

        # Also good for assassination: place near opponent's usual area
        near_opp_area = self._md(final, opp.loc) <= 10

        if not has_distant and not near_opp_area:
            return actions

        # Place it
        actions[last_move_idx] = Action.Move(
            direction=move.direction,
            move_type=MoveType.REGULAR,
            place_beacon=True
        )
        self._beacon_cooldown = 8
        return actions

    def commentate(self, board, player_parity, time_left):
        p = board.get_player(player_parity)
        o = board.get_opponent(player_parity)
        mon = self._opp_monitor
        ob = len(self._opp_beacons)
        ub = len(self._our_beacons)
        bt = 'Y' if mon.get('beacon_threat') else 'N'
        return (
            f"ns4.2 h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} oTR={mon['threat_range']}"
            f" bcn={ub}v{ob} bt={bt} r={board.current_round}"
        )
