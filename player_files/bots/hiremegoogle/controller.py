from collections import deque
from typing import Union, Iterable, Dict, Optional
import math

from game import *


class PlayerController:
    """
    Domination-focused bot.
    Strategy: Rush hills fast, capture efficiently, defend lead, expand for tiebreak.
    No wasted paint on uncontested cells — every stamina point serves a purpose.
    """

    def __init__(self, player_parity: int, time_left):
        self.pp = player_parity
        self._committed_hill_id = None  # hill ID we're committed to capturing
        self._map_info = None  # cached map classification
        self._pos_history = []  # last N positions for oscillation detection
        self._oscillation_override = False  # when True, skip safety backtrack
        self._prev_opp_dist = 999  # for opponent approach detection
        self._opp_approaching_count = 0  # consecutive turns opponent gets closer

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        return 0

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
        opp_reach = min(3, 1 + opp.stamina // GameConstants.EXTRA_MOVE_COST)
        round_num = board.current_round
        approaching_sudden_death = round_num >= 400
        in_sudden_death = board.turn_count > GameConstants.GLOBAL_DECAY_TURN_THRESHOLD

        # === MAP CLASSIFICATION (cached) — must be early for wall_ratio usage ===
        if self._map_info is None:
            self._map_info = self._classify_map(board)
        mi = self._map_info
        wall_ratio = mi['wall_ratio']
        maze_map = mi['maze']

        # === ANTI-OSCILLATION: detect if we're stuck bouncing between same positions ===
        self._pos_history.append(pos)
        if len(self._pos_history) > 6:
            self._pos_history = self._pos_history[-6:]
        # Oscillating if last 4+ positions only visit 2-3 unique cells
        if len(self._pos_history) >= 4:
            recent = self._pos_history[-4:]
            unique = set(recent)
            self._oscillation_override = len(unique) <= 2
        else:
            self._oscillation_override = False

        # === OPPONENT APPROACH DETECTION ===
        curr_opp_dist = self._md(pos, opp.loc)
        if curr_opp_dist < self._prev_opp_dist:
            self._opp_approaching_count += 1
        else:
            self._opp_approaching_count = 0
        self._prev_opp_dist = curr_opp_dist
        opp_approaching = self._opp_approaching_count >= 3  # 3+ turns getting closer

        # === OPENING PHASE: first 15 turns, paint aggressively ===
        is_opening = board.turn_count < 30  # ~15 rounds

        # === KILL CHECK (always first) ===
        kill = self._try_kill(board, me, opp, pp, stamina)
        if kill is not None:
            return kill

        # === Check if near a hill (suppress flee behavior) ===
        cell = board.cells[pos.r][pos.c]
        near_any_hill = self._near_hill(board, pos, pp, radius=4)

        # === ESCAPE: on enemy territory near opponent — but NOT if near a hill ===
        # On maze maps, increase escape trigger range since corridors limit escape routes
        # If opponent is approaching (3+ turns getting closer), widen escape range
        escape_range = 4 if maze_map else 3
        if opp_approaching and cell.owner_parity != pp:
            escape_range = max(escape_range, 5)
        if not near_any_hill and cell.owner_parity == -pp and self._md(pos, opp.loc) <= escape_range:
            esc = self._escape_dir(board, pos, opp.loc, pp)
            if esc:
                return [Action.Move(esc)]

        # === LOW STAMINA RETREAT — but NOT if near a hill ===
        if not near_any_hill and stamina < 35 and cell.owner_parity != pp:
            ret = self._retreat_dir(board, pos, opp.loc, pp)
            if ret:
                return [Action.Move(ret)]

        # mi already set at top of play()

        # === DISTANCES ===
        my_dist = self._bfs(board, pos, pp)
        opp_dist = self._bfs_raw(board, opp.loc)

        total_hills = len(board.hills)
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)

        # === NO HILLS: pure territory game ===
        if total_hills == 0:
            return self._play_territory(board, pp, me, opp, my_dist, stamina)

        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD)

        # === TARGET SELECTION (with commitment) ===
        defend = self._defend_target(board, pp, me, opp, my_dist, dom_needed, their_hills)
        attack, attack_hid = self._pick_attack_target(board, pp, me, opp, my_dist, opp_dist, dom_needed)

        # Prefer attack for faster closing — only defend when opponent threatens domination
        target = None
        chosen_hid = None
        if defend and attack:
            dd = my_dist.get(defend, 999)
            ad = my_dist.get(attack, 999)
            if their_hills + 1 >= dom_needed:
                # Must defend — opponent is about to dominate
                target = defend
                chosen_hid = None
            elif our_hills + 1 >= dom_needed and ad <= dd + 4:
                # We're close to domination — push for the win
                target = attack
                chosen_hid = attack_hid
            elif dd <= ad + 2:
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

        # Update hill commitment
        if chosen_hid is not None:
            self._committed_hill_id = chosen_hid
        elif target is None:
            self._committed_hill_id = None

        # All hills ours — territory for tiebreak
        if target is None:
            if our_hills >= dom_needed:
                return self._play_territory(board, pp, me, opp, my_dist, stamina, in_sudden_death)
            # No hill target — fall back to territory mode
            return self._play_territory(board, pp, me, opp, my_dist, stamina, in_sudden_death)

        # TIEBREAK AWARENESS: tiebreak resolves hills-first, then territory.
        # If we're heading toward tiebreak and have fewer hills, prioritize hill capture.
        approaching_tiebreak = round_num >= 350
        if approaching_tiebreak and our_hills <= their_hills and attack:
            target = attack
            chosen_hid = attack_hid
            td = my_dist.get(target, 999)

        # In sudden death with hill lead — maximize territory for regen instead of risky attacks
        if in_sudden_death and our_hills > their_hills and not (their_hills + 1 >= dom_needed):
            return self._play_territory(board, pp, me, opp, my_dist, stamina, in_sudden_death)

        td = my_dist.get(target, 999)

        # === POWERUP DETOUR: only if directly on our path (1 step) and stamina low ===
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

        # contested: opponent is ON or right next to the target hill
        contested = opp_hill_dist <= 4
        # approaching: opponent is heading toward the hill
        approaching = 4 < opp_hill_dist <= 10
        # danger_zone: WE are close to the opponent — need collision safety
        danger_zone = dist_to_opp <= 3
        # reinforce: add layers when opponent threatens the hill
        reinforce = contested or approaching

        # === NEUTRAL HILLS CHECK ===
        # Are there uncaptured hills that neither player controls?
        neutral_hills_exist = any(
            h.controller_parity == 0 for h in board.hills.values()
        )
        # All hills are claimed (by us or opponent) — contest + territory mode
        all_hills_claimed = not neutral_hills_exist and total_hills > 0

        # === MAX LAYERS — don't over-reinforce ===
        # Small maps: cap at 2 layers for speed, dominate early
        # Neutral hills exist: cap at 2 — rush to claim them instead of layering
        # All hills claimed + contested: reinforce to 3 (opponent is actively fighting)
        # All hills claimed + not contested: cap at 2, spend stamina on territory
        if neutral_hills_exist:
            max_layers = 2  # always rush neutral hills over reinforcing
        elif not large_map:
            if contested and stamina >= 60:
                max_layers = 4
            elif contested:
                max_layers = 3
            else:
                max_layers = 2
        elif contested and stamina >= 60:
            max_layers = 4  # full reinforcement when contesting and can afford it
        elif contested:
            max_layers = 3
        else:
            max_layers = 2  # default: don't over-layer

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

        # === CHOKEPOINT: if path to hill has a bottleneck, secure it ===
        # When opponent is also near the chokepoint, controlling it is critical
        if td > 2 and not on_target_hill and target_cell and target_cell.hill_id != 0:
            choke, choke_dist = self._find_chokepoint_on_path(board, pos, target)
            if choke and choke_dist <= td:
                choke_cell = board.cells[choke.r][choke.c]
                opp_to_choke = self._md(opp.loc, choke)
                # Fight for the chokepoint if opponent could also reach it
                if opp_to_choke <= choke_dist + 4 and choke_cell.owner_parity != pp:
                    target = choke
                    td = my_dist.get(target, 999)

        # === STAMINA BUDGET ===
        # Aggressive budgets for faster closing — spend stamina to win, not hoard it
        # Scale buffer by wall density: more walls = longer real paths = need more stamina reserve
        if maze_map:
            buf = 20 if td > 3 else 10
        elif td > 5:
            buf = 40
        elif td > 2:
            buf = 20
        else:
            buf = 10

        if large_map and td > 8 and not maze_map:
            buf = 30

        if on_target_hill and contested:
            buf = 5

        # Adaptive wall density scaling: 0% walls → 1.0x, 30% → 1.3x, 45% → 1.45x, cap 1.5x
        wall_mult = min(1.0 + wall_ratio, 1.5)
        buf = int(buf * wall_mult)

        # When approaching sudden death or in it, be more aggressive to close out
        if approaching_sudden_death and not in_sudden_death:
            buf = max(buf - 10, 5)

        # During opening, minimize buffer — spend stamina on painting like top players
        if is_opening:
            buf = min(buf, 5)

        # === SHOULD WE PAINT TERRITORY EN ROUTE? ===
        # Data shows: in WINS P1 moves more (1.46 mpt), in LOSSES P1 paints more (1.31 ppt).
        # When racing for uncaptured hills, prioritize MOVEMENT over painting.
        # When all hills claimed or defending, paint territory for regen.
        if neutral_hills_exist and td > 2:
            paint_territory_en_route = False  # RUSH hills, don't paint trail
        else:
            paint_territory_en_route = all_hills_claimed or danger_zone

        # Double-layer trail when near opponent — single layer gets erased in one step
        trail_layers = 2 if danger_zone else 1

        # === PRE-MOVE PAINTING ===
        actions, stamina = self._paint_hill_cells(board, pp, pos, actions, stamina, painted, buf, reinforce, max_layers)

        # Trail painting: always on maze maps (regen critical in corridors),
        # when contesting claimed hills, danger zone, or large maps
        if paint_territory_en_route or maze_map:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target, trail_layers)
        elif large_map and td > 2:
            actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)
        elif stamina > 70 and td > 2:
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

        def _make_move(direction, from_pos, current_stamina, current_buf):
            """Build move action, using erase on 3+ layer opponent hill cells.
            NEVER erase near opponent (neutral cell = collision death)."""
            nonlocal stamina
            dest = from_pos + direction
            # Erase cannot initiate collision, and avoid erasing near opponent
            if not board.oob(dest) and dest != opp.loc and self._should_erase(board, pp, dest, opp.loc, current_stamina, current_buf):
                stamina -= GameConstants.ERASE_STEP_EXTRA_COST
                return Action.Move(direction, move_type=MoveType.ERASE)
            return Action.Move(direction)

        def _safe_to_move(from_pos, direction):
            """NEVER move onto opponent's cell unless we own it. #1 cause of losses."""
            dest = from_pos + direction
            if board.oob(dest):
                return False
            if dest == opp.loc:
                c = board.cells[dest.r][dest.c]
                return c.owner_parity == pp  # only collide on OUR cell
            return True

        d1 = _pick_step(pos)
        if d1 and _safe_to_move(pos, d1):
            # Pre-paint destination if neutral and opponent within reach
            nxt_pos = pos + d1
            if not board.oob(nxt_pos):
                nxt_c = board.cells[nxt_pos.r][nxt_pos.c]
                if (nxt_c.owner_parity == 0 and nxt_c.beacon_parity == 0
                        and not nxt_c.is_wall
                        and self._md(nxt_pos, opp.loc) <= opp_reach
                        and stamina >= GameConstants.PAINT_STAMINA_COST + buf):
                    actions.append(Action.Paint(nxt_pos))
                    stamina -= GameConstants.PAINT_STAMINA_COST
                    painted[nxt_pos] = painted.get(nxt_pos, 0) + 1
            actions.append(_make_move(d1, pos, stamina, buf))
            p1 = pos + d1

            actions, stamina = self._paint_hill_cells(board, pp, p1, actions, stamina, painted, buf, reinforce, max_layers)

            if paint_territory_en_route or maze_map or (large_map and td > 3):
                actions, stamina = self._paint_trail(board, pp, p1, actions, stamina, painted, buf, target, trail_layers)

            # 2nd move — lower threshold for faster closing
            # On high-wall maps, require more headroom (corridors are longer)
            move2_threshold = GameConstants.EXTRA_MOVE_COST + buf + (10 if maze_map else 0) + int(15 * wall_ratio)
            if td > 1 and stamina >= move2_threshold:
                d2 = _pick_step(p1)
                if d2 and _safe_to_move(p1, d2) and (targeting_hill or not self._is_dangerous(board, p1 + d2, opp.loc, pp)):
                    # Pre-paint destination if neutral and opponent within reach
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
                    actions.append(_make_move(d2, p1, stamina, buf))
                    stamina -= GameConstants.EXTRA_MOVE_COST
                    p2 = p1 + d2

                    actions, stamina = self._paint_hill_cells(board, pp, p2, actions, stamina, painted, max(buf - 10, 10), reinforce, max_layers)

                    if paint_territory_en_route or (large_map and td > 5):
                        actions, stamina = self._paint_trail(board, pp, p2, actions, stamina, painted, max(buf - 10, 15), target, trail_layers)

                    # 3rd move — top players use 3 moves from turn 1
                    # More aggressive threshold, especially during opening
                    move3_threshold = GameConstants.EXTRA_MOVE_COST * 2 + max(buf - 10, 5) + int(20 * wall_ratio)
                    if is_opening:
                        move3_threshold = GameConstants.EXTRA_MOVE_COST * 2 + 5
                    if not maze_map and td > 2 and stamina >= move3_threshold:
                        d3 = _pick_step(p2)
                        if d3 and _safe_to_move(p2, d3) and (targeting_hill or not self._is_dangerous(board, p2 + d3, opp.loc, pp)):
                            # Pre-paint destination if neutral and opponent within reach
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
                            actions.append(_make_move(d3, p2, stamina, buf))
                            stamina -= GameConstants.EXTRA_MOVE_COST * 2
                            p3 = p2 + d3
                            actions, stamina = self._paint_hill_cells(board, pp, p3, actions, stamina, painted, 10, reinforce, max_layers)
                            actions, stamina = self._paint_trail(board, pp, p3, actions, stamina, painted, 5, target, trail_layers)

        # === POST-MOVE ===
        final = self._simpos(board, pos, actions)
        actions, stamina = self._paint_hill_cells(board, pp, final, actions, stamina, painted, 10, reinforce, max_layers)

        if board.cells[final.r][final.c].hill_id != 0 and stamina > 40:
            actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target, trail_layers)
        elif paint_territory_en_route or (large_map and stamina > 30):
            actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target, trail_layers)

        # === GUARANTEE MOVE ===
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        # === COLLISION SAFETY ===
        # Skip safety backtrack when oscillating — the safety logic IS what causes oscillation
        if not self._oscillation_override:
            actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        # FIX: ensure at least one Move survives
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

        # === FINAL COLLISION GUARD ===
        # NEVER end on a cell where the opponent is unless we own it.
        # This prevents the #1 cause of losses (3 of 7 collision suicides).
        final = self._simpos(board, pos, actions)
        if final == opp.loc:
            fc = board.cells[final.r][final.c]
            if fc.owner_parity != pp:
                # We'd die! Strip moves back until safe.
                for trim in range(len(actions) - 1, -1, -1):
                    if isinstance(actions[trim], Action.Move):
                        trimmed = actions[:trim]
                        test_pos = self._simpos(board, pos, trimmed)
                        if test_pos != opp.loc:
                            actions = trimmed
                            break
                # Re-ensure we have at least one move
                if not any(isinstance(a, Action.Move) for a in actions):
                    fb = self._fallback_move(board, pp, me, opp)
                    if fb:
                        actions.append(fb)
                    else:
                        for d in Direction.cardinals():
                            nxt = pos + d
                            if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall and nxt != opp.loc:
                                actions.append(Action.Move(d))
                                break

        return actions if actions else [Action.Move(Direction.UP)]

    # ===============================================================
    # TARGET SELECTION
    # ===============================================================

    def _pick_attack_target(self, board, pp, me, opp, my_dist, opp_dist, dom_needed):
        """Pick best uncaptured hill to rush — race-aware scoring with commitment.
        Returns (target_loc, hill_id) or (None, None)."""
        best = None
        best_hid = None
        best_score = -9999
        our_hills = len(me.controlled_hills)

        # Validate current commitment: abandon if hill is now ours or doesn't exist
        committed = self._committed_hill_id
        if committed is not None:
            if committed not in board.hills or board.hills[committed].controller_parity == pp:
                self._committed_hill_id = None
                committed = None

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue

            # Distance to nearest ANY hill cell (including ours) = true proximity
            hill_dist = 999
            # Distance to nearest unpainted cell = where we need to go
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

            # Opponent race
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

            # Cap race_adv so a far-away "easy" hill can't beat a nearby one
            race_adv = min(opp_cost - our_cost, 30)
            size_bonus = max(0, 10 - hs) * 3
            close_to_dom = 80 if (our_hills + 1 >= dom_needed) else 0
            steal_bonus = 40 if hill.controller_parity == -pp else 0
            late_bonus = 30 if board.current_round > 400 else 0

            # Commitment bonus: strongly prefer the hill we're already heading to.
            commit_bonus = 60 if (hid == committed) else 0

            # Proximity bonus: strongly favor hills we're right next to.
            proximity_bonus = max(0, 100 - hill_dist * 20)

            # Opponent camping penalty: if opponent is sitting on this hill,
            # deprioritize it — go capture neutral hills first for domination.
            # Only contest camped hills when no better options exist.
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
                urgency += 0.5  # about to flip
            # Extra urgency if losing this hill lets opponent dominate
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
        If uncontested: only paint NEW cells (single layer, maximize coverage).
        If contested: also reinforce existing cells up to max_layers.
        max_layers: cap reinforcement depth (None = game max)."""
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
                    continue  # hill cells only

                base = abs(c.paint_value) if c.owner_parity == pp else 0
                added = painted.get(t, 0)
                if base + added >= cap:
                    continue

                is_new = (c.owner_parity == 0 and added == 0)

                if is_new:
                    score = 200
                elif contested:
                    # Reinforce only when opponent is nearby contesting
                    score = 80 + (cap - base - added) * 10
                else:
                    # Uncontested: skip reinforcement — save stamina for coverage
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
        """Paint ALL adjacent paintable cells — new cells first, then reinforce up to reinforce_to.
        This is the key difference vs top players who paint 4-6 cells per position."""
        COST = GameConstants.PAINT_STAMINA_COST

        # Collect all paintable candidates with scores
        candidates = []
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
            candidates.append((score, t))

        # Sort by score descending and paint ALL that we can afford
        candidates.sort(key=lambda x: -x[0])
        for _score, t in candidates:
            if stamina - COST < buf:
                break
            actions.append(Action.Paint(t))
            painted[t] = painted.get(t, 0) + 1
            stamina -= COST

        return actions, stamina

    def _paint_expand(self, board, pp, pos, actions, stamina, painted, buf, toward=None):
        """Paint best adjacent cell — hill cells first, then new cells, then reinforce.
        Used in territory/expand phase."""
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
                    # Only reinforce 1 layer at a time for efficiency
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

    def _play_territory(self, board, pp, me, opp, my_dist, stamina, in_sudden_death=False):
        """Expand territory for tiebreak. Used when no hills or all hills captured.
        In sudden death: prioritize staying near friendly territory for regen."""
        actions = []
        painted = {}
        pos = me.loc

        # Also defend any contested hills
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

        # In sudden death: keep tighter stamina buffer and prefer painting near self for regen
        buf = 25 if in_sudden_death else 15

        # Paint at current position
        actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, buf)

        # Find nearest frontier cell — in sudden death, prefer cells near existing territory
        target = self._frontier_target(board, pp, my_dist, pos, prefer_adjacent=in_sudden_death)
        if target:
            d1 = self._step_toward(board, pos, target, pp, opp.loc)
            if d1:
                actions.append(Action.Move(d1))
                p1 = pos + d1
                actions, stamina = self._paint_expand(board, pp, p1, actions, stamina, painted, buf)

                td = my_dist.get(target, 999)
                if td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + buf + 5:
                    d2 = self._step_toward(board, p1, target, pp, opp.loc)
                    if d2 and not self._is_dangerous(board, p1 + d2, opp.loc, pp):
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

    def _frontier_target(self, board, pp, my_dist, pos, prefer_adjacent=False):
        """Find best unowned cell adjacent to our territory — for expansion.
        prefer_adjacent: in sudden death, heavily favor cells next to friendly territory for regen."""
        best = None
        best_score = -9999

        for loc, d in my_dist.items():
            c = board.cells[loc.r][loc.c]
            if c.is_wall or c.owner_parity == pp or loc == pos:
                continue

            adj_friendly = 0
            for dr in Direction.cardinals():
                nl = loc + dr
                if not board.oob(nl) and board.cells[nl.r][nl.c].owner_parity == pp:
                    adj_friendly += 1

            score = -d * 3
            if c.owner_parity == 0:
                score += 20
            if adj_friendly > 0:
                score += 25
                if prefer_adjacent:
                    score += adj_friendly * 15  # strongly prefer dense friendly areas for regen
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
                # Direct: opponent on our cell
                if c.owner_parity == pp:
                    return [Action.Move(d)]
                # Paint-then-kill: opponent on neutral cell
                if c.owner_parity == 0 and c.beacon_parity == 0 and stam >= GameConstants.PAINT_STAMINA_COST:
                    return [Action.Paint(nxt), Action.Move(d)]

        # === MULTI-STEP KILLS: opponent on our cell ===
        if oc.owner_parity == pp:
            if dist == 2 and stam >= GameConstants.EXTRA_MOVE_COST + 20:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    # Don't walk through opponent territory to reach kill
                    if board.cells[mid.r][mid.c].owner_parity == -pp:
                        continue
                    for d2 in Direction.cardinals():
                        if mid + d2 == opp.loc:
                            return [Action.Move(d1), Action.Move(d2)]

        # === HUNT: opponent on neutral cell within range — paint & rush ===
        # Only attempt if we're confident: adjacent kill with paint is safe,
        # but multi-step hunts are risky and caused 3/5 collision deaths vs jmc
        if oc.owner_parity == 0 and oc.beacon_parity == 0 and dist == 2:
            if stam >= GameConstants.EXTRA_MOVE_COST + GameConstants.PAINT_STAMINA_COST + 15:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    # Only hunt through safe intermediate cells (our territory or neutral)
                    mc = board.cells[mid.r][mid.c]
                    if mc.owner_parity == -pp:
                        continue
                    if self._md(mid, opp.loc) == 1:
                        for d2 in Direction.cardinals():
                            if mid + d2 == opp.loc:
                                return [Action.Move(d1), Action.Paint(opp.loc), Action.Move(d2)]

        return None

    # ===============================================================
    # SAFETY
    # ===============================================================

    def _ensure_safe_ending(self, board, pp, start_pos, opp, actions, stamina, painted):
        """ABSOLUTE RULE: never end turn on a non-owned cell if opponent can reach us."""
        final = self._simpos(board, start_pos, actions)
        fc = board.cells[final.r][final.c]
        opp_reach = min(3, 1 + opp.stamina // GameConstants.EXTRA_MOVE_COST)
        mi = self._map_info
        maze_map = mi['maze'] if mi else False

        def _is_corridor(loc):
            """A cell with <=2 passable neighbors — trapped in a corridor."""
            passable = 0
            for d in Direction.cardinals():
                nxt = loc + d
                if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                    passable += 1
            return passable <= 2

        def _is_safe(loc):
            if board.oob(loc):
                return False
            c = board.cells[loc.r][loc.c]
            if c.owner_parity == pp and c.beacon_parity != -pp:
                # On maze maps, corridor cells near opponent are unsafe even if painted
                # because we have limited escape routes
                if maze_map and _is_corridor(loc) and self._md(loc, opp.loc) <= 4:
                    return False
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
            """Strip stale paints after move_idx, add one valid paint from new position."""
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

        # Strategy 1: pre-paint final cell (position unchanged, paints still valid)
        if (fc.owner_parity == 0 and fc.beacon_parity == 0
                and not fc.is_wall
                and stamina >= GameConstants.PAINT_STAMINA_COST
                and self._md(pre_pos, final) == 1):
            actions.insert(last_move_idx, Action.Paint(final))
            stamina -= GameConstants.PAINT_STAMINA_COST
            painted[final] = painted.get(final, 0) + 1
            return actions, stamina

        # Strategy 2: swap last move to safe cell + clean stale paints
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

        # Strategy 4: remove last move, stay at pre_pos
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

    def _should_erase(self, board, pp, loc, opp_loc, stamina, buf):
        """Return True if we should erase-step onto this cell instead of regular step.
        Use erase on opponent hill cells with 3+ layers — saves 2-3 revisits.
        NEVER erase if opponent is adjacent (erased cell becomes neutral = collision death)."""
        if board.oob(loc):
            return False
        c = board.cells[loc.r][loc.c]
        if c.beacon_parity != 0:
            return False
        if c.owner_parity != -pp:
            return False
        if c.hill_id == 0:
            return False
        if abs(c.paint_value) < 3:
            return False
        if stamina < GameConstants.ERASE_STEP_EXTRA_COST + buf:
            return False
        # CRITICAL: erase makes cell neutral. If opponent is adjacent, they can
        # walk in and we die on neutral. Don't erase when opponent is within 2.
        if self._md(loc, opp_loc) <= 2:
            return False
        return True

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
        """BFS from start, only walls blocked. For opponent distance estimation."""
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
        """BFS first-step toward target. Tries safe path first, relaxes if needed.
        If hill_target=True, skip danger avoidance — we want to contest the hill.
        Always avoids opponent beacon cells (lethal collision traps)."""
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
                # Never step onto opponent beacon (always lethal)
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
        """Step toward target, preferring our own painted cells when near opponent.
        Avoids stepping on neutral/opponent cells within opponent's collision range."""
        if start == target:
            return None

        mi = self._map_info
        # On maze maps, increase conservative collision range since corridors limit escape
        opp_reach = 3 if (mi and mi['maze']) else 2

        # Score each possible first step, then BFS-verify it leads to target
        candidates = []
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            # Never step onto opponent beacon
            if c.beacon_parity == -pp:
                continue
            if nxt == opp_loc and c.owner_parity != pp:
                continue

            nd = self._md(nxt, opp_loc)
            # Safety score: strongly prefer our own cells when near opponent
            safety = 0
            if c.owner_parity == pp and c.beacon_parity != -pp:
                safety = 20  # safe — we win collisions here
            elif c.owner_parity == -pp:
                safety = -60  # opponent cell = lethal near opp
            elif c.owner_parity == 0 and nd <= opp_reach:
                safety = -15  # neutral in range = vulnerable
            else:
                safety = 5  # neutral but out of range

            # Direction score: prefer moves toward target
            progress = self._md(start, target) - self._md(nxt, target)
            candidates.append((d, safety + progress * 10, nxt))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[1])

        # Try each candidate — verify it can actually reach the target via BFS
        for d, _score, nxt in candidates:
            if nxt == target:
                return d
            if self._can_reach(board, nxt, target):
                return d

        # Fallback: any direction toward target
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

    # ===============================================================
    # MAP CLASSIFICATION
    # ===============================================================

    def _classify_map(self, board):
        """Classify map characteristics once at start. Cached in self._map_info."""
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
            'maze': wall_ratio >= 0.40,  # 40%+ walls = true maze (spiral/disjoint are corridors, not mazes)
            'wall_ratio': wall_ratio,
            'num_hills': len(board.hills),
        }

    # ===============================================================
    # CHOKEPOINT DETECTION
    # ===============================================================

    def _is_chokepoint(self, board, loc):
        """A cell is a chokepoint if it has exactly 2 passable neighbors (corridor)."""
        passable = 0
        for d in Direction.cardinals():
            nxt = loc + d
            if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                passable += 1
        return passable <= 2

    def _find_chokepoint_on_path(self, board, start, target):
        """BFS from start to target, return the first chokepoint cell on the path.
        Returns (chokepoint_loc, distance) or (None, 999)."""
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
        # Trace path back, find chokepoints
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

    def _nearest_unpainted_hill_cell(self, board, pp, hill, my_dist):
        """Find nearest unpainted cell on a specific hill — for walking across it."""
        best = None
        best_d = 999
        for hloc in hill.cells:
            hc = board.cells[hloc.r][hloc.c]
            if hc.owner_parity == pp:
                continue  # already ours, skip
            d = my_dist.get(hloc, 999)
            if d < best_d:
                best_d = d
                best = hloc
        return best

    def _nearest_unclaimed_hill_cell(self, board, pp, hill, my_dist, contested):
        """Find next hill cell to claim/reclaim. Prioritizes:
        - Neutral cells (easy to paint)
        - Opponent cells we can step on to weaken
        - Prefers cells we can reach without collision risk when contested."""
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
            # Prefer neutral cells (paintable) over opponent cells (need stepping)
            if hc.owner_parity == 0:
                score = 100 - d * 3
            else:
                # Opponent cell — need to step on it to weaken
                layers = abs(hc.paint_value)
                score = 60 - d * 3 - layers * 5
            # When contested, avoid cells right next to opponent (collision risk)
            if contested and self._md(hloc, opp.loc) <= 1:
                score -= 50
            if score > best_score:
                best_score = score
                best = hloc
        return best

    def _safe_hill_step(self, board, start, target, pp, opp_loc):
        """Step toward target on a contested hill, avoiding collision death.
        Never step onto opponent beacons or opponent cells near opponent."""
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
            # Never step onto opponent beacon (always lethal)
            if c.beacon_parity == -pp:
                continue
            # Never walk onto opponent cell adjacent to opponent (collision = death)
            if c.owner_parity == -pp and self._md(nxt, opp_loc) <= 1:
                continue
            # Score: prefer moves toward target, on friendly/neutral cells
            score = -self._md(nxt, target) * 10
            if c.owner_parity == pp and c.beacon_parity != -pp:
                score += 15  # safe cell
            elif c.owner_parity == 0:
                score += 5
            elif c.owner_parity == -pp:
                score -= 10  # opponent cell at safe distance — still risky
            if nxt == target:
                score += 50
            if score > best_score:
                best_score = score
                best = d
        return best

    def _near_hill(self, board, pos, pp, radius=4):
        """Check if any uncaptured hill cell is within radius steps."""
        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            for hloc in hill.cells:
                if self._md(pos, hloc) <= radius:
                    return True
        return False

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
        """Any valid move, preferring safe and friendly cells. Avoids collision traps."""
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
            # Never step onto opponent beacon
            if c.beacon_parity == -pp:
                continue
            nd = self._md(nxt, opp.loc)
            if c.owner_parity == pp and c.beacon_parity != -pp:
                p = 20  # our cell — collision safe
            elif c.owner_parity == 0 and nd > opp_reach:
                p = 10  # neutral out of reach
            elif c.owner_parity == 0:
                p = -5  # neutral in reach — dangerous
            elif c.owner_parity == -pp:
                p = -20  # opponent cell
            else:
                p = 0
            if c.owner_parity == -pp and nd <= 2:
                p -= 20  # extra penalty for opponent cell near them
            if p > best_p:
                best_p = p
                best = d
        return Action.Move(best) if best else None

    def commentate(self, board, player_parity, time_left):
        p = board.get_player(player_parity)
        o = board.get_opponent(player_parity)
        return (
            f"dom h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} r={board.current_round}"
        )
