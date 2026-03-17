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

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        # First-move advantage is worth ~6 stamina for hill rushing
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

        # Prioritize defense when opponent is close to domination
        target = None
        chosen_hid = None
        if defend and attack:
            dd = my_dist.get(defend, 999)
            ad = my_dist.get(attack, 999)
            if their_hills + 1 >= dom_needed or dd <= ad + 2:
                target = defend
                chosen_hid = None  # defense doesn't set commitment
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
                return self._play_territory(board, pp, me, opp, my_dist, stamina)
            # No hill target — fall back to territory mode (powerups found naturally)
            return self._play_territory(board, pp, me, opp, my_dist, stamina)

        td = my_dist.get(target, 999)

        # === POWERUP DETOUR: only if directly on our path (1 step) and stamina low ===
        if stamina < 50:
            pu = self._powerup_nearby(board, my_dist, 1)
            if pu:
                target = pu
                td = my_dist.get(pu, 999)

        # === MAP SIZE ===
        area = board.board_size.r * board.board_size.c
        large_map = area >= 400  # ~20x20+

        # === IS THE TARGET HILL CONTESTED? ===
        # contested = opponent close enough to threaten the hill
        # approaching = opponent heading toward hill but not yet on it
        # racing = both us and opponent heading for the same uncaptured hill
        contested = False
        approaching = False
        racing = False
        opp_hill_dist = 999
        target_cell = board.cells[target.r][target.c] if not board.oob(target) else None
        target_hill_obj = None
        if target_cell and target_cell.hill_id != 0:
            target_hid = target_cell.hill_id
            target_hill_obj = board.hills.get(target_hid)
            if target_hill_obj:
                opp_hill_dist = min(
                    (self._md(hloc, opp.loc) for hloc in target_hill_obj.cells),
                    default=999
                )
                if opp_hill_dist <= 4:
                    contested = True
                elif opp_hill_dist <= 10:
                    approaching = True
                # Racing: both heading for same hill, opponent is close enough to compete
                if opp_hill_dist <= td + 4 and target_hill_obj.controller_parity != pp:
                    racing = True

        # === OPPONENT PROXIMITY — collision-aware mode ===
        opp_nearby = self._md(pos, opp.loc) <= 4

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

        # === REINFORCE MODE ===
        reinforce = contested or approaching

        # === STAMINA BUDGET ===
        if racing:
            # Racing: minimal buffer — every stamina point goes to movement
            buf = 15
        elif td > 5:
            buf = 55
        elif td > 2:
            buf = 30
        else:
            buf = 15

        if large_map and td > 8 and not racing:
            buf = 35

        if on_target_hill and contested:
            buf = 10

        # === PRE-MOVE PAINTING ===
        # When racing, skip non-hill painting to save stamina for multi-moves
        actions, stamina = self._paint_hill_cells(board, pp, pos, actions, stamina, painted, buf, reinforce)

        if not racing:
            if large_map and td > 2:
                actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)
            elif stamina > 70 and td > 2:
                actions, stamina = self._paint_trail(board, pp, pos, actions, stamina, painted, buf, target)

        # === MOVEMENT ===
        targeting_hill = board.cells[target.r][target.c].hill_id != 0 if not board.oob(target) else False

        # Choose step function based on context
        def _pick_step(from_pos):
            if on_target_hill and contested:
                return self._safe_hill_step(board, from_pos, target, pp, opp.loc)
            elif opp_nearby:
                # When opponent is close, prefer stepping on our own paint
                return self._collision_aware_step(board, from_pos, target, pp, opp.loc)
            else:
                return self._step_toward(board, from_pos, target, pp, opp.loc, hill_target=targeting_hill)

        d1 = _pick_step(pos)
        if d1:
            actions.append(Action.Move(d1))
            p1 = pos + d1

            actions, stamina = self._paint_hill_cells(board, pp, p1, actions, stamina, painted, buf, reinforce)

            if not racing and large_map and td > 3:
                actions, stamina = self._paint_trail(board, pp, p1, actions, stamina, painted, buf, target)

            # 2nd move — more aggressive threshold when racing
            move2_cost = GameConstants.EXTRA_MOVE_COST + buf + (0 if racing else 5)
            if td > 2 and stamina >= move2_cost:
                d2 = _pick_step(p1)
                if d2 and (targeting_hill or racing or not self._is_dangerous(board, p1 + d2, opp.loc, pp)):
                    actions.append(Action.Move(d2))
                    stamina -= GameConstants.EXTRA_MOVE_COST
                    p2 = p1 + d2

                    actions, stamina = self._paint_hill_cells(board, pp, p2, actions, stamina, painted, max(buf - 10, 10), reinforce)

                    if not racing and large_map and td > 5:
                        actions, stamina = self._paint_trail(board, pp, p2, actions, stamina, painted, max(buf - 10, 15), target)

                    # 3rd move — always attempt when racing
                    move3_cost = GameConstants.EXTRA_MOVE_COST * 2 + (10 if racing else max(buf - 15, 15))
                    if (td > 5 or racing) and stamina >= move3_cost:
                        d3 = _pick_step(p2)
                        if d3 and (targeting_hill or racing or not self._is_dangerous(board, p2 + d3, opp.loc, pp)):
                            actions.append(Action.Move(d3))
                            stamina -= GameConstants.EXTRA_MOVE_COST * 2
                            p3 = p2 + d3
                            actions, stamina = self._paint_hill_cells(board, pp, p3, actions, stamina, painted, 10, reinforce)

                            # 4th move — only when racing and can afford it
                            if racing and td > 6 and stamina >= GameConstants.EXTRA_MOVE_COST * 3 + 10:
                                d4 = _pick_step(p3)
                                if d4:
                                    actions.append(Action.Move(d4))
                                    stamina -= GameConstants.EXTRA_MOVE_COST * 3

        # === POST-MOVE: paint remaining hill cells ===
        final = self._simpos(board, pos, actions)
        actions, stamina = self._paint_hill_cells(board, pp, final, actions, stamina, painted, 10, reinforce)

        if not racing:
            if board.cells[final.r][final.c].hill_id != 0 and stamina > 40:
                actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target)
            elif large_map and stamina > 30:
                actions, stamina = self._paint_trail(board, pp, final, actions, stamina, painted, 20, target)

        # === GUARANTEE MOVE ===
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        # === COLLISION SAFETY ===
        # Never end turn on a neutral cell within opponent's striking range.
        # Opponent can multi-move (up to 3-4 steps) and kill us on neutral cells.
        actions, stamina = self._ensure_safe_ending(board, pp, pos, opp, actions, stamina, painted)

        return actions if actions else []

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
            # A hill 1 step away gets +80, 2 steps +60, 3 steps +40, etc.
            proximity_bonus = max(0, 100 - hill_dist * 20)

            score = (race_adv * 8 + size_bonus + close_to_dom + steal_bonus
                     + late_bonus + commit_bonus + proximity_bonus
                     - to_paint * 5)

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

    def _paint_hill_cells(self, board, pp, pos, actions, stamina, painted, buf, contested=False):
        """Paint adjacent unpainted hill cells.
        If uncontested: only paint NEW cells (single layer, maximize coverage).
        If contested: also reinforce existing cells with layers."""
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
                if c.hill_id == 0:
                    continue  # hill cells only

                base = abs(c.paint_value) if c.owner_parity == pp else 0
                added = painted.get(t, 0)
                if base + added >= MV:
                    continue

                is_new = (c.owner_parity == 0 and added == 0)

                if is_new:
                    score = 200
                elif contested:
                    # Reinforce only when opponent is nearby contesting
                    score = 80 + (MV - base - added) * 10
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

    def _paint_trail(self, board, pp, pos, actions, stamina, painted, buf, toward=None):
        """Paint at most 1 new (unowned) adjacent cell — builds regen without waste."""
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
            # Only paint NEW cells (no layering on uncontested territory)
            if c.is_wall or c.beacon_parity != 0 or c.owner_parity != 0:
                continue
            if painted.get(t, 0) > 0:
                continue

            score = 10
            if c.hill_id != 0:
                score = 100  # always paint new hill cells
            if toward and self._md(t, toward) < self._md(pos, toward):
                score += 8  # bias toward our target
            if score > best_score:
                best_score = score
                best = t

        if best:
            actions.append(Action.Paint(best))
            painted[best] = painted.get(best, 0) + 1
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

    def _play_territory(self, board, pp, me, opp, my_dist, stamina):
        """Expand territory for tiebreak. Used when no hills or all hills captured."""
        actions = []
        painted = {}
        pos = me.loc

        # Also defend any contested hills
        if len(me.controlled_hills) > 0:
            for hid in me.controlled_hills:
                hill = board.hills[hid]
                oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
                if oc > 0:
                    # Find nearest contested cell
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

        # Paint at current position
        actions, stamina = self._paint_expand(board, pp, pos, actions, stamina, painted, 15)

        # Find nearest frontier cell (unowned, adjacent to our territory)
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
                    if d2 and not self._is_dangerous(board, p1 + d2, opp.loc, pp):
                        actions.append(Action.Move(d2))
                        stamina -= GameConstants.EXTRA_MOVE_COST
                        p2 = p1 + d2
                        actions, stamina = self._paint_expand(board, pp, p2, actions, stamina, painted, 15)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp)
            if fb:
                actions.append(fb)

        return actions if actions else []

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

        return actions if actions else []

    def _frontier_target(self, board, pp, my_dist, pos):
        """Find best unowned cell adjacent to our territory — for expansion."""
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
                score += 25  # grow outward from existing territory
            if c.hill_id != 0:
                score += 40
            if c.powerup and d <= 2:
                score += 20  # mild bonus, don't chase far

            if score > best_score:
                best_score = score
                best = loc

        return best

    # ===============================================================
    # KILLS
    # ===============================================================

    def _try_kill(self, board, me, opp, pp, stam):
        """Check kill opportunities: adjacent on our cell, paint-then-kill, 2-step."""
        # Direct: opponent adjacent on our cell
        for d in Direction.cardinals():
            nxt = me.loc + d
            if board.oob(nxt) or nxt != opp.loc:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.owner_parity == pp:
                return [Action.Move(d)]

        # Paint-then-kill: opponent adjacent on neutral cell
        if stam >= GameConstants.PAINT_STAMINA_COST:
            for d in Direction.cardinals():
                nxt = me.loc + d
                if board.oob(nxt) or nxt != opp.loc:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                if c.owner_parity == 0 and c.beacon_parity == 0:
                    return [Action.Paint(nxt), Action.Move(d)]

        # 2-step: opponent on our cell, we're 2 away
        if stam >= GameConstants.EXTRA_MOVE_COST + 20:
            oc = board.cells[opp.loc.r][opp.loc.c]
            if oc.owner_parity == pp:
                for d1 in Direction.cardinals():
                    mid = me.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        if mid + d2 == opp.loc:
                            return [Action.Move(d1), Action.Move(d2)]

        # 3-step: opponent on our cell, we're 3 away
        if stam >= GameConstants.EXTRA_MOVE_COST + GameConstants.EXTRA_MOVE_COST * 2 + 20:
            oc = board.cells[opp.loc.r][opp.loc.c]
            if oc.owner_parity == pp and self._md(me.loc, opp.loc) == 3:
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

        return None

    # ===============================================================
    # SAFETY
    # ===============================================================

    def _ensure_safe_ending(self, board, pp, start_pos, opp, actions, stamina, painted):
        """Prevent ending turn on a neutral cell within opponent's striking range.
        Opponent can multi-move up to 3 steps to collide on neutral = we die."""
        final = self._simpos(board, start_pos, actions)
        fc = board.cells[final.r][final.c]
        opp_dist_to_final = self._md(final, opp.loc)

        # Safe if: we own the cell, or opponent can't reach us next turn
        opp_reach = min(3, 1 + opp.stamina // GameConstants.EXTRA_MOVE_COST)
        if fc.owner_parity == pp or opp_dist_to_final > opp_reach:
            return actions, stamina

        # Near a hill — don't reroute away. Only try pre-painting.
        near_hill = fc.hill_id != 0 or self._near_hill(board, final, pp, radius=3)

        # Strategy 1: pre-paint the final cell before the last move.
        if fc.owner_parity == 0 and fc.beacon_parity == 0 and not fc.is_wall:
            last_move_idx = None
            for i in range(len(actions) - 1, -1, -1):
                if isinstance(actions[i], Action.Move):
                    last_move_idx = i
                    break

            if last_move_idx is not None and stamina >= GameConstants.PAINT_STAMINA_COST:
                pre_pos = self._simpos(board, start_pos, actions[:last_move_idx])
                if self._md(pre_pos, final) == 1:
                    actions.insert(last_move_idx, Action.Paint(final))
                    stamina -= GameConstants.PAINT_STAMINA_COST
                    painted[final] = painted.get(final, 0) + 1
                    return actions, stamina

        # Near a hill: don't reroute — we'd lose the race. Accept the risk.
        if near_hill:
            return actions, stamina

        # Strategy 2: swap last move to end on a friendly cell instead.
        # Only used away from hills where rerouting doesn't cost us.
        last_move_idx = None
        for i in range(len(actions) - 1, -1, -1):
            if isinstance(actions[i], Action.Move):
                last_move_idx = i
                break

        if last_move_idx is not None:
            pre_pos = self._simpos(board, start_pos, actions[:last_move_idx])
            best_dir = None
            best_score = -999
            for d in Direction.cardinals():
                nxt = pre_pos + d
                if board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                    continue
                score = 0
                nd = self._md(nxt, opp.loc)
                if c.owner_parity == pp:
                    score += 50
                elif c.owner_parity == 0 and nd > opp_reach:
                    score += 30
                elif c.owner_parity == -pp:
                    score -= 100
                else:
                    score -= 20
                score += nd * 2
                if score > best_score:
                    best_score = score
                    best_dir = d
            if best_dir and best_score > 0:
                actions[last_move_idx] = Action.Move(best_dir)

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
        If hill_target=True, skip danger avoidance — we want to contest the hill."""
        if start == target:
            return None

        # When rushing a hill, don't avoid opponent territory — go straight there
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

        opp_reach = 2  # conservative collision range

        # Score each possible first step, then BFS-verify it leads to target
        candidates = []
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if nxt == opp_loc and c.owner_parity != pp:
                continue

            nd = self._md(nxt, opp_loc)
            # Safety score: strongly prefer our own cells when near opponent
            safety = 0
            if c.owner_parity == pp:
                safety = 20  # safe — we win collisions here
            elif c.owner_parity == 0 and nd <= opp_reach:
                safety = -15  # neutral in range = vulnerable
            elif c.owner_parity == -pp and nd <= opp_reach:
                safety = -40  # opponent cell in range = death
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
        Don't step onto opponent-painted cells adjacent to opponent."""
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
            # Never walk onto opponent cell adjacent to opponent (collision = death)
            if c.owner_parity == -pp and self._md(nxt, opp_loc) <= 1:
                continue
            # Score: prefer moves toward target, on friendly/neutral cells
            score = -self._md(nxt, target) * 10
            if c.owner_parity == pp:
                score += 15  # safe cell
            elif c.owner_parity == 0:
                score += 5
            else:
                score -= 5  # opponent cell but safe distance
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

    def _simpos(self, board, start, actions):
        loc = start
        for a in actions:
            if isinstance(a, Action.Move) and a.direction is not None:
                nxt = loc + a.direction
                if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                    loc = nxt
        return loc

    def _fallback_move(self, board, pp, me, opp):
        """Any valid move, preferring safe and friendly cells."""
        best = None
        best_p = -999
        for d in Direction.cardinals():
            nxt = me.loc + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                continue
            p = 5 if c.owner_parity == pp else (3 if c.owner_parity == 0 else 0)
            if c.owner_parity == -pp and self._md(nxt, opp.loc) <= 2:
                p -= 10
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
