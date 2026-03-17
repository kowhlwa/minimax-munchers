"""
googleplz-timchan — Hybrid of timchan (proven base) + JMC insights

From timchan: phase system, danger avoidance, pathfinding, escape/retreat,
  kill detection, bid 6, opponent BFS race detection, directional painting,
  map-scaled thresholds.

From JMC:
  1. Defensive hill stacking — higher hill thicken priority (120 + gap*15)
  2. Regen-aware multi-moves — when local >= 12 ("stamina-infinite"),
     lower thresholds for 2nd/3rd moves and lower stamina buffer
  3. Softer race penalty — only penalize when gap > 2, skip in emergency
"""
from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict, List, Tuple
import math

from game import *

DANGER_STRICT = 2
DANGER_MODERATE = 1
MIN_REGEN_BASE = 6
REGEN_INFINITE = 12  # JMC: local cells where you become "stamina-infinite"


class PlayerController:

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 6

    def play(
        self, board: Board, player_parity: int, time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        opp_loc = opponent.loc
        actions: list = []
        stamina = player.stamina
        ex: Dict[Location, int] = {}

        # ---- Kill opportunities ----
        kill = self._try_kill(board, player, opponent, player_parity, stamina)
        if kill is not None:
            return kill

        # ---- Escape danger ----
        cur_cell = board.cells[player.loc.r][player.loc.c]
        dist_opp = self._md(player.loc, opp_loc)
        if cur_cell.owner_parity == -player_parity and dist_opp <= 3:
            esc = self._escape(board, player.loc, player_parity)
            if esc:
                actions.append(Action.Move(esc))
                np = player.loc + esc
                if not board.oob(np):
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 30)
                return actions

        # ---- Stamina conservation ----
        if stamina < 40 and cur_cell.owner_parity != player_parity:
            ret = self._retreat(board, player.loc, player_parity)
            if ret:
                actions.append(Action.Move(ret))
                np = player.loc + ret
                if not board.oob(np):
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 25)
                return actions

        # ---- Determine phase ----
        distances = self._bfs_all(board, player.loc, player_parity)
        opp_distances = self._opp_distances(board, player_parity)
        local = self._count_local(board, player.loc, player_parity)
        max_local = self._count_local_avail(board, player.loc)
        phase = self._phase(board, player, opponent, local, max_local, distances)
        has_regen = local >= REGEN_INFINITE  # JMC: stamina-infinite state

        # ---- RUSH / CONTEST / EMERGENCY ----
        if phase in ("rush", "contest", "emergency"):
            target = self._pick_hill_target(board, player_parity, distances, phase, opponent, opp_distances)
            # JMC: lower buffer when stamina-infinite
            buf = 15 if has_regen else 20
            # Always paint while moving — continuous territory trail
            actions, stamina = self._paint_smart(board, player_parity, player.loc, actions, stamina, ex, buf, target_loc=target)

            if target:
                td = distances.get(target)
                d1 = self._safe_step(board, player.loc, target, player_parity)
                if d1:
                    actions.append(Action.Move(d1))

                    # 2nd move — JMC: lower threshold when stamina-infinite
                    td_thresh_2 = 1 if has_regen else 2
                    if td and td > td_thresh_2 and stamina >= GameConstants.EXTRA_MOVE_COST + buf + 5:
                        p2 = self._simpos(board, player.loc, actions)
                        d2 = self._safe_step(board, p2, target, player_parity)
                        if d2 and not self._is_dangerous(board, p2 + d2, player_parity, 2):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST

                            # 3rd move — JMC: enabled when stamina-infinite or emergency
                            td_thresh_3 = 3 if has_regen else 4
                            if td and td > td_thresh_3 and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + buf:
                                p3 = self._simpos(board, player.loc, actions)
                                d3 = self._safe_step(board, p3, target, player_parity)
                                if d3 and not self._is_dangerous(board, p3 + d3, player_parity, 2):
                                    actions.append(Action.Move(d3))
                                    stamina -= GameConstants.EXTRA_MOVE_COST * 2

                                    # 4th move in emergency only
                                    if (phase == "emergency" and td > 6
                                            and stamina >= GameConstants.EXTRA_MOVE_COST * 3 + buf):
                                        p4 = self._simpos(board, player.loc, actions)
                                        d4 = self._safe_step(board, p4, target, player_parity)
                                        if d4 and not self._is_dangerous(board, p4 + d4, player_parity, 2):
                                            actions.append(Action.Move(d4))
                                            stamina -= GameConstants.EXTRA_MOVE_COST * 3

                    # Post-move painting
                    np = self._simpos(board, player.loc, actions)
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, buf, target_loc=target)

        # ---- FARM ----
        elif phase == "farm":
            actions, stamina = self._paint_smart(board, player_parity, player.loc, actions, stamina, ex, 15)
            ftarget = self._farm_target(board, player_parity, player.loc, opp_loc)
            if ftarget is None:
                ftarget = self._expand_target(board, player_parity, distances, player, opponent, opp_distances)
            if ftarget:
                d1 = self._safe_step(board, player.loc, ftarget, player_parity)
                if d1:
                    actions.append(Action.Move(d1))
                    np = self._simpos(board, player.loc, actions)
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 15)

        # ---- TERRITORY (endgame flood-fill) ----
        elif phase == "territory":
            actions, stamina = self._paint_smart(board, player_parity, player.loc, actions, stamina, ex, 10)
            etarget = self._territory_target(board, player_parity, distances, player)
            if etarget:
                d1 = self._safe_step(board, player.loc, etarget, player_parity)
                if d1:
                    actions.append(Action.Move(d1))
                    td = distances.get(etarget)
                    if td and td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + 15:
                        p2 = self._simpos(board, player.loc, actions)
                        d2 = self._safe_step(board, p2, etarget, player_parity)
                        if d2 and not self._is_dangerous(board, p2 + d2, player_parity, 1):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                    np = self._simpos(board, player.loc, actions)
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 10)

        # ---- EXPAND ----
        else:
            etarget = self._expand_target(board, player_parity, distances, player, opponent, opp_distances)
            actions, stamina = self._paint_smart(board, player_parity, player.loc, actions, stamina, ex, 15, target_loc=etarget)
            if etarget:
                d1 = self._safe_step(board, player.loc, etarget, player_parity)
                if d1:
                    actions.append(Action.Move(d1))
                    td = distances.get(etarget)
                    if td and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST + 25:
                        p2 = self._simpos(board, player.loc, actions)
                        d2 = self._safe_step(board, p2, etarget, player_parity)
                        if d2 and not self._is_dangerous(board, p2 + d2, player_parity, 1):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                    np = self._simpos(board, player.loc, actions)
                    actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 15, target_loc=etarget)

        # ---- Guarantee a move ----
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._any_move(board, player_parity)
            if fb:
                actions.append(fb)
                np = self._simpos(board, player.loc, actions)
                actions, stamina = self._paint_smart(board, player_parity, np, actions, stamina, ex, 15)
            else:
                return []

        return actions

    # ================================================================== #
    #  PHASE                                                              #
    # ================================================================== #

    def _phase(self, board, player, opponent, local, max_local, distances):
        total = len(board.hills)
        ours = len(player.controlled_hills)
        theirs = len(opponent.controlled_hills)
        has_regen = (local >= MIN_REGEN_BASE)
        area = board.board_size.r * board.board_size.c
        area_scale = max(0.6, min(1.3, max(board.board_size.r, board.board_size.c) / 20))

        if total == 0:
            if board.current_round > int(700 * area_scale) and area >= 400:
                return "territory"
            return "farm" if local < max(max_local * 0.65, 8) else "expand"

        dom = math.ceil(total * GameConstants.DOMINATION_WIN_THRESHOLD)

        if theirs + 1 >= dom and theirs > ours:
            return "emergency"

        if area >= 800 and not has_regen and local < 3:
            nearest = 999
            for loc in distances:
                c = board.cells[loc.r][loc.c]
                if c.hill_id != 0 and board.hills[c.hill_id].controller_parity != player.parity:
                    if distances[loc] < nearest:
                        nearest = distances[loc]
            if nearest > 12:
                return "farm"

        if board.current_round > int(400 * area_scale) and theirs > ours:
            return "emergency"
        if theirs > ours:
            return "contest"
        if ours == 0:
            for loc in distances:
                c = board.cells[loc.r][loc.c]
                if c.hill_id != 0 and board.hills[c.hill_id].controller_parity != player.parity:
                    return "rush"
        if total > 0 and ours + 1 >= dom:
            return "rush"

        if board.current_round > int(650 * area_scale) and ours > 0 and ours > theirs and area >= 400:
            return "territory"

        return "farm" if local < max(max_local * 0.65, 8) else "expand"

    # ================================================================== #
    #  HILL TARGET                                                        #
    # ================================================================== #

    def _pick_hill_target(self, board, pp, distances, phase, opponent, opp_distances=None):
        player = board.get_player(pp)
        best = None
        best_score = -9999.0
        area_scale = max(0.6, min(1.3, max(board.board_size.r, board.board_size.c) / 20))
        late_game = board.current_round > int(600 * area_scale)

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            opp_holds = (hill.controller_parity == opponent.parity)
            nd = nl = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == pp:
                    continue
                d = distances[hloc]
                if nd is None or d < nd:
                    nd = d
                    nl = hloc
            if nl is None or nd is None:
                continue
            hs = len(hill.cells)
            needed = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD) + 1
            our_count = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == pp)
            opp_count = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            to_paint = max(0, needed - our_count)
            cost = nd + to_paint * 1.5
            if cost <= 0:
                cost = 0.1
            eff = -cost + max(0, 12 - hs) * 2.0

            # JMC-softened race analysis: only penalize significant gaps, skip in emergency
            race_penalty = 0
            if opp_distances is not None and phase != "emergency":
                opp_nd = None
                opp_to_paint = max(0, needed - opp_count)
                for hloc in hill.cells:
                    if hloc not in opp_distances:
                        continue
                    hc = board.cells[hloc.r][hloc.c]
                    if hc.owner_parity == -pp:
                        continue
                    d = opp_distances[hloc]
                    if opp_nd is None or d < opp_nd:
                        opp_nd = d
                if opp_nd is not None:
                    opp_cost = opp_nd + opp_to_paint * 1.5
                    race_diff = cost - max(0.1, opp_cost)
                    # Only penalize/reward when gap > 2 turns
                    if abs(race_diff) > 2:
                        race_penalty = max(-80, min(80, -(race_diff - 2 * (1 if race_diff > 0 else -1)) * 15))

            if phase == "emergency":
                score = 1000 + eff * 30 + (200 if opp_holds else 0)
            elif phase == "contest":
                score = 500 + eff * 25 + (100 if opp_holds else 0) + race_penalty
            else:
                score = 300 + eff * 20 + race_penalty
            if late_game:
                score += 50 + (100 if opp_holds else 0)
            if score > best_score:
                best_score = score
                best = nl

        for hid in player.controlled_hills:
            hill = board.hills[hid]
            opp_cells = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            if opp_cells == 0:
                continue
            bl = bd = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == pp:
                    continue
                d = distances[hloc]
                if bd is None or d < bd:
                    bd = d
                    bl = hloc
            if bl:
                urg = opp_cells / len(hill.cells)
                score = 250 + urg * 250 - bd * 2
                if late_game:
                    score += 100
                if score > best_score:
                    best_score = score
                    best = bl

        for loc, dist in distances.items():
            if board.cells[loc.r][loc.c].powerup and dist <= 4:
                return loc

        return best

    # ================================================================== #
    #  FARM / EXPAND / TERRITORY TARGETS                                  #
    # ================================================================== #

    def _farm_target(self, board, pp, pos, opp_loc):
        best = None
        best_score = 0
        for d in Direction.cardinals():
            nxt = pos + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp_loc and c.owner_parity != pp):
                continue
            if self._danger(board, nxt, opp_loc, pp, DANGER_MODERATE):
                continue
            score = self._paint_value(board, pp, nxt)
            if score > best_score:
                best_score = score
                best = nxt
        return best

    def _paint_value(self, board, pp, pos):
        score = 0
        MV = GameConstants.MAX_PAINT_VALUE
        for d in Direction.cardinals():
            t = pos + d
            if board.oob(t):
                continue
            c = board.cells[t.r][t.c]
            if c.is_wall or c.beacon_parity != 0 or c.owner_parity == -pp:
                continue
            if c.owner_parity == 0:
                score += 20
                if c.hill_id != 0:
                    score += 50
            elif c.owner_parity == pp and abs(c.paint_value) < MV:
                score += (MV - abs(c.paint_value)) * 3
                if c.hill_id != 0:
                    score += 30
        return score

    def _expand_target(self, board, pp, distances, player, opponent, opp_distances=None):
        total = len(board.hills)
        ours = len(player.controlled_hills)
        theirs = len(opponent.controlled_hills)
        dom = math.ceil(total * GameConstants.DOMINATION_WIN_THRESHOLD) if total > 0 else 0
        close_dom = (total > 0 and ours + 1 >= dom)
        area_scale = max(0.6, min(1.3, max(board.board_size.r, board.board_size.c) / 20))
        late_game = board.current_round > int(600 * area_scale)
        hill_boost = 80.0 if late_game and ours <= theirs else 0.0

        best = None
        best_score = -9999.0

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            nd = nl = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == pp:
                    continue
                d = distances[hloc]
                if nd is None or d < nd:
                    nd = d
                    nl = hloc
            if nl is None or nd is None:
                continue
            hs = len(hill.cells)
            needed = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD) + 1
            oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == pp)
            opp_oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            cost = nd + max(0, needed - oc) * 1.5
            eff = -cost + max(0, 12 - hs) * 2.0

            race_penalty = 0
            if opp_distances is not None:
                opp_nd = None
                opp_to_paint = max(0, needed - opp_oc)
                for hloc in hill.cells:
                    if hloc not in opp_distances:
                        continue
                    hc = board.cells[hloc.r][hloc.c]
                    if hc.owner_parity == -pp:
                        continue
                    d = opp_distances[hloc]
                    if opp_nd is None or d < opp_nd:
                        opp_nd = d
                if opp_nd is not None:
                    opp_cost = opp_nd + opp_to_paint * 1.5
                    race_diff = cost - max(0.1, opp_cost)
                    if abs(race_diff) > 2:
                        race_penalty = max(-60, min(60, -(race_diff - 2 * (1 if race_diff > 0 else -1)) * 12))

            score = (500.0 if close_dom else 200.0) + eff * 20 + hill_boost + race_penalty
            if score > best_score:
                best_score = score
                best = nl

        for hid in player.controlled_hills:
            hill = board.hills[hid]
            oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            if oc == 0:
                continue
            bl = bd = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                if board.cells[hloc.r][hloc.c].owner_parity == pp:
                    continue
                d = distances[hloc]
                if bd is None or d < bd:
                    bd = d
                    bl = hloc
            if bl:
                score = 150 + (oc / len(hill.cells)) * 150 - bd * 2
                if score > best_score:
                    best_score = score
                    best = bl

        for loc, dist in distances.items():
            if board.cells[loc.r][loc.c].powerup:
                score = 140 - dist * 3
                if score > best_score:
                    best_score = score
                    best = loc

        for loc, dist in distances.items():
            c = board.cells[loc.r][loc.c]
            if c.hill_id != 0 or c.powerup or c.owner_parity != 0 or loc == player.loc:
                continue
            adj = any(
                not board.oob(loc + d) and board.cells[(loc + d).r][(loc + d).c].owner_parity == pp
                for d in Direction.cardinals()
            )
            score = (30.0 if adj else 8.0) - dist * 2.0
            if score > best_score:
                best_score = score
                best = loc

        return best

    def _territory_target(self, board, pp, distances, player):
        best = None
        best_score = -9999.0

        for hid in player.controlled_hills:
            hill = board.hills[hid]
            oc = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            if oc == 0:
                continue
            bl = bd = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                if board.cells[hloc.r][hloc.c].owner_parity == pp:
                    continue
                d = distances[hloc]
                if bd is None or d < bd:
                    bd = d
                    bl = hloc
            if bl:
                score = 300 + (oc / len(hill.cells)) * 200 - bd * 2
                if score > best_score:
                    best_score = score
                    best = bl

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp:
                continue
            nd = nl = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hc = board.cells[hloc.r][hloc.c]
                if hc.owner_parity == pp:
                    continue
                d = distances[hloc]
                if nd is None or d < nd:
                    nd = d
                    nl = hloc
            if nl and nd is not None:
                score = 250 - nd * 5
                if score > best_score:
                    best_score = score
                    best = nl

        for loc, dist in distances.items():
            c = board.cells[loc.r][loc.c]
            if c.is_wall or c.owner_parity != 0 or loc == player.loc:
                continue
            score = 50.0 - dist * 3.0
            if self._adj_friendly(board, loc, pp):
                score += 20.0
            if c.hill_id != 0:
                score += 30.0
            if score > best_score:
                best_score = score
                best = loc

        return best

    # ================================================================== #
    #  PAINT — JMC defensive stacking on hills                            #
    # ================================================================== #

    def _paint_smart(self, board, pp, pos, actions, stamina, ex, buf, target_loc=None):
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
                added = ex.get(t, 0)
                eff = base + added
                if eff >= MV:
                    continue
                is_new = (c.owner_parity == 0 and added == 0)
                is_hill = (c.hill_id != 0)
                score = 0
                if is_hill and is_new:
                    score = 200
                elif is_hill:
                    # JMC: defensive stacking — higher priority on hill thickening
                    score = 120 + (MV - eff) * 15
                elif is_new:
                    score = 50
                else:
                    score = 10 + (MV - eff) * 3
                # Directional bias toward target
                if target_loc is not None and score > 0:
                    if self._md(t, target_loc) < self._md(pos, target_loc):
                        score += 15
                if score > best_score:
                    best_score = score
                    best = t
            if best is None:
                break
            actions.append(Action.Paint(best))
            ex[best] = ex.get(best, 0) + 1
            stamina -= COST
        return actions, stamina

    # ================================================================== #
    #  KILLS                                                              #
    # ================================================================== #

    def _try_kill(self, board, player, opp, pp, stam):
        for d in Direction.cardinals():
            nxt = player.loc + d
            if board.oob(nxt) or nxt != opp.loc:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall:
                continue
            if c.owner_parity == pp:
                return [Action.Move(d)]
        if stam >= GameConstants.PAINT_STAMINA_COST:
            for d in Direction.cardinals():
                nxt = player.loc + d
                if board.oob(nxt) or nxt != opp.loc:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                if c.owner_parity == 0 and c.beacon_parity == 0:
                    return [Action.Paint(nxt), Action.Move(d)]
        if stam >= GameConstants.EXTRA_MOVE_COST + 20:
            oc = board.cells[opp.loc.r][opp.loc.c]
            if oc.owner_parity == pp:
                for d1 in Direction.cardinals():
                    mid = player.loc + d1
                    if board.oob(mid) or board.cells[mid.r][mid.c].is_wall or mid == opp.loc:
                        continue
                    for d2 in Direction.cardinals():
                        if mid + d2 == opp.loc:
                            return [Action.Move(d1), Action.Move(d2)]
        if stam >= 50:
            oc = board.cells[opp.loc.r][opp.loc.c]
            if oc.owner_parity == pp and self._md(player.loc, opp.loc) == 3:
                for d1 in Direction.cardinals():
                    p1 = player.loc + d1
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

    # ================================================================== #
    #  SAFETY                                                             #
    # ================================================================== #

    def _danger(self, board, loc, opp_loc, pp, radius):
        c = board.cells[loc.r][loc.c]
        is_opp = (c.owner_parity == -pp)
        if not is_opp:
            pv = c.paint_value
            if (pp == 1 and pv <= -2) or (pp == -1 and pv >= 2):
                is_opp = True
        if not is_opp:
            return False
        return self._md(loc, opp_loc) <= radius

    def _is_dangerous(self, board, dest, pp, radius):
        if board.oob(dest):
            return True
        return self._danger(board, dest, board.get_opponent(pp).loc, pp, radius)

    def _escape(self, board, start, pp):
        opp = board.get_opponent(pp)
        visited = {start}
        queue = deque()
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt) or nxt in visited:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or nxt == opp.loc:
                continue
            visited.add(nxt)
            if not self._danger(board, nxt, opp.loc, pp, DANGER_STRICT):
                return d
            queue.append((nxt, d))
        while queue:
            loc, fd = queue.popleft()
            for d in Direction.cardinals():
                nxt = loc + d
                if board.oob(nxt) or nxt in visited:
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or nxt == opp.loc:
                    continue
                visited.add(nxt)
                if not self._danger(board, nxt, opp.loc, pp, DANGER_STRICT):
                    return fd
                queue.append((nxt, fd))
        return None

    def _retreat(self, board, start, pp):
        opp = board.get_opponent(pp)
        visited = {start}
        queue = deque()
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt) or nxt in visited:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or nxt == opp.loc:
                continue
            if self._danger(board, nxt, opp.loc, pp, DANGER_MODERATE):
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
                if c.is_wall or nxt == opp.loc:
                    continue
                visited.add(nxt)
                if c.owner_parity == pp:
                    return fd
                queue.append((nxt, fd))
        return None

    # ================================================================== #
    #  PATHFINDING                                                        #
    # ================================================================== #

    def _bfs_all(self, board, start, pp):
        opp = board.get_opponent(pp)
        dist = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for dir in Direction.cardinals():
                nxt = loc + dir
                if nxt in dist or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                    continue
                dist[nxt] = d + 1
                queue.append((nxt, d + 1))
        return dist

    def _safe_step(self, board, start, target, pp):
        if start == target:
            return None
        r = self._bfs_step(board, start, target, pp, DANGER_STRICT)
        if r:
            return r
        r = self._bfs_step(board, start, target, pp, DANGER_MODERATE)
        if r:
            return r
        return self._bfs_step(board, start, target, pp, 0)

    def _bfs_step(self, board, start, target, pp, dr):
        opp = board.get_opponent(pp)
        opp_loc = opp.loc
        fsr = max(dr, DANGER_MODERATE)
        visited = {start}
        queue = deque()
        for d in Direction.cardinals():
            nxt = start + d
            if board.oob(nxt) or nxt in visited:
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp_loc and c.owner_parity != pp):
                continue
            if self._danger(board, nxt, opp_loc, pp, fsr):
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
                if c.is_wall or (nxt == opp_loc and c.owner_parity != pp):
                    continue
                if dr > 0 and self._danger(board, nxt, opp_loc, pp, dr):
                    continue
                visited.add(nxt)
                if nxt == target:
                    return fd
                queue.append((nxt, fd))
        return None

    # ================================================================== #
    #  UTILITIES                                                          #
    # ================================================================== #

    def _opp_distances(self, board, pp):
        opp = board.get_opponent(pp)
        dist = {opp.loc: 0}
        queue = deque([(opp.loc, 0)])
        while queue:
            loc, d = queue.popleft()
            for dir in Direction.cardinals():
                nxt = loc + dir
                if nxt in dist or board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall:
                    continue
                dist[nxt] = d + 1
                queue.append((nxt, d + 1))
        return dist

    def _md(self, a, b):
        return abs(a.r - b.r) + abs(a.c - b.c)

    def _simpos(self, board, start, actions):
        loc = start
        for a in actions:
            if isinstance(a, Action.Move) and a.direction is not None:
                c = loc + a.direction
                if not board.oob(c) and not board.cells[c.r][c.c].is_wall:
                    loc = c
        return loc

    def _count_local(self, board, loc, pp):
        r = GameConstants.ADJACENCY_RADIUS
        return sum(
            1 for dr in range(-r, r + 1) for dc in range(-r, r + 1)
            if not board.oob(Location(loc.r + dr, loc.c + dc))
            and board.cells[loc.r + dr][loc.c + dc].owner_parity == pp
        )

    def _count_local_avail(self, board, loc):
        r = GameConstants.ADJACENCY_RADIUS
        return sum(
            1 for dr in range(-r, r + 1) for dc in range(-r, r + 1)
            if not board.oob(Location(loc.r + dr, loc.c + dc))
            and not board.cells[loc.r + dr][loc.c + dc].is_wall
        )

    def _adj_friendly(self, board, loc, pp):
        for d in Direction.cardinals():
            n = loc + d
            if not board.oob(n) and board.cells[n.r][n.c].owner_parity == pp:
                return True
        return False

    def _any_move(self, board, pp):
        player = board.get_player(pp)
        opp = board.get_opponent(pp)
        best = None
        best_p = -999
        for d in Direction.cardinals():
            nxt = player.loc + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp.loc and c.owner_parity != pp):
                continue
            p = 5 if c.owner_parity == pp else (3 if c.owner_parity == 0 else 0)
            if self._danger(board, nxt, opp.loc, pp, DANGER_STRICT):
                p -= 10
            if p > best_p:
                best_p = p
                best = d
        return Action.Move(best) if best else None

    def commentate(self, board, player_parity, time_left):
        p = board.get_player(player_parity)
        o = board.get_opponent(player_parity)
        return (
            f"gplz-tc h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} r={board.current_round}"
        )
