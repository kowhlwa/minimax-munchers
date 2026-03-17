"""
googleplz v2 — JMC Logistical Hill Dominator (corrected)

Phase flow (the real JMC sequence):
  1. RUSH to nearest hill immediately (triple-move T1 with 100 stamina)
  2. ANCHOR at hill — stop and paint solid 5x5 until local >= 12 (+24 regen)
  3. DOMINATE — now "stamina-infinite", sprint 3 moves/turn to next hill
  4. Repeat until 75% domination win

Key mechanics:
  - Bid 0: keep full 100 stamina for T1 triple-move
  - Path-paint every cell you traverse (umbilical cord = regen + collision safety)
  - Defensive stack hill cells to max layers
  - Danger avoidance: DANGER_STRICT=2 with cascading fallback (proven from timchan)
"""
from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Dict, List, Tuple, Optional
import math

from game import *

DANGER_STRICT = 2
DANGER_MODERATE = 1
ANCHOR_TARGET = 12  # local cells needed before leaving hill (~+24 regen/turn)


class PlayerController:

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def play(
        self, board: Board, player_parity: int, time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        pp = player_parity
        player = board.get_player(pp)
        opponent = board.get_opponent(pp)
        opp_loc = opponent.loc
        stamina = player.stamina
        actions: list = []
        ex: Dict[Location, int] = {}
        area = board.board_size.r * board.board_size.c
        small_map = area < 200

        # ---- 1. Kill opportunities ----
        kill = self._try_kill(board, player, opponent, pp, stamina)
        if kill is not None:
            return kill

        # ---- 2. Escape danger ----
        cur_cell = board.cells[player.loc.r][player.loc.c]
        dist_opp = self._md(player.loc, opp_loc)
        if cur_cell.owner_parity == -pp and dist_opp <= 3:
            esc = self._escape(board, player.loc, pp)
            if esc:
                actions.append(Action.Move(esc))
                np = self._simpos(board, player.loc, actions)
                actions, stamina = self._paint_smart(board, pp, np, actions, stamina, ex, 20)
                return actions

        # ---- 3. Stamina conservation ----
        if stamina < 40 and cur_cell.owner_parity != pp:
            ret = self._retreat(board, player.loc, pp)
            if ret:
                actions.append(Action.Move(ret))
                np = self._simpos(board, player.loc, actions)
                actions, stamina = self._paint_smart(board, pp, np, actions, stamina, ex, 25)
                return actions

        # ---- 4. Compute state ----
        distances = self._bfs_all(board, player.loc, pp)
        local = self._count_local(board, player.loc, pp)
        phase = self._phase(board, player, opponent, local, distances, small_map)

        # ---- 5. Execute phase ----
        if phase == "anchor":
            return self._do_anchor(board, pp, player, opponent, distances, actions, stamina, ex)
        elif phase in ("rush", "emergency", "dominate"):
            has_regen = local >= ANCHOR_TARGET
            return self._do_rush(board, pp, player, opponent, distances, actions,
                                 stamina, ex, phase, small_map, has_regen)
        else:  # expand
            return self._do_expand(board, pp, player, opponent, distances, actions, stamina, ex)

    # ================================================================== #
    #  PHASE                                                              #
    # ================================================================== #

    def _phase(self, board, player, opponent, local, distances, small_map):
        total = len(board.hills)
        ours = len(player.controlled_hills)
        theirs = len(opponent.controlled_hills)

        if total == 0:
            return "expand"

        dom = math.ceil(total * GameConstants.DOMINATION_WIN_THRESHOLD)

        # Emergency: opponent near domination
        if theirs + 1 >= dom and theirs > ours:
            return "emergency"
        if board.current_round > 400 and theirs > ours:
            return "emergency"

        # Do we have a regen base? (the "stamina-infinite" check)
        has_regen = local >= ANCHOR_TARGET

        # We own at least one hill — should we anchor here?
        if ours > 0 and not has_regen and not small_map:
            # Are we near a hill we control?
            for hid in player.controlled_hills:
                hill = board.hills[hid]
                for hloc in hill.cells:
                    if self._md(player.loc, hloc) <= 3:
                        return "anchor"

        # No hills yet — rush to first one
        if ours == 0:
            return "rush"

        # Close to domination — sprint
        if ours + 1 >= dom:
            return "dominate"

        # Behind on hills — contest
        if theirs > ours:
            return "rush"

        # Ahead/tied with regen base — go capture more
        if has_regen:
            return "dominate"

        # Ahead but no regen base and not near our hill — rush anyway
        return "rush"

    # ================================================================== #
    #  RUSH — Beeline to nearest uncaptured hill                          #
    # ================================================================== #

    def _do_rush(self, board, pp, player, opponent, distances, actions,
                 stamina, ex, phase, small_map, has_regen):
        target = self._pick_hill_target(board, pp, distances, player, opponent, phase)

        if target is None:
            return self._do_expand(board, pp, player, opponent, distances, actions, stamina, ex)

        td = distances.get(target)
        # Determine how many moves we can afford
        # With regen base (stamina-infinite): always try 3 moves
        # Without: be more conservative
        if has_regen or small_map:
            buf = 15
        else:
            buf = 25

        # Paint before moving (umbilical cord)
        actions, stamina = self._paint_smart(board, pp, player.loc, actions, stamina, ex, buf, target_loc=target)

        d1 = self._safe_step(board, player.loc, target, pp)
        if d1:
            actions.append(Action.Move(d1))

            # 2nd move
            can_2nd = td and td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + buf + 5
            if has_regen:
                can_2nd = td and td > 1 and stamina >= GameConstants.EXTRA_MOVE_COST + buf
            if can_2nd:
                p2 = self._simpos(board, player.loc, actions)
                d2 = self._safe_step(board, p2, target, pp)
                if d2 and not self._is_dangerous(board, p2 + d2, pp, 2):
                    actions.append(Action.Move(d2))
                    stamina -= GameConstants.EXTRA_MOVE_COST

                    # 3rd move — with regen base or emergency or small map
                    can_3rd = (td and td > 3
                               and stamina >= GameConstants.EXTRA_MOVE_COST * 2 + buf
                               and (has_regen or phase == "emergency" or small_map))
                    if can_3rd:
                        p3 = self._simpos(board, player.loc, actions)
                        d3 = self._safe_step(board, p3, target, pp)
                        if d3 and not self._is_dangerous(board, p3 + d3, pp, 2):
                            actions.append(Action.Move(d3))
                            stamina -= GameConstants.EXTRA_MOVE_COST * 2

                            # 4th move in emergency only
                            if (phase == "emergency" and td > 5
                                    and stamina >= GameConstants.EXTRA_MOVE_COST * 3 + buf):
                                p4 = self._simpos(board, player.loc, actions)
                                d4 = self._safe_step(board, p4, target, pp)
                                if d4 and not self._is_dangerous(board, p4 + d4, pp, 2):
                                    actions.append(Action.Move(d4))
                                    stamina -= GameConstants.EXTRA_MOVE_COST * 3

            # Post-move painting
            np = self._simpos(board, player.loc, actions)
            actions, stamina = self._paint_smart(board, pp, np, actions, stamina, ex, buf, target_loc=target)

        # Fallback
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._any_move(board, pp)
            if fb:
                actions.append(fb)
                np = self._simpos(board, player.loc, actions)
                actions, stamina = self._paint_smart(board, pp, np, actions, stamina, ex, 15)
            else:
                return []

        return actions

    # ================================================================== #
    #  ANCHOR — Paint 5x5 at hill until regen base is built               #
    # ================================================================== #

    def _do_anchor(self, board, pp, player, opponent, distances, actions, stamina, ex):
        """Stay near hill and paint dense 5x5. Move only to maximize painting."""
        buf = 10
        opp_loc = opponent.loc

        # Paint current position first
        actions, stamina = self._paint_anchor(board, pp, player.loc, actions, stamina, ex, buf)

        # Move to adjacent cell with most painting opportunities
        # Stay near our hill — prefer moves that keep us within 3 of a hill cell
        best_dir = None
        best_score = -1
        for d in Direction.cardinals():
            nxt = player.loc + d
            if board.oob(nxt):
                continue
            c = board.cells[nxt.r][nxt.c]
            if c.is_wall or (nxt == opp_loc and c.owner_parity != pp):
                continue
            # Danger check
            if self._danger(board, nxt, opp_loc, pp, DANGER_STRICT):
                continue
            # Stay near our hills
            near_hill = False
            for hid in player.controlled_hills:
                for hloc in board.hills[hid].cells:
                    if self._md(nxt, hloc) <= 3:
                        near_hill = True
                        break
                if near_hill:
                    break
            if not near_hill:
                continue
            score = self._anchor_move_score(board, pp, nxt)
            if score > best_score:
                best_score = score
                best_dir = d

        # Fallback: any safe move
        if best_dir is None:
            for d in Direction.cardinals():
                nxt = player.loc + d
                if board.oob(nxt):
                    continue
                c = board.cells[nxt.r][nxt.c]
                if c.is_wall or (nxt == opp_loc and c.owner_parity != pp):
                    continue
                score = self._anchor_move_score(board, pp, nxt)
                if score > best_score:
                    best_score = score
                    best_dir = d

        if best_dir:
            actions.append(Action.Move(best_dir))
            np = player.loc + best_dir
            if not board.oob(np):
                actions, stamina = self._paint_anchor(board, pp, np, actions, stamina, ex, buf)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._any_move(board, pp)
            if fb:
                actions.append(fb)
            else:
                return []

        return actions

    def _anchor_move_score(self, board, pp, pos):
        """Score position for anchoring: count unpainted/underthickened cells."""
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

    # ================================================================== #
    #  EXPAND                                                             #
    # ================================================================== #

    def _do_expand(self, board, pp, player, opponent, distances, actions, stamina, ex):
        actions, stamina = self._paint_smart(board, pp, player.loc, actions, stamina, ex, 15)

        best = None
        best_score = -9999.0
        for loc, dist in distances.items():
            c = board.cells[loc.r][loc.c]
            if c.is_wall or loc == player.loc:
                continue
            if c.powerup:
                score = 200 - dist * 3
            elif c.hill_id != 0 and c.owner_parity != pp:
                score = 150 - dist * 4
            elif c.owner_parity == 0:
                adj = self._adj_friendly(board, loc, pp)
                score = (30.0 if adj else 8.0) - dist * 2.0
            else:
                continue
            if score > best_score:
                best_score = score
                best = loc

        if best:
            d1 = self._safe_step(board, player.loc, best, pp)
            if d1:
                actions.append(Action.Move(d1))
                np = self._simpos(board, player.loc, actions)
                actions, stamina = self._paint_smart(board, pp, np, actions, stamina, ex, 15)

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._any_move(board, pp)
            if fb:
                actions.append(fb)
            else:
                return []

        return actions

    # ================================================================== #
    #  HILL TARGETING                                                     #
    # ================================================================== #

    def _pick_hill_target(self, board, pp, distances, player, opponent, phase):
        best = None
        best_score = -9999.0
        late_game = board.current_round > 600

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
            to_paint = max(0, needed - our_count)
            cost = nd + to_paint * 1.5
            if cost <= 0:
                cost = 0.1
            eff = -cost + max(0, 12 - hs) * 2.0

            if phase == "emergency":
                score = 1000 + eff * 30 + (200 if opp_holds else 0)
            elif phase == "dominate":
                score = 500 + eff * 25 + (150 if opp_holds else 0)
            else:
                score = 300 + eff * 20
            if late_game:
                score += 50 + (100 if opp_holds else 0)
            if score > best_score:
                best_score = score
                best = nl

        # Defend our hills under attack
        for hid in player.controlled_hills:
            hill = board.hills[hid]
            opp_cells = sum(1 for h in hill.cells if board.cells[h.r][h.c].owner_parity == -pp)
            if opp_cells == 0:
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
                urg = opp_cells / len(hill.cells)
                score = 250 + urg * 250 - bd * 2
                if late_game:
                    score += 100
                if score > best_score:
                    best_score = score
                    best = bl

        # Nearby powerups
        for loc, dist in distances.items():
            if board.cells[loc.r][loc.c].powerup and dist <= 4:
                return loc

        return best

    # ================================================================== #
    #  PAINTING                                                           #
    # ================================================================== #

    def _paint_smart(self, board, pp, pos, actions, stamina, ex, buf, target_loc=None):
        """General painting: hill priority + directional bias + defensive layering."""
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
                    score = 120 + (MV - eff) * 15  # Defensive stacking on hills
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

    def _paint_anchor(self, board, pp, pos, actions, stamina, ex, buf):
        """Anchor painting: maximize local density, heavy hill stacking."""
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
                    score = 250
                elif is_hill:
                    score = 150 + (MV - eff) * 15
                elif is_new:
                    score = 60
                else:
                    score = 15 + (MV - eff) * 5
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
        # 3-step kill
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
    #  SAFETY (borrowed from timchan — proven)                            #
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
    #  PATHFINDING (borrowed from timchan — proven)                       #
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
            f"googleplz h={len(p.controlled_hills)}v{len(o.controlled_hills)}"
            f" t={board.get_territory_count(player_parity)}v{board.get_territory_count(-player_parity)}"
            f" s={p.stamina}/{p.max_stamina} r={board.current_round}"
        )
