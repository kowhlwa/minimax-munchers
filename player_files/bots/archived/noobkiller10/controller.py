from collections import deque
from typing import Union, Iterable, Optional, Dict, Set
import math

from game import *


class PlayerController:
    """
    noobkiller10 — pivoted strategy
    Large maps: rush hills fast, avoid opponent contact, end turns on owned tiles.
    Small maps: safety-first, paint-then-step to avoid neutral collisions.
    """

    def __init__(self, player_parity: int, time_left):
        self.pp = player_parity
        self._map_info = None
        self._turn = 0
        # Opponent model
        self._opp_prev_loc = None
        self._opp_moves = []
        self._opp_on_hill = 0
        self._opp_close = 0
        self._opp_style = None
        # Commitment
        self._committed_hill_id = None
        # BFS cache
        self._bfs_cache_turn = -1
        self._bfs_cache = None
        self._opp_bfs_cache_turn = -1
        self._opp_bfs_cache = None

    # ===============================================================
    # Entry points
    # ===============================================================

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        if not board.hills:
            return 0
        mi = self._classify_map(board)
        min_dist = self._min_hill_dist(board, board.get_player(player_parity).loc)
        if mi["small"]:
            if min_dist <= 3:
                return 12
            if min_dist <= 6:
                return 8
            return 5
        if mi["large"]:
            return 3
        return 5

    def play(
        self, board: Board, player_parity: int, time_left,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        pp = player_parity
        me = board.get_player(pp)
        opp = board.get_opponent(pp)
        pos = me.loc
        stamina = me.stamina
        actions: list = []
        self._turn += 1

        # Update opponent model
        self._track_opponent(board, opp, me)

        # Map info + distances
        if self._map_info is None:
            self._map_info = self._classify_map(board)
        mi = self._map_info

        my_dist = self._bfs_cached(board, pos)
        opp_dist = self._opp_bfs_cached(board, opp.loc)

        total_hills = len(board.hills)
        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD) if total_hills > 0 else 0
        our_hills = len(me.controlled_hills)
        their_hills = len(opp.controlled_hills)

        opp_reach = self._opponent_reach(opp.stamina)
        dist_to_opp = self._md(pos, opp.loc)
        small_map = mi["small"]
        aggressive = small_map and dist_to_opp <= 6

        # Mode selection
        mode = self._select_mode(total_hills, our_hills, their_hills, dom_needed, mi, dist_to_opp, aggressive)

        # No hills: territory play
        if total_hills == 0:
            return self._play_territory(board, pp, me, opp, my_dist, opp_reach, stamina)

        # Choose target
        target, target_hid = self._choose_target(board, pp, me, opp, my_dist, opp_dist, mode, dom_needed)
        if aggressive:
            pressure = self._pressure_target(board, pp, pos, opp.loc, my_dist)
            if pressure is not None:
                target, target_hid = pressure, None
        if target_hid is not None:
            self._committed_hill_id = target_hid
        elif target is None:
            self._committed_hill_id = None

        if target is None:
            return self._play_territory(board, pp, me, opp, my_dist, opp_reach, stamina)

        # If we're on a hill we're contesting, sweep unowned cells
        target = self._hill_sweep_target(board, pp, pos, target, my_dist)

        # Movement budget
        td = my_dist.get(target, 999)
        move_budget = self._move_budget(mi, td, stamina, dist_to_opp, aggressive)

        # Small-map aggression: don't default to safety-paint, pressure instead

        # Execute moves
        for move_index in range(move_budget):
            cur = self._simpos(pos, actions)
            step = self._choose_step(board, pp, cur, target, my_dist, opp.loc, opp_reach, mi, aggressive)
            if step is None:
                break
            nxt = cur + step
            if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall:
                break

            extra_cost = GameConstants.EXTRA_MOVE_COST * move_index
            if move_index >= 1 and stamina < extra_cost:
                break

            # If last move would end on risky neutral, try paint-then-step
            if move_index == move_budget - 1:
                if self._is_risky_end(board, pp, nxt, opp.loc, opp_reach, aggressive):
                    if self._can_paint(board, pp, cur, nxt, stamina, extra_cost):
                        actions.append(Action.Paint(nxt))
                        stamina -= GameConstants.PAINT_STAMINA_COST
                    else:
                        alt = self._best_safe_step(board, pp, cur, opp.loc, opp_reach)
                        if alt is not None:
                            step = alt
                            nxt = cur + step

            cell = board.cells[nxt.r][nxt.c]
            # Erase if this is a contested hill cell with thick opponent paint
            if (cell.hill_id != 0 and cell.owner_parity == -pp
                    and abs(cell.paint_value) >= 2
                    and stamina >= extra_cost + GameConstants.ERASE_STEP_EXTRA_COST + 5):
                actions.append(Action.Move(step, move_type=MoveType.ERASE))
                stamina -= extra_cost + GameConstants.ERASE_STEP_EXTRA_COST
            else:
                actions.append(Action.Move(step))
                if move_index >= 1:
                    stamina -= extra_cost

            # After moving, paint adjacent hill cell if available (speed capture)
            cur = cur + step
            actions, stamina = self._paint_adjacent_hill(board, pp, cur, actions, stamina)

        # Guarantee at least one move
        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp, opp_reach)
            if fb:
                actions.append(fb)

        return actions if actions else [self._any_move(board, pp, me, opp)]

    # ===============================================================
    # Mode + Targeting
    # ===============================================================

    def _select_mode(self, total_hills, our_hills, their_hills, dom_needed, mi, dist_to_opp, aggressive):
        if total_hills == 0:
            return "territory"
        if their_hills + 1 >= dom_needed:
            return "defend"
        if our_hills == 0:
            return "rush"
        if aggressive:
            return "pressure"
        if mi["large"] and dist_to_opp > 8:
            return "avoid_rush"
        return "rush" if our_hills < their_hills else "expand"

    def _choose_target(self, board, pp, me, opp, my_dist, opp_dist, mode, dom_needed):
        if mode == "defend":
            defend = self._defend_target(board, pp, me, my_dist)
            if defend is not None:
                return defend, None

        best = None
        best_hid = None
        best_score = -9999
        committed = self._committed_hill_id

        for hid, hill in board.hills.items():
            if hill.controller_parity == pp and mode != "defend":
                continue

            hs = len(hill.cells)
            need = math.ceil(hs * GameConstants.HILL_CONTROL_THRESHOLD)
            our_cells = 0
            opp_cells = 0
            nearest_loc = None
            nearest_dist = 999
            opp_nearest = 999
            sep = 999

            for hloc in hill.cells:
                cell = board.cells[hloc.r][hloc.c]
                if cell.owner_parity == pp:
                    our_cells += 1
                elif cell.owner_parity == -pp:
                    opp_cells += 1

                d = my_dist.get(hloc, 999)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_loc = hloc

                od = opp_dist.get(hloc, 999)
                if od < opp_nearest:
                    opp_nearest = od

                md = self._md(hloc, opp.loc)
                if md < sep:
                    sep = md

            if nearest_loc is None:
                continue

            our_need = max(0, need - our_cells)
            opp_need = max(0, need - opp_cells)
            our_cost = nearest_dist + our_need * 1.5
            opp_cost = opp_nearest + opp_need * 1.5

            race_margin = opp_cost - our_cost
            separation_bonus = sep * 2
            steal_bonus = 25 if hill.controller_parity == -pp else 0
            dom_bonus = 35 if (len(me.controlled_hills) + 1 >= dom_needed) else 0
            commit_bonus = 40 if committed is not None and hid == committed else 0

            score = race_margin * 6 + separation_bonus + steal_bonus + dom_bonus + commit_bonus - our_need * 4

            # Avoid switching hills unless clearly better
            if committed is not None and hid != committed and race_margin < 6:
                score -= 30

            if score > best_score:
                best_score = score
                best = nearest_loc
                best_hid = hid

        return best, best_hid

    def _pressure_target(self, board, pp, cur, opp_loc, my_dist):
        candidates = []
        for d in Direction.cardinals():
            loc = opp_loc + d
            if board.oob(loc) or board.cells[loc.r][loc.c].is_wall:
                continue
            candidates.append(loc)
        if not candidates:
            return None

        # Prefer our-owned adjacent squares, then neutral, then any
        owned = [c for c in candidates if board.cells[c.r][c.c].owner_parity == pp]
        neutral = [c for c in candidates if board.cells[c.r][c.c].owner_parity == 0]
        pool = owned or neutral or candidates
        return min(pool, key=lambda l: my_dist.get(l, 999))

    def _defend_target(self, board, pp, me, my_dist):
        worst = None
        worst_score = -1
        for hid in me.controlled_hills:
            hill = board.hills[hid]
            opp_cells = [hloc for hloc in hill.cells if board.cells[hloc.r][hloc.c].owner_parity == -pp]
            if not opp_cells:
                continue
            # urgency by opponent presence on our hill
            urgency = len(opp_cells) / len(hill.cells)
            if urgency > worst_score:
                # target nearest contested cell
                nearest = min(opp_cells, key=lambda l: my_dist.get(l, 999))
                worst = nearest
                worst_score = urgency
        return worst

    def _hill_sweep_target(self, board, pp, pos, target, my_dist):
        cur_hid = board.cells[pos.r][pos.c].hill_id
        if cur_hid == 0:
            return target
        hill = board.hills.get(cur_hid)
        if hill is None or hill.controller_parity == pp:
            return target
        best = None
        best_d = 999
        for hloc in hill.cells:
            if board.cells[hloc.r][hloc.c].owner_parity == pp:
                continue
            d = my_dist.get(hloc, 999)
            if d < best_d:
                best = hloc
                best_d = d
        return best if best is not None else target

    # ===============================================================
    # Movement + Safety
    # ===============================================================

    def _choose_step(self, board, pp, cur, target, my_dist, opp_loc, opp_reach, mi, aggressive):
        best = None
        best_cost = 999999
        for d in Direction.cardinals():
            nxt = cur + d
            if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall:
                continue
            if nxt == opp_loc:
                continue
            dist = my_dist.get(nxt, 9999)
            cell = board.cells[nxt.r][nxt.c]
            risk = 0
            if cell.owner_parity == -pp:
                risk += 12 if not aggressive else 4
            elif cell.owner_parity == 0:
                if self._md(nxt, opp_loc) <= opp_reach:
                    risk += 6 if not aggressive else 1
            if self._md(nxt, opp_loc) <= 2:
                risk += 4 if not aggressive else -2
            if mi["large"]:
                # prioritize separation on large maps
                risk += max(0, 6 - self._md(nxt, opp_loc))
            cost = dist + risk
            if cost < best_cost:
                best_cost = cost
                best = d
        return best

    def _best_safe_step(self, board, pp, cur, opp_loc, opp_reach) -> Optional[Direction]:
        best = None
        best_score = -9999
        for d in Direction.cardinals():
            nxt = cur + d
            if board.oob(nxt) or board.cells[nxt.r][nxt.c].is_wall:
                continue
            if nxt == opp_loc:
                continue
            cell = board.cells[nxt.r][nxt.c]
            dist = self._md(nxt, opp_loc)
            if cell.owner_parity == -pp:
                score = -1000
            elif cell.owner_parity == pp:
                score = 100 + dist
            else:
                score = dist - (50 if dist <= opp_reach else 0)
            if score > best_score:
                best_score = score
                best = d
        return best

    def _fallback_move(self, board, pp, me, opp, opp_reach) -> Optional[Action.Move]:
        step = self._best_safe_step(board, pp, me.loc, opp.loc, opp_reach)
        if step is not None:
            return Action.Move(step)
        return None

    def _any_move(self, board, pp, me, opp) -> Action.Move:
        for d in Direction.cardinals():
            nxt = me.loc + d
            if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall:
                return Action.Move(d)
        return Action.Move(Direction.UP)

    def _is_risky_end(self, board, pp, loc, opp_loc, opp_reach, aggressive) -> bool:
        cell = board.cells[loc.r][loc.c]
        if cell.owner_parity == pp:
            return False
        if cell.owner_parity == -pp:
            return True
        if aggressive:
            return self._md(loc, opp_loc) <= max(1, opp_reach - 1)
        return self._md(loc, opp_loc) <= opp_reach

    def _can_paint(self, board, pp, cur, target, stamina, extra_cost) -> bool:
        if stamina < GameConstants.PAINT_STAMINA_COST + extra_cost:
            return False
        if board.oob(target):
            return False
        cell = board.cells[target.r][target.c]
        if cell.is_wall or cell.beacon_parity != 0:
            return False
        if cell.owner_parity != 0 and cell.owner_parity != pp:
            return False
        md = self._md(cur, target)
        return md == 1

    # ===============================================================
    # Painting helpers
    # ===============================================================

    def _paint_adjacent_hill(self, board, pp, pos, actions, stamina):
        if stamina < GameConstants.PAINT_STAMINA_COST:
            return actions, stamina
        for d in Direction.cardinals():
            loc = pos + d
            if board.oob(loc):
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall or cell.hill_id == 0:
                continue
            if cell.owner_parity == 0 and cell.beacon_parity == 0:
                actions.append(Action.Paint(loc))
                stamina -= GameConstants.PAINT_STAMINA_COST
                break
        return actions, stamina

    def _paint_adjacent_safe(self, board, pp, pos, actions, stamina):
        if stamina < GameConstants.PAINT_STAMINA_COST:
            return actions, stamina
        for d in Direction.cardinals():
            loc = pos + d
            if board.oob(loc):
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == 0:
                actions.append(Action.Paint(loc))
                stamina -= GameConstants.PAINT_STAMINA_COST
                break
        return actions, stamina

    # ===============================================================
    # Territory mode
    # ===============================================================

    def _play_territory(self, board, pp, me, opp, my_dist, opp_reach, stamina):
        pos = me.loc
        actions = []

        # Paint adjacent neutral if safe
        for d in Direction.cardinals():
            loc = pos + d
            if board.oob(loc):
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall:
                continue
            if cell.owner_parity == 0 and stamina >= GameConstants.PAINT_STAMINA_COST:
                actions.append(Action.Paint(loc))
                stamina -= GameConstants.PAINT_STAMINA_COST
                break

        # Move to nearest neutral but avoid risky endpoints
        target = None
        best_d = 999
        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                cell = board.cells[r][c]
                if cell.is_wall:
                    continue
                if cell.owner_parity == 0:
                    loc = Location(r, c)
                    d = my_dist.get(loc, 999)
                    if d < best_d:
                        best_d = d
                        target = loc

        if target is not None:
            step = self._choose_step(board, pp, pos, target, my_dist, opp.loc, opp_reach, {"large": False})
            if step is not None:
                nxt = pos + step
                if self._is_risky_end(board, pp, nxt, opp.loc, opp_reach, False):
                    alt = self._best_safe_step(board, pp, pos, opp.loc, opp_reach)
                    if alt is not None:
                        step = alt
                actions.append(Action.Move(step))

        if not any(isinstance(a, Action.Move) for a in actions):
            fb = self._fallback_move(board, pp, me, opp, opp_reach)
            if fb:
                actions.append(fb)

        return actions if actions else [self._any_move(board, pp, me, opp)]

    # ===============================================================
    # Map + Opponent Model
    # ===============================================================

    def _classify_map(self, board: Board) -> Dict[str, object]:
        r, c = board.board_size.r, board.board_size.c
        size = r * c
        walls = 0
        for rr in range(r):
            for cc in range(c):
                if board.cells[rr][cc].is_wall:
                    walls += 1
        wall_ratio = walls / max(1, size)
        return {
            "small": size <= 15 * 15,
            "large": size >= 20 * 20,
            "maze": wall_ratio >= 0.18,
        }

    def _min_hill_dist(self, board, pos):
        best = 999
        for hill in board.hills.values():
            for hloc in hill.cells:
                d = self._md(pos, hloc)
                if d < best:
                    best = d
        return best

    def _track_opponent(self, board: Board, opp, me) -> None:
        if self._opp_prev_loc is not None:
            d = self._md(self._opp_prev_loc, opp.loc)
            self._opp_moves.append(d)
        self._opp_prev_loc = opp.loc

        if board.cells[opp.loc.r][opp.loc.c].hill_id != 0:
            self._opp_on_hill += 1
        if self._md(opp.loc, me.loc) <= 3:
            self._opp_close += 1

        if len(self._opp_moves) >= 10:
            avg = sum(self._opp_moves[-10:]) / 10
            hill_focus = self._opp_on_hill / max(1, self._turn)
            close = self._opp_close / max(1, self._turn)
            if hill_focus > 0.45:
                self._opp_style = "hill_rusher"
            elif close > 0.35:
                self._opp_style = "aggressor"
            elif avg <= 1.0:
                self._opp_style = "camper"
            else:
                self._opp_style = "balanced"

    # ===============================================================
    # BFS + utils
    # ===============================================================

    def _bfs_cached(self, board, start: Location) -> Dict[Location, int]:
        if board.turn_count == self._bfs_cache_turn and self._bfs_cache is not None:
            return self._bfs_cache
        dist = self._bfs(board, start)
        self._bfs_cache_turn = board.turn_count
        self._bfs_cache = dist
        return dist

    def _opp_bfs_cached(self, board, start: Location) -> Dict[Location, int]:
        if board.turn_count == self._opp_bfs_cache_turn and self._opp_bfs_cache is not None:
            return self._opp_bfs_cache
        dist = self._bfs(board, start)
        self._opp_bfs_cache_turn = board.turn_count
        self._opp_bfs_cache = dist
        return dist

    def _bfs(self, board, start: Location) -> Dict[Location, int]:
        dist: Dict[Location, int] = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for d in Direction.cardinals():
                nxt = cur + d
                if board.oob(nxt) or nxt in dist:
                    continue
                if board.cells[nxt.r][nxt.c].is_wall:
                    continue
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
        return dist

    def _simpos(self, start, actions) -> Location:
        pos = start
        for a in actions:
            if isinstance(a, Action.Move):
                if a.move_type == MoveType.BEACON_TRAVEL and a.beacon_target is not None:
                    pos = a.beacon_target
                elif a.direction is not None:
                    pos = pos + a.direction
        return pos

    def _md(self, a: Location, b: Location) -> int:
        return abs(a.r - b.r) + abs(a.c - b.c)

    def _opponent_reach(self, opp_stamina: int) -> int:
        if opp_stamina >= 60:
            return 4
        if opp_stamina >= 30:
            return 3
        if opp_stamina >= 10:
            return 2
        return 1

    def _move_budget(self, mi, td, stamina, dist_to_opp, aggressive):
        if stamina < 20:
            return 1
        if mi["small"] and aggressive and stamina >= 35:
            return 2
        if mi["small"] and dist_to_opp <= 4:
            return 1
        if mi["large"] and td >= 6 and stamina >= 60 and dist_to_opp > 6:
            return 3
        if td >= 3 and stamina >= 35:
            return 2
        return 1
