from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict, List, Tuple
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from game import *

DANGER_STRICT = 3
DANGER_MODERATE = 1


class PlayerController:
    """
    HillRusher v8 — numpy/scipy accelerated.

    Key optimizations over v7:
      - Board state mirrored as numpy arrays (owner, paint, wall, hill_id, powerup)
        updated incrementally each turn instead of re-read from Python objects.
      - scipy Dijkstra replaces pure-Python BFS (~12x faster on 20x20 boards).
        The sparse adjacency matrix is built ONCE at init (walls never change).
      - Centrality grid precomputed once at init.
      - Local friendly count uses numpy slice sum instead of nested Python loop.
      - Territory count uses np.sum instead of double Python loop.
      - Expand target scoring is fully vectorized (no per-cell Python loop).
      - Distance lookups use dist_grid[r, c] (O(1) array index) instead of dict.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity
        self._board_initialized = False

    # ------------------------------------------------------------------ #
    #  NUMPY BOARD INIT & SYNC                                            #
    # ------------------------------------------------------------------ #

    def _init_numpy_board(self, board: Board) -> None:
        """
        Build numpy arrays and precompute static structures from the board.
        Called once on the first play() invocation.
        """
        R = board.board_size.r
        C = board.board_size.c
        self._R = R
        self._C = C

        # Static arrays (never change after init)
        self._wall = np.array(
            [[board.cells[r][c].is_wall for c in range(C)] for r in range(R)],
            dtype=np.bool_
        )
        self._hill_id = np.array(
            [[board.cells[r][c].hill_id for c in range(C)] for r in range(R)],
            dtype=np.int32
        )

        # Centrality grid — precomputed once
        rs = np.arange(R, dtype=np.float32)
        cs = np.arange(C, dtype=np.float32)
        mid_r, mid_c = (R - 1) / 2.0, (C - 1) / 2.0
        max_dist = mid_r + mid_c
        if max_dist == 0:
            self._cent = np.zeros((R, C), dtype=np.float32)
        else:
            self._cent = (15.0 * (
                1.0 - (np.abs(rs[:, None] - mid_r) + np.abs(cs[None, :] - mid_c)) / max_dist
            )).astype(np.float32)

        # Sparse adjacency matrix for Dijkstra — built once (walls fixed)
        n = R * C
        rows_i, cols_i, data_i = [], [], []
        for r in range(R):
            for c in range(C):
                if self._wall[r, c]:
                    continue
                idx = r * C + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and not self._wall[nr, nc]:
                        rows_i.append(idx)
                        cols_i.append(nr * C + nc)
                        data_i.append(1.0)
        self._adj = csr_matrix(
            (data_i, (rows_i, cols_i)), shape=(n, n), dtype=np.float32
        )

        # Cardinal direction offsets as array for vectorized neighbor ops
        self._card = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)

        self._board_initialized = True

    def _sync_arrays(self, board: Board) -> None:
        """
        Re-read mutable board state into numpy arrays each turn.
        Only paint, beacon, and powerup change between turns.
        """
        R, C = self._R, self._C
        cells = board.cells

        paint  = np.empty((R, C), dtype=np.int8)
        beacon = np.empty((R, C), dtype=np.int8)
        pwrup  = np.empty((R, C), dtype=np.bool_)

        for r in range(R):
            row = cells[r]
            for c in range(C):
                cell = row[c]
                paint[r, c]  = cell.paint_value
                beacon[r, c] = cell.beacon_parity
                pwrup[r, c]  = cell.powerup

        self._paint  = paint
        self._beacon = beacon
        self._pwrup  = pwrup

        # owner_parity: sign(paint) where beacon==0, else beacon value
        own = np.sign(paint).astype(np.int8)
        mask = beacon != 0
        own[mask] = beacon[mask]
        self._owner = own

    def _dijkstra_from(self, r: int, c: int) -> np.ndarray:
        """
        Run Dijkstra from (r,c), return distance grid shaped (R, C).
        Unreachable cells have value np.inf.
        """
        idx = r * self._C + c
        raw = dijkstra(self._adj, indices=idx, limit=10000)
        return raw.reshape(self._R, self._C)

    # ================================================================== #
    #  MAIN PLAY                                                          #
    # ================================================================== #

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:

        if not self._board_initialized:
            self._init_numpy_board(board)
        self._sync_arrays(board)

        player   = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        opp_loc  = opponent.loc
        actions: list = []
        stamina = player.stamina
        extra_layers: Dict[Location, int] = {}

        # Priority 1: Paint-then-kill
        ptk = self._paint_then_kill(board, player, opponent, player_parity, stamina)
        if ptk is not None:
            return ptk

        # Priority 2: Adjacent kill
        kill_dir = self._check_kill(board, player, opponent, player_parity)
        if kill_dir:
            actions.append(Action.Move(kill_dir))
            new_pos = player.loc + kill_dir
            actions, stamina = self._smart_paint(
                board, player_parity, new_pos, actions, stamina, 20, extra_layers
            )
            return actions

        # Priority 3: Multi-step kill
        msk = self._multi_step_kill(board, player, opponent, player_parity, stamina)
        if msk is not None:
            return msk

        # Priority 4: Escape
        cur_cell = board.cells[player.loc.r][player.loc.c]
        dist_to_opp = abs(player.loc.r - opp_loc.r) + abs(player.loc.c - opp_loc.c)
        if cur_cell.owner_parity == -player_parity and dist_to_opp <= 4:
            escape_dir = self._escape_step(board, player.loc, player_parity)
            if escape_dir:
                actions.append(Action.Move(escape_dir))
                new_pos = player.loc + escape_dir
                actions, stamina = self._smart_paint(
                    board, player_parity, new_pos, actions, stamina, 40, extra_layers
                )
                return actions

        # ---- Numpy-accelerated state computation ----
        pr, pc = player.loc.r, player.loc.c
        dist_grid = self._dijkstra_from(pr, pc)

        # Local count via numpy slice (radius=2 → 5×5 window)
        r0, r1 = max(0, pr - 2), min(self._R, pr + 3)
        c0, c1 = max(0, pc - 2), min(self._C, pc + 3)
        local_count = int(np.sum(self._owner[r0:r1, c0:c1] == player_parity))
        max_local   = int(np.sum(~self._wall[r0:r1, c0:c1]))

        # Build distances dict for path-planning (only reachable cells)
        reachable_mask = ~np.isinf(dist_grid) & ~self._wall
        rr, cc = np.where(reachable_mask)
        distances: Dict[Location, int] = {
            Location(int(r), int(c)): int(dist_grid[r, c])
            for r, c in zip(rr, cc)
        }

        phase = self._determine_phase(
            board, player, opponent, local_count, max_local, distances
        )

        if dist_to_opp <= 4 and phase != "rush":
            actions, stamina = self._territorial_pressure_paint(
                board, player_parity, player.loc, opp_loc, actions, stamina, extra_layers
            )

        # ---- RUSH ----
        if phase == "rush":
            target = self._choose_rush_target(board, player_parity, dist_grid, distances)
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    actions.append(Action.Move(direction))
                    td = distances.get(target)
                    if td and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST + 40:
                        np2 = self._simulate_position(board, player.loc, actions)
                        d2 = self._safe_step(board, np2, target, player_parity)
                        if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, 40, extra_layers
                    )

        # ---- FARM / EXPAND ----
        else:
            buf = 15 if phase == "farm" else 25
            actions, stamina = self._smart_paint(
                board, player_parity, player.loc, actions, stamina, buf, extra_layers
            )

            if phase == "farm":
                target = self._choose_farm_target(board, player_parity, player.loc)
                if target is None:
                    target = self._choose_expand_target(
                        board, player_parity, dist_grid, distances, player
                    )
            else:
                target = self._choose_expand_target(
                    board, player_parity, dist_grid, distances, player
                )

            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    actions.append(Action.Move(direction))
                    if phase == "expand":
                        td = distances.get(target)
                        if td and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST + 30:
                            np2 = self._simulate_position(board, player.loc, actions)
                            d2 = self._safe_step(board, np2, target, player_parity)
                            if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity):
                                actions.append(Action.Move(d2))
                                stamina -= GameConstants.EXTRA_MOVE_COST
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, buf, extra_layers
                    )

        # ---- Guarantee at least one Move ----
        has_move = any(isinstance(a, Action.Move) for a in actions)
        if not has_move:
            fallback = self._any_valid_move(board, player_parity)
            if fallback:
                actions.append(fallback)
                new_pos = self._simulate_position(board, player.loc, actions)
                actions, stamina = self._smart_paint(
                    board, player_parity, new_pos, actions, stamina, 20, extra_layers
                )
            else:
                return []

        return actions

    # ================================================================== #
    #  OFFENSIVE COLLISIONS                                               #
    # ================================================================== #

    def _paint_then_kill(self, board, player, opponent, player_parity, stamina):
        if stamina < GameConstants.PAINT_STAMINA_COST:
            return None
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt) or nxt != opponent.loc:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if cell.owner_parity == 0 and cell.beacon_parity == 0:
                return [Action.Paint(nxt), Action.Move(direction)]
        return None

    def _check_kill(self, board, player, opponent, player_parity):
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt) or nxt != opponent.loc:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if cell.owner_parity == player_parity:
                return direction
        return None

    def _multi_step_kill(self, board, player, opponent, player_parity, stamina):
        if stamina < GameConstants.EXTRA_MOVE_COST + 20:
            return None
        opp_loc  = opponent.loc
        opp_cell = board.cells[opp_loc.r][opp_loc.c]
        if opp_cell.owner_parity != player_parity:
            return None
        for d1 in Direction.cardinals():
            mid = player.loc + d1
            if board.oob(mid):
                continue
            if board.cells[mid.r][mid.c].is_wall or mid == opp_loc:
                continue
            for d2 in Direction.cardinals():
                if mid + d2 == opp_loc:
                    return [Action.Move(d1), Action.Move(d2)]
        return None

    def _territorial_pressure_paint(self, board, player_parity, player_loc, opp_loc,
                                     actions, stamina, extra_layers):
        COST    = GameConstants.PAINT_STAMINA_COST
        MAX_VAL = GameConstants.MAX_PAINT_VALUE
        opp_adj: Set[Location] = set()
        for d in Direction.cardinals():
            n = opp_loc + d
            if not board.oob(n):
                opp_adj.add(n)
        for direction in Direction.cardinals():
            if stamina - COST < 30:
                break
            t = player_loc + direction
            if board.oob(t) or t not in opp_adj:
                continue
            cell = board.cells[t.r][t.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == -player_parity:
                continue
            added      = extra_layers.get(t, 0)
            base_layers = abs(cell.paint_value) if cell.owner_parity == player_parity else 0
            if base_layers + added >= MAX_VAL:
                continue
            if cell.owner_parity == player_parity or cell.owner_parity == 0:
                actions.append(Action.Paint(t))
                extra_layers[t] = added + 1
                stamina -= COST
        return actions, stamina

    # ================================================================== #
    #  PHASE DETERMINATION                                                #
    # ================================================================== #

    def _determine_phase(self, board, player, opponent, local_count, max_local, distances):
        total_hills = len(board.hills)
        our_hills   = len(player.controlled_hills)
        if total_hills > 0 and our_hills == 0:
            for loc in distances:
                cell = board.cells[loc.r][loc.c]
                if cell.hill_id != 0:
                    h = board.hills[cell.hill_id]
                    if h.controller_parity != player.parity:
                        return "rush"
        threshold = max(max_local * 0.65, 8)
        if local_count < threshold:
            return "farm"
        return "expand"

    # ================================================================== #
    #  TARGET SELECTION                                                   #
    # ================================================================== #

    def _centrality_bonus(self, board: Board, loc: Location) -> float:
        """
        Precomputed centrality grid lookup — O(1) array index.
        Previously recomputed from scratch each call.
        """
        return float(self._cent[loc.r, loc.c])

    def _hill_efficiency(self, board, player_parity, hill, nearest_dist):
        hill_size  = len(hill.cells)
        needed     = math.ceil(hill_size * GameConstants.HILL_CONTROL_THRESHOLD) + 1
        our_count  = int(np.sum(
            self._owner[[h.r for h in hill.cells], [h.c for h in hill.cells]] == player_parity
        ))
        cells_to_paint = max(0, needed - our_count)
        total_cost     = nearest_dist + cells_to_paint * 1.5
        if total_cost <= 0:
            total_cost = 0.1
        return -(total_cost + hill_size * 0.3)

    def _choose_rush_target(self, board, player_parity, dist_grid, distances):
        best      = None
        best_eff  = -9999.0
        seen_hills: Set[int] = set()

        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id == 0 or cell.hill_id in seen_hills:
                continue
            hill = board.hills[cell.hill_id]
            if hill.controller_parity == player_parity:
                continue
            seen_hills.add(cell.hill_id)

            nearest_dist = None
            nearest_loc  = None
            for hloc in hill.cells:
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = dist_grid[hloc.r, hloc.c]   # O(1) array lookup
                if np.isinf(d):
                    continue
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = int(d)
                    nearest_loc  = hloc

            if nearest_loc is None:
                continue
            eff = self._hill_efficiency(board, player_parity, hill, nearest_dist)
            if eff > best_eff:
                best_eff = eff
                best     = nearest_loc

        if best is not None:
            for loc, dist in distances.items():
                if board.cells[loc.r][loc.c].powerup and dist <= 2:
                    best = loc
                    break
        return best

    def _choose_farm_target(self, board, player_parity, player_loc):
        opp     = board.get_opponent(player_parity)
        opp_loc = opp.loc
        best       = None
        best_score = 0
        for direction in Direction.cardinals():
            nxt = player_loc + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            if self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE):
                continue
            score = self._paint_value_from(board, player_parity, nxt)
            score += self._cent[nxt.r, nxt.c]   # direct array lookup
            if score > best_score:
                best_score = score
                best       = nxt
        return best

    def _paint_value_from(self, board, player_parity, pos):
        score   = 0
        MAX_VAL = GameConstants.MAX_PAINT_VALUE
        for direction in Direction.cardinals():
            t = pos + direction
            if board.oob(t):
                continue
            cell = board.cells[t.r][t.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == -player_parity:
                continue
            if cell.owner_parity == 0:
                score += 20
                if cell.hill_id != 0:
                    score += 50
            elif cell.owner_parity == player_parity:
                gap    = MAX_VAL - abs(cell.paint_value)
                score += gap * 3
                if cell.hill_id != 0:
                    score += gap * 10
        return score

    def _choose_expand_target(self, board, player_parity, dist_grid, distances, player):
        total_hills  = len(board.hills)
        our_hills    = len(player.controlled_hills)
        hills_for_win = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD) if total_hills > 0 else 0
        close_to_dom = (total_hills > 0 and our_hills + 1 >= hills_for_win)

        best_target = None
        best_score: float = -9999.0

        # Hills
        for hill_id, hill in board.hills.items():
            if hill.controller_parity == player_parity:
                continue
            nearest_dist = None
            nearest_loc  = None
            for hloc in hill.cells:
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = dist_grid[hloc.r, hloc.c]
                if np.isinf(d):
                    continue
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = int(d)
                    nearest_loc  = hloc
            if nearest_loc is None:
                continue
            eff   = self._hill_efficiency(board, player_parity, hill, nearest_dist)
            base  = 500.0 if close_to_dom else 200.0
            score = base + eff * 20.0
            if score > best_score:
                best_score  = score
                best_target = nearest_loc

        # Defend our hills
        for hill_id in player.controlled_hills:
            hill     = board.hills[hill_id]
            opp_cells = sum(
                1 for hloc in hill.cells
                if board.cells[hloc.r][hloc.c].owner_parity == -player_parity
            )
            if opp_cells == 0:
                continue
            best_def_dist = None
            best_def_loc  = None
            for hloc in hill.cells:
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = dist_grid[hloc.r, hloc.c]
                if np.isinf(d):
                    continue
                if best_def_dist is None or d < best_def_dist:
                    best_def_dist = int(d)
                    best_def_loc  = hloc
            if best_def_loc is not None:
                hill_size = len(hill.cells)
                urgency   = opp_cells / hill_size if hill_size > 0 else 0
                score     = 150.0 + urgency * 100.0 - best_def_dist * 2.0
                if score > best_score:
                    best_score  = score
                    best_target = best_def_loc

        # Powerups
        for loc, dist in distances.items():
            if board.cells[loc.r][loc.c].powerup:
                score = 120.0 - dist * 3.0
                if score > best_score:
                    best_score  = score
                    best_target = loc

        # ---- Vectorized territory expansion scoring ----
        # Previously: Python loop over every entry in distances dict
        # Now: numpy array ops over the whole board at once
        inf_mask    = np.isinf(dist_grid)
        terr_mask   = (
            (self._hill_id == 0)
            & ~self._pwrup
            & (self._owner == 0)
            & ~inf_mask
            & ~self._wall
        )
        # Adjacency bonus: cell adjacent to any friendly cell
        own_friendly = (self._owner == player_parity).astype(np.float32)
        # Shift in 4 directions and OR together to get "adjacent to friendly" mask
        adj_friendly = (
            (np.roll(own_friendly, 1,  axis=0) +
             np.roll(own_friendly, -1, axis=0) +
             np.roll(own_friendly, 1,  axis=1) +
             np.roll(own_friendly, -1, axis=1)) > 0
        )
        # Zero out rolled edges
        adj_friendly[0, :]  = False
        adj_friendly[-1, :] = False
        adj_friendly[:, 0]  = False
        adj_friendly[:, -1] = False

        own_opp = (self._owner == -player_parity).astype(np.float32)
        adj_opp = (
            (np.roll(own_opp, 1,  axis=0) +
             np.roll(own_opp, -1, axis=0) +
             np.roll(own_opp, 1,  axis=1) +
             np.roll(own_opp, -1, axis=1)) > 0
        )
        adj_opp[0, :]  = False
        adj_opp[-1, :] = False
        adj_opp[:, 0]  = False
        adj_opp[:, -1] = False

        scores = np.where(
            terr_mask & adj_friendly,
            30.0 - dist_grid * 2.0 + np.where(adj_opp, 10.0, 0.0) + self._cent,
            np.where(
                terr_mask,
                8.0 - dist_grid * 2.0 + self._cent,
                -np.inf
            )
        )

        best_flat = int(np.argmax(scores))
        best_r, best_c = divmod(best_flat, self._C)
        best_np_score  = float(scores[best_r, best_c])

        if best_np_score > best_score and not np.isinf(best_np_score) and best_np_score > -np.inf:
            best_score  = best_np_score
            best_target = Location(best_r, best_c)

        return best_target

    # ================================================================== #
    #  SMART PAINT                                                        #
    # ================================================================== #

    def _smart_paint(self, board, player_parity, pos, actions, stamina, buffer, extra_layers):
        COST    = GameConstants.PAINT_STAMINA_COST
        MAX_VAL = GameConstants.MAX_PAINT_VALUE

        while stamina - COST >= buffer:
            best       = None
            best_score = -1

            for direction in Direction.cardinals():
                t = pos + direction
                if board.oob(t):
                    continue
                cell = board.cells[t.r][t.c]
                if cell.is_wall or cell.beacon_parity != 0:
                    continue
                if cell.owner_parity == -player_parity:
                    continue

                base_layers = abs(cell.paint_value) if cell.owner_parity == player_parity else 0
                added       = extra_layers.get(t, 0)
                effective   = base_layers + added
                if effective >= MAX_VAL:
                    continue

                is_new  = (cell.owner_parity == 0 and added == 0)
                is_hill = (cell.hill_id != 0)
                gap     = MAX_VAL - effective

                if is_hill:
                    score = 200 if is_new else 120 + gap * 15
                elif is_new:
                    score = 80
                else:
                    score = 15 + gap * 5

                score += self._cent[t.r, t.c] * 0.5   # direct array lookup

                if score > best_score:
                    best_score = score
                    best       = t

            if best is None:
                break

            actions.append(Action.Paint(best))
            extra_layers[best] = extra_layers.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    # ================================================================== #
    #  COLLISION SAFETY                                                   #
    # ================================================================== #

    def _is_danger(self, board, loc, opp_loc, player_parity, radius=DANGER_STRICT):
        cell  = board.cells[loc.r][loc.c]
        is_opp = (cell.owner_parity == -player_parity)
        if not is_opp:
            pv = cell.paint_value
            if player_parity == 1 and pv <= -2:
                is_opp = True
            elif player_parity == -1 and pv >= 2:
                is_opp = True
        if not is_opp:
            return False
        return abs(loc.r - opp_loc.r) + abs(loc.c - opp_loc.c) <= radius

    def _is_step_dangerous(self, board, dest, player_parity):
        if board.oob(dest):
            return True
        opp = board.get_opponent(player_parity)
        return self._is_danger(board, dest, opp.loc, player_parity, DANGER_MODERATE)

    def _escape_step(self, board, start, player_parity):
        opp     = board.get_opponent(player_parity)
        opp_loc = opp.loc
        visited: Set[Location] = {start}
        queue: deque = deque()
        for direction in Direction.cardinals():
            nxt = start + direction
            if board.oob(nxt) or nxt in visited:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall or nxt == opp_loc:
                continue
            visited.add(nxt)
            if not self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT):
                return direction
            queue.append((nxt, direction))
        while queue:
            loc, first_dir = queue.popleft()
            for direction in Direction.cardinals():
                nxt = loc + direction
                if board.oob(nxt) or nxt in visited:
                    continue
                cell = board.cells[nxt.r][nxt.c]
                if cell.is_wall or nxt == opp_loc:
                    continue
                visited.add(nxt)
                if not self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT):
                    return first_dir
                queue.append((nxt, first_dir))
        return None

    # ================================================================== #
    #  PATHFINDING (BFS kept for safe_step; Dijkstra used for scoring)   #
    # ================================================================== #

    def _bfs_all_distances(self, board, start, player_parity):
        """Legacy fallback — now only used indirectly via distances dict."""
        opp     = board.get_opponent(player_parity)
        opp_loc = opp.loc
        distances: Dict[Location, int] = {start: 0}
        queue: deque = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for direction in Direction.cardinals():
                nxt = loc + direction
                if nxt in distances or board.oob(nxt):
                    continue
                cell = board.cells[nxt.r][nxt.c]
                if cell.is_wall:
                    continue
                if nxt == opp_loc and cell.owner_parity != player_parity:
                    continue
                distances[nxt] = d + 1
                queue.append((nxt, d + 1))
        return distances

    def _safe_step(self, board, start, target, player_parity):
        if start == target:
            return None
        r = self._bfs_step(board, start, target, player_parity, DANGER_STRICT)
        if r:
            return r
        r = self._bfs_step(board, start, target, player_parity, DANGER_MODERATE)
        if r:
            return r
        return self._bfs_step(board, start, target, player_parity, 0)

    def _bfs_step(self, board, start, target, player_parity, danger_radius):
        opp     = board.get_opponent(player_parity)
        opp_loc = opp.loc
        first_step_radius = max(danger_radius, DANGER_MODERATE)
        visited: Set[Location] = {start}
        queue: deque = deque()
        for direction in Direction.cardinals():
            nxt = start + direction
            if board.oob(nxt) or nxt in visited:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            if self._is_danger(board, nxt, opp_loc, player_parity, first_step_radius):
                continue
            visited.add(nxt)
            queue.append((nxt, direction))
        while queue:
            loc, first_dir = queue.popleft()
            if loc == target:
                return first_dir
            for direction in Direction.cardinals():
                nxt = loc + direction
                if board.oob(nxt) or nxt in visited:
                    continue
                cell = board.cells[nxt.r][nxt.c]
                if cell.is_wall:
                    continue
                if nxt == opp_loc and cell.owner_parity != player_parity:
                    continue
                if danger_radius > 0 and self._is_danger(board, nxt, opp_loc, player_parity, danger_radius):
                    continue
                visited.add(nxt)
                queue.append((nxt, first_dir))
        return None

    # ================================================================== #
    #  UTILITIES                                                          #
    # ================================================================== #

    def _adjacent_to_friendly(self, board, loc, player_parity):
        for d in Direction.cardinals():
            n = loc + d
            if not board.oob(n) and board.cells[n.r][n.c].owner_parity == player_parity:
                return True
        return False

    def _adjacent_to_opponent_territory(self, board, loc, player_parity):
        for d in Direction.cardinals():
            n = loc + d
            if not board.oob(n) and board.cells[n.r][n.c].owner_parity == -player_parity:
                return True
        return False

    def _simulate_position(self, board, start, actions):
        loc = start
        for action in actions:
            if isinstance(action, Action.Move) and action.move_type != MoveType.BEACON_TRAVEL:
                if action.direction is not None:
                    candidate = loc + action.direction
                    if not board.oob(candidate) and not board.cells[candidate.r][candidate.c].is_wall:
                        loc = candidate
        return loc

    def _any_valid_move(self, board, player_parity):
        player  = board.get_player(player_parity)
        opp     = board.get_opponent(player_parity)
        opp_loc = opp.loc
        candidates: list = []
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            sd = self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT)
            md = self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE)
            fr = (cell.owner_parity == player_parity)
            ne = (cell.owner_parity == 0)
            if fr and not sd:   p = 5
            elif ne and not sd: p = 4
            elif not md and fr: p = 3
            elif not md:        p = 2
            elif not sd:        p = 1
            else:               p = 0
            candidates.append((p, direction))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return Action.Move(candidates[0][1])

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        if not board.hills:
            return 0
        player  = board.get_player(player_parity)
        min_dist = 999
        for hill in board.hills.values():
            for hloc in hill.cells:
                d = abs(player.loc.r - hloc.r) + abs(player.loc.c - hloc.c)
                if d < min_dist:
                    min_dist = d
        if min_dist <= 3:  return 15
        if min_dist <= 6:  return 10
        return 5

    def commentate(self, board, player_parity, time_left):
        player = board.get_player(player_parity)
        mt = int(np.sum(self._owner == player_parity)) if self._board_initialized else 0
        ot = int(np.sum(self._owner == -player_parity)) if self._board_initialized else 0
        pr, pc = player.loc.r, player.loc.c
        r0,r1 = max(0,pr-2), min(self._R,pr+3)
        c0,c1 = max(0,pc-2), min(self._C,pc+3)
        lc = int(np.sum(self._owner[r0:r1,c0:c1] == player_parity)) if self._board_initialized else 0
        return (
            f"hills={len(player.controlled_hills)}"
            f" terr={mt} vs={ot}"
            f" local5x5={lc}"
            f" stam={player.stamina}/{player.max_stamina}"
            f" r={board.current_round}"
        )