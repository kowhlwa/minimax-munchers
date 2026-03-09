from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict

from game import *


class PlayerController:
    """
    HillRusher v3:
    - Danger-zone collision avoidance (avoid opponent territory near opponent)
    - Offensive kills (step onto opponent when they're on our cell)
    - Multi-move when traveling far (2nd step for 10 stamina)
    - Per-hill scoring (size, contestedness, distance)
    - Hill defense (retarget our hills under attack)
    - Greedy multi-paint before AND after each move
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 10

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        actions: list = []
        stamina = player.stamina

        # ---- Instant kill: opponent adjacent on a cell we own ----
        kill_dir = self._check_kill(board, player, opponent, player_parity)
        if kill_dir:
            actions.append(Action.Move(kill_dir))
            new_pos = player.loc + kill_dir
            actions, stamina = self._greedy_paint(
                board, player_parity, new_pos, set(), actions, stamina
            )
            return actions

        # ---- Pre-compute BFS distances ----
        distances = self._bfs_all_distances(board, player.loc, player_parity)

        # ---- Mode: rush first hill, full-paint otherwise ----
        needs_hill = len(board.hills) > 0 and len(player.controlled_hills) == 0
        in_hill_area = self._in_hill_area(board, player.loc)
        hill_rush = needs_hill and not in_hill_area

        # ---- Paint phase 1: around current position ----
        painted: Set[Location] = set()
        if not hill_rush:
            actions, stamina = self._greedy_paint(
                board, player_parity, player.loc, painted, actions, stamina
            )

        # ---- Move phase ----
        target = self._choose_target(board, player_parity, distances)
        moved = False
        if target:
            direction = self._safe_step(board, player.loc, target, player_parity)
            if direction:
                actions.append(Action.Move(direction))
                moved = True

                # Multi-move: 2nd step (costs 10) when target is far
                target_dist = distances.get(target)
                if (target_dist is not None and target_dist > 3
                        and stamina >= GameConstants.EXTRA_MOVE_COST + 30):
                    new_pos = self._simulate_position(board, player.loc, actions)
                    dir2 = self._safe_step(board, new_pos, target, player_parity)
                    if dir2:
                        actions.append(Action.Move(dir2))
                        stamina -= GameConstants.EXTRA_MOVE_COST

        # ---- Paint phase 2: around destination ----
        if moved:
            new_pos = self._simulate_position(board, player.loc, actions)
            buf = 40 if hill_rush else 20
            actions, stamina = self._greedy_paint(
                board, player_parity, new_pos, set(), actions, stamina, buffer=buf
            )

        # ---- Fallback ----
        if not actions:
            fallback = self._any_valid_move(board, player_parity)
            if fallback:
                return [fallback]
            return []

        return actions

    # ------------------------------------------------------------------ #
    # Kill detection                                                       #
    # ------------------------------------------------------------------ #

    def _check_kill(
        self, board: Board, player: Player, opponent: Player,
        player_parity: int,
    ) -> Optional[Direction]:
        """Return direction to kill opponent if they're adjacent on our cell."""
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

    # ------------------------------------------------------------------ #
    # Greedy paint                                                         #
    # ------------------------------------------------------------------ #

    def _greedy_paint(
        self,
        board: Board,
        player_parity: int,
        pos: Location,
        already_painted: Set[Location],
        actions: list,
        stamina: int,
        buffer: int = 20,
    ):
        COST = GameConstants.PAINT_STAMINA_COST
        while True:
            if stamina - COST < buffer:
                break
            t = self._find_paint_target_at(board, player_parity, pos, already_painted)
            if t is None:
                break
            actions.append(Action.Paint(t))
            already_painted.add(t)
            stamina -= COST
        return actions, stamina

    def _find_paint_target_at(
        self,
        board: Board,
        player_parity: int,
        pos: Location,
        exclude: Set[Location],
    ) -> Optional[Location]:
        best: Optional[Location] = None
        best_score = -1
        for direction in Direction.cardinals():
            target = pos + direction
            if board.oob(target) or target in exclude:
                continue
            cell = board.cells[target.r][target.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == -player_parity:
                continue
            if (
                cell.owner_parity == player_parity
                and abs(cell.paint_value) >= GameConstants.MAX_PAINT_VALUE
            ):
                continue
            score = 0
            if cell.hill_id != 0:
                score += 50
            if cell.owner_parity == 0:
                score += 20
            elif cell.owner_parity == player_parity:
                score += 5
            if score > best_score:
                best_score = score
                best = target
        return best

    # ------------------------------------------------------------------ #
    # BFS — full distance map                                              #
    # ------------------------------------------------------------------ #

    def _bfs_all_distances(
        self, board: Board, start: Location, player_parity: int,
    ) -> Dict[Location, int]:
        opponent = board.get_opponent(player_parity)
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
                if nxt == opponent.loc and cell.owner_parity != player_parity:
                    continue
                distances[nxt] = d + 1
                queue.append((nxt, d + 1))
        return distances

    # ------------------------------------------------------------------ #
    # Target selection                                                     #
    # ------------------------------------------------------------------ #

    def _choose_target(
        self,
        board: Board,
        player_parity: int,
        distances: Dict[Location, int],
    ) -> Optional[Location]:
        player = board.get_player(player_parity)
        total_hills = len(board.hills)
        our_hills = len(player.controlled_hills)

        # How badly do we need more hills?
        if our_hills == 0 and total_hills > 0:
            hill_base = 200.0
        elif total_hills > 0 and our_hills < (total_hills * 3 + 3) // 4:
            hill_base = 170.0  # push for domination
        else:
            hill_base = 80.0   # already dominating or no hills

        best_target: Optional[Location] = None
        best_score: float = -9999.0

        # --- Uncaptured hills (per-hill evaluation) ---
        seen_hills: Set[int] = set()
        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id == 0 or cell.hill_id in seen_hills:
                continue
            hill = board.hills[cell.hill_id]
            if hill.controller_parity == player_parity:
                continue
            seen_hills.add(cell.hill_id)
            hill_size = len(hill.cells)

            best_hill_dist: Optional[int] = None
            best_hill_loc: Optional[Location] = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = distances[hloc]
                if best_hill_dist is None or d < best_hill_dist:
                    best_hill_dist = d
                    best_hill_loc = hloc

            if best_hill_loc is None:
                continue

            opp_cells = sum(
                1 for hloc in hill.cells
                if board.cells[hloc.r][hloc.c].owner_parity == -player_parity
            )
            size_bonus = max(0.0, 40.0 - hill_size * 3.5)
            contest_penalty = opp_cells * 5.0
            score = hill_base + size_bonus - contest_penalty - best_hill_dist * 2.0

            if score > best_score:
                best_score = score
                best_target = best_hill_loc

        # --- Hill defense: our hills under attack ---
        for hill_id in player.controlled_hills:
            hill = board.hills[hill_id]
            opp_cells = sum(
                1 for hloc in hill.cells
                if board.cells[hloc.r][hloc.c].owner_parity == -player_parity
            )
            if opp_cells == 0:
                continue
            best_def_dist: Optional[int] = None
            best_def_loc: Optional[Location] = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = distances[hloc]
                if best_def_dist is None or d < best_def_dist:
                    best_def_dist = d
                    best_def_loc = hloc
            if best_def_loc is not None:
                score = 130.0 - best_def_dist * 2.0
                if score > best_score:
                    best_score = score
                    best_target = best_def_loc

        # --- Non-hill targets ---
        for loc, dist in distances.items():
            if loc == player.loc:
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id != 0:
                continue

            score: float = 0.0
            if cell.powerup:
                score = 100.0 - dist
            elif cell.owner_parity == 0:
                if self._adjacent_to_friendly(board, loc, player_parity):
                    score = 25.0 - dist * 2.0
                else:
                    score = 10.0 - dist * 2.0

            if score > best_score:
                best_score = score
                best_target = loc

        return best_target

    # ------------------------------------------------------------------ #
    # Safe pathfinding — avoids opponent territory near opponent            #
    # ------------------------------------------------------------------ #

    def _safe_step(
        self, board: Board, start: Location, target: Location,
        player_parity: int,
    ) -> Optional[Direction]:
        if start == target:
            return None
        # Phase 1: try avoiding danger zones
        result = self._bfs_step(board, start, target, player_parity, avoid_danger=True)
        if result:
            return result
        # Phase 2: take any path
        return self._bfs_step(board, start, target, player_parity, avoid_danger=False)

    def _bfs_step(
        self, board: Board, start: Location, target: Location,
        player_parity: int, avoid_danger: bool = False,
    ) -> Optional[Direction]:
        opponent = board.get_opponent(player_parity)
        opp_loc = opponent.loc

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
            if avoid_danger and self._is_danger(cell, nxt, opp_loc, player_parity):
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
                if avoid_danger and self._is_danger(cell, nxt, opp_loc, player_parity):
                    continue
                visited.add(nxt)
                queue.append((nxt, first_dir))

        return None

    def _is_danger(
        self, cell: CellState, loc: Location, opp_loc: Location,
        player_parity: int,
    ) -> bool:
        """Opponent-owned cell within Manhattan distance 2 of opponent."""
        if cell.owner_parity != -player_parity:
            return False
        return abs(loc.r - opp_loc.r) + abs(loc.c - opp_loc.c) <= 2

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _adjacent_to_friendly(
        self, board: Board, loc: Location, player_parity: int,
    ) -> bool:
        for direction in Direction.cardinals():
            neighbor = loc + direction
            if board.oob(neighbor):
                continue
            if board.cells[neighbor.r][neighbor.c].owner_parity == player_parity:
                return True
        return False

    def _simulate_position(
        self, board: Board, start: Location, actions: list,
    ) -> Location:
        loc = start
        for action in actions:
            if isinstance(action, Action.Move) and action.move_type != MoveType.BEACON_TRAVEL:
                if action.direction is not None:
                    candidate = loc + action.direction
                    if (not board.oob(candidate)
                            and not board.cells[candidate.r][candidate.c].is_wall):
                        loc = candidate
        return loc

    def _any_valid_move(
        self, board: Board, player_parity: int,
    ) -> Optional[Action.Move]:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        opp_loc = opponent.loc
        best: Optional[Action.Move] = None
        best_safe = False
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            safe = (cell.owner_parity != -player_parity)
            if best is None or (safe and not best_safe):
                best = Action.Move(direction)
                best_safe = safe
                if safe:
                    break
        return best

    def _in_hill_area(self, board: Board, loc: Location) -> bool:
        if board.cells[loc.r][loc.c].hill_id != 0:
            return True
        for direction in Direction.cardinals():
            neighbor = loc + direction
            if board.oob(neighbor):
                continue
            if board.cells[neighbor.r][neighbor.c].hill_id != 0:
                return True
        return False

    def commentate(
        self, board: Board, player_parity: int, time_left: Callable,
    ) -> str:
        player = board.get_player(player_parity)
        my_territory = board.get_territory_count(player_parity)
        opp_territory = board.get_territory_count(-player_parity)
        return (
            f"hills={len(player.controlled_hills)}"
            f" territory={my_territory}"
            f" vs_opp={opp_territory}"
            f" stamina={player.stamina}/{player.max_stamina}"
            f" round={board.current_round}"
        )
