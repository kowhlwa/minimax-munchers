from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict

from game import *


class PlayerController:
    """
    HillRusher v2:
    - Greedy multi-paint before AND after each move (up to all 4 neighbours)
    - Single BFS per turn for real path distances (no Manhattan heuristic)
    - No cache on target — recomputed every turn from BFS distances
    - No beacon placement (was causing INVALID_TURN)
    - No extra-move logic (stamina better spent on paint)
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
        actions: list = []
        stamina = player.stamina

        # ---- Pre-compute BFS distances from current position ----
        distances = self._bfs_all_distances(board, player.loc, player_parity)

        # ---- Paint phase 1: paint neighbours of current position ----
        painted: Set[Location] = set()
        actions, stamina = self._greedy_paint(board, player_parity, player.loc,
                                               painted, actions, stamina)

        # ---- Move phase ----
        target = self._choose_target(board, player_parity, distances)
        moved = False
        if target:
            direction = self._bfs_next_step_from_distances(
                board, player.loc, target, player_parity
            )
            if direction:
                actions.append(Action.Move(direction))
                moved = True

        # ---- Paint phase 2: paint neighbours of new position ----
        if moved:
            new_pos = self._simulate_position(board, player.loc, actions)
            actions, stamina = self._greedy_paint(board, player_parity, new_pos,
                                                   set(), actions, stamina)

        # ---- Guarantee we always return at least one action ----
        if not actions:
            fallback = self._any_valid_move(board, player_parity)
            if fallback:
                return [fallback]
            return []

        return actions

    # ------------------------------------------------------------------ #
    # Greedy paint helper                                                  #
    # ------------------------------------------------------------------ #

    def _greedy_paint(
        self,
        board: Board,
        player_parity: int,
        pos: Location,
        already_painted: Set[Location],
        actions: list,
        stamina: int,
    ):
        """
        Paint as many adjacent cells as possible from `pos` while
        keeping a 20-stamina buffer. Returns updated (actions, stamina).
        """
        COST = GameConstants.PAINT_STAMINA_COST
        BUFFER = 20
        while True:
            if stamina - COST < BUFFER:
                break
            t = self._find_paint_target_at(board, player_parity, pos, already_painted)
            if t is None:
                break
            actions.append(Action.Paint(t))
            already_painted.add(t)
            stamina -= COST
        return actions, stamina

    # ------------------------------------------------------------------ #
    # Paint target selection (position-explicit)                          #
    # ------------------------------------------------------------------ #

    def _find_paint_target_at(
        self,
        board: Board,
        player_parity: int,
        pos: Location,
        exclude: Set[Location],
    ) -> Optional[Location]:
        """
        Return the best adjacent cell to paint from `pos`.
        Score: hill=50, neutral=20, own underpainted=5.
        Skips: walls, beacons, opponent cells, own max-painted cells, already queued.
        """
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
    # BFS — full distance map from one source                             #
    # ------------------------------------------------------------------ #

    def _bfs_all_distances(
        self,
        board: Board,
        start: Location,
        player_parity: int,
    ) -> Dict[Location, int]:
        """
        BFS from `start` to every reachable cell.
        Returns {Location: steps} for all reachable, non-wall cells.
        Avoids the opponent's cell if they own it (collision death).
        """
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
                if nxt == opponent.loc and cell.owner_parity == -player_parity:
                    continue
                distances[nxt] = d + 1
                queue.append((nxt, d + 1))

        return distances

    # ------------------------------------------------------------------ #
    # Target selection using real BFS distances                           #
    # ------------------------------------------------------------------ #

    def _choose_target(
        self,
        board: Board,
        player_parity: int,
        distances: Dict[Location, int],
    ) -> Optional[Location]:
        """
        Score every reachable cell using actual BFS distances.

        Scoring table (dist = BFS steps):
          Powerup                           : 100 - dist
          Hill cell (hill not ours)         :  80 - dist * 1.5
          Neutral adjacent to our territory :  25 - dist * 2
          Any neutral cell                  :  10 - dist * 2
          Opponent cell (free erosion)      :   5 - dist * 3
        """
        player = board.get_player(player_parity)
        best_target: Optional[Location] = None
        best_score: float = -9999.0

        for loc, dist in distances.items():
            if loc == player.loc:
                continue
            cell = board.cells[loc.r][loc.c]

            score: float = 0.0

            if cell.powerup:
                score = 100.0 - dist

            elif cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    score = 80.0 - dist * 1.5

            elif cell.owner_parity == 0:
                if self._adjacent_to_friendly(board, loc, player_parity):
                    score = 25.0 - dist * 2.0
                else:
                    score = 10.0 - dist * 2.0

            elif cell.owner_parity == -player_parity:
                score = 5.0 - dist * 3.0

            if score > best_score:
                best_score = score
                best_target = loc

        return best_target

    # ------------------------------------------------------------------ #
    # Pathfinding — extract first step toward target                      #
    # ------------------------------------------------------------------ #

    def _bfs_next_step_from_distances(
        self,
        board: Board,
        start: Location,
        target: Location,
        player_parity: int,
    ) -> Optional[Direction]:
        """
        Given the board state, BFS from start and return the first direction
        that lies on a shortest path to target.
        Uses a fresh BFS so we can recover the parent direction.
        """
        if start == target:
            return None

        opponent = board.get_opponent(player_parity)
        visited: Set[Location] = {start}
        queue: deque = deque()

        for direction in Direction.cardinals():
            nxt = start + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opponent.loc and cell.owner_parity == -player_parity:
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
                if nxt == opponent.loc and cell.owner_parity == -player_parity:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_dir))

        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _adjacent_to_friendly(
        self, board: Board, loc: Location, player_parity: int
    ) -> bool:
        for direction in Direction.cardinals():
            neighbor = loc + direction
            if board.oob(neighbor):
                continue
            if board.cells[neighbor.r][neighbor.c].owner_parity == player_parity:
                return True
        return False

    def _simulate_position(
        self, board: Board, start: Location, actions: list
    ) -> Location:
        loc = start
        for action in actions:
            if isinstance(action, Action.Move) and action.move_type != MoveType.BEACON_TRAVEL:
                if action.direction is not None:
                    candidate = loc + action.direction
                    if not board.oob(candidate) and not board.cells[candidate.r][candidate.c].is_wall:
                        loc = candidate
        return loc

    def _any_valid_move(self, board: Board, player_parity: int) -> Optional[Action.Move]:
        player = board.get_player(player_parity)
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt):
                continue
            if not board.cells[nxt.r][nxt.c].is_wall:
                return Action.Move(direction)
        return None

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        player = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        my_territory = board.get_territory_count(player_parity)
        opp_territory = board.get_territory_count(-player_parity)
        return (
            f"hills={len(player.controlled_hills)}"
            f" territory={my_territory}"
            f" vs_opp={opp_territory}"
            f" stamina={player.stamina}/{player.max_stamina}"
            f" round={board.current_round}"
        )
