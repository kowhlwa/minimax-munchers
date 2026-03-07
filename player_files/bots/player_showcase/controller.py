from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set

from game import *


class PlayerController:
    """
    HillRusher: Captures hills for stamina bonuses, paints densely for
    local regen, and deploys free beacons as control anchors.

    Win path: Hill Dominance (75% of hills) via superior stamina economy.
    Fallback: Stamina collapse from territory denial.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity
        self._cached_target: Optional[Location] = None

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        # Small bid to win initiative — going first is a marginal advantage.
        return 10

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        player = board.get_player(player_parity)
        actions = []
        stamina = player.stamina

        # --- Paint phase ---
        # Paint adjacent cells before moving so we paint our current neighborhood.
        painted: Set[Location] = set()

        paint1 = self._find_paint_target(board, player_parity, painted)
        if paint1 and stamina >= GameConstants.PAINT_STAMINA_COST + 15:
            actions.append(Action.Paint(paint1))
            stamina -= GameConstants.PAINT_STAMINA_COST
            painted.add(paint1)

        # Second paint only if stamina is comfortable (keep ≥30 for move + buffer)
        if stamina >= GameConstants.PAINT_STAMINA_COST * 2 + 30:
            paint2 = self._find_paint_target(board, player_parity, painted)
            if paint2:
                actions.append(Action.Paint(paint2))
                stamina -= GameConstants.PAINT_STAMINA_COST
                painted.add(paint2)

        # --- Move phase ---
        target = self._choose_target(board, player_parity)
        place_beacon = False

        if target:
            direction = self._bfs_next_step(board, player.loc, target, player_parity)
            if direction:
                # Check if we should place a beacon when we arrive at that cell.
                next_loc = player.loc + direction
                if (
                    not board.oob(next_loc)
                    and stamina >= 0
                    and self._can_place_beacon_at(board, next_loc, player_parity)
                ):
                    place_beacon = True

                actions.append(Action.Move(direction, place_beacon=place_beacon))
                stamina -= 0  # first move is free

        # --- Extra move(s) if stamina allows ---
        # Each additional move costs 10 * move_number.
        extra_move_num = 1
        while stamina >= GameConstants.EXTRA_MOVE_COST * extra_move_num + 30:
            if target:
                # Simulate position after moves so far
                simulated_loc = self._simulate_position(board, player.loc, actions)
                if simulated_loc == target:
                    break
                direction = self._bfs_next_step(board, simulated_loc, target, player_parity)
                if direction:
                    actions.append(Action.Move(direction))
                    stamina -= GameConstants.EXTRA_MOVE_COST * extra_move_num
                    extra_move_num += 1
                    continue
            break

        if not actions:
            # Fallback: any valid move
            fallback = self._any_valid_move(board, player_parity)
            if fallback:
                return fallback

        return actions

    # ------------------------------------------------------------------ #
    # Target selection                                                     #
    # ------------------------------------------------------------------ #

    def _choose_target(self, board: Board, player_parity: int) -> Optional[Location]:
        """
        Score every non-wall cell and return the best target.

        Priority order:
          1. Hill cells not yet controlled by us          (+100, -dist*1)
          2. Powerup cells                                (+60,  -dist*1)
          3. Neutral cells adjacent to our territory      (+20,  -dist*2)
          4. Any neutral cell                             (+10,  -dist*2)
          5. Opponent cells (free erosion by walking)     (+5,   -dist*3)

        The cache is kept until we arrive at the target or it becomes invalid.
        """
        player = board.get_player(player_parity)

        # Validate cached target
        if self._cached_target is not None:
            if self._cached_target == player.loc:
                self._cached_target = None
            else:
                ct = self._cached_target
                if not board.oob(ct):
                    cell = board.cells[ct.r][ct.c]
                    # Re-evaluate: if it's a hill already captured or no longer interesting, drop cache
                    hill_captured = (
                        cell.hill_id != 0
                        and board.hills[cell.hill_id].controller_parity == player_parity
                        and cell.owner_parity == player_parity
                    )
                    if not hill_captured:
                        return self._cached_target

        best_target: Optional[Location] = None
        best_score = -9999

        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                cell = board.cells[r][c]
                if cell.is_wall:
                    continue
                loc = Location(r, c)
                if loc == player.loc:
                    continue

                dist = self._manhattan(player.loc, loc)
                score = 0

                # Hill cells not already controlled by us
                if cell.hill_id != 0:
                    hill = board.hills[cell.hill_id]
                    if hill.controller_parity != player_parity:
                        score += 100 - dist

                # Powerup
                if cell.powerup:
                    score += 60 - dist

                # Neutral territory
                if cell.owner_parity == 0 and score == 0:
                    # Bonus if adjacent to our paint
                    if self._adjacent_to_friendly(board, loc, player_parity):
                        score += 20 - dist * 2
                    else:
                        score += 10 - dist * 2

                # Opponent territory (weaken by walking)
                if cell.owner_parity == -player_parity and score == 0:
                    score += 5 - dist * 3

                if score > best_score:
                    best_score = score
                    best_target = loc

        self._cached_target = best_target
        return best_target

    # ------------------------------------------------------------------ #
    # Paint target selection                                               #
    # ------------------------------------------------------------------ #

    def _find_paint_target(
        self,
        board: Board,
        player_parity: int,
        already_painting: Set[Location],
    ) -> Optional[Location]:
        """
        Find the best adjacent cell to paint this turn.

        Score: hill cell=50, neutral=20, own underpainted=5.
        Skips: walls, beacons, opponent cells, fully-painted own cells, already queued.
        """
        player = board.get_player(player_parity)
        best: Optional[Location] = None
        best_score = -1

        for direction in Direction.cardinals():
            target = player.loc + direction
            if board.oob(target) or target in already_painting:
                continue

            cell = board.cells[target.r][target.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            # Can't paint opponent cells
            if cell.owner_parity == -player_parity:
                continue
            # Skip own fully-painted cells
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
    # Pathfinding                                                          #
    # ------------------------------------------------------------------ #

    def _bfs_next_step(
        self,
        board: Board,
        start: Location,
        target: Location,
        player_parity: int,
    ) -> Optional[Direction]:
        """
        BFS from start to target. Returns the first direction to take.

        Avoids: out-of-bounds, walls, opponent's current cell if they own it
        (moving there = collision loss).
        """
        if start == target:
            return None

        opponent = board.get_opponent(player_parity)

        visited: Set[Location] = {start}
        # Queue: (location, first_direction_taken)
        queue: deque = deque()

        for direction in Direction.cardinals():
            next_loc = start + direction
            if board.oob(next_loc):
                continue
            cell = board.cells[next_loc.r][next_loc.c]
            if cell.is_wall:
                continue
            # Avoid stepping into opponent's cell while they're standing on it and own it
            if next_loc == opponent.loc and cell.owner_parity == -player_parity:
                continue
            visited.add(next_loc)
            queue.append((next_loc, direction))

        while queue:
            loc, first_dir = queue.popleft()
            if loc == target:
                return first_dir

            for direction in Direction.cardinals():
                next_loc = loc + direction
                if board.oob(next_loc) or next_loc in visited:
                    continue
                cell = board.cells[next_loc.r][next_loc.c]
                if cell.is_wall:
                    continue
                if next_loc == opponent.loc and cell.owner_parity == -player_parity:
                    continue
                visited.add(next_loc)
                queue.append((next_loc, first_dir))

        return None  # No path found

    # ------------------------------------------------------------------ #
    # Beacon placement check                                               #
    # ------------------------------------------------------------------ #

    def _can_place_beacon_at(
        self, board: Board, loc: Location, player_parity: int
    ) -> bool:
        """Check if placing a beacon at loc would succeed (7/9 cells painted)."""
        if board.oob(loc):
            return False
        cell = board.cells[loc.r][loc.c]
        if cell.owner_parity != player_parity or cell.beacon_parity != 0:
            return False

        radius = GameConstants.BEACON_WINDOW_SIZE_P // 2
        friendly_count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                neighbor = Location(loc.r + dr, loc.c + dc)
                if board.oob(neighbor):
                    continue
                n_cell = board.cells[neighbor.r][neighbor.c]
                if n_cell.beacon_parity == 0 and n_cell.owner_parity == player_parity:
                    friendly_count += 1

        return friendly_count >= GameConstants.BEACON_REQUIREMENT_Q

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _adjacent_to_friendly(
        self, board: Board, loc: Location, player_parity: int
    ) -> bool:
        """Return True if any cardinal neighbor of loc is owned by player."""
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
        """Compute where the player would be after executing the given actions."""
        loc = start
        for action in actions:
            if isinstance(action, Action.Move) and action.move_type != MoveType.BEACON_TRAVEL:
                if action.direction is not None:
                    candidate = loc + action.direction
                    if not board.oob(candidate) and not board.cells[candidate.r][candidate.c].is_wall:
                        loc = candidate
        return loc

    def _any_valid_move(self, board: Board, player_parity: int) -> Optional[Action.Move]:
        """Return any valid non-wall move as a fallback."""
        player = board.get_player(player_parity)
        for direction in Direction.cardinals():
            next_loc = player.loc + direction
            if board.oob(next_loc):
                continue
            if not board.cells[next_loc.r][next_loc.c].is_wall:
                return Action.Move(direction)
        return None

    @staticmethod
    def _manhattan(a: Location, b: Location) -> int:
        return abs(a.r - b.r) + abs(a.c - b.c)

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        player = board.get_player(player_parity)
        territory = board.get_territory_count(player_parity)
        hills = len(player.controlled_hills)
        return f"hills={hills} territory={territory} stamina={player.stamina}/{player.max_stamina}"
