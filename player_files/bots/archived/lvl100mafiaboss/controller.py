from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set

from game import *


class PlayerController:
    """
    Hill-rush bot: prioritize taking hill control as fast as possible.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        """
        Called at the start of each round. Return the number of stamina you
        want to bid for initiative. Defaults to zero.
        """
        return 0

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        """
        Rush the nearest hill cell that is not controlled by us. If adjacent
        to an unowned hill cell, paint it.
        """
        paint_action = self._paint_adjacent_hill(board, player_parity)
        if paint_action is not None:
            return paint_action

        targets = self._hill_targets(board, player_parity)
        if targets:
            direction = self._bfs_first_step(board, board.get_player(player_parity).loc, targets, player_parity)
            if direction is not None:
                return Action.Move(direction)

        fallback = self._any_valid_move(board, player_parity)
        if fallback is not None:
            return fallback
        return []

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        """
        Allows for you to display a string at the end of the match on our online
        portal for your own statistics usage. Be careful, your opponents will be
        able to see this as well.
        """
        return ""

    def _paint_adjacent_hill(self, board: Board, player_parity: int) -> Optional[Action.Paint]:
        player = board.get_player(player_parity)
        if player.stamina < GameConstants.PAINT_STAMINA_COST:
            return None

        # Prefer unowned hill cells
        for direction in Direction.cardinals():
            loc = player.loc + direction
            if board.oob(loc):
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall:
                continue
            if cell.hill_id == 0:
                continue
            if cell.owner_parity == 0:
                return Action.Paint(loc)

        return None

    def _hill_targets(self, board: Board, player_parity: int) -> Set[Location]:
        targets: Set[Location] = set()
        if not board.hills:
            return targets

        for hill in board.hills.values():
            for loc in hill.cells:
                if board.oob(loc):
                    continue
                cell = board.cells[loc.r][loc.c]
                if cell.is_wall:
                    continue
                if cell.owner_parity != player_parity:
                    targets.add(loc)

        if targets:
            return targets

        # If all hills are already controlled by us, move among hill cells anyway.
        for hill in board.hills.values():
            for loc in hill.cells:
                if board.oob(loc):
                    continue
                cell = board.cells[loc.r][loc.c]
                if not cell.is_wall:
                    targets.add(loc)

        return targets

    def _bfs_first_step(
        self,
        board: Board,
        start: Location,
        targets: Set[Location],
        player_parity: int,
    ) -> Optional[Direction]:
        if not targets:
            return None

        opponent_loc = board.get_opponent(player_parity).loc
        visited: Set[Location] = {start}
        queue: deque = deque()

        for direction in Direction.cardinals():
            nxt = start + direction
            if board.oob(nxt) or nxt in visited:
                continue
            if nxt == opponent_loc:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt in targets:
                return direction
            visited.add(nxt)
            queue.append((nxt, direction))

        while queue:
            loc, first_dir = queue.popleft()
            for direction in Direction.cardinals():
                nxt = loc + direction
                if board.oob(nxt) or nxt in visited:
                    continue
                if nxt == opponent_loc:
                    continue
                cell = board.cells[nxt.r][nxt.c]
                if cell.is_wall:
                    continue
                if nxt in targets:
                    return first_dir
                visited.add(nxt)
                queue.append((nxt, first_dir))

        return None

    def _any_valid_move(self, board: Board, player_parity: int) -> Optional[Action.Move]:
        player = board.get_player(player_parity)
        opponent_loc = board.get_opponent(player_parity).loc
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt):
                continue
            if nxt == opponent_loc:
                continue
            if board.cells[nxt.r][nxt.c].is_wall:
                continue
            return Action.Move(direction)
        return None
