from collections.abc import Callable, Iterable
from typing import Union, Optional

from game import *
from .player_board import PlayerBoard
import random


class PlayerController:
    """
    Simple example controller: move around and paint nearby squares.
    
    1. Paint nearby squares
    2. Move to a random valid adjacent square
    
    More methods as needed to improve your strategy.
    """
    
    def __init__(self, player_parity: int, time_left: Callable):
        """Initialize the controller. Called once at the start of the game."""
        pass
    
    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        """
        Decide how much stamina to bid for the first move.
        
        Higher bid = you go first but start with less stamina.
        """
        return 10
    
    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        """
        Main game logic. Called once per turn.
        
        Returns either:
        - A single action (Action.Move or Action.Paint)
        - A list of actions to take in sequence
        """
        player = board.get_player(player_parity)
        
        # Build a list of actions to perform this turn
        actions = []
        
        # Step 1: Try to paint a nearby square that you own or is neutral
        if player.stamina >= GameConstants.PAINT_STAMINA_COST:
            paint_target = self._find_nearby_paint_target(board, player_parity)
            if paint_target:
                actions.append(Action.Paint(paint_target))
        
        # Step 2: Move to a random valid adjacent square
        move = self._find_valid_move(board, player_parity)
        if move:
            actions.append(move)
        
        # Return the actions we planned, or just a move if nothing else
        return actions if actions else self._find_valid_move(board, player_parity)
    
    def _find_nearby_paint_target(self, board: Board, player_parity: int) -> Optional[Location]:
        """
        Find a nearby square to paint.
        
        Prioritize:
        1. Your own squares that aren't fully painted yet
        2. Neutral squares (not owned by anyone)
        
        Only paint squares within PAINT_RANGE of your current position.
        Paint range is measured using Manhattan distance (|dr| + |dc|).
        """
        player = board.get_player(player_parity)
        best_target = None
        best_priority = -1
        
        # Check all squares within paint range
        for dr in range(-GameConstants.PAINT_RANGE, GameConstants.PAINT_RANGE + 1):
            for dc in range(-GameConstants.PAINT_RANGE, GameConstants.PAINT_RANGE + 1):
                # Skip your current position
                if dr == 0 and dc == 0:
                    continue
                
                # Check Manhattan distance - only paint within range
                manhattan_dist = abs(dr) + abs(dc)
                if manhattan_dist > GameConstants.PAINT_RANGE:
                    continue
                
                target_pos = Location(player.loc.r + dr, player.loc.c + dc)
                
                # Skip if out of bounds
                if board.oob(target_pos):
                    continue
                
                cell = board.cells[target_pos.r][target_pos.c]
                
                # Skip walls and beacons
                if cell.is_wall or cell.beacon_parity != 0:
                    continue
                
                # Only paint squares you own or neutral squares
                if cell.owner_parity == player_parity:
                    # Your own square - paint if not fully painted
                    if abs(cell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                        priority = 10  # Higher priority than neutral
                        if priority > best_priority:
                            best_priority = priority
                            best_target = target_pos
                elif cell.owner_parity == 0:
                    # Neutral square - good to paint
                    priority = 5
                    if priority > best_priority:
                        best_priority = priority
                        best_target = target_pos
        
        return best_target
    
    def _find_valid_move(self, board: Board, player_parity: int) -> Optional[Action.Move]:
        """
        Find a valid move to an adjacent square.
        
        Randomly pick from available cardinal directions that:
        - Don't go out of bounds
        - Don't hit a wall
        """
        player = board.get_player(player_parity)
        valid_moves = []
        
        # Try all four cardinal directions
        for direction in Direction.cardinals():
            next_pos = player.loc + direction
            
            # Skip if out of bounds
            if board.oob(next_pos):
                continue
            
            cell = board.cells[next_pos.r][next_pos.c]
            
            # Skip if wall
            if cell.is_wall:
                continue
            
            # This is a valid move
            valid_moves.append(Action.Move(direction))
        
        # Return a random valid move, or None if trapped
        return random.choice(valid_moves) if valid_moves else None
    
    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        """
        Optional: Return a string to display after the game ends.
        Your opponents will see this, so keep it professional!
        """
        return ""
        
