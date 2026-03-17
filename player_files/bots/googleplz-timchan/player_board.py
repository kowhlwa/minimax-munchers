from game import Board
from game import GameConstants, Direction, Location, Action, MoveType, Parity

class PlayerBoard:
    def __init__(self, board: Board, player_parity: int):
        self.player_parity = player_parity
        self.opponent_parity = self.player_parity * -1
        self.board = board
