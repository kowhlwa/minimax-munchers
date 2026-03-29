from collections.abc import Callable, Iterable
from typing import Union
from .target_selection import Algorithms
from game import *
# from .player_board import PlayerBoard
import random


class PlayerController:
	"""
	You may add functions, however, __init__, bid, and play are the entry
	points for your program and should not be changed.
	"""
	
	def __init__(self, player_parity:int, time_left: Callable):
		return
	
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
		Pathfinds to closest hill cell.
		"""
		staminaBudget = board.get_player(player_parity).stamina
		currentLoc = board.get_player(player_parity).loc
		moveList = []
		
		# First finds target hill
		targetHill = Algorithms.TargetSelector().bfs(player_parity, board, currentLoc)
		
		# Second returns list of 3 moves towards target hill to take
		targetHillObject = board.hills[targetHill]
		nextLoc = Algorithms.ShortestPathSelector().bfs_next(player_parity, board, set(targetHillObject.cells), currentLoc)
		for i in range(len(nextLoc) - 1):
			curr = nextLoc[i]
			nxt = nextLoc[i + 1]
			nxtCell = board.cells[nxt.r][nxt.c]
			direction = Direction((nxt.r - curr.r, nxt.c - curr.c))

			###### This is the logic for painting free territory around us before moving. ######
			for toPaint in curr.neighbors():
				if board.oob(toPaint):
					continue
				toPaintCell = board.cells[toPaint.r][toPaint.c]
				if staminaBudget >= 15 and not toPaintCell.is_wall and toPaintCell.owner_parity == 0:
					moveList.append(Action.Paint(toPaint))
					staminaBudget -= 15

			####### This is the logic for erase step ########################
			if nxtCell.owner_parity == Parity.get_opponent_parity(player_parity) and nxtCell.paint_value > 1:
				if staminaBudget >= 40:
					moveList.append(Action.Move(direction = direction, move_type = MoveType.ERASE))
			else:
				moveList.append(Action.Move(direction=direction))
		return moveList
	
	def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
		"""
		Allows for you to display a string at the end of the match on our online
		portal for your own statistics usage. Be careful, your opponents will be 
		able to see this as well.
		"""
		return ""
		
