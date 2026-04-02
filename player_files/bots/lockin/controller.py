from collections.abc import Callable, Iterable
from collections import deque
from typing import Union
from .target_selection import Algorithms
from game import *
import numpy as np
# from .player_board import PlayerBoard
import random


class PlayerController:
	"""
	You may add functions, however, __init__, bid, and play are the entry
	points for your program and should not be changed.
	"""
	
	def __init__(self, player_parity:int, time_left: Callable):
		# Precompute distances from hills
		self.hill_precomputed_matrices = {} # self.hill_precomputed_matrices is where you hold arrays representing distance maps from each hill.
		return
	
	def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
		if board.current_round == 1:
			for computing_hill in board.hill_list:
				distance_map = np.full((board.board_size.r, board.board_size.c), np.inf) # infinity 2d array dims r,c
				self.bfs_floodfill(board, distance_map, computing_hill) # bfs floodfill fills the matrix with distance from hill.
				self.hill_precomputed_matrices[computing_hill.id] = distance_map
		return 0
	
	def play(
		self,
		board: Board,
		player_parity: int,
		time_left: Callable,
	) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
		return None
		
	def bfs_floodfill(self, board: Board, distance_matrix: NDArray, start: Hill):
		"""
		BFS floodfill is used to fill the distance matrix with distance from a specific hill.
		Used in precomputing during first bid phase.
		"""
		dist = 0
		q = deque() # deque contains (location, dist to start)
		for cell in start.cells:
			q.append((cell, 0))

		while q:
			loc, distance = q.popleft()
			distance_matrix[loc.r][loc.c] = distance
			for newLoc in loc.neighbors():
				if not board.oob(newLoc) and not board.cells[newLoc.r][newLoc.c].is_wall and distance_matrix[newLoc.r][newLoc.c] == np.inf:
					distance_matrix[loc.r][loc.c] = distance + 1
					q.append((newLoc, distance + 1))
		return

	def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
		"""
		Allows for you to display a string at the end of the match on our online
		portal for your own statistics usage. Be careful, your opponents will be 
		able to see this as well.
		"""
		return ""
		
