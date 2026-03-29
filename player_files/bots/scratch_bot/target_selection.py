from collections import deque
from game import *

class Algorithms:
    class TargetSelector:
        # bfs to the first hill it finds.
        def bfs(self, player_parity: int, board: Board, current_pos: Location):
            seen = {(current_pos.r, current_pos.c)}
            q = deque([current_pos])

            while q:
                nextPos = q.popleft()
                cellState = board.cells[nextPos.r][nextPos.c]
                if (cellState.hill_id and cellState.hill_id not in board.get_player(player_parity).controlled_hills):
                    return cellState.hill_id
                else: # keep searching for a hill.
                    for neighborLoc in nextPos.neighbors():
                        if (not board.oob(neighborLoc) and not board.cells[neighborLoc.r][neighborLoc.c].is_wall and (neighborLoc.r, neighborLoc.c) not in seen):
                            seen.add((neighborLoc.r, neighborLoc.c))
                            q.append(neighborLoc)

    class ShortestPathSelector:
        # dijkstra's shortest 3 move path towards targeted hill
        def bfs_next(self, player_parity: int, board: Board, hill_cell_set: set[Location], player_pos: Location):
            q = deque([(player_pos, [player_pos])]) # queue of tuple(position, path up to position)
            visited = {(player_pos.r, player_pos.c)}

            while q:
                curPos, curPath = q.popleft()
                if curPos in hill_cell_set and curPos != player_pos and board.cells[curPos.r][curPos.c].owner_parity != player_parity:
                    return curPath[:2]
                else:
                    for nextPos in curPos.neighbors():
                        if (not board.oob(nextPos) and not board.cells[nextPos.r][nextPos.c].is_wall and (nextPos.r, nextPos.c) not in visited):
                            visited.add((nextPos.r, nextPos.c))
                            q.append((nextPos, curPath + [nextPos]))