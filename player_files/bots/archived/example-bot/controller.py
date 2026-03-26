from collections import deque
from typing import Union, Iterable, Dict, List
from game import *

class PlayerController:
    """
    MASTER DOMINATOR - Hybrid v16
    Designed to counter high-ELO passive-regen strategies.
    """
    def __init__(self, player_parity: int, time_left):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left) -> int:
        # Counter JMC's 0-bid by sniping initiative when near hills
        player = board.get_player(player_parity)
        dist = self._min_dist_to_hill(board, player.loc)
        if dist <= 2: return 35  # Snatch the hill
        if dist <= 5: return 15
        return 0

    def play(self, board: Board, player_parity: int, time_left) -> Union[Action, Iterable[Action]]:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        actions = []
        stamina = player.stamina

        # 1. KILL CHECK (Punish neighbors)
        kill = self._check_offensive_kill(board, player, opponent)
        if kill: return kill

        # 2. EMERGENCY DEFENSE (Teleport to contested hills)
        if self._hill_under_threat(board):
            target = self._get_threatened_hill(board)
            # Use beacon travel if possible (from bangv14 logic)
            # ... implementation omitted for brevity ...

        # 3. PHASE LOGIC: Speed vs Regen
        owned = len([h for h in board.hills.values() if h.controller_parity == player_parity])
        local_regen = self._count_local(board, player.loc)
        
        # If we have no hills, RUSH at max speed (ignore farming)
        if owned == 0:
            target = self._get_best_hill(board, player.loc)
            path = self._bfs(board, player.loc, target)
            if path:
                # TAKE 3 MOVES immediately to beat JMC to the spot
                for i in range(min(3, len(path))):
                    actions.append(Action.Move(self._dir(player.loc if i==0 else path[i-1], path[i])))
                # Place beacon on arrival for permanent lock
                actions[-1].place_beacon = True 
                return actions

        # 4. DEFAULT: Move-First then Post-Paint (bangv15 optimization)
        target = self._get_best_hill(board, player.loc)
        path = self._bfs(board, player.loc, target)
        if path:
            actions.append(Action.Move(self._dir(player.loc, path[0])))
            # Post-paint current cell for regen while moving
            actions.append(Action.Paint(player.loc))
            
            # Spend excess stamina on speed (Cost 10 for 2nd move)
            if stamina > 40 and len(path) > 1:
                actions.append(Action.Move(self._dir(path[0], path[1])))
                
        return actions

    # --- Heuristics ---

    def _check_offensive_kill(self, board, player, opponent):
        # If opponent is adjacent and we can win collision (ours/neutral), STEP ON THEM
        for d in Direction.cardinals():
            target = player.loc + d
            if target == opponent.loc:
                cell = board.cells[target.r][target.c]
                # If cell is ours or neutral, regular move is a kill
                if cell.owner_parity == self.player_parity or cell.owner_parity == 0:
                    return Action.Move(d)
                # If cell is THEIRS, use ERASE move (50 stamina) to kill them
                if player.stamina >= 50:
                    return Action.Move(d, move_type=MoveType.ERASE)
        return None

    def _get_best_hill(self, board, loc):
        # Prioritize small hills (higher ROI) and unowned ones
        best, min_score = None, 999
        for hill in board.hills.values():
            if hill.controller_parity != self.player_parity:
                dist = self._md(loc, list(hill.cells)[0])
                score = dist + (len(hill.cells) * 2) # Weighted distance
                if score < min_score:
                    min_score = score
                    best = list(hill.cells)[0]
        return best

    def _count_local(self, board, loc):
        count = 0
        for r in range(loc.r-2, loc.r+3):
            for c in range(loc.c-2, loc.c+3):
                if not board.oob(Location(r,c)) and board.cells[r][c].owner_parity == self.player_parity:
                    count += 1
        return count

    def _md(self, l1, l2): return abs(l1.r - l2.r) + abs(l1.c - l2.c)
    
    def _dir(self, start, end):
        if end.r < start.r: return Direction.UP
        if end.r > start.r: return Direction.DOWN
        if end.c < start.c: return Direction.LEFT
        return Direction.RIGHT

    def _min_dist_to_hill(self, board, loc):
        dists = [self._md(loc, list(h.cells)[0]) for h in board.hills.values()]
        return min(dists) if dists else 999

    def _bfs(self, board, start, target):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == target: return path
            for d in Direction.cardinals():
                nxt = curr + d
                if not board.oob(nxt) and not board.cells[nxt.r][nxt.c].is_wall and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None