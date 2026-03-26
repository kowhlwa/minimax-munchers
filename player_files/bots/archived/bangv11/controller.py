from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict, List, Tuple
import math

from game import *

DANGER_STRICT = 3
DANGER_MODERATE = 1


class PlayerController:
    """
    HillRusher v7 — collision-offensive + hill-efficiency rebuild.

    Collision model (board.py _resolve_collision):
      Resolves when co-located at START of a move. Cell owner wins.
      On neutral the "moving_player" (whose turn triggers check) wins,
      meaning the WALKER dies on neutral.

      Offensive plays:
        - Adjacent kill: step onto opponent on our cell (existing)
        - Multi-step kill: opponent 2 cells away on our territory → double-move
        - Paint-then-kill: opponent adjacent on neutral → paint cell, step on
        - Territorial pressure: paint cells around opponent to create kill zones

    Hill strategy:
      All hills give identical reward (+40 max stamina, same domination weight).
      Efficiency = reward / (travel_cost + capture_cost).
      Small hills >> large hills in ROI. Thick paint makes them nearly free to hold.

    Phases:
      RUSH  → no hills owned, sprint to most efficient uncaptured hill
      FARM  → local 5×5 under-saturated, paint thick for regen snowball
      EXPAND → push territory, capture additional hills, territorial pressure
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        if not board.hills:
            return 0
        player = board.get_player(player_parity)
        min_dist = 999
        for hill in board.hills.values():
            for hloc in hill.cells:
                d = abs(player.loc.r - hloc.r) + abs(player.loc.c - hloc.c)
                if d < min_dist:
                    min_dist = d
        if min_dist <= 3:
            return 15
        if min_dist <= 6:
            return 10
        return 5

    # ================================================================== #
    #  MAIN PLAY                                                          #
    # ================================================================== #

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        opp_loc = opponent.loc
        actions: list = []
        stamina = player.stamina
        extra_layers: Dict[Location, int] = {}

        # ---- Priority 1: Paint-then-kill combo ----
        # Opponent adjacent on neutral cell → paint it (now ours) → step on
        ptk = self._paint_then_kill(board, player, opponent, player_parity, stamina)
        if ptk is not None:
            return ptk

        # ---- Priority 2: Adjacent kill (opponent on our cell) ----
        kill_dir = self._check_kill(board, player, opponent, player_parity)
        if kill_dir:
            actions.append(Action.Move(kill_dir))
            new_pos = player.loc + kill_dir
            actions, stamina = self._smart_paint(
                board, player_parity, new_pos, actions, stamina, 20, extra_layers
            )
            return actions

        # ---- Priority 3: Multi-step kill (opponent 2 away on our territory) ----
        msk = self._multi_step_kill(board, player, opponent, player_parity, stamina)
        if msk is not None:
            return msk

        # ---- Priority 4: Escape (on opponent territory near opponent) ----
        cur_cell = board.cells[player.loc.r][player.loc.c]
        dist_to_opp = abs(player.loc.r - opp_loc.r) + abs(player.loc.c - opp_loc.c)
        if cur_cell.owner_parity == -player_parity and dist_to_opp <= 4:
            escape_dir = self._escape_step(board, player.loc, player_parity)
            if escape_dir:
                actions.append(Action.Move(escape_dir))
                new_pos = player.loc + escape_dir
                actions, stamina = self._smart_paint(
                    board, player_parity, new_pos, actions, stamina, 40, extra_layers
                )
                return actions

        # ---- State computation ----
        distances = self._bfs_all_distances(board, player.loc, player_parity)
        local_count = self._count_local_controlled(board, player.loc, player_parity)
        max_local = self._count_local_available(board, player.loc)
        phase = self._determine_phase(
            board, player, opponent, local_count, max_local, distances
        )

        # ---- Territorial pressure: near opponent, paint their surroundings ----
        if dist_to_opp <= 4 and phase != "rush":
            actions, stamina = self._territorial_pressure_paint(
                board, player_parity, player.loc, opp_loc, actions, stamina, extra_layers
            )

        # ---- RUSH ----
        if phase == "rush":
            target = self._choose_rush_target(board, player_parity, distances)
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    actions.append(Action.Move(direction))
                    td = distances.get(target)
                    if td and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST + 40:
                        np2 = self._simulate_position(board, player.loc, actions)
                        d2 = self._safe_step(board, np2, target, player_parity)
                        if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, 40, extra_layers
                    )

        # ---- FARM / EXPAND ----
        else:
            buf = 15 if phase == "farm" else 25

            actions, stamina = self._smart_paint(
                board, player_parity, player.loc, actions, stamina, buf, extra_layers
            )

            if phase == "farm":
                target = self._choose_farm_target(board, player_parity, player.loc)
                if target is None:
                    target = self._choose_expand_target(
                        board, player_parity, distances, player
                    )
            else:
                target = self._choose_expand_target(
                    board, player_parity, distances, player
                )

            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    actions.append(Action.Move(direction))

                    if phase == "expand":
                        td = distances.get(target)
                        if td and td > 3 and stamina >= GameConstants.EXTRA_MOVE_COST + 30:
                            np2 = self._simulate_position(board, player.loc, actions)
                            d2 = self._safe_step(board, np2, target, player_parity)
                            if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity):
                                actions.append(Action.Move(d2))
                                stamina -= GameConstants.EXTRA_MOVE_COST

                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, buf, extra_layers
                    )

        # ---- Guarantee at least one Move ----
        has_move = any(isinstance(a, Action.Move) for a in actions)
        if not has_move:
            fallback = self._any_valid_move(board, player_parity)
            if fallback:
                actions.append(fallback)
                new_pos = self._simulate_position(board, player.loc, actions)
                actions, stamina = self._smart_paint(
                    board, player_parity, new_pos, actions, stamina, 20, extra_layers
                )
            else:
                return []

        return actions

    # ================================================================== #
    #  OFFENSIVE COLLISIONS                                               #
    # ================================================================== #

    def _paint_then_kill(
        self, board: Board, player: Player, opponent: Player,
        player_parity: int, stamina: int,
    ) -> Optional[list]:
        """
        If opponent is adjacent on a NEUTRAL cell and we have stamina to
        paint + move: paint the cell (makes it ours) then step on them.
        Engine executes actions sequentially, so paint resolves before move.
        On our cell → collision on their turn → they die.
        """
        if stamina < GameConstants.PAINT_STAMINA_COST:
            return None

        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt) or nxt != opponent.loc:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if cell.owner_parity == 0 and cell.beacon_parity == 0:
                return [Action.Paint(nxt), Action.Move(direction)]

        return None

    def _check_kill(
        self, board: Board, player: Player, opponent: Player,
        player_parity: int,
    ) -> Optional[Direction]:
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

    def _multi_step_kill(
        self, board: Board, player: Player, opponent: Player,
        player_parity: int, stamina: int,
    ) -> Optional[list]:
        """
        Opponent is 2 cells away and the destination cell (opponent's cell)
        is ours → double-move for a kill. Costs EXTRA_MOVE_COST for 2nd step.
        """
        if stamina < GameConstants.EXTRA_MOVE_COST + 20:
            return None

        opp_loc = opponent.loc
        opp_cell = board.cells[opp_loc.r][opp_loc.c]

        if opp_cell.owner_parity != player_parity:
            return None

        for d1 in Direction.cardinals():
            mid = player.loc + d1
            if board.oob(mid):
                continue
            mid_cell = board.cells[mid.r][mid.c]
            if mid_cell.is_wall:
                continue
            if mid == opp_loc:
                continue

            for d2 in Direction.cardinals():
                dest = mid + d2
                if dest != opp_loc:
                    continue
                return [Action.Move(d1), Action.Move(d2)]

        return None

    def _territorial_pressure_paint(
        self,
        board: Board,
        player_parity: int,
        player_loc: Location,
        opp_loc: Location,
        actions: list,
        stamina: int,
        extra_layers: Dict[Location, int],
    ):
        """
        When near the opponent, prioritize painting cells adjacent to
        both us and the opponent to create kill zones. If we paint cells
        around the opponent, any direction they move puts them on our turf.
        """
        COST = GameConstants.PAINT_STAMINA_COST
        MAX_VAL = GameConstants.MAX_PAINT_VALUE

        opp_adj: Set[Location] = set()
        for d in Direction.cardinals():
            n = opp_loc + d
            if not board.oob(n):
                opp_adj.add(n)

        for direction in Direction.cardinals():
            if stamina - COST < 30:
                break
            t = player_loc + direction
            if board.oob(t) or t not in opp_adj:
                continue
            cell = board.cells[t.r][t.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == -player_parity:
                continue

            added = extra_layers.get(t, 0)
            base_layers = abs(cell.paint_value) if cell.owner_parity == player_parity else 0
            if base_layers + added >= MAX_VAL:
                continue

            is_paintable = (cell.owner_parity == player_parity or cell.owner_parity == 0)
            if is_paintable:
                actions.append(Action.Paint(t))
                extra_layers[t] = added + 1
                stamina -= COST

        return actions, stamina

    # ================================================================== #
    #  PHASE DETERMINATION                                                #
    # ================================================================== #

    def _determine_phase(
        self, board: Board, player: Player, opponent: Player,
        local_count: int, max_local: int, distances: Dict[Location, int],
    ) -> str:
        total_hills = len(board.hills)
        our_hills = len(player.controlled_hills)

        if total_hills > 0 and our_hills == 0:
            for loc in distances:
                cell = board.cells[loc.r][loc.c]
                if cell.hill_id != 0:
                    h = board.hills[cell.hill_id]
                    if h.controller_parity != player.parity:
                        return "rush"

        threshold = max(max_local * 0.65, 8)
        if local_count < threshold:
            return "farm"

        return "expand"

    def _count_local_controlled(
        self, board: Board, loc: Location, player_parity: int,
    ) -> int:
        radius = GameConstants.ADJACENCY_RADIUS
        count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nloc = Location(loc.r + dr, loc.c + dc)
                if board.oob(nloc):
                    continue
                if board.cells[nloc.r][nloc.c].owner_parity == player_parity:
                    count += 1
        return count

    def _count_local_available(
        self, board: Board, loc: Location,
    ) -> int:
        radius = GameConstants.ADJACENCY_RADIUS
        count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nloc = Location(loc.r + dr, loc.c + dc)
                if board.oob(nloc):
                    continue
                if not board.cells[nloc.r][nloc.c].is_wall:
                    count += 1
        return count

    # ================================================================== #
    #  TARGET SELECTION                                                   #
    # ================================================================== #

    def _hill_efficiency(
        self, board: Board, player_parity: int,
        hill: Hill, nearest_dist: int,
    ) -> float:
        """
        Score a hill by capture efficiency.
        All hills give identical reward, so minimize cost.
        cost = travel_turns + cells_to_paint * paint_cost_equivalent
        Return negative cost (higher = better).
        """
        hill_size = len(hill.cells)
        threshold = GameConstants.HILL_CONTROL_THRESHOLD
        needed = math.ceil(hill_size * threshold) + 1
        our_count = sum(
            1 for hloc in hill.cells
            if board.cells[hloc.r][hloc.c].owner_parity == player_parity
        )
        cells_to_paint = max(0, needed - our_count)

        paint_time = cells_to_paint
        travel_cost = nearest_dist

        total_cost = travel_cost + paint_time * 1.5
        if total_cost <= 0:
            total_cost = 0.1

        defense_cost = hill_size * 0.3

        return -(total_cost + defense_cost)

    def _choose_rush_target(
        self, board: Board, player_parity: int, distances: Dict[Location, int],
    ) -> Optional[Location]:
        """Rush to the most EFFICIENT hill, not just the nearest."""
        best: Optional[Location] = None
        best_eff = -9999.0

        seen_hills: Set[int] = set()
        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id == 0 or cell.hill_id in seen_hills:
                continue
            hill = board.hills[cell.hill_id]
            if hill.controller_parity == player_parity:
                continue
            seen_hills.add(cell.hill_id)

            nearest_dist: Optional[int] = None
            nearest_loc: Optional[Location] = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = distances[hloc]
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_loc = hloc

            if nearest_loc is None or nearest_dist is None:
                continue

            eff = self._hill_efficiency(board, player_parity, hill, nearest_dist)
            if eff > best_eff:
                best_eff = eff
                best = nearest_loc

        if best is not None:
            for loc, dist in distances.items():
                cell = board.cells[loc.r][loc.c]
                if cell.powerup and dist <= 2:
                    best = loc
                    break

        return best

    def _choose_farm_target(
        self, board: Board, player_parity: int, player_loc: Location,
    ) -> Optional[Location]:
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc

        best: Optional[Location] = None
        best_score = 0

        for direction in Direction.cardinals():
            nxt = player_loc + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            if self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE):
                continue

            score = self._paint_value_from(board, player_parity, nxt)
            if score > best_score:
                best_score = score
                best = nxt

        return best

    def _paint_value_from(
        self, board: Board, player_parity: int, pos: Location,
    ) -> int:
        score = 0
        MAX_VAL = GameConstants.MAX_PAINT_VALUE
        for direction in Direction.cardinals():
            t = pos + direction
            if board.oob(t):
                continue
            cell = board.cells[t.r][t.c]
            if cell.is_wall or cell.beacon_parity != 0:
                continue
            if cell.owner_parity == -player_parity:
                continue

            if cell.owner_parity == 0:
                score += 20
                if cell.hill_id != 0:
                    score += 50
            elif cell.owner_parity == player_parity:
                layers = abs(cell.paint_value)
                if layers < MAX_VAL:
                    gap = MAX_VAL - layers
                    score += gap * 3
                    if cell.hill_id != 0:
                        score += gap * 10
        return score

    def _choose_expand_target(
        self, board: Board, player_parity: int,
        distances: Dict[Location, int], player: Player,
    ) -> Optional[Location]:
        total_hills = len(board.hills)
        our_hills = len(player.controlled_hills)
        hills_for_win = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD) if total_hills > 0 else 0
        close_to_dom = (total_hills > 0 and our_hills + 1 >= hills_for_win)

        best_target: Optional[Location] = None
        best_score: float = -9999.0

        # ---- Hills scored by efficiency ----
        seen_hills: Set[int] = set()
        for hill_id, hill in board.hills.items():
            if hill.controller_parity == player_parity:
                continue
            seen_hills.add(hill_id)

            nearest_dist: Optional[int] = None
            nearest_loc: Optional[Location] = None
            for hloc in hill.cells:
                if hloc not in distances:
                    continue
                hcell = board.cells[hloc.r][hloc.c]
                if hcell.owner_parity == player_parity:
                    continue
                d = distances[hloc]
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_loc = hloc

            if nearest_loc is None or nearest_dist is None:
                continue

            eff = self._hill_efficiency(board, player_parity, hill, nearest_dist)

            base = 500.0 if close_to_dom else 200.0
            score = base + eff * 20.0

            if score > best_score:
                best_score = score
                best_target = nearest_loc

        # ---- Defend our hills under attack ----
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
                hill_size = len(hill.cells)
                urgency = opp_cells / hill_size if hill_size > 0 else 0
                score = 150.0 + urgency * 100.0 - best_def_dist * 2.0
                if score > best_score:
                    best_score = score
                    best_target = best_def_loc

        # ---- Powerups ----
        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.powerup:
                score = 120.0 - dist * 3.0
                if score > best_score:
                    best_score = score
                    best_target = loc

        # ---- Territory expansion (connected, regen-aware) ----
        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id != 0 or cell.powerup:
                continue
            if cell.owner_parity != 0:
                continue
            if loc == board.get_player(player_parity).loc:
                continue

            score: float = 0.0
            if self._adjacent_to_friendly(board, loc, player_parity):
                score = 30.0 - dist * 2.0
                if self._adjacent_to_opponent_territory(board, loc, player_parity):
                    score += 10.0
            else:
                score = 8.0 - dist * 2.0

            if score > best_score:
                best_score = score
                best_target = loc

        return best_target

    # ================================================================== #
    #  SMART PAINT — thickness-aware, regen-optimized                     #
    # ================================================================== #

    def _smart_paint(
        self,
        board: Board,
        player_parity: int,
        pos: Location,
        actions: list,
        stamina: int,
        buffer: int,
        extra_layers: Dict[Location, int],
    ):
        COST = GameConstants.PAINT_STAMINA_COST
        MAX_VAL = GameConstants.MAX_PAINT_VALUE

        while stamina - COST >= buffer:
            best: Optional[Location] = None
            best_score = -1

            for direction in Direction.cardinals():
                t = pos + direction
                if board.oob(t):
                    continue
                cell = board.cells[t.r][t.c]
                if cell.is_wall or cell.beacon_parity != 0:
                    continue
                if cell.owner_parity == -player_parity:
                    continue

                base_layers = abs(cell.paint_value) if cell.owner_parity == player_parity else 0
                added = extra_layers.get(t, 0)
                effective = base_layers + added
                if effective >= MAX_VAL:
                    continue

                is_new = (cell.owner_parity == 0 and added == 0)
                is_hill = (cell.hill_id != 0)
                gap = MAX_VAL - effective

                score = 0
                if is_hill:
                    score = 200 if is_new else 120 + gap * 15
                elif is_new:
                    score = 80
                else:
                    score = 15 + gap * 5

                if score > best_score:
                    best_score = score
                    best = t

            if best is None:
                break

            actions.append(Action.Paint(best))
            extra_layers[best] = extra_layers.get(best, 0) + 1
            stamina -= COST

        return actions, stamina

    # ================================================================== #
    #  COLLISION SAFETY                                                   #
    # ================================================================== #

    def _is_danger(
        self, board: Board, loc: Location, opp_loc: Location,
        player_parity: int, radius: int = DANGER_STRICT,
    ) -> bool:
        cell = board.cells[loc.r][loc.c]
        is_opp = (cell.owner_parity == -player_parity)
        if not is_opp:
            pv = cell.paint_value
            if player_parity == 1 and pv <= -2:
                is_opp = True
            elif player_parity == -1 and pv >= 2:
                is_opp = True
        if not is_opp:
            return False
        return abs(loc.r - opp_loc.r) + abs(loc.c - opp_loc.c) <= radius

    def _is_step_dangerous(
        self, board: Board, dest: Location, player_parity: int,
    ) -> bool:
        if board.oob(dest):
            return True
        opp = board.get_opponent(player_parity)
        return self._is_danger(board, dest, opp.loc, player_parity, DANGER_MODERATE)

    def _escape_step(
        self, board: Board, start: Location, player_parity: int,
    ) -> Optional[Direction]:
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc

        visited: Set[Location] = {start}
        queue: deque = deque()

        for direction in Direction.cardinals():
            nxt = start + direction
            if board.oob(nxt) or nxt in visited:
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall or nxt == opp_loc:
                continue
            visited.add(nxt)
            if not self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT):
                return direction
            queue.append((nxt, direction))

        while queue:
            loc, first_dir = queue.popleft()
            for direction in Direction.cardinals():
                nxt = loc + direction
                if board.oob(nxt) or nxt in visited:
                    continue
                cell = board.cells[nxt.r][nxt.c]
                if cell.is_wall or nxt == opp_loc:
                    continue
                visited.add(nxt)
                if not self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT):
                    return first_dir
                queue.append((nxt, first_dir))

        return None

    # ================================================================== #
    #  PATHFINDING                                                        #
    # ================================================================== #

    def _bfs_all_distances(
        self, board: Board, start: Location, player_parity: int,
    ) -> Dict[Location, int]:
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc
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
                if nxt == opp_loc and cell.owner_parity != player_parity:
                    continue
                distances[nxt] = d + 1
                queue.append((nxt, d + 1))
        return distances

    def _safe_step(
        self, board: Board, start: Location, target: Location,
        player_parity: int,
    ) -> Optional[Direction]:
        if start == target:
            return None
        r = self._bfs_step(board, start, target, player_parity, DANGER_STRICT)
        if r:
            return r
        r = self._bfs_step(board, start, target, player_parity, DANGER_MODERATE)
        if r:
            return r
        return self._bfs_step(board, start, target, player_parity, 0)

    def _bfs_step(
        self, board: Board, start: Location, target: Location,
        player_parity: int, danger_radius: int,
    ) -> Optional[Direction]:
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc
        first_step_radius = max(danger_radius, DANGER_MODERATE)

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
            if self._is_danger(board, nxt, opp_loc, player_parity, first_step_radius):
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
                if danger_radius > 0 and self._is_danger(board, nxt, opp_loc, player_parity, danger_radius):
                    continue
                visited.add(nxt)
                queue.append((nxt, first_dir))

        return None

    # ================================================================== #
    #  UTILITIES                                                          #
    # ================================================================== #

    def _adjacent_to_friendly(
        self, board: Board, loc: Location, player_parity: int,
    ) -> bool:
        for d in Direction.cardinals():
            n = loc + d
            if not board.oob(n) and board.cells[n.r][n.c].owner_parity == player_parity:
                return True
        return False

    def _adjacent_to_opponent_territory(
        self, board: Board, loc: Location, player_parity: int,
    ) -> bool:
        for d in Direction.cardinals():
            n = loc + d
            if not board.oob(n) and board.cells[n.r][n.c].owner_parity == -player_parity:
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
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc

        candidates: list = []
        for direction in Direction.cardinals():
            nxt = player.loc + direction
            if board.oob(nxt):
                continue
            cell = board.cells[nxt.r][nxt.c]
            if cell.is_wall:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue

            sd = self._is_danger(board, nxt, opp_loc, player_parity, DANGER_STRICT)
            md = self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE)
            fr = (cell.owner_parity == player_parity)
            ne = (cell.owner_parity == 0)

            if fr and not sd:
                p = 5
            elif ne and not sd:
                p = 4
            elif not md and fr:
                p = 3
            elif not md:
                p = 2
            elif not sd:
                p = 1
            else:
                p = 0

            candidates.append((p, direction))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return Action.Move(candidates[0][1])

    def commentate(
        self, board: Board, player_parity: int, time_left: Callable,
    ) -> str:
        player = board.get_player(player_parity)
        mt = board.get_territory_count(player_parity)
        ot = board.get_territory_count(-player_parity)
        lc = self._count_local_controlled(board, player.loc, player_parity)
        return (
            f"hills={len(player.controlled_hills)}"
            f" terr={mt} vs={ot}"
            f" local5x5={lc}"
            f" stam={player.stamina}/{player.max_stamina}"
            f" r={board.current_round}"
        )
