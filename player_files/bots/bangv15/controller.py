from collections import deque
from collections.abc import Callable, Iterable
from typing import Union, Optional, Set, Dict, List, Tuple
import math

from game import *

DANGER_STRICT = 3
DANGER_MODERATE = 1
MIN_REGEN_BASE = 6  # v15: same as v13/v14 — preserves territory/regen for long rushes


class PlayerController:
    """
    HillRusher v15.

    Key changes from v14:
      - MOVE-FIRST in rush/contest/emergency: removed pre-paint before first move.
        v14's hybrid approach left only ~40 stamina for extra-move checks (threshold 50),
        so extra moves NEVER fired. Now: move first (full 100 stamina), check extra moves
        from full stamina, then post-paint after all moves. This enables 2-3 moves/turn
        reliably, matching and exceeding v12's speed to hills.
      - Extra move thresholds fixed: check from full stamina with lower threshold
        (EXTRA_MOVE_COST + 30 = 40) instead of broken (EXTRA_MOVE_COST + buf + 10 = 50).
      - Stricter danger check for extra moves: radius=2 (was radius=1 DANGER_MODERATE).
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    # ================================================================== #
    #  BID                                                                #
    # ================================================================== #

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
        if min_dist <= 2:
            return 35
        if min_dist <= 4:
            return 25
        if min_dist <= 6:
            return 15
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
        ptk = self._paint_then_kill(board, player, opponent, player_parity, stamina)
        if ptk is not None:
            return ptk

        # ---- Priority 2: Adjacent kill ----
        kill_dir = self._check_kill(board, player, opponent, player_parity)
        if kill_dir:
            actions.append(Action.Move(kill_dir))
            new_pos = player.loc + kill_dir
            actions, stamina = self._smart_paint(
                board, player_parity, new_pos, actions, stamina, 20, extra_layers
            )
            return actions

        # ---- Priority 3: Multi-step kill ----
        msk = self._multi_step_kill(board, player, opponent, player_parity, stamina)
        if msk is not None:
            return msk

        # ---- Priority 4: Escape ----
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

        # ---- Priority 5: Stamina conservation ----
        if stamina < 40 and cur_cell.owner_parity != player_parity:
            retreat_dir = self._retreat_to_friendly(board, player.loc, player_parity)
            if retreat_dir:
                actions.append(Action.Move(retreat_dir))
                new_pos = player.loc + retreat_dir
                actions, stamina = self._smart_paint(
                    board, player_parity, new_pos, actions, stamina, 30, extra_layers
                )
                return actions

        # ---- State ----
        distances = self._bfs_all_distances(board, player.loc, player_parity)
        local_count = self._count_local_controlled(board, player.loc, player_parity)
        max_local = self._count_local_available(board, player.loc)
        phase = self._determine_phase(board, player, opponent, local_count, max_local, distances)

        # ---- Priority 6: Beacon teleportation (emergency/contest/rush only) ----
        if phase in ("emergency", "contest", "rush"):
            btp = self._beacon_teleport(board, player, player_parity, phase, distances)
            if btp is not None:
                return [btp]

        # ---- Territorial pressure when near opponent ----
        if dist_to_opp <= 4 and phase not in ("rush", "emergency", "contest"):
            actions, stamina = self._territorial_pressure_paint(
                board, player_parity, player.loc, opp_loc, actions, stamina, extra_layers
            )

        # ---- EMERGENCY / CONTEST / RUSH ----
        if phase in ("emergency", "contest", "rush"):
            target = self._choose_hill_target(board, player_parity, distances, phase, opponent)
            move_buf = 30  # stamina threshold for taking extra moves (sustainability)
            paint_buf = 25  # stamina buffer to keep after post-paint (allows 1-2 extra cells)

            # v15: MOVE-FIRST — skip pre-paint to preserve full stamina for extra moves.
            # v14's hybrid pre-paint depleted stamina to ~40 before moves, making the
            # extra-move check (>= 50) never pass. Now extra moves fire reliably.
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    nxt1 = player.loc + direction
                    beacon_m1 = self._with_beacon_if_possible(board, player_parity, direction, nxt1)
                    actions.append(beacon_m1)
                    stamina_rem = stamina  # full stamina — no pre-paint depleted it

                    td = distances.get(target)
                    # When opponent is close, limit to 2nd move only (avoid rushing into
                    # opponent's zone — 3rd move lands too far from safety and
                    # opponent can intercept on their turn as mover on neutral cell).
                    close_to_opp = (dist_to_opp <= 4)
                    # Second move: check from full stamina, lower threshold (no +10 margin)
                    if td and td > 2 and stamina_rem >= GameConstants.EXTRA_MOVE_COST + move_buf:
                        np2 = self._simulate_position(board, player.loc, actions)
                        d2 = self._safe_step(board, np2, target, player_parity)
                        if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity, 2):
                            actions.append(Action.Move(d2))
                            stamina_rem -= GameConstants.EXTRA_MOVE_COST

                            # Third move: only when opponent is not close
                            if (not close_to_opp
                                    and td > 3
                                    and stamina_rem >= GameConstants.EXTRA_MOVE_COST * 2 + move_buf):
                                np3 = self._simulate_position(board, player.loc, actions)
                                d3 = self._safe_step(board, np3, target, player_parity)
                                if d3 and not self._is_step_dangerous(board, np3 + d3, player_parity, 2):
                                    actions.append(Action.Move(d3))
                                    stamina_rem -= GameConstants.EXTRA_MOVE_COST * 2

                                    # Fourth move in emergency only — when opponent about to win
                                    if (phase == "emergency" and td > 4
                                            and stamina_rem >= GameConstants.EXTRA_MOVE_COST * 3 + move_buf):
                                        np4 = self._simulate_position(board, player.loc, actions)
                                        d4 = self._safe_step(board, np4, target, player_parity)
                                        if d4 and not self._is_step_dangerous(board, np4 + d4, player_parity, 2):
                                            actions.append(Action.Move(d4))
                                            stamina_rem -= GameConstants.EXTRA_MOVE_COST * 3

                    # Post-paint after all moves — use paint_buf (lower than move_buf)
                    # to squeeze 1 extra cell painted when stamina is in 40-70 range.
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina_rem, paint_buf, extra_layers
                    )

        # ---- FARM ----
        elif phase == "farm":
            actions, stamina = self._smart_paint(
                board, player_parity, player.loc, actions, stamina, 15, extra_layers
            )
            target = self._choose_farm_target(board, player_parity, player.loc)
            if target is None:
                target = self._choose_expand_target(board, player_parity, distances, player, opponent)
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    nxt = player.loc + direction
                    beacon_m = self._with_beacon_if_possible(board, player_parity, direction, nxt)
                    actions.append(beacon_m)
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, 15, extra_layers
                    )

        # ---- ENDGAME TERRITORY ----
        elif phase == "territory":
            actions, stamina = self._smart_paint(
                board, player_parity, player.loc, actions, stamina, 10, extra_layers
            )
            target = self._choose_territory_target(board, player_parity, distances, player)
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    nxt = player.loc + direction
                    beacon_m = self._with_beacon_if_possible(board, player_parity, direction, nxt)
                    actions.append(beacon_m)
                    td = distances.get(target)
                    if td and td > 2 and stamina >= GameConstants.EXTRA_MOVE_COST + 15:
                        np2 = self._simulate_position(board, player.loc, actions)
                        d2 = self._safe_step(board, np2, target, player_parity)
                        if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity):
                            actions.append(Action.Move(d2))
                            stamina -= GameConstants.EXTRA_MOVE_COST
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, 10, extra_layers
                    )

        # ---- EXPAND ----
        else:
            actions, stamina = self._smart_paint(
                board, player_parity, player.loc, actions, stamina, 25, extra_layers
            )
            target = self._choose_expand_target(board, player_parity, distances, player, opponent)
            if target:
                direction = self._safe_step(board, player.loc, target, player_parity)
                if direction:
                    nxt = player.loc + direction
                    beacon_m = self._with_beacon_if_possible(board, player_parity, direction, nxt)
                    actions.append(beacon_m)
                    td = distances.get(target)
                    target_is_hill = (not board.oob(target)
                                      and board.cells[target.r][target.c].hill_id != 0)
                    stamina_after_m1 = stamina
                    # Double-move when target is 4+ steps away (radius=2 safety same as rush)
                    if td and td > 3 and stamina_after_m1 >= GameConstants.EXTRA_MOVE_COST + 30:
                        np2 = self._simulate_position(board, player.loc, actions)
                        d2 = self._safe_step(board, np2, target, player_parity)
                        if d2 and not self._is_step_dangerous(board, np2 + d2, player_parity, 2):
                            actions.append(Action.Move(d2))
                            stamina_after_m1 -= GameConstants.EXTRA_MOVE_COST
                            # Triple move when expanding to a hill
                            if (target_is_hill and td > 4
                                    and stamina_after_m1 >= GameConstants.EXTRA_MOVE_COST * 2 + 30):
                                np3 = self._simulate_position(board, player.loc, actions)
                                d3 = self._safe_step(board, np3, target, player_parity)
                                if d3 and not self._is_step_dangerous(board, np3 + d3, player_parity, 2):
                                    actions.append(Action.Move(d3))
                    new_pos = self._simulate_position(board, player.loc, actions)
                    actions, stamina = self._smart_paint(
                        board, player_parity, new_pos, actions, stamina, 25, extra_layers
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
    #  PHASE DETERMINATION — REGEN-GATED                                  #
    # ================================================================== #

    def _determine_phase(
        self, board: Board, player: Player, opponent: Player,
        local_count: int, max_local: int, distances: Dict[Location, int],
    ) -> str:
        total_hills = len(board.hills)
        our_hills = len(player.controlled_hills)
        opp_hills = len(opponent.controlled_hills)
        have_regen_base = (local_count >= MIN_REGEN_BASE)

        if total_hills == 0:
            # No hills: endgame territory if late, else farm/expand
            if board.current_round > 750:
                return "territory"
            threshold = max(max_local * 0.65, 8)
            if local_count < threshold:
                return "farm"
            return "expand"

        dom_needed = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD)

        # EMERGENCY: opponent 1 hill from domination — override regen gate
        if opp_hills + 1 >= dom_needed and opp_hills > our_hills:
            return "emergency"

        # If we don't have a regen base yet, farm first.
        if not have_regen_base:
            return "farm"

        # Late game: behind on hills with regen base → emergency
        if board.current_round > 400 and opp_hills > our_hills:
            return "emergency"

        # CONTEST: opponent has more hills (we have regen base)
        if opp_hills > our_hills:
            return "contest"

        # RUSH: 0 hills (we have regen base)
        if our_hills == 0:
            reachable = False
            for loc in distances:
                cell = board.cells[loc.r][loc.c]
                if cell.hill_id != 0 and board.hills[cell.hill_id].controller_parity != player.parity:
                    reachable = True
                    break
            if reachable:
                return "rush"

        # Endgame territory push: leading/tied on hills (and we actually have hills), late game
        if board.current_round > 750 and our_hills > 0 and our_hills >= opp_hills:
            return "territory"

        # Mid-game territory push: when hills are tied and all hills are captured,
        # avoid futile hill re-challenges (opponent defends with layers/beacons).
        # Flood neutral territory instead to win tiebreaks.
        if (board.current_round > 500 and our_hills == opp_hills and our_hills > 0):
            any_neutral_hill = any(
                board.hills[cell.hill_id].controller_parity == 0
                for loc in distances
                for cell in [board.cells[loc.r][loc.c]]
                if cell.hill_id != 0
            )
            if not any_neutral_hill:
                return "territory"

        # Ahead or tied — farm or expand
        threshold = max(max_local * 0.65, 8)
        if local_count < threshold:
            return "farm"

        return "expand"

    # ================================================================== #
    #  BEACON HELPERS (v14 new)                                           #
    # ================================================================== #

    def _can_place_beacon_at(
        self, board: Board, target: Location, player_parity: int
    ) -> bool:
        """Return True if a beacon can be placed at target (our movement destination)."""
        if board.oob(target):
            return False
        target_cell = board.cells[target.r][target.c]
        if target_cell.owner_parity != player_parity:
            return False
        if target_cell.beacon_parity != 0:
            return False
        window = target.square_region(1)  # 3×3 centered on target
        friendly = sum(
            1 for loc in window
            if not board.oob(loc)
            and board.cells[loc.r][loc.c].beacon_parity == 0
            and Parity.owned(board.cells[loc.r][loc.c].paint_value, player_parity)
        )
        return friendly >= GameConstants.BEACON_REQUIREMENT_Q

    def _with_beacon_if_possible(
        self,
        board: Board,
        player_parity: int,
        direction: Direction,
        target: Location,
    ) -> Action.Move:
        """
        Return a Move with place_beacon=True if beacon deployment is valid at target.
        ONLY call this for the FIRST move of a turn — subsequent moves must use plain
        Action.Move because the board state is stale after move 1 executes.
        Since BEACON_COST=0, deploying is always worth it: permanent anchor the
        opponent can never remove.
        """
        if self._can_place_beacon_at(board, target, player_parity):
            return Action.Move(direction, place_beacon=True)
        return Action.Move(direction)

    def _beacon_teleport(
        self,
        board: Board,
        player: Player,
        player_parity: int,
        phase: str,
        distances: Dict[Location, int],
    ) -> Optional[Action.Move]:
        """
        If standing on own beacon and another beacon exists at a better location,
        return a BEACON_TRAVEL move. Used for emergency repositioning.
        """
        cur_cell = board.cells[player.loc.r][player.loc.c]
        if cur_cell.beacon_parity != player_parity:
            return None
        if player.beacon_count < 2:
            return None

        opponent = board.get_opponent(player_parity)
        best_target: Optional[Location] = None
        best_score = 50.0  # minimum threshold to bother teleporting

        for r in range(board.board_size.r):
            for c in range(board.board_size.c):
                loc = Location(r, c)
                if loc == player.loc:
                    continue
                cell = board.cells[r][c]
                if cell.beacon_parity != player_parity:
                    continue

                score = 0.0

                # Bonus: beacon near contested hill we own
                for hill_id in player.controlled_hills:
                    hill = board.hills[hill_id]
                    opp_cells = sum(
                        1 for hloc in hill.cells
                        if board.cells[hloc.r][hloc.c].owner_parity == -player_parity
                    )
                    if opp_cells > 0:
                        for hloc in hill.cells:
                            d = abs(loc.r - hloc.r) + abs(loc.c - hloc.c)
                            if d <= 3:
                                score += 250 + opp_cells * 60 - d * 15
                            break  # only count closest cell

                # Bonus: beacon near uncaptured hill (rush/contest/emergency)
                if phase in ("emergency", "contest", "rush"):
                    for hill_id, hill in board.hills.items():
                        if hill.controller_parity == player_parity:
                            continue
                        for hloc in hill.cells:
                            d = abs(loc.r - hloc.r) + abs(loc.c - hloc.c)
                            if d <= 5:
                                score += 180 - d * 20
                            break

                # Penalty: beacon near opponent
                dist_to_opp = abs(loc.r - opponent.loc.r) + abs(loc.c - opponent.loc.c)
                if dist_to_opp <= 2:
                    score -= 400

                # Current position distance penalty (must be worth the jump)
                cur_dist_to_opp_hill = min(
                    (distances.get(hloc, 999)
                     for hid, hill in board.hills.items()
                     if hill.controller_parity != player_parity
                     for hloc in hill.cells),
                    default=999,
                )
                score -= cur_dist_to_opp_hill * 5

                if score > best_score:
                    best_score = score
                    best_target = loc

        if best_target is not None:
            return Action.Move(None, MoveType.BEACON_TRAVEL, beacon_target=best_target)
        return None

    # ================================================================== #
    #  OFFENSIVE COLLISIONS                                               #
    # ================================================================== #

    def _paint_then_kill(
        self, board: Board, player: Player, opponent: Player,
        player_parity: int, stamina: int,
    ) -> Optional[list]:
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
            if mid_cell.is_wall or mid == opp_loc:
                continue
            for d2 in Direction.cardinals():
                if mid + d2 == opp_loc:
                    return [Action.Move(d1), Action.Move(d2)]
        return None

    def _territorial_pressure_paint(
        self, board: Board, player_parity: int, player_loc: Location,
        opp_loc: Location, actions: list, stamina: int,
        extra_layers: Dict[Location, int],
    ):
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
            if cell.is_wall or cell.beacon_parity != 0 or cell.owner_parity == -player_parity:
                continue
            added = extra_layers.get(t, 0)
            base_layers = abs(cell.paint_value) if cell.owner_parity == player_parity else 0
            if base_layers + added >= MAX_VAL:
                continue
            actions.append(Action.Paint(t))
            extra_layers[t] = added + 1
            stamina -= COST
        return actions, stamina

    # ================================================================== #
    #  TARGET SELECTION                                                   #
    # ================================================================== #

    def _hill_efficiency(
        self, board: Board, player_parity: int,
        hill: Hill, nearest_dist: int,
    ) -> float:
        hill_size = len(hill.cells)
        threshold = GameConstants.HILL_CONTROL_THRESHOLD
        needed = math.ceil(hill_size * threshold) + 1
        our_count = sum(
            1 for hloc in hill.cells
            if board.cells[hloc.r][hloc.c].owner_parity == player_parity
        )
        cells_to_paint = max(0, needed - our_count)
        total_cost = nearest_dist + cells_to_paint * 1.5
        if total_cost <= 0:
            total_cost = 0.1
        return -total_cost

    def _choose_hill_target(
        self, board: Board, player_parity: int,
        distances: Dict[Location, int], phase: str, opponent: Player,
    ) -> Optional[Location]:
        player = board.get_player(player_parity)
        best: Optional[Location] = None
        best_score = -9999.0

        for hill_id, hill in board.hills.items():
            if hill.controller_parity == player_parity:
                continue
            opp_holds = (hill.controller_parity == opponent.parity)
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
            if phase == "emergency":
                score = 1000.0 + eff * 30.0
                if opp_holds:
                    score += 200.0
            elif phase == "contest":
                score = 500.0 + eff * 25.0
                if opp_holds:
                    score += 100.0
            else:
                score = 300.0 + eff * 20.0
            if score > best_score:
                best_score = score
                best = nearest_loc

        # Defend our hills under attack
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
                score = 300.0 + urgency * 300.0 - best_def_dist * 2.0
                if score > best_score:
                    best_score = score
                    best = best_def_loc

        # Grab close powerups en route (v14: raised range from 2 to 4)
        if best is not None:
            for loc, dist in distances.items():
                cell = board.cells[loc.r][loc.c]
                if cell.powerup and dist <= 4:
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
            # Never farm into opponent-owned cells — too dangerous
            if cell.owner_parity == -player_parity:
                continue
            if nxt == opp_loc and cell.owner_parity != player_parity:
                continue
            if self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE):
                continue
            # Avoid landing on neutral cell adjacent to opponent (paint-then-kill trap)
            if (cell.owner_parity == 0 and cell.beacon_parity == 0
                    and abs(nxt.r - opp_loc.r) + abs(nxt.c - opp_loc.c) == 1):
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
            if cell.is_wall or cell.beacon_parity != 0 or cell.owner_parity == -player_parity:
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
        distances: Dict[Location, int], player: Player, opponent: Player,
    ) -> Optional[Location]:
        total_hills = len(board.hills)
        our_hills = len(player.controlled_hills)
        opp_hills = len(opponent.controlled_hills)
        hills_for_win = math.ceil(total_hills * GameConstants.DOMINATION_WIN_THRESHOLD) if total_hills > 0 else 0
        opp_loc = opponent.loc
        close_to_dom = (total_hills > 0 and our_hills + 1 >= hills_for_win)
        late_game = board.current_round > 600
        hill_boost = 100.0 if late_game and our_hills <= opp_hills else 0.0

        best_target: Optional[Location] = None
        best_score: float = -9999.0

        for hill_id, hill in board.hills.items():
            if hill.controller_parity == player_parity:
                continue
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
            # Direct distance-based scoring: hills always beat adjacent territory
            base = 800.0 if close_to_dom else 500.0
            score = base + hill_boost - nearest_dist * 6.0
            if score > best_score:
                best_score = score
                best_target = nearest_loc

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
                score = 150.0 + urgency * 150.0 - best_def_dist * 2.0
                if score > best_score:
                    best_score = score
                    best_target = best_def_loc

        # Powerups (v14: raised range from 3 to 4 steps)
        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.powerup:
                score = 140.0 - dist * 3.0
                if score > best_score:
                    best_score = score
                    best_target = loc

        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.hill_id != 0 or cell.powerup:
                continue
            if cell.owner_parity != 0 or loc == player.loc:
                continue
            # Skip neutral cells immediately adjacent to opponent (paint-then-kill risk)
            if abs(loc.r - opp_loc.r) + abs(loc.c - opp_loc.c) <= 1:
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

    def _choose_territory_target(
        self, board: Board, player_parity: int,
        distances: Dict[Location, int], player: Player,
    ) -> Optional[Location]:
        """
        v14 endgame: pick the closest unowned cell to flood-fill territory.
        Prioritizes cells adjacent to our existing territory for connected expansion.
        Avoids cells immediately adjacent to opponent (paint-then-kill risk).
        """
        opp_loc = board.get_opponent(player_parity).loc
        best_target: Optional[Location] = None
        best_score: float = -9999.0

        for loc, dist in distances.items():
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall or cell.owner_parity != 0 or loc == player.loc:
                continue
            # Skip neutral cells adjacent to opponent (paint-then-kill vulnerability)
            if abs(loc.r - opp_loc.r) + abs(loc.c - opp_loc.c) <= 1:
                continue
            score = 50.0 - dist * 3.0
            if self._adjacent_to_friendly(board, loc, player_parity):
                score += 20.0
            if cell.hill_id != 0:
                score += 30.0
            if score > best_score:
                best_score = score
                best_target = loc

        return best_target

    # ================================================================== #
    #  SMART PAINT                                                        #
    # ================================================================== #

    def _smart_paint(
        self, board: Board, player_parity: int, pos: Location,
        actions: list, stamina: int, buffer: int,
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
                if cell.is_wall or cell.beacon_parity != 0 or cell.owner_parity == -player_parity:
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
        radius: int = DANGER_MODERATE,
    ) -> bool:
        if board.oob(dest):
            return True
        opp = board.get_opponent(player_parity)
        opp_loc = opp.loc
        # Never step directly onto opponent's current location.
        if dest == opp_loc:
            return True
        return self._is_danger(board, dest, opp_loc, player_parity, radius)

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

    def _retreat_to_friendly(
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
            if self._is_danger(board, nxt, opp_loc, player_parity, DANGER_MODERATE):
                continue
            visited.add(nxt)
            if cell.owner_parity == player_parity:
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
                if cell.owner_parity == player_parity:
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
        opp = board.get_opponent(player_parity)
        lc = self._count_local_controlled(board, player.loc, player_parity)
        la = self._count_local_available(board, player.loc)
        phase = self._determine_phase(board, player, opp, lc, la, {})
        mt = board.get_territory_count(player_parity)
        ot = board.get_territory_count(-player_parity)
        return (
            f"[{phase}] hills={len(player.controlled_hills)}v{len(opp.controlled_hills)}"
            f" terr={mt}v{ot}"
            f" local={lc}/{la}"
            f" beacons={player.beacon_count}"
            f" stam={player.stamina}/{player.max_stamina}"
            f" r={board.current_round}"
        )
