#!/usr/bin/env python3
"""
ByteFight 2026: Paint — Replay Analyzer

Processes game replay JSONs and produces structured text summaries
that Claude Code can consume to learn from matches.

Usage:
    python3 replay_analyzer.py <replay.json> [--verbose]
    python3 replay_analyzer.py <directory_of_jsons> [--verbose]

Output: For each replay, writes a .analysis.txt file alongside the JSON,
        plus prints a summary to stdout.
"""

import json
import sys
import os
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Game constants (from game_constants.py)
BASE_MAX_STAMINA = 100
HILL_MAX_STAMINA_BONUS = 40
EXTRA_MOVE_COST = 10
PAINT_STAMINA_COST = 15
MAX_PAINT_VALUE = 4
ERASE_STEP_EXTRA_COST = 50
HILL_CONTROL_THRESHOLD = 0.5
DOMINATION_WIN_THRESHOLD = 0.75
BASE_STAMINA_REGEN = 5
ADJACENT_REGEN_BONUS = 2
ADJACENCY_RADIUS = 2
GLOBAL_PAINT_REGEN_RATIO = 8
GLOBAL_PAINT_REGEN_CAP = 50
STAMINA_POWERUP_AMOUNT = 35
GLOBAL_DECAY_TURN_THRESHOLD = 1000
GLOBAL_DECAY_INTERVAL = 100
GLOBAL_DECAY_REGEN_PENALTY = 1


class ReplayAnalyzer:
    """Analyzes a single game replay JSON."""

    def __init__(self, data: dict):
        self.data = data
        self.num_turns = data.get("turn_count", 0)
        self.result = data.get("result", "UNKNOWN")
        self.reason = data.get("reason", "UNKNOWN")
        self.map_name = data.get("map_string", "unknown")
        # extract just the map name if it's a full map string
        if isinstance(self.map_name, str) and "\n" in self.map_name:
            self.map_name = self.map_name.split("\n")[0].strip()

        self.walls = data.get("walls", [])
        self.hill_mapping = data.get("hill_mapping", [])
        self.board_rows = len(self.walls) if self.walls else 0
        self.board_cols = len(self.walls[0]) if self.walls and self.walls[0] else 0

        self.p1_bid = data.get("p1_bid", 0)
        self.p2_bid = data.get("p2_bid", 0)

        # Per-turn arrays
        self.p1_loc = data.get("p1_loc", [])
        self.p2_loc = data.get("p2_loc", [])
        self.p1_stamina = data.get("p1_stamina", [])
        self.p2_stamina = data.get("p2_stamina", [])
        self.p1_max_stamina = data.get("p1_max_stamina", [])
        self.p2_max_stamina = data.get("p2_max_stamina", [])
        self.p1_territory = data.get("p1_territory", [])
        self.p2_territory = data.get("p2_territory", [])
        self.parity_playing = data.get("parity_playing", [])
        self.paint_updates = data.get("paint_updates", [])
        self.beacon_updates = data.get("beacon_updates", [])
        self.hill_updates = data.get("hill_updates", [])
        self.powerup_updates = data.get("powerup_updates", [])
        self.actions = data.get("actions", [])

        # Derived
        self.wall_count = sum(cell for row in self.walls for cell in row) if self.walls else 0
        self.total_cells = self.board_rows * self.board_cols
        self.passable_cells = self.total_cells - self.wall_count
        self.wall_ratio = self.wall_count / self.total_cells if self.total_cells > 0 else 0

        # Hill info
        self.hills = self._parse_hills()
        self.num_hills = len(self.hills)

    def _parse_hills(self) -> Dict[int, List[Tuple[int, int]]]:
        """Extract hill cells from hill_mapping."""
        hills = defaultdict(list)
        for r, row in enumerate(self.hill_mapping):
            for c, hid in enumerate(row):
                if hid != 0:
                    hills[hid].append((r, c))
        return dict(hills)

    def _cell_from_index(self, idx: int) -> Tuple[int, int]:
        """Convert flattened cell index to (row, col)."""
        return (idx // self.board_cols, idx % self.board_cols)

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _loc_tuple(self, loc) -> Tuple[int, int]:
        """Normalize location to tuple."""
        if isinstance(loc, (list, tuple)):
            return (loc[0], loc[1])
        return (0, 0)

    # ── Event Detection ──────────────────────────────────────────────

    def detect_deaths(self) -> List[dict]:
        """Detect player deaths by stamina going to -1 or location jumping to spawn."""
        deaths = []
        for t in range(1, len(self.p1_stamina)):
            # P1 death: stamina drops to -1 or respawns (stamina jumps down to ~60% of max)
            if self.p1_stamina[t] == -1 or (
                t + 1 < len(self.p1_stamina) and
                self.p1_stamina[t] >= 0 and
                self.p1_stamina[t] < self.p1_stamina[t - 1] * 0.5 and
                self._loc_tuple(self.p1_loc[t]) != self._loc_tuple(self.p1_loc[t - 1])
            ):
                # Check if opponent was at same location (collision)
                p1_pos = self._loc_tuple(self.p1_loc[t - 1])
                p2_pos = self._loc_tuple(self.p2_loc[t])
                p1_curr = self._loc_tuple(self.p1_loc[t])

                deaths.append({
                    "turn": t,
                    "player": "P1",
                    "stamina_before": self.p1_stamina[t - 1],
                    "stamina_after": self.p1_stamina[t],
                    "location": p1_pos,
                })

            if self.p2_stamina[t] == -1 or (
                t + 1 < len(self.p2_stamina) and
                self.p2_stamina[t] >= 0 and
                self.p2_stamina[t] < self.p2_stamina[t - 1] * 0.5 and
                self._loc_tuple(self.p2_loc[t]) != self._loc_tuple(self.p2_loc[t - 1])
            ):
                p2_pos = self._loc_tuple(self.p2_loc[t - 1])

                deaths.append({
                    "turn": t,
                    "player": "P2",
                    "stamina_before": self.p2_stamina[t - 1],
                    "stamina_after": self.p2_stamina[t],
                    "location": p2_pos,
                })

        return deaths

    def detect_collisions(self) -> List[dict]:
        """Detect collisions where players occupy the same cell."""
        collisions = []
        for t in range(len(self.p1_loc)):
            p1 = self._loc_tuple(self.p1_loc[t])
            p2 = self._loc_tuple(self.p2_loc[t])
            if p1 == p2:
                # Determine who died
                who_died = None
                if t < len(self.p1_stamina) and self.p1_stamina[t] == -1:
                    who_died = "P1"
                elif t < len(self.p2_stamina) and self.p2_stamina[t] == -1:
                    who_died = "P2"

                collisions.append({
                    "turn": t,
                    "location": p1,
                    "who_died": who_died,
                    "parity_playing": self.parity_playing[t] if t < len(self.parity_playing) else None,
                })
        return collisions

    def detect_hill_captures(self) -> List[dict]:
        """Extract hill control changes from hill_updates."""
        captures = []
        for t, updates in enumerate(self.hill_updates):
            if not updates:
                continue
            for hill_id_str, new_controller in updates.items():
                hill_id = int(hill_id_str)
                controller_name = "P1" if new_controller == 1 else ("P2" if new_controller == -1 else "Neutral")
                captures.append({
                    "turn": t,
                    "hill_id": hill_id,
                    "new_controller": controller_name,
                    "new_controller_parity": new_controller,
                })
        return captures

    def detect_powerup_pickups(self) -> List[dict]:
        """Detect powerup collections from powerup_updates."""
        pickups = []
        for t, updates in enumerate(self.powerup_updates):
            if not updates:
                continue
            for cell_idx_str, val in updates.items():
                if val is False or val == 0:  # powerup consumed
                    cell = self._cell_from_index(int(cell_idx_str))
                    # Determine who picked it up by proximity
                    p1 = self._loc_tuple(self.p1_loc[t]) if t < len(self.p1_loc) else None
                    p2 = self._loc_tuple(self.p2_loc[t]) if t < len(self.p2_loc) else None
                    picker = None
                    if p1 and self._manhattan(p1, cell) <= 1:
                        picker = "P1"
                    elif p2 and self._manhattan(p2, cell) <= 1:
                        picker = "P2"
                    pickups.append({
                        "turn": t,
                        "location": cell,
                        "picked_by": picker,
                    })
        return pickups

    # ── Phase Analysis ───────────────────────────────────────────────

    def analyze_phases(self) -> List[dict]:
        """Break the game into phases based on territory momentum shifts."""
        phases = []
        if not self.p1_territory or not self.p2_territory:
            return phases

        window = max(10, self.num_turns // 20)
        phase_start = 0
        prev_leader = None

        for t in range(0, len(self.p1_territory), window):
            t1 = self.p1_territory[min(t, len(self.p1_territory) - 1)]
            t2 = self.p2_territory[min(t, len(self.p2_territory) - 1)]
            leader = "P1" if t1 > t2 else ("P2" if t2 > t1 else "Tied")

            if leader != prev_leader and prev_leader is not None:
                phases.append({
                    "start_turn": phase_start,
                    "end_turn": t,
                    "leader": prev_leader,
                    "p1_territory_start": self.p1_territory[phase_start],
                    "p1_territory_end": self.p1_territory[min(t, len(self.p1_territory) - 1)],
                    "p2_territory_start": self.p2_territory[phase_start],
                    "p2_territory_end": self.p2_territory[min(t, len(self.p2_territory) - 1)],
                })
                phase_start = t
            prev_leader = leader

        # Final phase
        if prev_leader is not None:
            phases.append({
                "start_turn": phase_start,
                "end_turn": len(self.p1_territory) - 1,
                "leader": prev_leader,
                "p1_territory_start": self.p1_territory[phase_start],
                "p1_territory_end": self.p1_territory[-1],
                "p2_territory_start": self.p2_territory[phase_start],
                "p2_territory_end": self.p2_territory[-1],
            })

        return phases

    # ── Action Analysis ──────────────────────────────────────────────

    def analyze_actions_per_turn(self) -> dict:
        """Analyze action patterns: moves per turn, paints per turn, move types."""
        stats = {
            "p1_moves_per_turn": [],
            "p2_moves_per_turn": [],
            "p1_paints_per_turn": [],
            "p2_paints_per_turn": [],
            "p1_total_moves": 0,
            "p2_total_moves": 0,
            "p1_total_paints": 0,
            "p2_total_paints": 0,
            "p1_erase_count": 0,
            "p2_erase_count": 0,
            "p1_beacon_travel_count": 0,
            "p2_beacon_travel_count": 0,
            "p1_beacon_place_count": 0,
            "p2_beacon_place_count": 0,
            "p1_turns_played": 0,
            "p2_turns_played": 0,
        }

        for t, action_list in enumerate(self.actions):
            if not action_list or action_list == "NONE":
                continue

            parity = self.parity_playing[t] if t < len(self.parity_playing) else 0
            prefix = "p1" if parity == 1 else "p2"
            stats[f"{prefix}_turns_played"] += 1

            moves = 0
            paints = 0
            if isinstance(action_list, list):
                for action in action_list:
                    if isinstance(action, dict):
                        name = action.get("name", "")
                        if name == "Move":
                            moves += 1
                            mt = action.get("move_type", "REGULAR")
                            if mt == "ERASE":
                                stats[f"{prefix}_erase_count"] += 1
                            elif mt == "BEACON_TRAVEL":
                                stats[f"{prefix}_beacon_travel_count"] += 1
                            if action.get("place_beacon", False):
                                stats[f"{prefix}_beacon_place_count"] += 1
                        elif name == "Paint":
                            paints += 1

            stats[f"{prefix}_moves_per_turn"].append(moves)
            stats[f"{prefix}_paints_per_turn"].append(paints)
            stats[f"{prefix}_total_moves"] += moves
            stats[f"{prefix}_total_paints"] += paints

        # Compute averages
        for p in ["p1", "p2"]:
            mpt = stats[f"{p}_moves_per_turn"]
            ppt = stats[f"{p}_paints_per_turn"]
            stats[f"{p}_avg_moves"] = sum(mpt) / len(mpt) if mpt else 0
            stats[f"{p}_avg_paints"] = sum(ppt) / len(ppt) if ppt else 0
            stats[f"{p}_max_moves"] = max(mpt) if mpt else 0
            # Distribution of move counts
            stats[f"{p}_move_distribution"] = {}
            for m in mpt:
                stats[f"{p}_move_distribution"][m] = stats[f"{p}_move_distribution"].get(m, 0) + 1

        return stats

    # ── Stamina Analysis ─────────────────────────────────────────────

    def analyze_stamina(self) -> dict:
        """Analyze stamina efficiency and patterns."""
        stats = {"p1": {}, "p2": {}}

        for prefix, stam_arr, max_stam_arr in [
            ("p1", self.p1_stamina, self.p1_max_stamina),
            ("p2", self.p2_stamina, self.p2_max_stamina),
        ]:
            valid = [s for s in stam_arr if s >= 0]
            stats[prefix]["avg_stamina"] = sum(valid) / len(valid) if valid else 0
            stats[prefix]["min_stamina"] = min(valid) if valid else 0
            stats[prefix]["max_stamina_reached"] = max(max_stam_arr) if max_stam_arr else BASE_MAX_STAMINA
            stats[prefix]["times_below_20"] = sum(1 for s in valid if s < 20)
            stats[prefix]["times_below_10"] = sum(1 for s in valid if s < 10)
            # Stamina at end
            stats[prefix]["final_stamina"] = stam_arr[-1] if stam_arr else 0

        return stats

    # ── Territory Analysis ───────────────────────────────────────────

    def analyze_territory(self) -> dict:
        """Analyze territory growth and control."""
        stats = {}
        if not self.p1_territory or not self.p2_territory:
            return stats

        stats["p1_max_territory"] = max(self.p1_territory)
        stats["p2_max_territory"] = max(self.p2_territory)
        stats["p1_final_territory"] = self.p1_territory[-1]
        stats["p2_final_territory"] = self.p2_territory[-1]

        # Territory at key moments (25%, 50%, 75%)
        for pct in [25, 50, 75]:
            idx = min(len(self.p1_territory) - 1, int(len(self.p1_territory) * pct / 100))
            stats[f"p1_territory_at_{pct}pct"] = self.p1_territory[idx]
            stats[f"p2_territory_at_{pct}pct"] = self.p2_territory[idx]

        # Territory growth rate (first 50 turns)
        early = min(50, len(self.p1_territory))
        if early > 1:
            stats["p1_early_growth_rate"] = round(self.p1_territory[early - 1] / early, 2)
            stats["p2_early_growth_rate"] = round(self.p2_territory[early - 1] / early, 2)

        # Painting efficiency: cells gained per paint action
        action_stats = self.analyze_actions_per_turn()
        for p in ["p1", "p2"]:
            total_paints = action_stats[f"{p}_total_paints"]
            max_terr = stats.get(f"{p}_max_territory", 0)
            stats[f"{p}_paint_efficiency"] = round(max_terr / total_paints, 2) if total_paints > 0 else 0

        return stats

    # ── Movement Pattern Analysis ────────────────────────────────────

    def analyze_movement_patterns(self) -> dict:
        """Analyze where players spend their time and how they move."""
        stats = {"p1": defaultdict(int), "p2": defaultdict(int)}

        # Heatmap: count turns spent in each cell
        for t, loc in enumerate(self.p1_loc):
            pos = self._loc_tuple(loc)
            stats["p1"][pos] += 1
        for t, loc in enumerate(self.p2_loc):
            pos = self._loc_tuple(loc)
            stats["p2"][pos] += 1

        # Top 10 most visited cells
        result = {}
        for p in ["p1", "p2"]:
            sorted_cells = sorted(stats[p].items(), key=lambda x: -x[1])[:10]
            result[f"{p}_top_cells"] = sorted_cells
            # Time on hills vs off hills
            on_hill = 0
            for pos, count in stats[p].items():
                if self.hill_mapping and pos[0] < len(self.hill_mapping) and pos[1] < len(self.hill_mapping[0]):
                    if self.hill_mapping[pos[0]][pos[1]] != 0:
                        on_hill += count
            total = sum(stats[p].values())
            result[f"{p}_pct_on_hills"] = round(100 * on_hill / total, 1) if total > 0 else 0

        # Player proximity over time
        close_turns = 0
        for t in range(min(len(self.p1_loc), len(self.p2_loc))):
            p1 = self._loc_tuple(self.p1_loc[t])
            p2 = self._loc_tuple(self.p2_loc[t])
            dist = self._manhattan(p1, p2)
            if dist <= 3:
                close_turns += 1
        result["turns_within_3_manhattan"] = close_turns
        result["pct_close_proximity"] = round(100 * close_turns / max(1, len(self.p1_loc)), 1)

        return result

    # ── Opening Analysis ─────────────────────────────────────────────

    def analyze_opening(self, depth: int = 30) -> dict:
        """Analyze the first N turns of the game."""
        n = min(depth, len(self.p1_loc))
        result = {
            "p1_bid": self.p1_bid,
            "p2_bid": self.p2_bid,
            "first_mover": "P1" if self.p1_bid > self.p2_bid else ("P2" if self.p2_bid > self.p1_bid else "Random"),
            "p1_opening_path": [self._loc_tuple(self.p1_loc[t]) for t in range(n)],
            "p2_opening_path": [self._loc_tuple(self.p2_loc[t]) for t in range(n)],
        }

        # Territory at turn 30
        if len(self.p1_territory) > n - 1:
            result["p1_territory_at_turn_30"] = self.p1_territory[n - 1]
            result["p2_territory_at_turn_30"] = self.p2_territory[n - 1]

        # First hill capture
        captures = self.detect_hill_captures()
        for c in captures:
            if "p1_first_hill_turn" not in result and c["new_controller"] == "P1":
                result["p1_first_hill_turn"] = c["turn"]
                result["p1_first_hill_id"] = c["hill_id"]
            if "p2_first_hill_turn" not in result and c["new_controller"] == "P2":
                result["p2_first_hill_turn"] = c["turn"]
                result["p2_first_hill_id"] = c["hill_id"]

        return result

    # ── Critical Moments ─────────────────────────────────────────────

    def detect_critical_moments(self) -> List[dict]:
        """Identify turning points: large territory swings, deaths, hill flips."""
        moments = []

        # Territory swings (>10 cell difference in 5 turns)
        for t in range(5, len(self.p1_territory)):
            p1_delta = self.p1_territory[t] - self.p1_territory[t - 5]
            p2_delta = self.p2_territory[t] - self.p2_territory[t - 5]
            swing = abs(p1_delta - p2_delta)
            if swing > 15:
                moments.append({
                    "turn": t,
                    "type": "territory_swing",
                    "p1_delta_5t": p1_delta,
                    "p2_delta_5t": p2_delta,
                    "swing": swing,
                    "favor": "P1" if p1_delta > p2_delta else "P2",
                })

        # Add collisions
        for c in self.detect_collisions():
            moments.append({
                "turn": c["turn"],
                "type": "collision",
                **c,
            })

        # Add hill captures
        for c in self.detect_hill_captures():
            moments.append({
                "turn": c["turn"],
                "type": "hill_capture",
                **c,
            })

        # Sort by turn
        moments.sort(key=lambda x: x["turn"])

        # Deduplicate nearby territory swings (keep only largest in 10-turn window)
        filtered = []
        last_swing_turn = -20
        for m in moments:
            if m["type"] == "territory_swing":
                if m["turn"] - last_swing_turn < 10:
                    if filtered and filtered[-1]["type"] == "territory_swing":
                        if m["swing"] > filtered[-1]["swing"]:
                            filtered[-1] = m
                        continue
                last_swing_turn = m["turn"]
            filtered.append(m)

        return filtered

    # ── Turn-by-Turn Reconstruction ──────────────────────────────────

    def reconstruct_board_at_turn(self, target_turn: int) -> Optional[List[List[int]]]:
        """Reconstruct the paint state of the board at a given turn."""
        if target_turn >= len(self.paint_updates):
            return None

        board = [[0] * self.board_cols for _ in range(self.board_rows)]

        for t in range(target_turn + 1):
            updates = self.paint_updates[t]
            if not updates:
                continue
            for cell_idx_str, paint_val in updates.items():
                r, c = self._cell_from_index(int(cell_idx_str))
                if 0 <= r < self.board_rows and 0 <= c < self.board_cols:
                    board[r][c] = paint_val

        return board

    # ── Hill Control Timeline ────────────────────────────────────────

    def get_hill_control_timeline(self) -> Dict[int, List[dict]]:
        """For each hill, produce a timeline of control changes."""
        timelines = defaultdict(list)
        for t, updates in enumerate(self.hill_updates):
            if not updates:
                continue
            for hill_id_str, parity in updates.items():
                hid = int(hill_id_str)
                controller = "P1" if parity == 1 else ("P2" if parity == -1 else "Neutral")
                timelines[hid].append({"turn": t, "controller": controller})
        return dict(timelines)

    # ── Full Analysis Report ─────────────────────────────────────────

    def generate_report(self, verbose: bool = False) -> str:
        """Generate a comprehensive text report for Claude Code consumption."""
        lines = []

        def section(title):
            lines.append(f"\n{'='*60}")
            lines.append(f"  {title}")
            lines.append(f"{'='*60}")

        def subsection(title):
            lines.append(f"\n--- {title} ---")

        # Header
        lines.append("BYTEFIGHT 2026: PAINT — GAME REPLAY ANALYSIS")
        lines.append(f"Map: {self.map_name} ({self.board_rows}x{self.board_cols})")
        lines.append(f"Walls: {self.wall_count}/{self.total_cells} ({self.wall_ratio*100:.1f}%)")
        lines.append(f"Passable cells: {self.passable_cells}")
        lines.append(f"Hills: {self.num_hills} (IDs: {list(self.hills.keys())})")
        for hid, cells in self.hills.items():
            lines.append(f"  Hill {hid}: {len(cells)} cells, center ~({sum(c[0] for c in cells)//len(cells)}, {sum(c[1] for c in cells)//len(cells)})")
        lines.append(f"Result: {self.result} by {self.reason} after {self.num_turns} turns")
        lines.append(f"Bids: P1={self.p1_bid}, P2={self.p2_bid}")

        # Opening
        section("OPENING (First 30 Turns)")
        opening = self.analyze_opening(30)
        lines.append(f"First mover: {opening['first_mover']}")
        lines.append(f"P1 opening path (first 10): {opening['p1_opening_path'][:10]}")
        lines.append(f"P2 opening path (first 10): {opening['p2_opening_path'][:10]}")
        if "p1_territory_at_turn_30" in opening:
            lines.append(f"Territory at turn 30: P1={opening['p1_territory_at_turn_30']}, P2={opening['p2_territory_at_turn_30']}")
        if "p1_first_hill_turn" in opening:
            lines.append(f"P1 first hill capture: hill {opening['p1_first_hill_id']} at turn {opening['p1_first_hill_turn']}")
        if "p2_first_hill_turn" in opening:
            lines.append(f"P2 first hill capture: hill {opening['p2_first_hill_id']} at turn {opening['p2_first_hill_turn']}")

        # Territory
        section("TERRITORY ANALYSIS")
        terr = self.analyze_territory()
        if terr:
            lines.append(f"P1 max territory: {terr.get('p1_max_territory', 0)}")
            lines.append(f"P2 max territory: {terr.get('p2_max_territory', 0)}")
            lines.append(f"P1 final territory: {terr.get('p1_final_territory', 0)}")
            lines.append(f"P2 final territory: {terr.get('p2_final_territory', 0)}")
            for pct in [25, 50, 75]:
                p1t = terr.get(f"p1_territory_at_{pct}pct", "?")
                p2t = terr.get(f"p2_territory_at_{pct}pct", "?")
                lines.append(f"  At {pct}% game: P1={p1t}, P2={p2t}")
            if "p1_early_growth_rate" in terr:
                lines.append(f"Early growth rate (first 50 turns): P1={terr['p1_early_growth_rate']}/turn, P2={terr['p2_early_growth_rate']}/turn")
            lines.append(f"Paint efficiency: P1={terr.get('p1_paint_efficiency', 0)} cells/paint, P2={terr.get('p2_paint_efficiency', 0)} cells/paint")

        # Actions
        section("ACTION ANALYSIS")
        act = self.analyze_actions_per_turn()
        for p, name in [("p1", "P1"), ("p2", "P2")]:
            subsection(f"{name} Actions")
            lines.append(f"Turns played: {act[f'{p}_turns_played']}")
            lines.append(f"Total moves: {act[f'{p}_total_moves']}, Total paints: {act[f'{p}_total_paints']}")
            lines.append(f"Avg moves/turn: {act[f'{p}_avg_moves']:.2f}, Avg paints/turn: {act[f'{p}_avg_paints']:.2f}")
            lines.append(f"Max moves in a turn: {act[f'{p}_max_moves']}")
            lines.append(f"Move count distribution: {dict(sorted(act[f'{p}_move_distribution'].items()))}")
            lines.append(f"Erases: {act[f'{p}_erase_count']}, Beacon travels: {act[f'{p}_beacon_travel_count']}, Beacon places: {act[f'{p}_beacon_place_count']}")

        # Stamina
        section("STAMINA ANALYSIS")
        stam = self.analyze_stamina()
        for p, name in [("p1", "P1"), ("p2", "P2")]:
            subsection(f"{name} Stamina")
            s = stam[p]
            lines.append(f"Avg stamina: {s['avg_stamina']:.1f}")
            lines.append(f"Min stamina: {s['min_stamina']}")
            lines.append(f"Max stamina reached: {s['max_stamina_reached']}")
            lines.append(f"Turns below 20 stamina: {s['times_below_20']}")
            lines.append(f"Turns below 10 stamina: {s['times_below_10']}")

        # Hill Control
        section("HILL CONTROL TIMELINE")
        hill_timeline = self.get_hill_control_timeline()
        for hid in sorted(hill_timeline.keys()):
            events = hill_timeline[hid]
            lines.append(f"\nHill {hid} ({len(self.hills.get(hid, []))} cells):")
            for evt in events:
                lines.append(f"  Turn {evt['turn']}: → {evt['controller']}")
            # Control duration
            if len(events) >= 2:
                for i in range(len(events) - 1):
                    dur = events[i + 1]["turn"] - events[i]["turn"]
                    lines.append(f"    {events[i]['controller']} held for {dur} turns")

        # Collisions
        section("COLLISIONS")
        collisions = self.detect_collisions()
        if collisions:
            lines.append(f"Total collisions: {len(collisions)}")
            for c in collisions:
                lines.append(f"  Turn {c['turn']}: at {c['location']}, {c['who_died']} died (acting: {'P1' if c['parity_playing']==1 else 'P2'})")
        else:
            lines.append("No collisions detected.")

        # Movement Patterns
        section("MOVEMENT PATTERNS")
        move = self.analyze_movement_patterns()
        for p, name in [("p1", "P1"), ("p2", "P2")]:
            subsection(f"{name} Movement")
            lines.append(f"% time on hills: {move[f'{p}_pct_on_hills']}%")
            lines.append(f"Top 5 cells: {move[f'{p}_top_cells'][:5]}")
        lines.append(f"\nProximity: {move['turns_within_3_manhattan']} turns within 3 Manhattan distance ({move['pct_close_proximity']}%)")

        # Critical Moments
        section("CRITICAL MOMENTS (Turning Points)")
        moments = self.detect_critical_moments()
        if moments:
            for m in moments[:30]:  # Cap at 30 to avoid overwhelming
                if m["type"] == "territory_swing":
                    lines.append(f"  Turn {m['turn']}: Territory swing favoring {m['favor']} (P1 Δ{m['p1_delta_5t']:+d}, P2 Δ{m['p2_delta_5t']:+d})")
                elif m["type"] == "collision":
                    lines.append(f"  Turn {m['turn']}: COLLISION at {m['location']} — {m['who_died']} died")
                elif m["type"] == "hill_capture":
                    lines.append(f"  Turn {m['turn']}: Hill {m['hill_id']} → {m['new_controller']}")
        else:
            lines.append("No significant moments detected.")

        # Game Phases
        section("GAME PHASES")
        phases = self.analyze_phases()
        for ph in phases:
            lines.append(f"  Turns {ph['start_turn']}-{ph['end_turn']}: {ph['leader']} leading")
            lines.append(f"    P1 territory: {ph['p1_territory_start']} → {ph['p1_territory_end']}")
            lines.append(f"    P2 territory: {ph['p2_territory_start']} → {ph['p2_territory_end']}")

        # Verbose: turn-by-turn action dump
        if verbose:
            section("TURN-BY-TURN ACTIONS (Verbose)")
            for t in range(min(len(self.actions), len(self.parity_playing))):
                act_list = self.actions[t]
                if not act_list or act_list == "NONE":
                    continue
                parity = self.parity_playing[t]
                player = "P1" if parity == 1 else "P2"
                p1_pos = self._loc_tuple(self.p1_loc[t]) if t < len(self.p1_loc) else "?"
                p2_pos = self._loc_tuple(self.p2_loc[t]) if t < len(self.p2_loc) else "?"
                p1_stam = self.p1_stamina[t] if t < len(self.p1_stamina) else "?"
                p2_stam = self.p2_stamina[t] if t < len(self.p2_stamina) else "?"
                lines.append(f"\nTurn {t} ({player}): P1@{p1_pos} stam={p1_stam} | P2@{p2_pos} stam={p2_stam}")
                if isinstance(act_list, list):
                    for a in act_list:
                        if isinstance(a, dict):
                            if a.get("name") == "Move":
                                lines.append(f"  → Move {a.get('direction', '?')} [{a.get('move_type', 'REGULAR')}]" +
                                              (f" +beacon" if a.get("place_beacon") else ""))
                            elif a.get("name") == "Paint":
                                lines.append(f"  → Paint {a.get('location', '?')}")

        # Summary for Claude Code
        section("SUMMARY FOR BOT DEVELOPMENT")
        winner = self.result
        loser = "P2" if "PLAYER_1" in winner else "P1"
        winner_name = "P1" if "PLAYER_1" in winner else "P2"

        lines.append(f"\nWinner: {winner_name} by {self.reason}")
        lines.append(f"Game length: {self.num_turns} turns ({self.num_turns // 2} rounds)")

        # Key takeaways
        lines.append("\nKEY OBSERVATIONS:")
        if collisions:
            p1_deaths = sum(1 for c in collisions if c["who_died"] == "P1")
            p2_deaths = sum(1 for c in collisions if c["who_died"] == "P2")
            lines.append(f"• Collision deaths: P1={p1_deaths}, P2={p2_deaths}")

        captures = self.detect_hill_captures()
        p1_captures = sum(1 for c in captures if c["new_controller"] == "P1")
        p2_captures = sum(1 for c in captures if c["new_controller"] == "P2")
        lines.append(f"• Hill captures: P1={p1_captures}, P2={p2_captures}")

        if terr:
            lines.append(f"• Territory dominance: {winner_name} peaked at {terr.get(f'{winner_name.lower()}_max_territory', '?')} cells")

        return "\n".join(lines)


def analyze_file(filepath: str, verbose: bool = False) -> str:
    """Analyze a single replay JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    analyzer = ReplayAnalyzer(data)
    return analyzer.generate_report(verbose=verbose)


def analyze_directory(dirpath: str, verbose: bool = False) -> str:
    """Analyze all replay JSONs in a directory."""
    results = []
    json_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".json")])

    for fname in json_files:
        filepath = os.path.join(dirpath, fname)
        try:
            report = analyze_file(filepath, verbose=verbose)
            results.append(f"\n{'#'*70}")
            results.append(f"# FILE: {fname}")
            results.append(f"{'#'*70}")
            results.append(report)

            # Write individual analysis
            out_path = filepath.replace(".json", ".analysis.txt")
            with open(out_path, "w") as f:
                f.write(report)
            print(f"✓ Analyzed {fname} → {os.path.basename(out_path)}")

        except Exception as e:
            results.append(f"\n# ERROR analyzing {fname}: {e}")
            print(f"✗ Error analyzing {fname}: {e}")

    return "\n".join(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 replay_analyzer.py <replay.json|directory> [--verbose]")
        sys.exit(1)

    target = sys.argv[1]
    verbose = "--verbose" in sys.argv

    if os.path.isdir(target):
        output = analyze_directory(target, verbose=verbose)
    elif os.path.isfile(target):
        output = analyze_file(target, verbose=verbose)
        out_path = target.replace(".json", ".analysis.txt")
        with open(out_path, "w") as f:
            f.write(output)
        print(f"✓ Analysis written to {out_path}")
    else:
        print(f"Error: {target} not found")
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()
