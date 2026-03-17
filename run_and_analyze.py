#!/usr/bin/env python3
"""
ByteFight 2026: Paint — Run games with replay recording + analysis.

Runs games between two bots, saves replay JSONs, and generates
analysis reports that Claude Code can consume.

Usage:
    python3 run_and_analyze.py <bot1> <bot2> [games_per_map] [--maps map1,map2,...] [--verbose]

Examples:
    python3 run_and_analyze.py xxxdominatorxxx bangv14 2
    python3 run_and_analyze.py xxxdominatorxxx bangv14 1 --maps spiral,matrix
    python3 run_and_analyze.py xxxdominatorxxx bangv14 2 --verbose

Output:
    replays/<bot1>_vs_<bot2>/<map>_game<N>.json     — raw replay
    replays/<bot1>_vs_<bot2>/<map>_game<N>.analysis.txt — analysis
    replays/<bot1>_vs_<bot2>/SUMMARY.txt            — aggregate summary
"""

import sys
import os
import json
import importlib
import importlib.util
import time
from datetime import datetime
from collections import defaultdict

# Add engine path
ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'client_terminal_version_v0.1.3')
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'player_files'))

from game import *
from game_runner.gen_board import get_board_from_map_string
from game_runner.game_controller import GameController, CustomEncoder
from replay_analyzer import ReplayAnalyzer


def load_bot(bot_name: str):
    """Load a bot controller from player_files/bots/<bot_name>/controller.py"""
    bots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'player_files', 'bots')
    bot_dir = os.path.join(bots_dir, bot_name)
    ctrl_path = os.path.join(bot_dir, 'controller.py')
    if not os.path.exists(ctrl_path):
        print(f"Error: Bot controller not found at {ctrl_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(f"{bot_name}_ctrl", ctrl_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlayerController


def run_game_with_replay(ctrl1_cls, ctrl2_cls, map_name: str, map_string: str,
                          p1_is_bot1: bool, bot1_name: str, bot2_name: str) -> dict:
    """
    Run a single game with full replay recording.
    Returns a dict with the replay data and metadata.
    """
    board = get_board_from_map_string(map_string)
    gc = GameController(board, time_limit=10000.0, record_history=True)

    if p1_is_bot1:
        ctrl1 = ctrl1_cls(1, lambda: 9999)
        ctrl2 = ctrl2_cls(-1, lambda: 9999)
        p1_name = bot1_name
        p2_name = bot2_name
    else:
        ctrl1 = ctrl2_cls(1, lambda: 9999)
        ctrl2 = ctrl1_cls(-1, lambda: 9999)
        p1_name = bot2_name
        p2_name = bot1_name

    # Bids
    bid1 = ctrl1.bid(gc.get_board_copy(), 1, lambda: 9999)
    bid2 = ctrl2.bid(gc.get_board_copy(), -1, lambda: 9999)
    gc.run_bid(bid1, bid2)

    start_time = time.time()
    turn_count = 0
    invalid_reason = None

    while not gc.is_game_over():
        parity = gc.board.parity_to_play
        board_copy = gc.get_board_copy()
        try:
            if parity == 1:
                actions = ctrl1.play(board_copy, 1, lambda: 9999)
            else:
                actions = ctrl2.play(board_copy, -1, lambda: 9999)
        except Exception as e:
            invalid_reason = f"{'P1' if parity == 1 else 'P2'} crashed: {e}"
            break

        if actions is None:
            actions = []

        success = gc.execute_turn(parity, actions, 0.001)
        turn_count += 1

        if not success:
            invalid_reason = f"{'P1' if parity == 1 else 'P2'} invalid turn at turn {turn_count}"
            break

        # Safety: prevent infinite games
        if turn_count > 2200:
            break

    elapsed = time.time() - start_time

    # Get result
    winner_result = gc.get_winner()
    if winner_result:
        result, reason = winner_result
    else:
        result = Result.TIE
        reason = WinReason.TIEBREAK

    # Build replay dict
    replay = gc.history.copy() if hasattr(gc, 'history') else {}
    replay["turn_count"] = gc.board.turn_count
    replay["result"] = result.name
    replay["reason"] = reason.name
    replay["map_string"] = map_name
    replay["elapsed_seconds"] = round(elapsed, 2)
    replay["p1_name"] = p1_name
    replay["p2_name"] = p2_name
    if invalid_reason:
        replay["invalid_reason"] = invalid_reason

    # Determine bot1 result
    if result == Result.PLAYER_1:
        bot1_result = 1 if p1_is_bot1 else -1
    elif result == Result.PLAYER_2:
        bot1_result = -1 if p1_is_bot1 else 1
    else:
        bot1_result = 0

    replay["_bot1_result"] = bot1_result
    replay["_p1_is_bot1"] = p1_is_bot1

    return replay


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    bot1_name = sys.argv[1]
    bot2_name = sys.argv[2]
    games_per_map = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('-') else 2
    verbose = "--verbose" in sys.argv

    # Parse --maps
    selected_maps = None
    for i, arg in enumerate(sys.argv):
        if arg == "--maps" and i + 1 < len(sys.argv):
            selected_maps = [m.strip() for m in sys.argv[i + 1].split(",")]

    # Load maps
    maps_path = os.path.join(ENGINE_DIR, 'config', 'maps.json')
    with open(maps_path) as f:
        all_maps = json.load(f)

    if selected_maps:
        maps = {k: v for k, v in all_maps.items() if k in selected_maps}
        if not maps:
            print(f"Error: No matching maps found. Available: {list(all_maps.keys())}")
            sys.exit(1)
    else:
        maps = all_maps

    # Load bots
    print(f"Loading {bot1_name}...")
    ctrl1_cls = load_bot(bot1_name)
    print(f"Loading {bot2_name}...")
    ctrl2_cls = load_bot(bot2_name)

    # Create output directory
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "replays", f"{bot1_name}_vs_{bot2_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Track results
    results_by_map = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "games": []})
    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_games = 0
    all_analyses = []

    print(f"\nRunning {bot1_name} vs {bot2_name}: {games_per_map} games/map on {len(maps)} maps")
    print("=" * 60)

    for map_name, map_string in maps.items():
        for i in range(games_per_map):
            p1_is_bot1 = (i % 2 == 0)
            side = "P1" if p1_is_bot1 else "P2"

            print(f"  {map_name} game {i+1}/{games_per_map} ({bot1_name} as {side})...", end=" ", flush=True)

            try:
                replay = run_game_with_replay(
                    ctrl1_cls, ctrl2_cls, map_name, map_string,
                    p1_is_bot1, bot1_name, bot2_name
                )
            except Exception as e:
                print(f"CRASH: {e}")
                continue

            bot1_result = replay.get("_bot1_result", 0)
            result_str = "WIN" if bot1_result == 1 else ("LOSS" if bot1_result == -1 else "TIE")
            print(f"{result_str} ({replay['reason']}, {replay['turn_count']}t, {replay['elapsed_seconds']}s)")

            # Save replay JSON
            fname = f"{map_name}_game{i+1}.json"
            replay_path = os.path.join(out_dir, fname)
            with open(replay_path, "w") as f:
                json.dump(replay, f, cls=CustomEncoder)

            # Analyze (reload from JSON to get serialized action dicts)
            try:
                with open(replay_path, "r") as f:
                    replay_serialized = json.load(f)
                analyzer = ReplayAnalyzer(replay_serialized)
                report = analyzer.generate_report(verbose=verbose)
                analysis_path = os.path.join(out_dir, f"{map_name}_game{i+1}.analysis.txt")
                with open(analysis_path, "w") as f:
                    f.write(report)
                all_analyses.append((fname, report, bot1_result, replay))
            except Exception as e:
                print(f"    Analysis error: {e}")

            # Track results
            if bot1_result == 1:
                results_by_map[map_name]["wins"] += 1
                total_wins += 1
            elif bot1_result == -1:
                results_by_map[map_name]["losses"] += 1
                total_losses += 1
            else:
                results_by_map[map_name]["ties"] += 1
                total_ties += 1
            total_games += 1
            results_by_map[map_name]["games"].append({
                "file": fname,
                "result": result_str,
                "reason": replay["reason"],
                "turns": replay["turn_count"],
                "side": side,
            })

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {bot1_name} vs {bot2_name}")
    print("=" * 60)

    summary_lines = []
    summary_lines.append(f"BYTEFIGHT BATCH ANALYSIS: {bot1_name} vs {bot2_name}")
    summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary_lines.append(f"Games per map: {games_per_map}")
    summary_lines.append("")

    for map_name in maps:
        r = results_by_map[map_name]
        total = r["wins"] + r["losses"] + r["ties"]
        pct = r["wins"] / total * 100 if total > 0 else 0
        line = f"  {map_name:15s}: {r['wins']}/{total} ({pct:.0f}%) W={r['wins']} L={r['losses']} T={r['ties']}"
        print(line)
        summary_lines.append(line)
        for g in r["games"]:
            detail = f"    {g['file']}: {g['result']} ({g['reason']}, {g['turns']}t, as {g['side']})"
            summary_lines.append(detail)

    overall_pct = total_wins / total_games * 100 if total_games > 0 else 0
    footer = f"\nOverall: {total_wins}/{total_games} ({overall_pct:.1f}%) wins"
    print(footer)
    summary_lines.append(footer)

    # Aggregate loss analysis
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("LOSS ANALYSIS (games where bot1 lost)")
    summary_lines.append("=" * 60)

    losses = [(fname, report, replay) for fname, report, result, replay in all_analyses if result == -1]
    if losses:
        for fname, report, replay in losses:
            summary_lines.append(f"\n--- {fname} ---")
            summary_lines.append(f"Map: {replay.get('map_string', '?')}")
            summary_lines.append(f"Result: {replay.get('result', '?')} by {replay.get('reason', '?')}")
            summary_lines.append(f"Turns: {replay.get('turn_count', '?')}")
            summary_lines.append(f"Bot1 was: {'P1' if replay.get('_p1_is_bot1') else 'P2'}")
            # Extract collision info from report
            for line in report.split("\n"):
                if "collision" in line.lower() or "COLLISION" in line:
                    summary_lines.append(f"  {line.strip()}")
                if "hill capture" in line.lower() or "Hill" in line and "→" in line:
                    summary_lines.append(f"  {line.strip()}")
            if replay.get("invalid_reason"):
                summary_lines.append(f"  INVALID: {replay['invalid_reason']}")
    else:
        summary_lines.append("No losses! 🎉")

    # Win analysis
    summary_lines.append("\n" + "=" * 60)
    summary_lines.append("WIN PATTERNS (common traits in wins)")
    summary_lines.append("=" * 60)

    wins = [(fname, report, replay) for fname, report, result, replay in all_analyses if result == 1]
    if wins:
        win_reasons = defaultdict(int)
        win_turn_counts = []
        for fname, report, replay in wins:
            win_reasons[replay.get("reason", "?")] += 1
            win_turn_counts.append(replay.get("turn_count", 0))

        summary_lines.append(f"Win reasons: {dict(win_reasons)}")
        if win_turn_counts:
            summary_lines.append(f"Avg win turn count: {sum(win_turn_counts)/len(win_turn_counts):.0f}")
            summary_lines.append(f"Fastest win: {min(win_turn_counts)} turns")
            summary_lines.append(f"Slowest win: {max(win_turn_counts)} turns")

    # Save summary
    summary_path = os.path.join(out_dir, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Replays saved to: {out_dir}/")


if __name__ == "__main__":
    main()
