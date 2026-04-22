#!/usr/bin/env python3
"""Refresh portal-sensitive inputs, rerun models, and rebuild the dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], *, required: bool = True) -> bool:
    label = " ".join(command)
    print(f"\n==> {label}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode == 0:
        return True
    message = f"step failed with exit code {completed.returncode}: {label}"
    if required:
        raise RuntimeError(message)
    print(f"warning: {message}", file=sys.stderr)
    return False


def latest_current_player_agg() -> Path | None:
    directory = PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "v1" / "player-agg-box"
    paths = sorted(directory.glob("stats_player_agg_box_competition_41097_v1_*.json"), reverse=True)
    return paths[0] if paths else None


def can_rebuild_upset_risk() -> bool:
    return (PROJECT_ROOT / "data" / "raw" / "hoopr" / "mbb_schedule_master.csv").exists()


def existing_upset_risk_board() -> bool:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "upset_risk"
        / "current_2027_guarantee_risk_board.csv"
    ).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-roster-status", action="store_true")
    parser.add_argument("--skip-upset-risk", action="store_true")
    parser.add_argument("--on3-year", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_network:
        run_step(
            [
                "python3",
                "scripts/fetch_on3_rankings.py",
                "--source",
                "hs",
                "--year",
                str(args.on3_year),
            ],
            required=False,
        )
        run_step(
            [
                "python3",
                "scripts/fetch_on3_rankings.py",
                "--source",
                "transfer",
                "--year",
                str(args.on3_year),
            ],
            required=False,
        )

    run_step(["python3", "scripts/build_on3_features.py"])

    if not args.skip_roster_status:
        player_agg = latest_current_player_agg()
        if player_agg:
            run_step(
                [
                    "python3",
                    "scripts/build_roster_status.py",
                    "--player-agg-json",
                    str(player_agg),
                    "--season",
                    "2026",
                ]
            )
        else:
            print("warning: no current CBB player aggregate snapshot found; skipping roster status")

    run_step(["python3", "scripts/predict_current_season.py"])
    if args.skip_upset_risk:
        print("skipping upset-risk rebuild")
    elif can_rebuild_upset_risk():
        run_step(["python3", "scripts/build_upset_risk.py"])
    elif existing_upset_risk_board():
        print(
            "warning: hoopR schedule master is missing; reusing existing upset-risk board",
            file=sys.stderr,
        )
    else:
        run_step(["python3", "scripts/build_upset_risk.py"])
    run_step(["python3", "scripts/build_dashboard.py"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
