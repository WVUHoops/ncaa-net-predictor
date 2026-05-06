#!/usr/bin/env python3
"""Refresh portal-sensitive inputs, rerun models, and rebuild the dashboard."""

from __future__ import annotations

import argparse
import csv
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


def current_player_roster_status_csv() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "roster_status"
        / "player_roster_status_2026.csv"
    )


def has_existing_current_roster_status() -> bool:
    return current_player_roster_status_csv().exists()


def latest_transfer_ledger_csv() -> Path | None:
    directories = [
        PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "transfer_portal" / "current",
        PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "transfer_portal",
    ]
    for directory in directories:
        if not directory.exists():
            continue
        paths = sorted(directory.glob("*.csv"), reverse=True)
        if paths:
            return paths[0]
    return None


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


def upset_risk_board_has_coach_history() -> bool:
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "upset_risk"
        / "current_2027_guarantee_risk_board.csv"
    )
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return False
    return any(
        row.get("away_coach_road_hm_upset_rate") not in (None, "")
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-roster-status", action="store_true")
    parser.add_argument("--skip-upset-risk", action="store_true")
    parser.add_argument("--on3-year", type=int, default=2026)
    parser.add_argument(
        "--current-cbb-competition-id",
        type=int,
        default=41097,
        help="CBB Analytics competition ID for the current men's D-I season player aggregate snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_network:
        run_step(
            [
                "python3",
                "scripts/fetch_cbb_transfer_portal_csv.py",
            ],
            required=False,
        )
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
        run_step(
            [
                "python3",
                "scripts/fetch_on3_hs_commit_players.py",
                "--year",
                str(args.on3_year),
            ],
            required=False,
        )
        run_step(
            [
                "python3",
                "scripts/fetch_hoopdirt_coaching_changes.py",
                "--season",
                "2026",
            ],
            required=False,
        )

    run_step(["python3", "scripts/build_on3_features.py"])
    run_step(["python3", "scripts/build_coach_features.py", "--season", "2026"])

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
        elif has_existing_current_roster_status():
            print(
                "warning: no current CBB player aggregate snapshot found; reusing existing roster-status snapshot",
                file=sys.stderr,
            )
        else:
            print(
                "warning: no current CBB player aggregate snapshot found and no existing roster-status snapshot is available",
                file=sys.stderr,
            )

        if has_existing_current_roster_status():
            transfer_ledger_csv = latest_transfer_ledger_csv()
            transfer_command = [
                "python3",
                "scripts/build_transfer_features.py",
                "--player-roster-status-csv",
                str(current_player_roster_status_csv()),
                "--source-season",
                "2026",
                "--output-dir",
                str(PROJECT_ROOT / "data" / "processed" / "transfer_features" / "current"),
            ]
            if transfer_ledger_csv:
                transfer_command.extend(
                    [
                        "--transfer-ledger-csv",
                        str(transfer_ledger_csv),
                        "--destination-season",
                        "2027",
                    ]
                )
            else:
                print(
                    "warning: no transfer ledger CSV found; falling back to roster-status transfer detection",
                    file=sys.stderr,
                )
                transfer_command.append("--current-roster-transfers")
            run_step(transfer_command)

    run_step(["python3", "scripts/predict_current_season.py"])
    if args.skip_upset_risk:
        print("skipping upset-risk rebuild")
    elif can_rebuild_upset_risk():
        run_step(["python3", "scripts/build_upset_risk.py"])
    elif existing_upset_risk_board():
        if not upset_risk_board_has_coach_history():
            raise RuntimeError(
                "existing upset-risk board is missing coach road-HM history; "
                "cannot safely deploy dashboard"
            )
        print(
            "warning: hoopR schedule master is missing; reusing existing upset-risk board",
            file=sys.stderr,
        )
    else:
        run_step(["python3", "scripts/build_upset_risk.py"])
    if not upset_risk_board_has_coach_history():
        raise RuntimeError("upset-risk board has no coach road-HM history")
    run_step(["python3", "scripts/build_dashboard.py"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
