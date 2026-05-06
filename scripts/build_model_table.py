#!/usr/bin/env python3
"""Build the first season-level NET modeling table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.model_table import (  # noqa: E402
    build_model_rows,
    kenpom_preseason_rows,
    read_csv_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kenpom-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
    )
    parser.add_argument("--min-season", type=int)
    parser.add_argument("--max-season", type=int)
    parser.add_argument(
        "--coach-history-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "coach_history"
        / "coach_history_features_2016_2026.csv",
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "targets" / "net_targets.csv",
    )
    parser.add_argument(
        "--roster-summary-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "roster_status"
        / "team_roster_summary_history.csv",
    )
    parser.add_argument(
        "--roster-player-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "roster_status"
        / "player_roster_status_history.csv",
    )
    parser.add_argument(
        "--on3-features-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "on3_features"
        / "on3_incoming_talent_features.csv",
    )
    parser.add_argument(
        "--on3-hs-recruit-players-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "on3_features"
        / "on3_hs_recruit_players.csv",
    )
    parser.add_argument(
        "--transfer-features-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "transfer_features"
        / "cbb_incoming_transfer_features.csv",
    )
    parser.add_argument(
        "--transfer-player-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "transfer_features"
        / "cbb_incoming_transfer_players.csv",
    )
    parser.add_argument(
        "--program-history-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "program_history"
        / "program_history_features_2017_2027.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "modeling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preseason = kenpom_preseason_rows(args.kenpom_dir, args.min_season, args.max_season)
    coach = read_csv_rows(args.coach_history_csv) if args.coach_history_csv.exists() else []
    targets = read_csv_rows(args.targets_csv) if args.targets_csv.exists() else []
    roster = read_csv_rows(args.roster_summary_csv) if args.roster_summary_csv.exists() else []
    roster_players = read_csv_rows(args.roster_player_csv) if args.roster_player_csv.exists() else []
    on3 = read_csv_rows(args.on3_features_csv) if args.on3_features_csv.exists() else []
    on3_hs_recruit_players = (
        read_csv_rows(args.on3_hs_recruit_players_csv) if args.on3_hs_recruit_players_csv.exists() else []
    )
    transfers = read_csv_rows(args.transfer_features_csv) if args.transfer_features_csv.exists() else []
    transfer_players = read_csv_rows(args.transfer_player_csv) if args.transfer_player_csv.exists() else []
    program = read_csv_rows(args.program_history_csv) if args.program_history_csv.exists() else []
    rows = build_model_rows(
        preseason,
        coach,
        targets,
        roster,
        roster_players,
        on3,
        transfers,
        transfer_players,
        on3_hs_recruit_players,
        program,
    )

    json_path = write_json(rows, args.output_dir / "modeling_table.json")
    csv_path = write_csv(rows, args.output_dir / "modeling_table.csv")
    target_rows = sum(1 for row in rows if row.get("target_net_rank") not in (None, ""))
    roster_rows = sum(1 for row in rows if row.get("prior_roster_source_season") not in (None, ""))
    on3_hs_rows = sum(1 for row in rows if row.get("incoming_on3_hs_rank") not in (None, ""))
    on3_transfer_rows = sum(1 for row in rows if row.get("incoming_on3_transfer_rank") not in (None, ""))
    cbb_transfer_rows = sum(1 for row in rows if row.get("incoming_cbb_transfer_players") not in (None, ""))
    program_rows = sum(1 for row in rows if row.get("program_prior_seasons") not in (None, ""))

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"model rows: {len(rows)}")
    print(f"rows with NET targets: {target_rows}")
    print(f"rows with prior roster features: {roster_rows}")
    print(f"rows with On3 HS features: {on3_hs_rows}")
    print(f"rows with On3 transfer features: {on3_transfer_rows}")
    print(f"rows with CBB incoming transfer features: {cbb_transfer_rows}")
    print(f"rows with program history features: {program_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
