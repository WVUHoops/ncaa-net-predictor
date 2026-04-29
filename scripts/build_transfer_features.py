#!/usr/bin/env python3
"""Build incoming-transfer production features from CBB Analytics and KenPom context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.transfer_features import (  # noqa: E402
    kenpom_context_by_team,
    player_transfer_rows,
    read_csv_rows,
    supplement_player_rows_with_roster_rows,
    transfer_rows_from_ledger,
    transfer_summary_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-roster-status-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "roster_status"
        / "player_roster_status_history.csv",
    )
    parser.add_argument(
        "--kenpom-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "transfer_features",
    )
    parser.add_argument(
        "--source-season",
        type=int,
        help="Use this source season when the roster-status CSV does not include a season column.",
    )
    parser.add_argument(
        "--current-roster-transfers",
        action="store_true",
        help="Treat is_transfer rows on current rosters as incoming transfers from prior_team_market.",
    )
    parser.add_argument(
        "--transfer-ledger-csv",
        type=Path,
        help="Optional downloaded transfer ledger CSV to use for source->destination movement.",
    )
    parser.add_argument(
        "--destination-season",
        type=int,
        help="Destination season for transfer-ledger rows. Defaults to source season + 1.",
    )
    parser.add_argument(
        "--competition-team-players-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "raw"
        / "cbb_analytics"
        / "v1"
        / "competition-team-players"
        / "competition_team_players_competition_41097_v1_2026-04-13.csv",
        help="Optional current-team roster file used to backfill zero-minute D-I players missing from player-agg.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player_rows = read_csv_rows(args.player_roster_status_csv)
    if args.source_season is not None:
        for row in player_rows:
            row.setdefault("season", str(args.source_season))
    context = kenpom_context_by_team(args.kenpom_dir)
    unmatched: list[dict[str, object]] = []
    if args.transfer_ledger_csv and args.transfer_ledger_csv.exists():
        roster_rows = (
            read_csv_rows(args.competition_team_players_csv)
            if args.competition_team_players_csv and args.competition_team_players_csv.exists()
            else []
        )
        roster_source_season = args.source_season
        if roster_source_season is None and args.destination_season is not None:
            roster_source_season = args.destination_season - 1
        player_rows = supplement_player_rows_with_roster_rows(
            player_rows,
            roster_rows,
            season=roster_source_season,
        )
        ledger_rows = read_csv_rows(args.transfer_ledger_csv)
        transfers, unmatched = transfer_rows_from_ledger(
            ledger_rows,
            player_rows,
            context,
            destination_season=args.destination_season,
        )
    else:
        transfers = player_transfer_rows(
            player_rows,
            context,
            current_roster_transfers=args.current_roster_transfers,
        )
    summaries = transfer_summary_rows(transfers)

    player_json = write_json(transfers, args.output_dir / "cbb_incoming_transfer_players.json")
    player_csv = write_csv(transfers, args.output_dir / "cbb_incoming_transfer_players.csv")
    summary_json = write_json(summaries, args.output_dir / "cbb_incoming_transfer_features.json")
    summary_csv = write_csv(summaries, args.output_dir / "cbb_incoming_transfer_features.csv")
    if args.transfer_ledger_csv and args.transfer_ledger_csv.exists():
        unmatched_json = write_json(unmatched, args.output_dir / "cbb_transfer_ledger_unmatched.json")
        unmatched_csv = write_csv(unmatched, args.output_dir / "cbb_transfer_ledger_unmatched.csv")
        print(f"saved {unmatched_json}")
        print(f"saved {unmatched_csv}")
        print(f"unmatched transfer ledger rows: {len(unmatched)}")

    print(f"saved {player_json}")
    print(f"saved {player_csv}")
    print(f"saved {summary_json}")
    print(f"saved {summary_csv}")
    print(f"incoming transfer player rows: {len(transfers)}")
    print(f"incoming transfer team-season rows: {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
