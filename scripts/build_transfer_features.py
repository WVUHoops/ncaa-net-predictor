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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player_rows = read_csv_rows(args.player_roster_status_csv)
    if args.source_season is not None:
        for row in player_rows:
            row.setdefault("season", str(args.source_season))
    context = kenpom_context_by_team(args.kenpom_dir)
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

    print(f"saved {player_json}")
    print(f"saved {player_csv}")
    print(f"saved {summary_json}")
    print(f"saved {summary_csv}")
    print(f"incoming transfer player rows: {len(transfers)}")
    print(f"incoming transfer team-season rows: {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
