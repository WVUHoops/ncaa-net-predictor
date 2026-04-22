#!/usr/bin/env python3
"""Build player roster-status rows and team continuity summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.roster_status import (  # noqa: E402
    player_status_rows,
    read_json_rows,
    team_summary_rows,
    write_csv,
    write_json,
)


DEFAULT_PLAYER_AGG = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cbb_analytics"
    / "v1"
    / "player-agg-box"
    / "stats_player_agg_box_competition_41097_v1_2026-04-13.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-agg-json",
        type=Path,
        default=DEFAULT_PLAYER_AGG,
        help="CBB Analytics player aggregate JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "roster_status",
    )
    parser.add_argument("--season", type=int, default=2026, help="Ending year for output names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player_rows = read_json_rows(args.player_agg_json)
    statuses = player_status_rows(player_rows)
    summaries = team_summary_rows(statuses)

    player_stem = f"player_roster_status_{args.season}"
    team_stem = f"team_roster_summary_{args.season}"

    player_json = write_json(statuses, args.output_dir / f"{player_stem}.json")
    player_csv = write_csv(statuses, args.output_dir / f"{player_stem}.csv")
    team_json = write_json(summaries, args.output_dir / f"{team_stem}.json")
    team_csv = write_csv(summaries, args.output_dir / f"{team_stem}.csv")

    print(f"saved {player_json}")
    print(f"saved {player_csv}")
    print(f"saved {team_json}")
    print(f"saved {team_csv}")
    print(f"player rows: {len(statuses)}")
    print(f"team rows: {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
