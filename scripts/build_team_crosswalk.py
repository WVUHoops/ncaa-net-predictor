#!/usr/bin/env python3
"""Build a reviewable team crosswalk centered on KenPom team IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.team_crosswalk import (  # noqa: E402
    build_crosswalk_rows,
    load_cbb_teams,
    load_hoopdirt_teams,
    load_kenpom_teams,
    load_net_teams,
    load_on3_teams,
    write_csv,
    write_json,
)


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def glob_existing(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(PROJECT_ROOT.glob(pattern))
    return sorted(path for path in paths if path.exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kenpom-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
        help="Directory containing per-season KenPom teams.json files.",
    )
    parser.add_argument("--max-kenpom-season", type=int, help="Latest KenPom season to use.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "team_crosswalk",
    )
    parser.add_argument("--review-threshold", type=float, default=0.94)
    return parser.parse_args()


def default_source_paths() -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    cbb_paths = glob_existing(
        [
            "data/processed/roster_status/team_roster_summary_*.csv",
            "data/raw/cbb_analytics/v1/player-agg-box/*.csv",
        ]
    )
    net_paths = glob_existing(
        [
            "data/raw/ncaa_net/*.csv",
            "data/raw/ncaa_net_selections/*.csv",
        ]
    )
    on3_paths = glob_existing(["data/raw/on3/*/*/*.csv"])
    hoopdirt_paths = glob_existing(["data/raw/hoopdirt/coaching_changes/*.csv"])
    return cbb_paths, net_paths, on3_paths, hoopdirt_paths


def main() -> int:
    args = parse_args()
    cbb_paths, net_paths, on3_paths, hoopdirt_paths = default_source_paths()
    kenpom_rows = load_kenpom_teams(args.kenpom_dir, max_season=args.max_kenpom_season)
    source_rows = [
        *load_cbb_teams(cbb_paths),
        *load_net_teams(net_paths),
        *load_on3_teams(on3_paths),
        *load_hoopdirt_teams(hoopdirt_paths),
    ]
    rows = build_crosswalk_rows(kenpom_rows, source_rows, review_threshold=args.review_threshold)
    review_rows = [row for row in rows if row["needs_review"]]

    json_path = write_json(rows, args.output_dir / "team_crosswalk_candidates.json")
    csv_path = write_csv(rows, args.output_dir / "team_crosswalk_candidates.csv")
    review_json_path = write_json(review_rows, args.output_dir / "team_crosswalk_review.json")
    review_csv_path = write_csv(review_rows, args.output_dir / "team_crosswalk_review.csv")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {review_json_path}")
    print(f"saved {review_csv_path}")
    print(f"kenpom rows: {len(kenpom_rows)}")
    print(f"source team rows: {len(source_rows)}")
    print(f"crosswalk rows: {len(rows)}")
    print(f"review rows: {len(review_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
