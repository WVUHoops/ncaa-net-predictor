#!/usr/bin/env python3
"""Build prior program-strength features from final KenPom seasons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.program_history import (  # noqa: E402
    load_kenpom_rating_rows,
    program_history_feature_rows,
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
        "--through-season",
        type=int,
        default=2027,
        help="Largest output season to build. Defaults to 2027 for the current 2026-27 board.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "program_history",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rating_rows = load_kenpom_rating_rows(args.kenpom_dir, args.min_season, args.max_season)
    rows = program_history_feature_rows(rating_rows, through_season=args.through_season)
    stem = f"program_history_features_{min(int(row['season']) for row in rows)}_{max(int(row['season']) for row in rows)}" if rows else "program_history_features_empty"
    json_path = write_json(rows, args.output_dir / f"{stem}.json")
    csv_path = write_csv(rows, args.output_dir / f"{stem}.csv")
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"program feature rows: {len(rows)}")
    print(f"rows with prior top-100 rate: {sum(1 for row in rows if row.get('prior_top100_rate') is not None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
