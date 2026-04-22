#!/usr/bin/env python3
"""Fetch a year range of On3 team recruiting or transfer ranking snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fetch_on3_rankings import output_stem  # noqa: E402
from net_predictor.on3 import On3Error, fetch_rankings, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hs", "transfer"), required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--first-page-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "on3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    for year in range(args.start_year, args.end_year + 1):
        try:
            rows, metadata = fetch_rankings(args.source, year, all_pages=not args.first_page_only)
        except On3Error as exc:
            print(f"skipped {year}: {exc}", file=sys.stderr)
            continue
        output_dir = args.output_dir / args.source / str(year)
        stem = output_stem(args.source, year, metadata.get("captured_at"))
        json_path = write_json(rows, output_dir / f"{stem}.json")
        csv_path = write_csv(rows, output_dir / f"{stem}.csv")
        print(f"saved {json_path}")
        print(f"saved {csv_path}")
        print(f"year {year} rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except On3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
