#!/usr/bin/env python3
"""Fetch HoopDirt D-I coaching-change tracker snapshots."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.hoopdirt import (  # noqa: E402
    DEFAULT_D1_TABLE_ID,
    HOOPDIRT_2026_TRACKER_URL,
    HoopDirtError,
    fetch_d1_coaching_changes,
    parse_ajax_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026, help="Ending year for the HoopDirt tracker.")
    parser.add_argument("--url", default=HOOPDIRT_2026_TRACKER_URL, help="HoopDirt tracker URL.")
    parser.add_argument(
        "--ajax-json-file",
        type=Path,
        help="Parse a saved HoopDirt AJAX JSON response instead of fetching live.",
    )
    parser.add_argument(
        "--table-id",
        default=DEFAULT_D1_TABLE_ID,
        help="Ninja Tables table id for the D-I coaching changes table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "hoopdirt" / "coaching_changes",
        help="Output directory.",
    )
    return parser.parse_args()


def output_stem(season: int, captured_at: object) -> str:
    if isinstance(captured_at, str) and captured_at:
        stamp = datetime.fromisoformat(captured_at).strftime("%Y-%m-%d")
    else:
        stamp = datetime.utcnow().strftime("%Y-%m-%d")
    return f"hoopdirt_d1_coaching_changes_{season}_{stamp}"


def main() -> int:
    args = parse_args()
    if args.ajax_json_file:
        rows, metadata = parse_ajax_file(args.ajax_json_file, season=args.season, source_url=args.url)
    else:
        rows, metadata = fetch_d1_coaching_changes(
            season=args.season,
            tracker_url=args.url,
            table_id=args.table_id,
        )

    stem = output_stem(args.season, metadata.get("captured_at"))
    json_path = write_json(rows, args.output_dir / f"{stem}.json")
    csv_path = write_csv(rows, args.output_dir / f"{stem}.csv")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"rows: {len(rows)}")
    print(f"source: {metadata.get('source_url')}")
    print(f"captured_at: {metadata.get('captured_at')}")
    if metadata.get("ajax_url"):
        print(f"ajax_url: {metadata.get('ajax_url')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HoopDirtError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
