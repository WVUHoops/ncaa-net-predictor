#!/usr/bin/env python3
"""Fetch or parse On3 team recruiting/transfer rankings as dated snapshots."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.on3 import (  # noqa: E402
    On3Error,
    fetch_rankings,
    parse_html_file,
    source_url,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("hs", "transfer"),
        required=True,
        help="On3 source: hs for recruiting class rankings, transfer for portal class rankings.",
    )
    parser.add_argument("--year", type=int, default=2026, help="Ranking year.")
    parser.add_argument(
        "--html-file",
        type=Path,
        help="Parse a saved On3 HTML file instead of fetching from the web.",
    )
    parser.add_argument(
        "--url",
        help="Override the source URL. Useful when parsing a saved HTML file.",
    )
    parser.add_argument(
        "--first-page-only",
        action="store_true",
        help="Fetch only the first On3 page instead of following pagination.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "on3",
        help="Base output directory.",
    )
    return parser.parse_args()


def output_stem(source: str, year: int, captured_at: object) -> str:
    if isinstance(captured_at, str) and captured_at:
        stamp = datetime.fromisoformat(captured_at).strftime("%Y-%m-%d")
    else:
        stamp = datetime.utcnow().strftime("%Y-%m-%d")
    return f"on3_{source}_{year}_{stamp}"


def main() -> int:
    args = parse_args()
    url = args.url or source_url(args.source, args.year)

    if args.html_file:
        rows, metadata = parse_html_file(args.source, args.html_file, url=url)
    else:
        rows, metadata = fetch_rankings(
            args.source,
            args.year,
            all_pages=not args.first_page_only,
        )

    output_dir = args.output_dir / args.source / str(args.year)
    stem = output_stem(args.source, args.year, metadata.get("captured_at"))
    json_path = write_json(rows, output_dir / f"{stem}.json")
    csv_path = write_csv(rows, output_dir / f"{stem}.csv")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"rows: {len(rows)}")
    print(f"source: {url}")
    print(f"captured_at: {metadata.get('captured_at')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except On3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
