#!/usr/bin/env python3
"""Fetch or parse NCAA men's basketball NET rankings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.ncaa_net import (  # noqa: E402
    NCAA_NET_RANKINGS_URL,
    NCAANETError,
    fetch_ncaa_net_html,
    parse_ncaa_net_html,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=NCAA_NET_RANKINGS_URL,
        help="NCAA.com rankings URL.",
    )
    parser.add_argument(
        "--html-file",
        type=Path,
        help="Parse an already-downloaded HTML file instead of fetching the URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ncaa_net",
        help="Output directory.",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save the fetched HTML next to the parsed outputs.",
    )
    return parser.parse_args()


def output_stem(through_games: object) -> str:
    if isinstance(through_games, str) and through_games:
        return f"net_rankings_{through_games}"
    return "net_rankings_current"


def main() -> int:
    args = parse_args()

    if args.html_file:
        html = args.html_file.read_text(encoding="utf-8")
    else:
        html = fetch_ncaa_net_html(args.url)

    rows, metadata = parse_ncaa_net_html(html, source_url=args.url)
    stem = output_stem(metadata.get("through_games"))

    json_path = write_json(rows, args.output_dir / f"{stem}.json")
    csv_path = write_csv(rows, args.output_dir / f"{stem}.csv")

    if args.save_html and not args.html_file:
        html_path = args.output_dir / f"{stem}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"saved {html_path}")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"rows: {metadata['row_count']}")
    print(f"through games: {metadata.get('through_games_label') or 'unknown'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NCAANETError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
