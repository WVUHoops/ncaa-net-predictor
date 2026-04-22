#!/usr/bin/env python3
"""Discover or parse official NCAA Statistics NET selection snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.ncaa_stats_net import (  # noqa: E402
    NCAA_STATS_NET_ARCHIVE_URL,
    NCAANETError,
    discover_selection_links,
    fetch_html,
    parse_selection_snapshot,
    save_selection_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-url",
        default=NCAA_STATS_NET_ARCHIVE_URL,
        help="NCAA Statistics archive index URL.",
    )
    parser.add_argument(
        "--index-html-file",
        type=Path,
        help="Parse a saved NCAA Statistics archive index instead of fetching it.",
    )
    parser.add_argument(
        "--list-selections",
        action="store_true",
        help="List discovered Selection Sunday snapshot links and exit.",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Ending year to fetch from discovered selections, e.g. 2025 for 2024-25.",
    )
    parser.add_argument(
        "--selection-url",
        help="Direct NCAA Statistics selection snapshot URL.",
    )
    parser.add_argument(
        "--selection-html-file",
        type=Path,
        help="Parse a saved NCAA Statistics selection snapshot instead of fetching it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ncaa_net_selections",
        help="Output directory.",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save fetched selection HTML next to parsed outputs.",
    )
    return parser.parse_args()


def read_index_html(args: argparse.Namespace) -> str:
    if args.index_html_file:
        return args.index_html_file.read_text(encoding="utf-8")
    return fetch_html(args.index_url)


def resolve_selection_url(args: argparse.Namespace) -> str | None:
    if args.selection_url:
        return args.selection_url
    if not args.season:
        return None

    links = discover_selection_links(read_index_html(args), base_url=args.index_url)
    for link in links:
        if link.season == args.season:
            return link.url

    available = ", ".join(str(link.season) for link in links) or "none"
    raise NCAANETError(f"No selections link found for season {args.season}. Available: {available}")


def snapshot_stem(metadata: dict[str, object]) -> str:
    season = metadata.get("season")
    thru_games = metadata.get("thru_games")
    if season and thru_games:
        return f"net_selections_{season}_{thru_games}"
    if season:
        return f"net_selections_{season}"
    return "net_selections"


def main() -> int:
    args = parse_args()

    if args.list_selections:
        links = discover_selection_links(read_index_html(args), base_url=args.index_url)
        for link in links:
            print(f"{link.season} {link.season_label} {link.thru_games} {link.url}")
        return 0

    source_url = resolve_selection_url(args)
    if args.selection_html_file:
        html = args.selection_html_file.read_text(encoding="utf-8")
    elif source_url:
        html = fetch_html(source_url)
    else:
        raise NCAANETError(
            "Provide --selection-html-file, --selection-url, or --season with an archive index."
        )

    rows, metadata = parse_selection_snapshot(html, source_url=source_url)
    stem = snapshot_stem(metadata)
    json_path, csv_path = save_selection_snapshot(rows, args.output_dir, stem)

    if args.save_html and source_url and not args.selection_html_file:
        html_path = args.output_dir / f"{stem}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        print(f"saved {html_path}")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"rows: {metadata['row_count']}")
    print(f"selection thru games: {metadata.get('thru_games') or 'unknown'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NCAANETError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
