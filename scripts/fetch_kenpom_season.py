#!/usr/bin/env python3
"""Fetch KenPom season data into local raw JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.kenpom import KenPomAPIError, KenPomClient  # noqa: E402


DEFAULT_ENDPOINTS = (
    "teams",
    "conferences",
    "conf-ratings",
    "ratings",
    "four-factors",
    "height",
    "misc-stats",
    "pointdist",
    "preseason-archive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="Ending year, e.g. 2025.")
    parser.add_argument(
        "--endpoint",
        choices=DEFAULT_ENDPOINTS,
        action="append",
        help="Endpoint to fetch. Repeat for multiple. Defaults to all core endpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
        help="Base output directory.",
    )
    return parser.parse_args()


def endpoint_request(endpoint: str, season: int) -> tuple[str, dict[str, object], str]:
    if endpoint == "preseason-archive":
        return "archive", {"y": season, "preseason": "true"}, "archive_preseason.json"
    return endpoint, {"y": season}, f"{endpoint.replace('-', '_')}.json"


def main() -> int:
    args = parse_args()
    endpoints = tuple(args.endpoint or DEFAULT_ENDPOINTS)
    season_dir = args.output_dir / str(args.season)
    client = KenPomClient()

    for endpoint in endpoints:
        api_endpoint, params, filename = endpoint_request(endpoint, args.season)
        output_path = season_dir / filename
        client.save_json(api_endpoint, output_path, **params)
        print(f"saved {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KenPomAPIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
