#!/usr/bin/env python3
"""Fetch a historical range of KenPom season endpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fetch_kenpom_season import DEFAULT_ENDPOINTS, endpoint_request  # noqa: E402
from net_predictor.kenpom import KenPomAPIError, KenPomClient  # noqa: E402


DEFAULT_HISTORY_ENDPOINTS = ("teams", "ratings", "preseason-archive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, required=True, help="First ending season to fetch.")
    parser.add_argument("--end-season", type=int, required=True, help="Last ending season to fetch.")
    parser.add_argument(
        "--endpoint",
        choices=DEFAULT_ENDPOINTS,
        action="append",
        help="Endpoint to fetch. Repeat for multiple. Defaults to teams, ratings, and preseason archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
        help="Base output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.end_season < args.start_season:
        raise ValueError("--end-season must be greater than or equal to --start-season")

    endpoints = tuple(args.endpoint or DEFAULT_HISTORY_ENDPOINTS)
    client = KenPomClient()

    for season in range(args.start_season, args.end_season + 1):
        season_dir = args.output_dir / str(season)
        for endpoint in endpoints:
            api_endpoint, params, filename = endpoint_request(endpoint, season)
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
