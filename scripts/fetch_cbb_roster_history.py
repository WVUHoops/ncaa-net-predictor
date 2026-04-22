#!/usr/bin/env python3
"""Fetch historical CBB Analytics player aggregate rows for roster features."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.cbb_analytics import CBBAnalyticsClient, CBBAnalyticsError, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, required=True, help="First competition ending year.")
    parser.add_argument("--end-season", type=int, required=True, help="Last competition ending year.")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument(
        "--competitions-json",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "raw"
        / "cbb_analytics"
        / "v1"
        / "competitions"
        / "competitions_v1_2026-04-13.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "cbb_analytics",
    )
    return parser.parse_args()


def load_competition_ids(path: Path, start_season: int, end_season: int) -> dict[int, int]:
    import json

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")

    mapping: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("gender") != "MALE":
            continue
        season = int(row.get("season") or row.get("endYear") or 0)
        competition_id = row.get("competitionId")
        if start_season <= season <= end_season and competition_id is not None:
            mapping[season] = int(competition_id)
    return dict(sorted(mapping.items()))


def output_stem(competition_id: int, version: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"stats_player_agg_box_competition_{competition_id}_{version}_{stamp}"


def main() -> int:
    args = parse_args()
    if args.end_season < args.start_season:
        raise ValueError("--end-season must be greater than or equal to --start-season")

    competition_ids = load_competition_ids(args.competitions_json, args.start_season, args.end_season)
    if not competition_ids:
        raise ValueError("No men's basketball competition IDs found for the requested season range.")

    client = CBBAnalyticsClient(version=args.version)
    output_dir = args.output_dir / args.version / "player-agg-box"

    for season, competition_id in competition_ids.items():
        rows = client.get_all(
            "/stats/player/agg-box",
            competitionIds=str(competition_id),
            splits="season",
            limit=args.limit,
            max_pages=args.max_pages,
        )
        stem = output_stem(competition_id, args.version)
        json_path = write_json(rows, output_dir / f"{stem}.json")
        csv_path = write_csv(rows, output_dir / f"{stem}.csv")
        print(f"saved {json_path}")
        print(f"saved {csv_path}")
        print(f"season {season} competition {competition_id} rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CBBAnalyticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
