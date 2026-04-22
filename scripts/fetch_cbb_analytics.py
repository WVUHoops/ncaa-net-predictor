#!/usr/bin/env python3
"""Fetch CBB Analytics API endpoints into raw JSON/CSV files."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.cbb_analytics import (  # noqa: E402
    CBBAnalyticsClient,
    CBBAnalyticsError,
    compact_endpoint_name,
    write_csv,
    write_json,
)


ENDPOINTS = {
    "competitions": "/competitions",
    "conferences": "/conferences",
    "teams": "/teams",
    "competition-teams": "/competition-teams",
    "players": "/players",
    "competition-team-players": "/competition-team-players",
    "team-agg-box": "/stats/team/agg-box",
    "player-agg-box": "/stats/player/agg-box",
    "team-game-box": "/stats/team/game-box",
    "player-game-box": "/stats/player/game-box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--competition-ids", help="Comma-separated competition IDs.")
    parser.add_argument("--conference-ids", help="Comma-separated conference IDs.")
    parser.add_argument("--team-ids", help="Comma-separated team IDs.")
    parser.add_argument("--player-ids", help="Comma-separated player IDs.")
    parser.add_argument("--game-ids", help="Comma-separated game IDs.")
    parser.add_argument("--division-ids", help="Comma-separated division IDs.")
    parser.add_argument("--gender", choices=("all", "male", "female"), help="Player/team gender filter.")
    parser.add_argument(
        "--splits",
        default="season",
        help="Comma-separated stat splits for aggregate endpoints, e.g. season,confReg.",
    )
    parser.add_argument(
        "--team-or-opponent",
        choices=("all", "team", "opponent"),
        help="Stats side for team endpoints.",
    )
    parser.add_argument(
        "--location",
        choices=("all", "neutral", "home", "away"),
        help="Location filter for game stats.",
    )
    parser.add_argument(
        "--exhibition",
        choices=("all", "exhibitionOnly", "nonExhibitionOnly"),
        help="Exhibition-game filter.",
    )
    parser.add_argument("--in-division", choices=("all", "true", "false"), help="D-I only filter.")
    parser.add_argument("--updated", help="Only rows updated on/after this RFC3339 time or date.")
    parser.add_argument("--sort-by", help="Sort field.")
    parser.add_argument("--sort-order", choices=("asc", "desc", "1", "-1"), help="Sort direction.")
    parser.add_argument("--limit", type=int, default=1000, help="Page size, max 1000.")
    parser.add_argument("--max-pages", type=int, help="Optional safety limit for pagination.")
    parser.add_argument(
        "--first-page-only",
        action="store_true",
        help="Fetch only the first page instead of paginating until exhausted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "cbb_analytics",
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> dict[str, object]:
    params = {
        "competitionIds": args.competition_ids,
        "conferenceIds": args.conference_ids,
        "teamIds": args.team_ids,
        "playerIds": args.player_ids,
        "gameIds": args.game_ids,
        "divisionIds": args.division_ids,
        "gender": args.gender,
        "splits": args.splits,
        "teamOrOpponent": args.team_or_opponent,
        "location": args.location,
        "exhibition": args.exhibition,
        "inDivision": args.in_division,
        "updated": args.updated,
        "sortBy": args.sort_by,
        "sortOrder": args.sort_order,
    }

    aggregate_endpoint = args.endpoint in {"team-agg-box", "player-agg-box"}
    if not aggregate_endpoint:
        params.pop("splits")

    return params


def output_stem(args: argparse.Namespace) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    endpoint_name = compact_endpoint_name(ENDPOINTS[args.endpoint])
    parts = [endpoint_name, args.version, stamp]
    if args.competition_ids:
        parts.insert(1, f"competition_{args.competition_ids.replace(',', '-')}")
    return "_".join(parts)


def main() -> int:
    args = parse_args()
    client = CBBAnalyticsClient(version=args.version)
    endpoint = ENDPOINTS[args.endpoint]
    params = build_params(args)

    if args.first_page_only:
        rows = client.get_all(endpoint, limit=args.limit, max_pages=1, **params)
    else:
        rows = client.get_all(endpoint, limit=args.limit, max_pages=args.max_pages, **params)

    output_dir = args.output_dir / args.version / args.endpoint
    stem = output_stem(args)
    json_path = write_json(rows, output_dir / f"{stem}.json")
    csv_path = write_csv(rows, output_dir / f"{stem}.csv")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CBBAnalyticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
