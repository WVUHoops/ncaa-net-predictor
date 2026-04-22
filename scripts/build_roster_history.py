#!/usr/bin/env python3
"""Build roster-status history from all CBB Analytics player aggregate snapshots."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.coach_factor import canonical_team_key  # noqa: E402
from net_predictor.roster_status import (  # noqa: E402
    player_status_rows,
    read_json_rows,
    team_summary_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-agg-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "v1" / "player-agg-box",
    )
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
        default=PROJECT_ROOT / "data" / "processed" / "roster_status",
    )
    parser.add_argument("--min-season", type=int)
    parser.add_argument("--max-season", type=int)
    return parser.parse_args()


def competition_id_from_path(path: Path) -> str | None:
    match = re.search(r"_competition_(\d+)_", path.name)
    return match.group(1) if match else None


def latest_files_by_competition(paths: list[Path]) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for path in sorted(paths):
        competition_id = competition_id_from_path(path)
        if competition_id is None:
            continue
        latest[competition_id] = path
    return latest


def load_competition_seasons(path: Path) -> dict[str, int]:
    rows = read_json_rows(path)
    mapping: dict[str, int] = {}
    for row in rows:
        competition_id = row.get("competitionId")
        season = row.get("season") or row.get("endYear")
        if competition_id is None or season in (None, ""):
            continue
        mapping[str(competition_id)] = int(season)
    return mapping


def add_history_keys(rows: list[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    keyed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["season"] = season
        item["team_key"] = canonical_team_key(item.get("team_market"))
        keyed.append(item)
    return keyed


def main() -> int:
    args = parse_args()
    competition_seasons = load_competition_seasons(args.competitions_json)
    latest_files = latest_files_by_competition(sorted(args.player_agg_dir.glob("*.json")))

    all_player_statuses: list[dict[str, Any]] = []
    all_team_summaries: list[dict[str, Any]] = []
    skipped_empty: list[tuple[int, str, Path]] = []

    for competition_id, path in sorted(
        latest_files.items(),
        key=lambda item: competition_seasons.get(item[0], 0),
    ):
        season = competition_seasons.get(competition_id)
        if season is None:
            continue
        if args.min_season is not None and season < args.min_season:
            continue
        if args.max_season is not None and season > args.max_season:
            continue

        player_rows = read_json_rows(path)
        if not player_rows:
            skipped_empty.append((season, competition_id, path))
            continue

        statuses = add_history_keys(player_status_rows(player_rows), season)
        summaries = add_history_keys(team_summary_rows(statuses), season)
        all_player_statuses.extend(statuses)
        all_team_summaries.extend(summaries)

    all_player_statuses.sort(
        key=lambda row: (
            int(row.get("season") or 0),
            str(row.get("team_market") or ""),
            str(row.get("player_name") or ""),
        )
    )
    all_team_summaries.sort(
        key=lambda row: (
            int(row.get("season") or 0),
            str(row.get("team_market") or ""),
            str(row.get("team_name") or ""),
        )
    )

    player_json = write_json(all_player_statuses, args.output_dir / "player_roster_status_history.json")
    player_csv = write_csv(all_player_statuses, args.output_dir / "player_roster_status_history.csv")
    team_json = write_json(all_team_summaries, args.output_dir / "team_roster_summary_history.json")
    team_csv = write_csv(all_team_summaries, args.output_dir / "team_roster_summary_history.csv")

    print(f"saved {player_json}")
    print(f"saved {player_csv}")
    print(f"saved {team_json}")
    print(f"saved {team_csv}")
    print(f"player rows: {len(all_player_statuses)}")
    print(f"team rows: {len(all_team_summaries)}")
    if skipped_empty:
        skipped = ", ".join(f"{season}:{competition_id}" for season, competition_id, _ in skipped_empty)
        print(f"skipped empty competitions: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
