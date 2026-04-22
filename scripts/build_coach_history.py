#!/usr/bin/env python3
"""Build historical KenPom coach performance metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.coach_history import (  # noqa: E402
    coach_history_feature_rows,
    coach_latest_summary_rows,
    coach_season_rows,
    load_kenpom_preseason_rows,
    load_kenpom_rating_rows,
    load_kenpom_team_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kenpom-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
        help="Directory containing per-season KenPom raw files.",
    )
    parser.add_argument("--min-season", type=int, help="Smallest ending season to include.")
    parser.add_argument("--max-season", type=int, help="Largest ending season to include.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "coach_history",
        help="Output directory.",
    )
    return parser.parse_args()


def season_stem(min_season: int | None, max_season: int | None, observed_seasons: list[int]) -> str:
    if observed_seasons:
        return f"{min(observed_seasons)}_{max(observed_seasons)}"
    if min_season is not None or max_season is not None:
        return f"{min_season or 'first'}_{max_season or 'last'}"
    return "all"


def main() -> int:
    args = parse_args()
    team_rows = load_kenpom_team_rows(args.kenpom_dir, args.min_season, args.max_season)
    rating_rows = load_kenpom_rating_rows(args.kenpom_dir, args.min_season, args.max_season)
    preseason_rows = load_kenpom_preseason_rows(args.kenpom_dir, args.min_season, args.max_season)
    seasons = sorted({int(row["Season"]) for row in rating_rows if row.get("Season") is not None})

    coach_seasons = coach_season_rows(team_rows, rating_rows, preseason_rows)
    history_features = coach_history_feature_rows(coach_seasons)
    latest_summaries = coach_latest_summary_rows(coach_seasons)
    stem = season_stem(args.min_season, args.max_season, seasons)

    seasons_json = write_json(coach_seasons, args.output_dir / f"coach_seasons_{stem}.json")
    seasons_csv = write_csv(coach_seasons, args.output_dir / f"coach_seasons_{stem}.csv")
    features_json = write_json(history_features, args.output_dir / f"coach_history_features_{stem}.json")
    features_csv = write_csv(history_features, args.output_dir / f"coach_history_features_{stem}.csv")
    latest_json = write_json(latest_summaries, args.output_dir / f"coach_latest_summary_{stem}.json")
    latest_csv = write_csv(latest_summaries, args.output_dir / f"coach_latest_summary_{stem}.csv")

    over_expected_rows = sum(1 for row in coach_seasons if row.get("adj_em_over_expected") is not None)
    print(f"saved {seasons_json}")
    print(f"saved {seasons_csv}")
    print(f"saved {features_json}")
    print(f"saved {features_csv}")
    print(f"saved {latest_json}")
    print(f"saved {latest_csv}")
    print(f"seasons: {seasons[0] if seasons else None}-{seasons[-1] if seasons else None}")
    print(f"coach-season rows: {len(coach_seasons)}")
    print(f"coach-season rows with preseason over/under: {over_expected_rows}")
    print(f"coach feature rows: {len(history_features)}")
    print(f"coach latest summary rows: {len(latest_summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
