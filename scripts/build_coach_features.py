#!/usr/bin/env python3
"""Build coach-change features by joining HoopDirt changes to KenPom teams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.coach_factor import (  # noqa: E402
    coach_feature_rows,
    read_csv_rows,
    read_json_rows,
    write_csv,
    write_json,
)


DEFAULT_KENPOM_TEAMS = (
    PROJECT_ROOT / "data" / "raw" / "kenpom" / "2025" / "teams.json",
    PROJECT_ROOT / "data" / "raw" / "kenpom" / "2026" / "teams.json",
)
DEFAULT_HOOPDIRT_CHANGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hoopdirt"
    / "coaching_changes"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kenpom-teams-json",
        type=Path,
        action="append",
        default=[],
        help="KenPom teams JSON file. Repeat for multiple seasons to estimate tenure/history.",
    )
    parser.add_argument(
        "--coaching-changes-csv",
        type=Path,
        help="HoopDirt D-I coaching changes CSV. Defaults to the latest dated snapshot.",
    )
    parser.add_argument(
        "--coaching-changes-dir",
        type=Path,
        default=DEFAULT_HOOPDIRT_CHANGES_DIR,
        help="Directory containing dated HoopDirt coaching changes CSV snapshots.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Ending year for output names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "coach_features",
        help="Output directory.",
    )
    return parser.parse_args()


def latest_coaching_changes_csv(directory: Path, season: int) -> Path:
    matches = sorted(directory.glob(f"hoopdirt_d1_coaching_changes_{season}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No HoopDirt coaching changes CSV found in {directory}. "
            "Run fetch_hoopdirt_coaching_changes.py first."
        )
    return matches[-1]


def main() -> int:
    args = parse_args()
    kenpom_paths = args.kenpom_teams_json or [path for path in DEFAULT_KENPOM_TEAMS if path.exists()]
    if not kenpom_paths:
        raise FileNotFoundError("No KenPom teams JSON files found. Run fetch_kenpom_season.py first.")

    kenpom_rows = []
    for path in kenpom_paths:
        kenpom_rows.extend(read_json_rows(path))
    coaching_changes_csv = args.coaching_changes_csv or latest_coaching_changes_csv(args.coaching_changes_dir, args.season)
    change_rows = read_csv_rows(coaching_changes_csv)

    features, unmatched = coach_feature_rows(kenpom_rows, change_rows)
    feature_stem = f"coach_features_{args.season}"
    unmatched_stem = f"coach_change_unmatched_{args.season}"

    feature_json = write_json(features, args.output_dir / f"{feature_stem}.json")
    feature_csv = write_csv(features, args.output_dir / f"{feature_stem}.csv")
    unmatched_json = write_json(unmatched, args.output_dir / f"{unmatched_stem}.json")
    unmatched_csv = write_csv(unmatched, args.output_dir / f"{unmatched_stem}.csv")

    changed = sum(1 for row in features if row["coach_changed"])
    old_coach_matched = sum(1 for row in features if row["former_coach_matches_kenpom"])
    print(f"saved {feature_json}")
    print(f"saved {feature_csv}")
    print(f"saved {unmatched_json}")
    print(f"saved {unmatched_csv}")
    print(f"team rows: {len(features)}")
    print(f"coach changes matched to KenPom teams: {changed}")
    print(f"former coaches matched to KenPom coach field: {old_coach_matched}")
    print(f"unmatched HoopDirt rows: {len(unmatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
