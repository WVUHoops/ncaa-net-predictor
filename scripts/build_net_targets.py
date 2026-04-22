#!/usr/bin/env python3
"""Build NET target variants from NCAA NET ranking snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.net_targets import load_target_rows, write_csv, write_json  # noqa: E402


def default_paths(include_current: bool) -> list[Path]:
    patterns = ["data/raw/ncaa_net_selections/*.csv"]
    if include_current:
        patterns.extend(["data/raw/ncaa_net/*.csv"])

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(PROJECT_ROOT.glob(pattern))
    return sorted(path for path in paths if path.exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net-file",
        type=Path,
        action="append",
        default=[],
        help="Parsed NCAA NET ranking CSV/JSON. Repeat for multiple snapshots.",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Include current NCAA.com NET snapshots in addition to Selection Sunday targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "targets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.net_file or default_paths(args.include_current)
    if not paths:
        raise FileNotFoundError(
            "No NET snapshot files found. Run fetch_ncaa_net_selections.py first, "
            "or pass --include-current to use current NCAA.com snapshots."
        )

    rows = load_target_rows(paths)
    json_path = write_json(rows, args.output_dir / "net_targets.json")
    csv_path = write_csv(rows, args.output_dir / "net_targets.csv")
    seasons = sorted({row["season"] for row in rows if row.get("season") is not None})

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"input files: {len(paths)}")
    print(f"target rows: {len(rows)}")
    print(f"seasons: {seasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
