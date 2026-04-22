#!/usr/bin/env python3
"""Build incoming-talent features from On3 recruiting and transfer snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.on3_features import build_on3_feature_rows, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--on3-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "on3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "on3_features",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.on3_dir.glob("*/*/*.json"))
    rows = build_on3_feature_rows(paths)

    json_path = write_json(rows, args.output_dir / "on3_incoming_talent_features.json")
    csv_path = write_csv(rows, args.output_dir / "on3_incoming_talent_features.csv")
    hs_rows = sum(1 for row in rows if row.get("on3_hs_rank") not in (None, ""))
    transfer_rows = sum(1 for row in rows if row.get("on3_transfer_rank") not in (None, ""))

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"feature rows: {len(rows)}")
    print(f"rows with HS rankings: {hs_rows}")
    print(f"rows with transfer rankings: {transfer_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
