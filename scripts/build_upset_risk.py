#!/usr/bin/env python3
"""Build and score a high-major guarantee-game upset risk model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.upset_risk import (  # noqa: E402
    LogisticRiskModel,
    build_low_major_road_high_major_rows,
    build_training_rows,
    current_risk_board,
    rolling_backtest,
    summarize_training,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "hoopr" / "mbb_schedule_master.csv",
        help="SportsDataverse/hoopR MBB schedule master CSV.",
    )
    parser.add_argument(
        "--kenpom-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom",
    )
    parser.add_argument(
        "--current-predictions-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "predictions"
        / "current_2027_schedule_predictions.csv",
    )
    parser.add_argument(
        "--coach-history-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "coach_history"
        / "coach_history_features_2016_2026.csv",
    )
    parser.add_argument(
        "--coach-latest-summary-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "coach_history"
        / "coach_latest_summary_2016_2026.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "upset_risk",
    )
    parser.add_argument(
        "--skip-current-board",
        action="store_true",
        help="Only build training/backtest outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coach_road_hm_rows = build_low_major_road_high_major_rows(args.schedule_csv, args.kenpom_dir)
    training_rows = build_training_rows(
        args.schedule_csv,
        args.kenpom_dir,
        args.coach_history_csv,
        coach_road_hm_rows,
    )
    output_dir = args.output_dir

    write_json(training_rows, output_dir / "guarantee_game_training_table.json")
    write_csv(training_rows, output_dir / "guarantee_game_training_table.csv")

    rolling_predictions, rolling_metrics = rolling_backtest(training_rows)
    write_json(rolling_predictions, output_dir / "rolling_predictions.json")
    write_csv(rolling_predictions, output_dir / "rolling_predictions.csv")
    write_json(rolling_metrics, output_dir / "rolling_metrics.json")
    write_csv(rolling_metrics, output_dir / "rolling_metrics.csv")

    model = LogisticRiskModel().fit(training_rows)
    coefficients = model.coefficients()
    write_json(coefficients, output_dir / "model_coefficients.json")
    write_csv(coefficients, output_dir / "model_coefficients.csv")

    current_rows = []
    if not args.skip_current_board:
        current_rows = current_risk_board(
            model,
            args.kenpom_dir,
            args.current_predictions_csv,
            args.coach_latest_summary_csv,
            training_rows,
            coach_road_hm_rows,
        )
        write_json(current_rows, output_dir / "current_2027_guarantee_risk_board.json")
        write_csv(current_rows, output_dir / "current_2027_guarantee_risk_board.csv")

    summary = summarize_training(training_rows)
    print(f"saved {output_dir / 'guarantee_game_training_table.csv'}")
    print(f"saved {output_dir / 'rolling_metrics.csv'}")
    print(f"saved {output_dir / 'current_2027_guarantee_risk_board.csv'}")
    print(f"training rows: {summary['rows']}")
    print(f"training upsets: {summary['upsets']}")
    print(f"training upset rate: {summary['upset_rate']:.4f}")
    print(f"rolling test seasons: {len(rolling_metrics)}")
    if rolling_metrics:
        average_auc = sum(row["auc"] for row in rolling_metrics if row["auc"] is not None) / len(
            [row for row in rolling_metrics if row["auc"] is not None]
        )
        print(f"average rolling AUC: {average_auc:.3f}")
    print(f"current candidate rows: {len(current_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
