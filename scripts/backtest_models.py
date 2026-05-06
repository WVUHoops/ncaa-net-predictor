#!/usr/bin/env python3
"""Run rolling-season NET prediction backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.backtest import (  # noqa: E402
    ModelConfig,
    backtest_band_metrics,
    backtest_metrics,
    backtest_slice_metrics,
    blended_predictions,
    calibrated_band_predictions,
    read_csv_rows,
    rolling_feature_selections,
    rolling_predictions,
    write_csv,
    write_json,
)


ROSTER_TALENT_PREFIXES = (
    "prior_roster_",
    "incoming_cbb_transfer_",
    "incoming_on3_hs_",
    "roster_talent_",
)
PROGRAM_PREFIXES = ("program_prior_",)
SCHEDULE_BUILDING_EXCLUDED_SUBSTRINGS = (
    "_sos",
    "_ncsos",
    "roster_talent_continuity_plus_incoming",
)
SCHEDULE_BUILDING_PREFIXES = (
    *PROGRAM_PREFIXES,
    "coach_coach_prior_",
    "coach_coach_first_",
    *ROSTER_TALENT_PREFIXES,
)
MODEL_CONFIGS = [
    ModelConfig("direct_ridge_roster_talent", ROSTER_TALENT_PREFIXES, "direct", alpha=100.0),
    ModelConfig(
        "direct_ridge_schedule_building",
        SCHEDULE_BUILDING_PREFIXES,
        "direct",
        alpha=100.0,
        excluded_substrings=SCHEDULE_BUILDING_EXCLUDED_SUBSTRINGS,
        forced_features=("prior_roster_probable_returner_minutes_pct",),
    ),
    ModelConfig(
        "direct_gbt_roster_talent",
        ROSTER_TALENT_PREFIXES,
        "direct",
        algorithm="gbt",
        max_features=55,
        estimators=28,
        learning_rate=0.05,
        max_depth=2,
        min_leaf=35,
        threshold_bins=6,
    ),
    ModelConfig(
        "direct_gbt_schedule_building",
        SCHEDULE_BUILDING_PREFIXES,
        "direct",
        algorithm="gbt",
        max_features=70,
        estimators=28,
        learning_rate=0.05,
        max_depth=2,
        min_leaf=35,
        threshold_bins=6,
        excluded_substrings=SCHEDULE_BUILDING_EXCLUDED_SUBSTRINGS,
        forced_features=("prior_roster_probable_returner_minutes_pct",),
    ),
    ModelConfig("direct_ridge_kenpom", ("kenpom_preseason_",), "direct", alpha=100.0),
    ModelConfig(
        "direct_ridge_kenpom_roster_talent",
        ("kenpom_preseason_", *ROSTER_TALENT_PREFIXES),
        "direct",
        alpha=100.0,
    ),
    ModelConfig(
        "direct_gbt_kenpom_roster_talent",
        ("kenpom_preseason_", *ROSTER_TALENT_PREFIXES),
        "direct",
        algorithm="gbt",
        max_features=55,
        estimators=28,
        learning_rate=0.05,
        max_depth=2,
        min_leaf=35,
        threshold_bins=6,
    ),
    ModelConfig("residual_ridge_kenpom", ("kenpom_preseason_",), "residual"),
    ModelConfig(
        "residual_ridge_kenpom_roster_talent",
        ("kenpom_preseason_", *ROSTER_TALENT_PREFIXES),
        "residual",
    ),
    ModelConfig(
        "residual_gbt_kenpom_roster_talent",
        ("kenpom_preseason_", *ROSTER_TALENT_PREFIXES),
        "residual",
        algorithm="gbt",
        max_features=55,
        estimators=28,
        learning_rate=0.05,
        max_depth=2,
        min_leaf=35,
        threshold_bins=6,
    ),
    ModelConfig(
        "residual_ridge_kenpom_coach",
        ("kenpom_preseason_", "coach_coach_prior_", "coach_coach_first_"),
        "residual",
    ),
    ModelConfig("residual_ridge_roster_talent", ROSTER_TALENT_PREFIXES, "residual"),
    ModelConfig(
        "residual_ridge_full",
        (
            "kenpom_preseason_",
            "coach_coach_prior_",
            "coach_coach_first_",
            *ROSTER_TALENT_PREFIXES,
        ),
        "residual",
    ),
    ModelConfig(
        "residual_gbt_full",
        (
            "kenpom_preseason_",
            "coach_coach_prior_",
            "coach_coach_first_",
            *ROSTER_TALENT_PREFIXES,
        ),
        "residual",
        algorithm="gbt",
        max_features=55,
        estimators=28,
        learning_rate=0.05,
        max_depth=2,
        min_leaf=35,
        threshold_bins=6,
    ),
]
BLEND_MODELS = {
    "blend_schedule_ridge_gbt": [
        "direct_ridge_schedule_building",
        "direct_gbt_schedule_building",
    ],
    "blend_baseline_gbt_roster_residual": [
        "kenpom_preseason_baseline",
        "residual_gbt_kenpom_roster_talent",
    ],
    "blend_ridge_gbt_roster_residual": [
        "residual_ridge_kenpom",
        "residual_gbt_kenpom_roster_talent",
    ],
    "blend_coach_gbt_roster_residual": [
        "residual_ridge_kenpom_coach",
        "residual_gbt_kenpom_roster_talent",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeling-table-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "modeling" / "modeling_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "backtests",
    )
    parser.add_argument("--alpha", type=float, default=10_000.0, help="Ridge regularization strength.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=80,
        help="Maximum numeric features selected per rolling train split before missing indicators and conference terms.",
    )
    parser.add_argument(
        "--first-test-season",
        type=int,
        default=2022,
        help="First season to evaluate. 2021 is skipped by default because there is no prior NET target season for training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.modeling_table_csv)
    predictions = rolling_predictions(
        rows,
        model_configs=MODEL_CONFIGS,
        alpha=args.alpha,
        max_features=args.max_features,
        first_test_season=args.first_test_season,
    )
    predictions.extend(blended_predictions(predictions, BLEND_MODELS))
    predictions = calibrated_band_predictions(predictions)
    metrics = backtest_metrics(predictions)
    band_metrics = backtest_band_metrics(predictions)
    slice_metrics = backtest_slice_metrics(predictions, rows)
    feature_selections = rolling_feature_selections(
        rows,
        model_configs=MODEL_CONFIGS,
        alpha=args.alpha,
        max_features=args.max_features,
        first_test_season=args.first_test_season,
    )

    predictions_csv = write_csv(predictions, args.output_dir / "rolling_predictions.csv")
    predictions_json = write_json(predictions, args.output_dir / "rolling_predictions.json")
    metrics_csv = write_csv(metrics, args.output_dir / "rolling_metrics.csv")
    metrics_json = write_json(metrics, args.output_dir / "rolling_metrics.json")
    band_metrics_csv = write_csv(band_metrics, args.output_dir / "rolling_band_metrics.csv")
    band_metrics_json = write_json(band_metrics, args.output_dir / "rolling_band_metrics.json")
    slice_metrics_csv = write_csv(slice_metrics, args.output_dir / "rolling_slice_metrics.csv")
    slice_metrics_json = write_json(slice_metrics, args.output_dir / "rolling_slice_metrics.json")
    selections_csv = write_csv(feature_selections, args.output_dir / "rolling_feature_selections.csv")
    selections_json = write_json(feature_selections, args.output_dir / "rolling_feature_selections.json")

    print(f"saved {predictions_csv}")
    print(f"saved {predictions_json}")
    print(f"saved {metrics_csv}")
    print(f"saved {metrics_json}")
    print(f"saved {band_metrics_csv}")
    print(f"saved {band_metrics_json}")
    print(f"saved {slice_metrics_csv}")
    print(f"saved {slice_metrics_json}")
    print(f"saved {selections_csv}")
    print(f"saved {selections_json}")
    print(f"prediction rows: {len(predictions)}")
    print(f"metric rows: {len(metrics)}")
    print(f"band metric rows: {len(band_metrics)}")
    print(f"slice metric rows: {len(slice_metrics)}")
    print(f"feature selection rows: {len(feature_selections)}")
    for row in metrics:
        if row["season"] == "overall":
            print(
                f"{row['model']}: MAE={float(row['rank_mae']):.2f}, "
                f"RMSE={float(row['rank_rmse']):.2f}, rows={row['rows']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
