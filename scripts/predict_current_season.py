#!/usr/bin/env python3
"""Predict the upcoming season with schedule-building-safe features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_models import SCHEDULE_BUILDING_PREFIXES  # noqa: E402
from net_predictor.backtest import (  # noqa: E402
    ModelConfig,
    as_float,
    best_band_threshold,
    config_feature_columns,
    exclusive_band_from_flags,
    fit_boosted_trees,
    fit_ridge,
    model_prediction_percentile,
    model_target,
    program_consistency_band,
    rank_from_percentile,
    read_csv_rows,
    select_feature_columns,
    stronger_band,
    target_rows,
    with_forced_features,
    write_csv,
    write_json,
)
from net_predictor.coach_factor import canonical_team_key, normalize_coach_name  # noqa: E402
from net_predictor.model_table import build_model_rows  # noqa: E402


CURRENT_MODEL_CONFIGS = [
    ModelConfig(
        "direct_ridge_schedule_building",
        SCHEDULE_BUILDING_PREFIXES,
        "direct",
        alpha=100.0,
        excluded_substrings=("_sos", "_ncsos", "roster_talent_continuity_plus_incoming"),
        forced_features=("prior_roster_probable_returner_minutes_pct",),
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
        excluded_substrings=("_sos", "_ncsos", "roster_talent_continuity_plus_incoming"),
        forced_features=("prior_roster_probable_returner_minutes_pct",),
    ),
]
CURRENT_BLEND_MODELS = {
    "blend_schedule_ridge_gbt": [
        "direct_ridge_schedule_building",
        "direct_gbt_schedule_building",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2027, help="Ending year for the season to predict.")
    parser.add_argument(
        "--modeling-table-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "modeling" / "modeling_table.csv",
    )
    parser.add_argument(
        "--team-universe-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom" / "2026" / "teams.json",
        help="Current D-I team universe and conference source. KenPom ratings are not used as model features.",
    )
    parser.add_argument(
        "--coach-features-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "coach_features" / "coach_features_2026.csv",
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
        "--roster-summary-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "roster_status" / "team_roster_summary_2026.csv",
    )
    parser.add_argument(
        "--on3-features-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "on3_features" / "on3_incoming_talent_features.csv",
    )
    parser.add_argument(
        "--transfer-features-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "transfer_features"
        / "current"
        / "cbb_incoming_transfer_features.csv",
    )
    parser.add_argument(
        "--program-history-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "program_history"
        / "program_history_features_2017_2027.csv",
    )
    parser.add_argument(
        "--rolling-predictions-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "backtests" / "rolling_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "predictions",
    )
    parser.add_argument("--max-features", type=int, default=80)
    return parser.parse_args()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [row for row in data if isinstance(row, dict)]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def indexed_by_team(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for row in rows:
        key = row.get("team_key") or canonical_team_key(row.get("team") or row.get("team_name"))
        if key:
            index[str(key)] = row
    return index


def projected_coach_row(
    team: dict[str, Any],
    *,
    season: int,
    change_by_team: dict[str, dict[str, Any]],
    summary_by_coach: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    team_name = team.get("TeamName")
    team_key = canonical_team_key(team_name)
    change = change_by_team.get(team_key, {})
    current_coach = change.get("kenpom_coach") or team.get("Coach")
    projected_coach = current_coach
    if bool_value(change.get("coach_changed")) and bool_value(change.get("new_coach_known")):
        projected_coach = change.get("new_coach") or projected_coach

    coach_key = normalize_coach_name(projected_coach)
    summary = dict(summary_by_coach.get(coach_key, {}))
    for key in ("coach", "coach_key"):
        summary.pop(key, None)

    return {
        "season": season,
        "team_id": team.get("TeamID"),
        "team_name": team_name,
        "team_key": team_key,
        "conference": team.get("ConfShort"),
        "coach": projected_coach,
        "coach_key": coach_key,
        **summary,
        "coach_changed": bool_value(change.get("coach_changed")),
        "coach_change_status": change.get("coach_change_status"),
    }


def current_model_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    teams = read_json_rows(args.team_universe_json)
    changes = read_csv_rows(args.coach_features_csv) if args.coach_features_csv.exists() else []
    summaries = read_csv_rows(args.coach_latest_summary_csv) if args.coach_latest_summary_csv.exists() else []
    roster_rows = read_csv_rows(args.roster_summary_csv) if args.roster_summary_csv.exists() else []
    on3_rows = read_csv_rows(args.on3_features_csv) if args.on3_features_csv.exists() else []
    transfer_rows = (
        read_csv_rows(args.transfer_features_csv) if args.transfer_features_csv.exists() else []
    )
    program_rows = read_csv_rows(args.program_history_csv) if args.program_history_csv.exists() else []

    change_by_team = indexed_by_team(changes)
    summary_by_coach = {normalize_coach_name(row.get("coach")): row for row in summaries}
    coach_rows = [
        projected_coach_row(
            team,
            season=args.season,
            change_by_team=change_by_team,
            summary_by_coach=summary_by_coach,
        )
        for team in teams
    ]

    preseason_rows = [
        {
            "season": args.season,
            "team": team.get("TeamName"),
            "team_key": canonical_team_key(team.get("TeamName")),
            "conference": team.get("ConfShort"),
        }
        for team in teams
    ]
    prior_roster_rows = []
    for row in roster_rows:
        team = row.get("team_market") or row.get("team_name")
        prior = dict(row)
        prior["season"] = str(args.season - 1)
        prior["team"] = team
        prior["team_key"] = canonical_team_key(team)
        prior_roster_rows.append(prior)

    rows = build_model_rows(
        preseason_rows,
        coach_rows,
        [],
        prior_roster_rows,
        on3_rows,
        transfer_rows,
        program_rows,
    )
    for row in rows:
        row["target_teams_ranked"] = len(rows)
    return rows


def fit_model(config: ModelConfig, train_rows: list[dict[str, Any]], max_features: int):
    target_values = [model_target(row, config.mode) for row in train_rows]
    feature_limit = config.max_features if config.max_features is not None else max_features
    columns = select_feature_columns(
        train_rows,
        config_feature_columns(train_rows, config),
        target_values,
        feature_limit,
    )
    columns = with_forced_features(
        columns,
        config_feature_columns(train_rows, config),
        config.forced_features,
    )
    if config.algorithm == "ridge":
        model = fit_ridge(train_rows, columns, target_values, config.alpha or 100.0)
    elif config.algorithm == "gbt":
        model = fit_boosted_trees(
            train_rows,
            columns,
            target_values,
            estimators=config.estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_leaf=config.min_leaf,
            threshold_bins=config.threshold_bins,
        )
    else:
        raise ValueError(f"Unsupported model algorithm: {config.algorithm}")
    return model, columns


def model_outputs(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    max_features: int,
) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {}
    teams_ranked = len(rows)
    for config in CURRENT_MODEL_CONFIGS:
        model, columns = fit_model(config, train_rows, max_features)
        model_rows = []
        for row in rows:
            percentile = model_prediction_percentile(row, model, config.mode)
            rank = rank_from_percentile(percentile, teams_ranked)
            model_rows.append(
                {
                    "model": config.name,
                    "season": row["season"],
                    "team": row["team"],
                    "team_key": row["team_key"],
                    "conference": row.get("conference"),
                    "projected_coach": row.get("coach_coach"),
                    "schedule_score_rank": rank,
                    "schedule_score_percentile": percentile,
                    "program_consistency_band": program_consistency_band(row),
                    "feature_count": len(columns),
                    "train_rows": len(train_rows),
                    "prior_roster_returning_minutes_pct": row.get(
                        "prior_roster_probable_returner_minutes_pct"
                    ),
                    "prior_roster_probable_returner_minutes_pct": row.get(
                        "prior_roster_probable_returner_minutes_pct"
                    ),
                    "prior_roster_expected_returning_minutes_pct": row.get(
                        "prior_roster_expected_returning_minutes_pct"
                    ),
                    "prior_roster_returning_top_7_minutes_share": row.get(
                        "prior_roster_returning_top_7_minutes_share"
                    ),
                    "roster_known_players": row.get("roster_talent_known_roster_players"),
                    "returner_roster_share": row.get("roster_talent_returner_roster_share"),
                    "newcomer_roster_share": row.get("roster_talent_newcomer_roster_share"),
                    "hs_newcomer_roster_share": row.get("roster_talent_hs_newcomer_roster_share"),
                    "transfer_newcomer_roster_share": row.get(
                        "roster_talent_transfer_newcomer_roster_share"
                    ),
                    "returner_impact_share": row.get("roster_talent_returner_impact_share"),
                    "newcomer_impact_share": row.get("roster_talent_newcomer_impact_share"),
                    "hs_newcomer_impact_share": row.get("roster_talent_hs_newcomer_impact_share"),
                    "transfer_newcomer_impact_share": row.get(
                        "roster_talent_transfer_newcomer_impact_share"
                    ),
                    "composition_weighted_roster_talent": row.get(
                        "roster_talent_continuity_plus_incoming"
                    ),
                    "roster_talent_returning_production_pct_avg": row.get(
                        "roster_talent_returning_production_pct_avg"
                    ),
                    "roster_talent_returning_quality_index": row.get(
                        "roster_talent_returning_quality_index"
                    ),
                    "roster_talent_returning_core_continuity": row.get(
                        "roster_talent_returning_core_continuity"
                    ),
                    "roster_talent_cbb_transfer_quality_index": row.get(
                        "roster_talent_cbb_transfer_quality_index"
                    ),
                    "roster_talent_incoming_hs_score": row.get(
                        "roster_talent_incoming_hs_score"
                    ),
                    "roster_talent_incoming_transfer_score": row.get(
                        "roster_talent_incoming_transfer_score"
                    ),
                    "roster_talent_incoming_hs_rank_percentile": row.get(
                        "roster_talent_incoming_hs_rank_percentile"
                    ),
                    "roster_talent_incoming_transfer_rank_percentile": row.get(
                        "roster_talent_incoming_transfer_rank_percentile"
                    ),
                    "roster_talent_incoming_transfer_production_percentile": row.get(
                        "roster_talent_incoming_transfer_production_percentile"
                    ),
                    "roster_talent_weighted_returning_core_continuity": row.get(
                        "roster_talent_weighted_returning_core_continuity"
                    ),
                    "roster_talent_weighted_hs_rank_percentile": row.get(
                        "roster_talent_weighted_hs_rank_percentile"
                    ),
                    "roster_talent_weighted_transfer_rank_percentile": row.get(
                        "roster_talent_weighted_transfer_rank_percentile"
                    ),
                    "roster_talent_continuity_plus_incoming": row.get(
                        "roster_talent_continuity_plus_incoming"
                    ),
                    "incoming_on3_hs_rank": row.get("incoming_on3_hs_rank"),
                    "incoming_on3_transfer_rank": row.get("incoming_on3_transfer_rank"),
                    "incoming_cbb_transfer_players": row.get("incoming_cbb_transfer_players"),
                    "incoming_cbb_transfer_production_percentile": row.get(
                        "incoming_cbb_transfer_production_percentile"
                    ),
                    "incoming_cbb_transfer_source_adjusted_warp": row.get(
                        "incoming_cbb_transfer_source_adjusted_warp"
                    ),
                    "incoming_cbb_transfer_minutes": row.get("incoming_cbb_transfer_minutes"),
                }
            )
        outputs[config.name] = model_rows

    for blend_name, components in CURRENT_BLEND_MODELS.items():
        blend_rows = []
        for index, row in enumerate(rows):
            component_rows = [outputs[component][index] for component in components]
            percentile = sum(as_float(item["schedule_score_percentile"]) or 0.0 for item in component_rows) / len(
                component_rows
            )
            rank = rank_from_percentile(percentile, teams_ranked)
            blend = dict(component_rows[0])
            blend["model"] = blend_name
            blend["schedule_score_rank"] = rank
            blend["schedule_score_percentile"] = percentile
            blend["feature_count"] = sum(int(as_float(item.get("feature_count")) or 0) for item in component_rows)
            blend_rows.append(blend)
        outputs[blend_name] = blend_rows

    return outputs


def add_calibrated_bands(
    rows: list[dict[str, Any]],
    rolling_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model = rows[0]["model"] if rows else ""
    prior = [row for row in rolling_predictions if row.get("model") == model]
    thresholds = {band: best_band_threshold(prior, band) if prior else float(band) for band in (25, 50, 75, 100, 135, 160, 200, 250, 300)}
    outputs = []
    for row in rows:
        output = dict(row)
        rank = as_float(output.get("schedule_score_rank"))
        if rank is None:
            rank = as_float(output.get("predicted_net_rank"))
        for band, threshold in thresholds.items():
            output[f"calibrated_threshold_top_{band}"] = threshold
            output[f"calibrated_top_{band}"] = rank <= threshold if rank is not None else False
        output["calibrated_schedule_band"] = exclusive_band_from_flags(output, "calibrated")
        output["opponent_quality_tier"] = stronger_band(
            output.get("calibrated_schedule_band"),
            output.get("program_consistency_band"),
        )
        outputs.append(output)
    return outputs


def main() -> int:
    args = parse_args()
    historical_rows = read_csv_rows(args.modeling_table_csv)
    train_rows = target_rows(historical_rows)
    current_rows = current_model_rows(args)
    outputs = model_outputs(current_rows, train_rows, max_features=args.max_features)
    rolling_predictions = (
        read_csv_rows(args.rolling_predictions_csv) if args.rolling_predictions_csv.exists() else []
    )

    all_rows = []
    for rows in outputs.values():
        all_rows.extend(add_calibrated_bands(rows, rolling_predictions))
    all_rows.sort(key=lambda row: (row["model"], as_float(row["schedule_score_rank"]) or 9999, row["team"]))

    stem = f"current_{args.season}_schedule_predictions"
    csv_path = write_csv(all_rows, args.output_dir / f"{stem}.csv")
    json_path = write_json(all_rows, args.output_dir / f"{stem}.json")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")
    print(f"prediction rows: {len(all_rows)}")
    print(f"teams predicted: {len(current_rows)}")
    print(f"rows with prior roster features: {sum(1 for row in current_rows if row.get('prior_roster_source_season'))}")
    print(f"rows with On3 HS features: {sum(1 for row in current_rows if row.get('incoming_on3_hs_rank'))}")
    print(f"rows with On3 transfer features: {sum(1 for row in current_rows if row.get('incoming_on3_transfer_rank'))}")
    print(
        "rows with CBB transfer features: "
        f"{sum(1 for row in current_rows if row.get('incoming_cbb_transfer_players'))}"
    )
    print(f"rows with program history features: {sum(1 for row in current_rows if row.get('program_prior_seasons'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
