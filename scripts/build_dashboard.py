#!/usr/bin/env python3
"""Build a static schedule-building dashboard from processed model outputs."""

from __future__ import annotations

import re
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.upset_risk import schedule_team_key  # noqa: E402

HOME_COURT_ADJ_EM = 3.5
WIN_PROBABILITY_SCALE = 6.5


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def latest_dated_snapshot(directory: Path, pattern: str) -> str | None:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    latest = candidates[-1]
    match = re.search(r"(\d{4}-\d{2}-\d{2})", latest.name)
    return match.group(1) if match else None


def file_mtime_date(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().date().isoformat()


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def latest_existing_file(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_float(value: Any, digits: int = 3) -> float | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def pct(value: Any) -> float | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return round(parsed * 100, 1)


def added_wab_proxy(schedule_score_rank: Any) -> float | None:
    """Approximate home-win WAB value from the schedule score rank.

    This is not official WAB. It is a monotonic schedule-value proxy so the
    dashboard can speak in a WAB-like language until a true WAB source is added.
    """
    rank = as_float(schedule_score_rank)
    if rank is None:
        return None
    rank = min(max(rank, 1.0), 365.0)
    bubble_home_win_probability = 1 / (1 + pow(2.718281828, -(rank - 95.0) / 58.0))
    return round(1 - bubble_home_win_probability, 3)


def canonical_team_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(".", "")


def tier_label(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "top_25":
        return "Top 25"
    if raw == "301_plus":
        return "301+"
    return raw.replace("_", "-")


def projected_adj_em_from_rank(rank: Any, adj_em_by_rank: list[float]) -> float | None:
    parsed = as_float(rank)
    if parsed is None or not adj_em_by_rank:
        return None
    parsed = min(max(parsed, 1.0), float(len(adj_em_by_rank)))
    lower_index = int(parsed) - 1
    upper_index = min(lower_index + 1, len(adj_em_by_rank) - 1)
    fraction = parsed - int(parsed)
    lower = adj_em_by_rank[lower_index]
    upper = adj_em_by_rank[upper_index]
    return round(lower + (upper - lower) * fraction, 2)


def average_adj_em_for_rank_range(
    adj_em_by_rank: list[float],
    start_rank: int,
    end_rank: int | None = None,
) -> float | None:
    if not adj_em_by_rank:
        return None
    end_rank = end_rank or len(adj_em_by_rank)
    start_index = max(start_rank - 1, 0)
    end_index = min(end_rank, len(adj_em_by_rank))
    values = adj_em_by_rank[start_index:end_index]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def tier_adj_em_benchmarks(adj_em_by_rank: list[float]) -> dict[str, float]:
    ranges = {
        "top_25": (1, 25),
        "26_50": (26, 50),
        "51_75": (51, 75),
        "76_100": (76, 100),
        "101_135": (101, 135),
        "136_160": (136, 160),
        "161_200": (161, 200),
        "201_250": (201, 250),
        "251_300": (251, 300),
        "301_plus": (301, None),
    }
    benchmarks = {}
    for tier, (start_rank, end_rank) in ranges.items():
        value = average_adj_em_for_rank_range(adj_em_by_rank, start_rank, end_rank)
        if value is not None:
            benchmarks[tier] = value
    return benchmarks


def wvu_home_upset_pct(
    row: dict[str, str],
    projections_by_team: dict[str, dict[str, Any]],
    host_row: dict[str, Any] | None,
) -> float | None:
    host_adj_em = as_float((host_row or {}).get("projected_adj_em"))
    opponent_row = projections_by_team.get(canonical_team_name(row.get("team")))
    opponent_adj_em = as_float((opponent_row or {}).get("projected_adj_em"))
    if host_adj_em is None or opponent_adj_em is None:
        return None

    wvu_home_gap = host_adj_em + HOME_COURT_ADJ_EM - opponent_adj_em
    wvu_win_probability = 1 / (1 + pow(2.718281828, -(wvu_home_gap / WIN_PROBABILITY_SCALE)))
    return round((1 - wvu_win_probability) * 100, 1)


def wvu_risk_bucket(upset_pct: Any) -> str:
    parsed = as_float(upset_pct)
    if parsed is None:
        return "unknown"
    if parsed >= 12:
        return "very_high"
    if parsed >= 8:
        return "high"
    if parsed >= 5:
        return "medium"
    if parsed >= 3:
        return "low"
    return "very_low"


def risk_sort_value(bucket: Any) -> int:
    return {
        "very_low": 1,
        "low": 2,
        "medium": 3,
        "high": 4,
        "very_high": 5,
    }.get(str(bucket or ""), 0)


def coach_signal_bucket(lift_pp: Any) -> str:
    parsed = as_float(lift_pp)
    if parsed is None:
        return "unknown"
    if parsed >= 8:
        return "very_high"
    if parsed >= 3:
        return "high"
    if parsed <= -2:
        return "low"
    return "neutral"


def slim_risk_row(
    row: dict[str, str],
    projections_by_team: dict[str, dict[str, Any]],
    host_row: dict[str, Any] | None,
) -> dict[str, Any]:
    upset_pct = pct(row.get("upset_probability_vs_median_high_major"))
    risk_bucket = wvu_risk_bucket(upset_pct)
    return {
        "team": row.get("team"),
        "projected_coach": row.get("projected_coach"),
        "coach_lift": compact_float(row.get("coach_upset_lift_pp"), 1),
        "coach_signal": coach_signal_bucket(row.get("coach_upset_lift_pp")),
        "conference": row.get("conference"),
        "tier": row.get("opponent_quality_tier"),
        "projected_net_tier": row.get("projected_net_tier") or row.get("opponent_quality_tier"),
        "program_band": row.get("program_consistency_band"),
        "upset_pct": upset_pct,
        "risk_bucket": risk_bucket,
        "risk_sort": risk_sort_value(risk_bucket),
        "recommendation": row.get("recommendation"),
        "danger_index": compact_float(row.get("danger_index"), 4),
        "projected_net_rank": compact_float(row.get("projected_net_rank") or row.get("schedule_score_rank"), 1),
        "schedule_score": compact_float(row.get("projected_net_rank") or row.get("schedule_score_rank"), 1),
        "added_wab": added_wab_proxy(row.get("projected_net_rank") or row.get("schedule_score_rank")),
        "three_rate": compact_float(row.get("away_three_point_attempt_rate"), 1),
        "experience": compact_float(row.get("away_experience"), 2),
        "adj_em": compact_float(row.get("away_adj_em"), 1),
        "adj_oe": compact_float(row.get("away_adj_oe"), 1),
        "adj_de": compact_float(row.get("away_adj_de"), 1),
        "tempo": compact_float(row.get("away_adj_tempo"), 1),
        "hs_rank": compact_float(row.get("incoming_on3_hs_rank"), 0),
        "transfer_rank": compact_float(row.get("incoming_on3_transfer_rank"), 0),
    }


def slim_schedule_row(
    row: dict[str, str],
    adj_em_by_rank: list[float],
    tier_benchmarks: dict[str, float],
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule_score_source = (
        override.get("projected_net_rank")
        if override and override.get("projected_net_rank") is not None
        else override.get("schedule_score")
        if override and override.get("schedule_score") is not None
        else row.get("projected_net_rank")
        if row.get("projected_net_rank") is not None
        else row.get("schedule_score_rank")
    )
    tier_source = (
        override.get("projected_net_tier")
        if override and override.get("projected_net_tier")
        else override.get("tier")
        if override and override.get("tier")
        else row.get("projected_net_tier")
        if row.get("projected_net_tier")
        else row.get("opponent_quality_tier")
    )
    program_band_source = (
        override.get("program_band") if override and override.get("program_band") else row.get("program_consistency_band")
    )
    rank = compact_float(schedule_score_source, 1)
    projected_adj_em = projected_adj_em_from_rank(schedule_score_source, adj_em_by_rank)
    tier_benchmark = tier_benchmarks.get(str(tier_source or ""))
    wab_adj_candidates = [value for value in (projected_adj_em, tier_benchmark) if value is not None]
    return {
        "team": row.get("team"),
        "team_key": row.get("team_key"),
        "conference": row.get("conference"),
        "tier": tier_source,
        "projected_net_tier": tier_source,
        "program_band": program_band_source,
        "projected_net_rank": rank,
        "schedule_score": rank,
        "schedule_percentile": compact_float(row.get("schedule_score_percentile"), 3),
        "added_wab": added_wab_proxy(schedule_score_source),
        "projected_adj_em": projected_adj_em,
        "wab_adj_em": round(max(wab_adj_candidates), 2) if wab_adj_candidates else None,
    }


def tier_placeholder_rows(
    rows: list[dict[str, Any]],
    tier_benchmarks: dict[str, float],
) -> list[dict[str, Any]]:
    tier_order = [
        "top_25",
        "26_50",
        "51_75",
        "76_100",
        "101_135",
        "136_160",
        "161_200",
        "201_250",
        "251_300",
        "301_plus",
    ]
    placeholders = []
    for tier in tier_order:
        tier_rows = [row for row in rows if row.get("tier") == tier]
        if not tier_rows:
            continue
        scores = [value for value in (as_float(row.get("schedule_score")) for row in tier_rows) if value is not None]
        adj_ems = [value for value in (as_float(row.get("projected_adj_em")) for row in tier_rows) if value is not None]
        if not scores or not adj_ems:
            continue
        wab_adj_em = tier_benchmarks.get(tier)
        label = tier_label(tier)
        placeholders.append(
            {
                "team": f"Placeholder: {label} Team",
                "team_key": f"placeholder_{tier}",
                "conference": "Tier",
                "tier": tier,
                "program_band": tier,
                "schedule_score": round(sum(scores) / len(scores), 1),
                "schedule_percentile": None,
                "added_wab": added_wab_proxy(sum(scores) / len(scores)),
                "projected_adj_em": round(sum(adj_ems) / len(adj_ems), 2),
                "wab_adj_em": wab_adj_em,
                "is_placeholder": True,
            }
        )
    return placeholders


def solve_linear_system(matrix: list[list[float]], targets: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [targets[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-9:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[index][size] for index in range(size)]


def fit_least_squares(rows: list[dict[str, float]]) -> list[float] | None:
    if not rows:
        return None
    feature_count = 4
    xtx = [[0.0] * feature_count for _ in range(feature_count)]
    xty = [0.0] * feature_count
    for row in rows:
        features = [
            1.0,
            row["avg_adj_em"],
            row["home_share"],
            row["neutral_share"],
        ]
        target = row["target_rank"]
        for left in range(feature_count):
            xty[left] += features[left] * target
            for right in range(feature_count):
                xtx[left][right] += features[left] * features[right]
    return solve_linear_system(xtx, xty)


def historical_ncsos_rows() -> list[dict[str, float]]:
    schedule_path = PROJECT_ROOT / "data" / "raw" / "hoopr" / "mbb_schedule_master.csv"
    selection_dir = PROJECT_ROOT / "data" / "raw" / "ncaa_net_selections"
    kenpom_dir = PROJECT_ROOT / "data" / "raw" / "kenpom"
    if not schedule_path.exists() or not selection_dir.exists() or not kenpom_dir.exists():
        return []

    targets: dict[tuple[int, str], float] = {}
    kenpom_adj_em_by_season: dict[int, dict[str, float]] = {}

    for selection_csv in sorted(selection_dir.glob("net_selections_*.csv")):
        match = re.search(r"net_selections_(\d{4})_", selection_csv.name)
        if not match:
            continue
        season = int(match.group(1))
        ratings_path = kenpom_dir / str(season) / "ratings.json"
        if not ratings_path.exists():
            continue
        for row in read_csv(selection_csv):
            team = row.get("team")
            target_rank = as_float(row.get("net_nonconference_sos"))
            if not team or target_rank is None:
                continue
            targets[(season, schedule_team_key(team))] = target_rank
        ratings_rows = read_json(ratings_path)
        kenpom_adj_em_by_season[season] = {
            schedule_team_key(row.get("TeamName") or row.get("TeamNameA")): as_float(row.get("AdjEM"))
            for row in ratings_rows
            if (row.get("TeamName") or row.get("TeamNameA")) and as_float(row.get("AdjEM")) is not None
        }

    aggregated: dict[tuple[int, str], dict[str, float]] = {}
    with schedule_path.open("r", encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            if raw.get("season_type") != "2" or raw.get("conference_competition") != "FALSE":
                continue
            season = int(raw.get("season") or 0)
            adj_em_by_team = kenpom_adj_em_by_season.get(season)
            if not adj_em_by_team:
                continue

            neutral = raw.get("neutral_site") == "TRUE"
            home_key = schedule_team_key(raw.get("home_short_display_name"))
            away_key = schedule_team_key(raw.get("away_short_display_name"))
            home_adj_em = adj_em_by_team.get(home_key)
            away_adj_em = adj_em_by_team.get(away_key)
            if home_adj_em is None or away_adj_em is None:
                continue

            if (season, home_key) in targets:
                entry = aggregated.setdefault((season, home_key), {"sum_adj_em": 0.0, "games": 0.0, "home_games": 0.0, "neutral_games": 0.0})
                entry["sum_adj_em"] += away_adj_em if neutral else away_adj_em - HOME_COURT_ADJ_EM
                entry["games"] += 1.0
                entry["home_games"] += 0.0 if neutral else 1.0
                entry["neutral_games"] += 1.0 if neutral else 0.0

            if (season, away_key) in targets:
                entry = aggregated.setdefault((season, away_key), {"sum_adj_em": 0.0, "games": 0.0, "home_games": 0.0, "neutral_games": 0.0})
                entry["sum_adj_em"] += home_adj_em if neutral else home_adj_em + HOME_COURT_ADJ_EM
                entry["games"] += 1.0
                entry["home_games"] += 0.0
                entry["neutral_games"] += 1.0 if neutral else 0.0

    rows: list[dict[str, float]] = []
    for (season, team_key), values in aggregated.items():
        games = values["games"]
        if not games:
            continue
        target_rank = targets.get((season, team_key))
        if target_rank is None:
            continue
        rows.append(
            {
                "season": float(season),
                "avg_adj_em": values["sum_adj_em"] / games,
                "home_share": values["home_games"] / games,
                "neutral_share": values["neutral_games"] / games,
                "games": games,
                "target_rank": target_rank,
            }
        )
    return rows


def build_ncsos_calibration() -> dict[str, Any] | None:
    rows = historical_ncsos_rows()
    coefficients = fit_least_squares(rows)
    if not rows or coefficients is None:
        return None
    seasons = sorted({int(row["season"]) for row in rows})
    predictions = [
        coefficients[0]
        + coefficients[1] * row["avg_adj_em"]
        + coefficients[2] * row["home_share"]
        + coefficients[3] * row["neutral_share"]
        for row in rows
    ]
    mae = sum(abs(prediction - row["target_rank"]) for prediction, row in zip(predictions, rows)) / len(rows)
    rmse = (
        sum((prediction - row["target_rank"]) ** 2 for prediction, row in zip(predictions, rows)) / len(rows)
    ) ** 0.5
    return {
        "coefficients": [round(value, 6) for value in coefficients],
        "row_count": len(rows),
        "seasons": seasons,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    risk_rows = read_csv(args.risk_board_csv)
    metric_rows = read_csv(args.metrics_csv)
    coefficient_rows = read_csv(args.coefficients_csv)
    rating_rows = read_json(args.kenpom_ratings_json)
    adj_em_by_rank = sorted(
        [value for value in (as_float(row.get("AdjEM")) for row in rating_rows) if value is not None],
        reverse=True,
    )
    tier_benchmarks = tier_adj_em_benchmarks(adj_em_by_rank)
    bubble_adj_em = average_adj_em_for_rank_range(adj_em_by_rank, 40, 60)
    ncsos_values = sorted(
        [value for value in (as_float(row.get("NCSOS")) for row in rating_rows) if value is not None],
        reverse=True,
    )
    ncsos_calibration = build_ncsos_calibration()
    on3_hs_snapshot = latest_dated_snapshot(
        PROJECT_ROOT / "data" / "raw" / "on3" / "hs" / "2026",
        "on3_hs_2026_*.json",
    )
    on3_transfer_snapshot = latest_dated_snapshot(
        PROJECT_ROOT / "data" / "raw" / "on3" / "transfer" / "2026",
        "on3_transfer_2026_*.json",
    )
    cbb_player_snapshot = latest_dated_snapshot(
        PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "v1" / "player-agg-box",
        "stats_player_agg_box_competition_41097_v1_*.json",
    )
    cbb_transfer_features_csv = (
        PROJECT_ROOT / "data" / "processed" / "transfer_features" / "current" / "cbb_incoming_transfer_features.csv"
    )
    transfer_ledger_path = latest_existing_file(
        [
            *sorted((PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "transfer_portal" / "current").glob("*.csv")),
            *sorted((PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "transfer_portal").glob("*.csv")),
        ]
    )
    cbb_transfer_feature_rows = csv_row_count(cbb_transfer_features_csv)
    cbb_transfer_feature_built_at = file_mtime_date(cbb_transfer_features_csv)
    cbb_transfer_ledger_snapshot = file_mtime_date(transfer_ledger_path) if transfer_ledger_path else None
    schedule_rows = [
        row
        for row in read_csv(args.schedule_predictions_csv)
        if row.get("model") == args.planner_model
    ]
    planner_rows = [slim_schedule_row(row, adj_em_by_rank, tier_benchmarks) for row in schedule_rows]
    planner_rows = sorted(planner_rows, key=lambda row: as_float(row.get("schedule_score")) or 9999)
    host_row = next(
        (
            row
            for row in planner_rows
            if canonical_team_name(row.get("team")) == canonical_team_name(args.planner_host)
        ),
        None,
    )
    projections_by_team = {canonical_team_name(row.get("team")): row for row in planner_rows}
    slimmed_risk_rows = [slim_risk_row(row, projections_by_team, host_row) for row in risk_rows]
    planner_overrides_by_team = {
        canonical_team_name(row.get("team")): row for row in slimmed_risk_rows
    }
    planner_rows = [
        slim_schedule_row(
            row,
            adj_em_by_rank,
            tier_benchmarks,
            override=planner_overrides_by_team.get(canonical_team_name(row.get("team"))),
        )
        for row in schedule_rows
    ]
    planner_rows = sorted(planner_rows, key=lambda row: as_float(row.get("schedule_score")) or 9999)
    planner_placeholder_rows = tier_placeholder_rows(planner_rows, tier_benchmarks)
    host_row = next(
        (
            row
            for row in planner_rows
            if canonical_team_name(row.get("team")) == canonical_team_name(args.planner_host)
        ),
        None,
    )
    projections_by_team = {canonical_team_name(row.get("team")): row for row in planner_rows}
    slimmed_risk_rows = [slim_risk_row(row, projections_by_team, host_row) for row in risk_rows]

    recommendations = Counter(row.get("recommendation") for row in slimmed_risk_rows)
    risk_buckets = Counter(row.get("risk_bucket") for row in slimmed_risk_rows)
    tiers = Counter(row.get("opponent_quality_tier") for row in risk_rows)
    avg_auc_values = [as_float(row.get("auc")) for row in metric_rows if as_float(row.get("auc")) is not None]
    avg_auc = sum(avg_auc_values) / len(avg_auc_values) if avg_auc_values else None
    training_rows = as_float(metric_rows[-1].get("train_rows")) if metric_rows else None
    latest_test_rows = as_float(metric_rows[-1].get("test_rows")) if metric_rows else None

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "risk_board_csv": str(args.risk_board_csv),
            "metrics_csv": str(args.metrics_csv),
            "coefficients_csv": str(args.coefficients_csv),
            "schedule_predictions_csv": str(args.schedule_predictions_csv),
            "kenpom_ratings_json": str(args.kenpom_ratings_json),
        },
        "input_status": {
            "on3_hs_snapshot": on3_hs_snapshot,
            "on3_transfer_snapshot": on3_transfer_snapshot,
            "cbb_player_snapshot": cbb_player_snapshot,
            "cbb_transfer_feature_rows": cbb_transfer_feature_rows,
            "cbb_transfer_feature_built_at": cbb_transfer_feature_built_at,
            "cbb_transfer_ledger_snapshot": cbb_transfer_ledger_snapshot,
        },
        "summary": {
            "candidate_count": len(risk_rows),
            "good_targets": recommendations.get("good_target", 0)
            + recommendations.get("strong_target", 0),
            "avoid_count": recommendations.get("avoid_bad_risk_reward", 0)
            + recommendations.get("avoid_unless_needed", 0),
            "avg_rolling_auc": round(avg_auc, 3) if avg_auc is not None else None,
            "latest_train_rows": int(training_rows) if training_rows is not None else None,
            "latest_test_rows": int(latest_test_rows) if latest_test_rows is not None else None,
            "schedule_prediction_teams": len(schedule_rows),
        },
        "recommendation_counts": dict(sorted(recommendations.items())),
        "risk_bucket_counts": dict(sorted(risk_buckets.items())),
        "tier_counts": dict(sorted(tiers.items())),
        "risk_rows": slimmed_risk_rows,
        "planner": {
            "host": args.planner_host,
            "model": args.planner_model,
            "host_projection": host_row,
            "teams": planner_placeholder_rows + planner_rows,
            "placeholder_teams": planner_placeholder_rows,
            "bubble_adj_em": bubble_adj_em,
            "tier_adj_em_benchmarks": tier_benchmarks,
            "ncsos_benchmarks": [round(value, 3) for value in ncsos_values],
            "ncsos_calibration": ncsos_calibration,
        },
        "metrics": metric_rows,
        "coefficients": [
            {
                "feature": row.get("feature"),
                "coefficient": compact_float(row.get("coefficient"), 4),
                "abs_coefficient": compact_float(row.get("abs_coefficient"), 4),
            }
            for row in coefficient_rows[:18]
        ],
    }


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def dashboard_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NCAA Schedule Builder</title>
  <style>
    :root {{
      --wvu-blue: #002855;
      --wvu-gold: #eaaa00;
      --wvu-gold-web: #eeaa00;
      --wvu-blue-light: #0062a3;
      --wvu-sky: #9ddae6;
      --ink: #1c2b39;
      --muted: #5b6672;
      --line: #d7dde4;
      --paper: #f7f7f7;
      --panel: #ffffff;
      --green: #0b6b4f;
      --teal: #0062a3;
      --gold: #7f6310;
      --red: #b91c1c;
      --soft-red: #fff1f1;
      --soft-green: #eef8f3;
      --soft-gold: #fff8df;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-size: 15px;
      line-height: 1.4;
    }}
    header {{
      padding: 24px;
      border-bottom: 4px solid var(--wvu-gold);
      background: var(--wvu-blue);
      color: #fff;
    }}
    main {{
      padding: 20px 24px 36px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{ margin: 6px 0 0; color: var(--muted); }}
    header p {{ color: #d9e6f2; }}
    .topline {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      max-width: 1500px;
      margin: 0 auto;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border: 1px solid rgba(234, 170, 0, 0.75);
      border-radius: 8px;
      background: rgba(234, 170, 0, 0.15);
      color: #fff;
      font-size: 13px;
      white-space: nowrap;
    }}
    .header-side {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-left: auto;
    }}
    .badge-stack {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .badge.subtle {{
      border-color: rgba(157, 218, 230, 0.7);
      background: rgba(157, 218, 230, 0.14);
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 0 0 18px;
    }}
    .tab-button {{
      min-height: 38px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--wvu-blue);
      font: inherit;
      cursor: pointer;
    }}
    .tab-button.active {{
      border-color: var(--wvu-blue);
      background: var(--wvu-blue);
      color: #fff;
      font-weight: 700;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .hero-mark-frame {{
      width: 64px;
      height: 64px;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid rgba(234, 170, 0, 0.75);
      background: #001f42;
    }}
    .hero-mark {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center top;
      display: block;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 4px solid var(--wvu-gold);
      border-radius: 8px;
      padding: 14px;
      min-height: 84px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 6px;
    }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(5, minmax(150px, 1fr));
      gap: 10px;
      margin: 18px 0 12px;
    }}
    input, select {{
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
    }}
    input:focus, select:focus, details.filter:focus-within {{
      outline: 2px solid rgba(234, 170, 0, 0.55);
      outline-offset: 2px;
      border-color: var(--wvu-blue);
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--wvu-gold);
      background: var(--wvu-gold);
      color: var(--wvu-blue);
      font-weight: 700;
    }}
    button.danger {{
      border-color: #ffd0d0;
      background: var(--soft-red);
      color: var(--red);
    }}
    details.filter {{
      position: relative;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    details.filter summary {{
      min-height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px;
      cursor: pointer;
      color: var(--ink);
      list-style: none;
    }}
    details.filter summary::-webkit-details-marker {{ display: none; }}
    details.filter summary::after {{
      content: "⌄";
      color: var(--muted);
    }}
    .filter-menu {{
      position: absolute;
      z-index: 4;
      top: calc(100% + 4px);
      left: 0;
      right: 0;
      max-height: 280px;
      overflow: auto;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 14px 28px rgba(0, 40, 85, 0.14);
    }}
    .filter-option {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 6px;
      border-radius: 8px;
      color: var(--ink);
    }}
    .filter-option:hover {{ background: #f4f7fb; }}
    .filter-option input {{ width: auto; min-height: auto; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 10px 24px rgba(0, 40, 85, 0.07);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1120px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: var(--wvu-blue);
      color: #fff;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      cursor: pointer;
      user-select: none;
    }}
    th.sorted::after {{
      content: attr(data-sort-mark);
      display: inline-block;
      margin-left: 6px;
      color: var(--wvu-gold);
    }}
    tbody tr:hover td {{ background: #f7fbff; }}
    tr:last-child td {{ border-bottom: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #f7f7f7;
      font-size: 12px;
    }}
    .tier-chip {{
      min-width: 74px;
      justify-content: center;
      font-weight: 700;
      color: var(--ink);
    }}
    .tier-top_25 {{ background: var(--wvu-blue); border-color: var(--wvu-blue); color: #fff; }}
    .tier-26_50 {{ background: #083d75; border-color: #083d75; color: #fff; }}
    .tier-51_75 {{ background: #e9f6fa; border-color: #9ddae6; color: var(--wvu-blue); }}
    .tier-76_100 {{ background: #fff4cc; border-color: var(--wvu-gold); color: var(--wvu-blue); }}
    .tier-101_135 {{ background: #f8e6a6; border-color: #d8a311; color: #473700; }}
    .tier-136_160 {{ background: #ece2c7; border-color: #b9ac77; color: #554741; }}
    .tier-161_200 {{ background: #e8edf2; border-color: #b8c4d2; color: #1c2b39; }}
    .tier-201_250 {{ background: #edf0f2; border-color: #c7cdd4; color: #354052; }}
    .tier-251_300 {{ background: #f2f2f2; border-color: #d4d4d4; color: #4d5962; }}
    .tier-301_plus {{ background: #fafafa; border-color: #d7dcde; color: #68727a; }}
    .good_target, .strong_target {{ background: #eff8fb; color: var(--wvu-blue); border-color: var(--wvu-sky); }}
    .avoid_bad_risk_reward, .avoid_unless_needed {{ background: var(--soft-red); color: var(--red); border-color: #ffd0d0; }}
    .monitor {{ background: var(--soft-gold); color: var(--gold); border-color: var(--wvu-gold); }}
    .low_value {{ color: var(--muted); }}
    .very_high, .high {{ color: var(--red); }}
    .medium {{ color: var(--gold); }}
    .low, .very_low {{ color: var(--green); }}
    .coach-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 220px;
      padding: 3px 8px;
      border-radius: 8px;
      border: 1px solid var(--line);
      font-weight: 700;
    }}
    .coach-chip .coach-name {{
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .coach-chip .coach-lift {{
      font-size: 11px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      opacity: 0.82;
    }}
    .coach-signal-very_high {{
      background: var(--soft-red);
      border-color: #ffd0d0;
      color: var(--red);
    }}
    .coach-signal-high {{
      background: var(--soft-gold);
      border-color: var(--wvu-gold);
      color: #5f4600;
    }}
    .coach-signal-neutral {{
      background: #f5f7fa;
      border-color: #dce4ec;
      color: var(--ink);
    }}
    .coach-signal-low {{
      background: #eff8fb;
      border-color: var(--wvu-sky);
      color: var(--wvu-blue);
    }}
    .coach-signal-unknown {{
      background: #f7f7f7;
      border-color: #d7dcde;
      color: var(--muted);
    }}
    .wab-swing {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      justify-content: flex-end;
      min-width: 98px;
      font-variant-numeric: tabular-nums;
    }}
    .wab-win {{ color: var(--green); font-weight: 700; }}
    .wab-loss {{ color: var(--red); font-weight: 700; }}
    .planner-shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      align-items: start;
      margin-top: 14px;
    }}
    .planner-actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0;
    }}
    .planner-actions button {{
      min-height: 38px;
      padding: 8px 12px;
    }}
    .planner-summary {{
      display: grid;
      gap: 10px;
    }}
    .planner-metric {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--wvu-gold);
      border-radius: 8px;
      background: #fff;
      padding: 13px;
    }}
    .planner-metric strong {{
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }}
    .planner-metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .planner-breakdown {{
      display: grid;
      gap: 8px;
      margin-top: 2px;
    }}
    .planner-breakdown-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      font-size: 14px;
    }}
    .planner-breakdown-row strong {{
      display: inline;
      font-size: 15px;
      margin: 0;
    }}
    .planner-breakdown-note {{
      margin-top: 6px;
      color: var(--text);
      font-size: 13px;
      line-height: 1.35;
    }}
    .planner-table table {{ min-width: 900px; }}
    .planner-table input, .planner-table select {{
      min-width: 145px;
    }}
    .planner-row-home td {{ background: #eff8fb; }}
    .planner-row-away td {{ background: #fff8df; }}
    .planner-row-neutral td {{ background: #f7f7f7; }}
    .planner-row-home:hover td,
    .planner-row-away:hover td,
    .planner-row-neutral:hover td {{ background: #eef4fb; }}
    .planner-table input[data-field="date"] {{ min-width: 96px; }}
    .remove-game {{
      min-height: 34px;
      padding: 6px 9px;
    }}
    .planner-note {{
      margin-top: 12px;
      font-size: 13px;
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 16px;
      margin-top: 22px;
    }}
    .section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(170px, 1fr) 3fr 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
    }}
    .bar-track {{
      height: 11px;
      border-radius: 8px;
      background: #e8edf2;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 8px;
      background: var(--wvu-blue-light);
    }}
    @media (max-width: 920px) {{
      header {{ padding: 18px; }}
      main {{ padding: 16px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .controls {{ grid-template-columns: 1fr; }}
      .planner-shell {{ grid-template-columns: 1fr; }}
      .grid-two {{ grid-template-columns: 1fr; }}
      .hero-mark-frame {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <h1>WVU Basketball Scheduling App</h1>
      </div>
      <div class="header-side">
        <span class="hero-mark-frame">
          <img class="hero-mark" alt="WVU Mountaineer" src="assets/mountaineer_face.png">
        </span>
        <div class="badge-stack">
          <span class="badge" id="updated"></span>
          <span class="badge subtle" id="recruitingFreshness"></span>
          <span class="badge subtle" id="transferFreshness"></span>
        </div>
      </div>
    </div>
  </header>
  <main>
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab-button active" type="button" data-tab="opponentBoard">Guarantee Game Target Board</button>
      <button class="tab-button" type="button" data-tab="schedulePlanner">WVU Schedule Planner</button>
    </nav>
    <section id="opponentBoard" class="tab-panel active">
      <div class="controls">
        <input id="search" type="search" placeholder="Search team or conference">
        <details class="filter" id="tierFilter">
          <summary data-label="Tiers">Tiers</summary>
          <div class="filter-menu" id="tierOptions"></div>
        </details>
        <details class="filter" id="riskFilter">
          <summary data-label="Risk Buckets">Risk Buckets</summary>
          <div class="filter-menu" id="riskOptions"></div>
        </details>
        <details class="filter" id="recommendationFilter">
          <summary data-label="Recommendations">Recommendations</summary>
          <div class="filter-menu" id="recommendationOptions"></div>
        </details>
        <details class="filter" id="conferenceFilter">
          <summary data-label="Conferences">Conferences</summary>
          <div class="filter-menu" id="conferenceOptions"></div>
        </details>
        <select id="sort">
          <option value="added_wab:desc">Added WAB high-low</option>
          <option value="upset_pct:desc">Highest upset risk</option>
          <option value="upset_pct:asc">Lowest upset risk</option>
          <option value="team:asc">Team A-Z</option>
          <option value="conference:asc">Conference A-Z</option>
          <option value="tier:asc">Tier A-Z</option>
          <option value="risk_sort:desc">Risk high-low</option>
          <option value="coach_lift:desc">Coach upset signal high-low</option>
          <option value="recommendation:asc">Recommendation A-Z</option>
          <option value="three_rate:desc">3PA rate high-low</option>
          <option value="experience:desc">Experience high-low</option>
          <option value="adj_em:desc">AdjEM high-low</option>
        </select>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th data-key="team">Team</th>
              <th data-key="coach_lift">Coach</th>
              <th data-key="conference">Conf</th>
              <th data-key="tier">Tier</th>
              <th data-key="upset_pct">Upset %</th>
              <th data-key="risk_sort">Risk</th>
              <th data-key="recommendation">Recommendation</th>
              <th data-key="added_wab">Added WAB</th>
              <th data-key="three_rate">3PA Rate</th>
              <th data-key="experience">Exp</th>
              <th data-key="adj_em">AdjEM</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
    <section id="schedulePlanner" class="tab-panel">
      <div class="planner-shell">
        <div>
          <div class="planner-actions">
            <button class="primary" type="button" id="addGame">Add Game</button>
            <button type="button" id="sampleSchedule">Load Defaults</button>
            <button class="danger" type="button" id="clearSchedule">Clear</button>
          </div>
          <div class="table-wrap planner-table">
            <table>
              <thead>
                <tr>
                  <th>2026 Date</th>
                  <th>Opponent</th>
                  <th>Location</th>
                  <th>Tier</th>
                  <th>WAB Win/Loss</th>
                  <th>WVU Win %</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="plannerRows"></tbody>
            </table>
          </div>
          <datalist id="opponentList"></datalist>
        </div>
        <aside class="planner-summary" aria-label="Schedule projection summary">
          <div class="planner-metric">
            <strong id="plannerGames">0</strong>
            <span>Non-conference games</span>
          </div>
          <div class="planner-metric">
            <strong id="plannerNcsosRank">—</strong>
            <span>Team sheet NCSOS rank</span>
          </div>
          <div class="planner-metric">
            <strong id="plannerWinPct">—</strong>
            <span>Predicted non-conference win %</span>
          </div>
          <div class="planner-metric">
            <strong id="plannerRecord">—</strong>
            <span>Expected non-conference record</span>
          </div>
          <div class="planner-metric">
            <strong id="plannerAvgOpp">—</strong>
            <span>Avg opponent difficulty</span>
          </div>
          <div class="planner-metric">
            <strong>Why It Lands There</strong>
            <div class="planner-breakdown">
              <div class="planner-breakdown-row">
                <span>Quality games</span>
                <strong id="plannerQualityBreakdown">—</strong>
              </div>
              <div class="planner-breakdown-row">
                <span>Buy-game drag</span>
                <strong id="plannerBuyBreakdown">—</strong>
              </div>
              <div class="planner-breakdown-row">
                <span>Site mix</span>
                <strong id="plannerSiteBreakdown">—</strong>
              </div>
            </div>
            <div class="planner-breakdown-note" id="plannerBreakdownNote">—</div>
          </div>
        </aside>
      </div>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const state = {{ sortKey: "added_wab", sortDir: "desc" }};
    const plannerState = {{ games: [] }};
    const plannerStorageKey = "wvu-nonconference-planner-v2";
    const winProbabilityScale = 6.5;
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 1 }});
    const fmt2 = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 2 }});
    const pctFmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 1 }});
    const tierOrder = ["top_25", "26_50", "51_75", "76_100", "101_135", "136_160", "161_200", "201_250", "251_300", "301_plus"];
    const riskOrder = ["very_high", "high", "medium", "low", "very_low"];
    const recommendationOrder = ["strong_target", "good_target", "monitor", "avoid_unless_needed", "avoid_bad_risk_reward", "low_value"];

    function text(value) {{ return value === null || value === undefined || value === "" ? "—" : value; }}
    function pctText(value) {{ return value === null || value === undefined || value === "" ? "—" : `${{text(value)}}%`; }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}
    function cls(value) {{ return String(value || "").replace(/[^a-z0-9_]+/gi, "_"); }}
    function normalizeTeam(value) {{ return String(value || "").trim().toLowerCase().replaceAll(".", ""); }}
    function tierLabel(value) {{
      return String(value || "—")
        .replace("top_25", "Top 25")
        .replace("301_plus", "301+")
        .replaceAll("_", "-");
    }}

    function coachLiftLabel(value) {{
      if (value === null || value === undefined || value === "") return "";
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "";
      return `${{numeric >= 0 ? "+" : ""}}${{numeric.toFixed(1)}}`;
    }}

    function coachTitle(row) {{
      const label = coachLiftLabel(row.coach_lift);
      if (!label) return "No coach upset signal available";
      return `Coach upset signal: ${{label}} percentage points vs a median candidate coach`;
    }}

    function coachChip(row) {{
      const coach = text(row.projected_coach);
      const lift = coachLiftLabel(row.coach_lift);
      const liftHtml = lift ? `<span class="coach-lift">${{lift}}</span>` : "";
      return `<span class="coach-chip coach-signal-${{cls(row.coach_signal)}}"
        title="${{escapeHtml(coachTitle(row))}}">
        <span class="coach-name">${{coach}}</span>${{liftHtml}}
      </span>`;
    }}

    function displayValue(value) {{
      const raw = String(value || "—");
      if (raw.includes("_") || raw.includes("top_") || raw.includes("plus")) return tierLabel(raw).replaceAll("-", " ");
      return raw;
    }}

    function plannerTeamIndex() {{
      return new Map((payload.planner?.teams || []).map(row => [normalizeTeam(row.team), row]));
    }}

    const teamsByName = plannerTeamIndex();

    function uniqueSorted(values) {{
      return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }}

    function orderedUnique(values, order = null) {{
      const unique = uniqueSorted(values);
      if (!order) return unique;
      const rank = new Map(order.map((value, index) => [value, index]));
      return unique.sort((a, b) => {{
        const left = rank.has(a) ? rank.get(a) : 999;
        const right = rank.has(b) ? rank.get(b) : 999;
        if (left !== right) return left - right;
        return String(a).localeCompare(String(b));
      }});
    }}

    function selectedValues(name) {{
      return new Set([...document.querySelectorAll(`input[name="${{name}}"]:checked`)].map(input => input.value));
    }}

    function updateFilterSummary(name, detailsId) {{
      const details = document.getElementById(detailsId);
      const summary = details.querySelector("summary");
      const label = summary.dataset.label;
      const count = selectedValues(name).size;
      summary.firstChild.nodeValue = count ? `${{label}} (${{count}})` : label;
    }}

    function populateFilter(name, optionsId, detailsId, values, formatter = displayValue, order = null) {{
      const container = document.getElementById(optionsId);
      container.innerHTML = orderedUnique(values, order).map(value => `
        <label class="filter-option">
          <input type="checkbox" name="${{name}}" value="${{value}}">
          <span>${{formatter(value)}}</span>
        </label>
      `).join("");
      container.querySelectorAll("input").forEach(input => {{
        input.addEventListener("change", () => {{
          updateFilterSummary(name, detailsId);
          renderRows();
        }});
      }});
      updateFilterSummary(name, detailsId);
    }}

    function initTimestamp() {{
      document.getElementById("updated").textContent = `Updated ${{new Date(payload.generated_at).toLocaleString()}}`;
      const hs = payload.input_status?.on3_hs_snapshot;
      const portal = payload.input_status?.on3_transfer_snapshot;
      const cbb = payload.input_status?.cbb_player_snapshot;
      const cbbTransferRows = Number(payload.input_status?.cbb_transfer_feature_rows || 0);
      const cbbTransferBuiltAt = payload.input_status?.cbb_transfer_feature_built_at;
      const cbbTransferLedgerAt = payload.input_status?.cbb_transfer_ledger_snapshot;
      const recruiting = document.getElementById("recruitingFreshness");
      const transfer = document.getElementById("transferFreshness");
      const dates = [hs, portal].filter(Boolean).sort();
      if (!dates.length) {{
        recruiting.textContent = "Recruiting data: cached";
      }} else {{
        recruiting.textContent = `Recruiting data: ${{dates[0]}}`;
      }}
      if (cbbTransferLedgerAt) {{
        transfer.textContent = `Transfer data: ${{cbbTransferLedgerAt}}`;
      }} else if (cbb) {{
        transfer.textContent = `Transfer data: ${{cbb}}`;
      }} else if (cbbTransferRows > 0 && cbbTransferBuiltAt) {{
        transfer.textContent = `Transfer data: cached ${{cbbTransferBuiltAt}}`;
      }} else {{
        transfer.textContent = "Transfer data: missing";
      }}
    }}

    function setSort(key, dir) {{
      state.sortKey = key;
      state.sortDir = dir;
      const select = document.getElementById("sort");
      const value = `${{key}}:${{dir}}`;
      if (![...select.options].some(option => option.value === value)) {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `${{key.replaceAll("_", " ")}} ${{dir === "asc" ? "low-high" : "high-low"}}`;
        select.appendChild(option);
      }}
      select.value = value;
      document.querySelectorAll("th[data-key]").forEach(th => {{
        th.classList.toggle("sorted", th.dataset.key === key);
        th.dataset.sortMark = th.dataset.key === key ? (dir === "asc" ? "↑" : "↓") : "";
      }});
    }}

    function filteredRows() {{
      const q = document.getElementById("search").value.trim().toLowerCase();
      const conferences = selectedValues("conference");
      const recommendations = selectedValues("recommendation");
      const tiers = selectedValues("tier");
      const risks = selectedValues("risk");
      const sortKey = state.sortKey;
      const sortDir = state.sortDir;
      return payload.risk_rows.filter(row => {{
        const haystack = `${{row.team}} ${{row.projected_coach}} ${{row.conference}}`.toLowerCase();
        return (!q || haystack.includes(q))
          && (!conferences.size || conferences.has(row.conference))
          && (!recommendations.size || recommendations.has(row.recommendation))
          && (!tiers.size || tiers.has(row.tier))
          && (!risks.size || risks.has(row.risk_bucket));
      }}).sort((a, b) => {{
        const left = typeof a[sortKey] === "number" ? a[sortKey] : String(a[sortKey] || "");
        const right = typeof b[sortKey] === "number" ? b[sortKey] : String(b[sortKey] || "");
        if (left < right) return sortDir === "asc" ? -1 : 1;
        if (left > right) return sortDir === "asc" ? 1 : -1;
        return String(a.team).localeCompare(String(b.team));
      }});
    }}

    function renderRows() {{
      const rows = filteredRows();
      document.getElementById("rows").innerHTML = rows.map(row => `
        <tr>
          <td><strong>${{text(row.team)}}</strong></td>
          <td>${{coachChip(row)}}</td>
          <td>${{text(row.conference)}}</td>
          <td><span class="pill tier-chip tier-${{cls(row.tier)}}">${{tierLabel(row.tier)}}</span></td>
          <td class="num"><strong>${{text(row.upset_pct)}}%</strong></td>
          <td><span class="${{cls(row.risk_bucket)}}">${{text(row.risk_bucket).replaceAll("_", " ")}}</span></td>
          <td><span class="pill ${{cls(row.recommendation)}}">${{text(row.recommendation).replaceAll("_", " ")}}</span></td>
          <td class="num">${{text(row.added_wab)}}</td>
          <td class="num">${{text(row.three_rate)}}</td>
          <td class="num">${{text(row.experience)}}</td>
          <td class="num">${{text(row.adj_em)}}</td>
        </tr>
      `).join("");
    }}

    function setActiveTab(tabId) {{
      document.querySelectorAll(".tab-button").forEach(button => {{
        button.classList.toggle("active", button.dataset.tab === tabId);
      }});
      document.querySelectorAll(".tab-panel").forEach(panel => {{
        panel.classList.toggle("active", panel.id === tabId);
      }});
    }}

    function blankGame() {{
      return {{
        id: `${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`,
        date: "",
        opponent: "",
        location: "Home"
      }};
    }}

    function defaultGames() {{
      return [
        {{ id: "default-1", date: "11/2", opponent: "Niagara", location: "Home" }},
        {{ id: "default-2", date: "11/12", opponent: "Pittsburgh", location: "Away" }},
        {{ id: "default-3", date: "11/17", opponent: "Placeholder: 51-75 Team", location: "Neutral" }},
        {{ id: "default-4", date: "11/18", opponent: "Placeholder: 26-50 Team", location: "Neutral" }},
        {{ id: "default-5", date: "11/19", opponent: "Placeholder: 26-50 Team", location: "Neutral" }},
        {{ id: "default-6", date: "12/1", opponent: "Mercyhurst", location: "Home" }},
        {{ id: "default-7", date: "12/5", opponent: "Virginia Tech", location: "Home" }},
        {{ id: "default-8", date: "12/13", opponent: "Coppin St.", location: "Home" }},
        {{ id: "default-9", date: "12/19", opponent: "Wake Forest", location: "Neutral" }}
      ];
    }}

    function savePlanner() {{
      localStorage.setItem(plannerStorageKey, JSON.stringify(plannerState.games));
    }}

    function loadPlanner() {{
      try {{
        const saved = JSON.parse(localStorage.getItem(plannerStorageKey) || "[]");
        plannerState.games = Array.isArray(saved) && saved.length ? saved : defaultGames();
      }} catch {{
        plannerState.games = defaultGames();
      }}
    }}

    function dateSortValue(value) {{
      const match = String(value || "").trim().match(/^(\\d{{1,2}})\\s*\\/\\s*(\\d{{1,2}})$/);
      if (!match) return 9999;
      const month = Number(match[1]);
      const day = Number(match[2]);
      if (!Number.isInteger(month) || !Number.isInteger(day) || month < 1 || month > 12 || day < 1 || day > 31) return 9999;
      return month * 100 + day;
    }}

    function sortPlannerGames() {{
      plannerState.games.sort((a, b) => {{
        const dateDiff = dateSortValue(a.date) - dateSortValue(b.date);
        if (dateDiff) return dateDiff;
        return String(a.opponent || "").localeCompare(String(b.opponent || ""));
      }});
    }}

    function opponentProjection(name) {{
      return teamsByName.get(normalizeTeam(name));
    }}

    function projectedWinProbability(opponent, location) {{
      const host = payload.planner?.host_projection;
      const hostAdj = Number(host?.projected_adj_em);
      const oppAdj = Number(opponent?.projected_adj_em);
      if (!Number.isFinite(hostAdj) || !Number.isFinite(oppAdj)) return null;
      const locationAdjustment = location === "Home" ? 3.5 : location === "Away" ? -3.5 : 0;
      return 1 / (1 + Math.exp(-((hostAdj + locationAdjustment - oppAdj) / winProbabilityScale)));
    }}

    function bubbleAdjEm() {{
      const value = Number(payload.planner?.bubble_adj_em);
      return Number.isFinite(value) ? value : 16;
    }}

    function bubbleWinProbability(opponent, location) {{
      const oppAdj = Number(opponent?.wab_adj_em ?? opponent?.projected_adj_em);
      if (!Number.isFinite(oppAdj)) return null;
      const locationAdjustment = location === "Home" ? 3.5 : location === "Away" ? -3.5 : 0;
      return 1 / (1 + Math.exp(-((bubbleAdjEm() + locationAdjustment - oppAdj) / winProbabilityScale)));
    }}

    function wabSwing(opponent, location) {{
      const bubbleWinProb = bubbleWinProbability(opponent, location);
      if (bubbleWinProb === null) return null;
      return {{
        win: 1 - bubbleWinProb,
        loss: -bubbleWinProb
      }};
    }}

    function wabSwingHtml(opponent, location) {{
      const swing = wabSwing(opponent, location);
      if (!swing) return "—";
      return `<span class="wab-swing"><span class="wab-win">+${{fmt2.format(swing.win)}}</span><span class="wab-loss">${{fmt2.format(swing.loss)}}</span></span>`;
    }}

    function ncsosRank(avgOpponentAdjEm, gameCounts) {{
      const calibration = payload.planner?.ncsos_calibration;
      if (
        calibration &&
        Array.isArray(calibration.coefficients) &&
        calibration.coefficients.length >= 4 &&
        Number.isFinite(avgOpponentAdjEm) &&
        gameCounts &&
        Number.isFinite(gameCounts.total) &&
        gameCounts.total > 0
      ) {{
        const coefficients = calibration.coefficients.map(Number);
        const homeShare = Number(gameCounts.home || 0) / gameCounts.total;
        const neutralShare = Number(gameCounts.neutral || 0) / gameCounts.total;
        const estimate =
          coefficients[0]
          + coefficients[1] * avgOpponentAdjEm
          + coefficients[2] * homeShare
          + coefficients[3] * neutralShare;
        if (Number.isFinite(estimate)) {{
          return Math.max(1, Math.min(365, Math.round(estimate)));
        }}
      }}
      const benchmarks = payload.planner?.ncsos_benchmarks || [];
      if (!benchmarks.length || !Number.isFinite(avgOpponentAdjEm)) return null;
      return 1 + benchmarks.filter(value => Number(value) > avgOpponentAdjEm).length;
    }}

    function ncsosBaseAdjEm(opponent) {{
      const raw = Number(opponent?.projected_adj_em);
      if (Number.isFinite(raw)) return raw;
      return null;
    }}

    function ncsosOpponentAdjEm(opponent, location) {{
      const raw = ncsosBaseAdjEm(opponent);
      if (!Number.isFinite(raw)) return null;
      const locationAdjustment = location === "Home" ? -3.5 : location === "Away" ? 3.5 : 0;
      return raw + locationAdjustment;
    }}

    function plannerProjectedRank(opponent) {{
      const value = Number(opponent?.projected_net_rank ?? opponent?.schedule_score);
      return Number.isFinite(value) ? value : null;
    }}

    function plannerBreakdown(validGames, gameCounts) {{
      const summary = {{
        top100: 0,
        top200: 0,
        buy251: 0,
        buy301: 0
      }};
      validGames.forEach(item => {{
        const rank = plannerProjectedRank(item.opponent);
        if (!Number.isFinite(rank)) return;
        if (rank <= 100) summary.top100 += 1;
        if (rank <= 200) summary.top200 += 1;
        if (rank > 250) summary.buy251 += 1;
        if (rank > 300) summary.buy301 += 1;
      }});

      const notes = [];
      if (summary.top100 >= 4) {{
        notes.push(`The top end is helping with ${{summary.top100}} top-100 caliber games.`);
      }} else if (summary.top100 <= 2) {{
        notes.push(`There are only ${{summary.top100}} top-100 caliber games carrying the sheet.`);
      }}

      if (summary.buy251 >= 4) {{
        notes.push(`${{summary.buy251}} games outside the top 250 are dragging the NCSOS rank down.`);
      }} else if (summary.buy301 >= 3) {{
        notes.push(`${{summary.buy301}} 301+ buy games are a meaningful drag.`);
      }}

      if (gameCounts.home >= 8) {{
        notes.push(`The slate is home-heavy at ${{gameCounts.home}} home games, which makes the NCAA-style difficulty read harsher.`);
      }} else if (gameCounts.neutral >= 4) {{
        notes.push(`${{gameCounts.neutral}} neutral-floor games are helping the schedule profile hold up.`);
      }}

      if (!notes.length) {{
        notes.push("The top-end opportunities and the lower-end buys are pulling against each other.");
      }}

      return {{
        quality: `${{summary.top100}} top-100 | ${{summary.top200}} top-200`,
        buy: `${{summary.buy251}} outside top 250 | ${{summary.buy301}} at 301+`,
        site: `${{gameCounts.home}} home | ${{gameCounts.neutral}} neutral | ${{gameCounts.away}} away`,
        note: notes.join(" ")
      }};
    }}

    function renderPlannerOptions() {{
      document.getElementById("opponentList").innerHTML = (payload.planner?.teams || [])
        .filter(row => normalizeTeam(row.team) !== normalizeTeam(payload.planner?.host))
        .map(row => {{
          const label = row.is_placeholder
            ? `${{row.team}} (${{tierLabel(row.tier)}})`
            : `${{row.team}} (${{row.conference || "—"}}, ${{tierLabel(row.tier)}})`;
          return `<option value="${{escapeHtml(row.team)}}" label="${{escapeHtml(label)}}">${{escapeHtml(label)}}</option>`;
        }})
        .join("");
    }}

    function renderPlannerRows() {{
      sortPlannerGames();
      const rows = plannerState.games;
      document.getElementById("plannerRows").innerHTML = rows.map(game => {{
        const opponent = opponentProjection(game.opponent);
        const winProb = projectedWinProbability(opponent, game.location);
        return `<tr class="planner-row-${{cls(game.location).toLowerCase()}}" data-id="${{escapeHtml(game.id)}}">
          <td><input type="text" data-field="date" inputmode="numeric" placeholder="MM/DD" value="${{escapeHtml(game.date)}}"></td>
          <td><input type="text" data-field="opponent" list="opponentList" placeholder="Opponent" value="${{escapeHtml(game.opponent)}}"></td>
          <td>
            <select data-field="location">
              ${{["Home", "Neutral", "Away"].map(location => `<option value="${{location}}" ${{game.location === location ? "selected" : ""}}>${{location}}</option>`).join("")}}
            </select>
          </td>
          <td>${{opponent ? `<span class="pill tier-chip tier-${{cls(opponent.tier)}}">${{tierLabel(opponent.tier)}}</span>` : "—"}}</td>
          <td class="num">${{wabSwingHtml(opponent, game.location)}}</td>
          <td class="num"><strong>${{winProb === null ? "—" : `${{pctFmt.format(winProb * 100)}}%`}}</strong></td>
          <td><button class="danger remove-game" type="button">Remove</button></td>
        </tr>`;
      }}).join("");

      document.querySelectorAll("#plannerRows tr").forEach(tr => {{
        const id = tr.dataset.id;
        tr.querySelectorAll("input, select").forEach(input => {{
          input.addEventListener("change", () => {{
            const game = plannerState.games.find(item => item.id === id);
            if (game) {{
              game[input.dataset.field] = input.value;
              savePlanner();
              renderPlannerRows();
              renderPlannerSummary();
            }}
          }});
        }});
        tr.querySelector(".remove-game").addEventListener("click", () => {{
          plannerState.games = plannerState.games.filter(item => item.id !== id);
          savePlanner();
          renderPlannerRows();
          renderPlannerSummary();
        }});
      }});
      renderPlannerSummary();
    }}

    function renderPlannerSummary() {{
      const validGames = plannerState.games
        .map(game => ({{ game, opponent: opponentProjection(game.opponent) }}))
        .filter(item => item.opponent);
      const winProbs = validGames
        .map(item => projectedWinProbability(item.opponent, item.game.location))
        .filter(value => value !== null);
      const oppAdjEms = validGames
        .map(item => ncsosOpponentAdjEm(item.opponent, item.game.location))
        .filter(value => value !== null);
      const gameCounts = validGames.reduce((counts, item) => {{
        counts.total += 1;
        if (item.game.location === "Home") counts.home += 1;
        else if (item.game.location === "Neutral") counts.neutral += 1;
        else if (item.game.location === "Away") counts.away += 1;
        return counts;
      }}, {{ total: 0, home: 0, neutral: 0, away: 0 }});

      if (!validGames.length || !oppAdjEms.length) {{
        document.getElementById("plannerGames").textContent = "0";
        document.getElementById("plannerNcsosRank").textContent = "—";
        document.getElementById("plannerWinPct").textContent = "—";
        document.getElementById("plannerRecord").textContent = "—";
        document.getElementById("plannerAvgOpp").textContent = "—";
        document.getElementById("plannerQualityBreakdown").textContent = "—";
        document.getElementById("plannerBuyBreakdown").textContent = "—";
        document.getElementById("plannerSiteBreakdown").textContent = "—";
        document.getElementById("plannerBreakdownNote").textContent = "—";
        return;
      }}

      const avgOppAdjEm = oppAdjEms.reduce((sum, value) => sum + value, 0) / oppAdjEms.length;
      const rank = ncsosRank(avgOppAdjEm, gameCounts);
      const expectedWins = winProbs.reduce((sum, value) => sum + value, 0);
      const winPct = winProbs.length ? expectedWins / winProbs.length : null;
      const breakdown = plannerBreakdown(validGames, gameCounts);

      document.getElementById("plannerGames").textContent = String(validGames.length);
      document.getElementById("plannerNcsosRank").textContent = rank ? `#${{rank}}` : "—";
      document.getElementById("plannerWinPct").textContent = winPct === null ? "—" : `${{pctFmt.format(winPct * 100)}}%`;
      document.getElementById("plannerRecord").textContent = `${{fmt2.format(expectedWins)}}-${{fmt2.format(validGames.length - expectedWins)}}`;
      document.getElementById("plannerAvgOpp").textContent = `${{fmt2.format(avgOppAdjEm)}} AdjEM`;
      document.getElementById("plannerQualityBreakdown").textContent = breakdown.quality;
      document.getElementById("plannerBuyBreakdown").textContent = breakdown.buy;
      document.getElementById("plannerSiteBreakdown").textContent = breakdown.site;
      document.getElementById("plannerBreakdownNote").textContent = breakdown.note;
    }}

    function renderBars(id, rows, key, labelKey, valueFormatter) {{
      const max = Math.max(...rows.map(row => Math.abs(Number(row[key]) || 0)), 0.001);
      document.getElementById(id).innerHTML = rows.map(row => {{
        const value = Number(row[key]) || 0;
        const width = Math.max(3, Math.abs(value) / max * 100);
        return `<div class="bar-row">
          <div>${{text(row[labelKey]).replaceAll("_", " ")}}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${{width}}%"></div></div>
          <div class="num">${{valueFormatter(value)}}</div>
        </div>`;
      }}).join("");
    }}

    function renderModelSections() {{
      renderBars("coefficients", payload.coefficients, "abs_coefficient", "feature", value => fmt2.format(value));
      const metricRows = payload.metrics.map(row => ({{
        label: row.season,
        auc: Number(row.auc || 0),
        top10: Number(row.upset_rate_top_10_pct_risk || 0)
      }}));
      document.getElementById("metrics").innerHTML = metricRows.map(row => `
        <div class="bar-row">
          <div>${{row.label}}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(3, row.auc * 100)}}%"></div></div>
          <div class="num">${{fmt2.format(row.auc)}}</div>
        </div>
      `).join("");
    }}

    function init() {{
      initTimestamp();
      document.querySelectorAll(".tab-button").forEach(button => {{
        button.addEventListener("click", () => setActiveTab(button.dataset.tab));
      }});
      populateFilter("tier", "tierOptions", "tierFilter", payload.risk_rows.map(row => row.tier), tierLabel, tierOrder);
      populateFilter("risk", "riskOptions", "riskFilter", payload.risk_rows.map(row => row.risk_bucket), displayValue, riskOrder);
      populateFilter("recommendation", "recommendationOptions", "recommendationFilter", payload.risk_rows.map(row => row.recommendation), displayValue, recommendationOrder);
      populateFilter("conference", "conferenceOptions", "conferenceFilter", payload.risk_rows.map(row => row.conference), value => value);
      ["search", "sort"].forEach(id => {{
        document.getElementById(id).addEventListener("input", () => {{
          if (id === "sort") {{
            const [key, dir] = document.getElementById("sort").value.split(":");
            setSort(key, dir);
          }}
          renderRows();
        }});
      }});
      document.querySelectorAll("th[data-key]").forEach(th => {{
        th.addEventListener("click", () => {{
          const key = th.dataset.key;
          const dir = state.sortKey === key && state.sortDir === "desc" ? "asc" : "desc";
          setSort(key, dir);
          renderRows();
        }});
      }});
      setSort(state.sortKey, state.sortDir);
      renderRows();
      renderPlannerOptions();
      loadPlanner();
      document.getElementById("addGame").addEventListener("click", () => {{
        plannerState.games.push(blankGame());
        savePlanner();
        renderPlannerRows();
      }});
      document.getElementById("sampleSchedule").addEventListener("click", () => {{
        plannerState.games = defaultGames();
        savePlanner();
        renderPlannerRows();
      }});
      document.getElementById("clearSchedule").addEventListener("click", () => {{
        plannerState.games = [blankGame()];
        savePlanner();
        renderPlannerRows();
      }});
      renderPlannerRows();
    }}
    init();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-board-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "upset_risk" / "current_2027_guarantee_risk_board.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "upset_risk" / "rolling_metrics.csv",
    )
    parser.add_argument(
        "--coefficients-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "upset_risk" / "model_coefficients.csv",
    )
    parser.add_argument(
        "--schedule-predictions-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "predictions"
        / "current_2027_schedule_predictions.csv",
    )
    parser.add_argument(
        "--kenpom-ratings-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kenpom" / "2026" / "ratings.json",
        help="Latest KenPom ratings file used to map projected rank to AdjEM and NCSOS benchmarks.",
    )
    parser.add_argument(
        "--planner-host",
        default="West Virginia",
        help="Hardcoded schedule planner team.",
    )
    parser.add_argument(
        "--planner-model",
        default="direct_ridge_schedule_building",
        help="Current-season projection model used by the schedule planner.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dashboard",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = write_json(payload, args.output_dir / "dashboard_payload.json")
    html_path = args.output_dir / "index.html"
    html_path.write_text(dashboard_html(payload), encoding="utf-8")
    print(f"saved {html_path}")
    print(f"saved {data_path}")
    print(f"candidates: {payload['summary']['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
