"""Rolling-season model backtests for preseason NET prediction."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_BANDS = (25, 50, 75, 100, 135, 160, 200, 250, 300)
EXCLUSIVE_BAND_LABELS = {
    25: "top_25",
    50: "26_50",
    75: "51_75",
    100: "76_100",
    135: "101_135",
    160: "136_160",
    200: "161_200",
    250: "201_250",
    300: "251_300",
}
EXCLUSIVE_BAND_ORDER = (
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
)

BASE_COLUMNS = {"season", "team", "team_key", "conference"}
TARGET_PREFIXES = ("target_",)
LEAKY_PREFIXES = ("coach_observed_",)
NON_FEATURE_SUBSTRINGS = ("source_file", "source_url", "captured_at", "ranking_date_updated")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_json(rows: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "true":
            return 1.0
        if stripped == "false":
            return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def feature_allowed(column: str) -> bool:
    if column in BASE_COLUMNS or column == "season":
        return False
    if column.startswith(TARGET_PREFIXES) or column.startswith(LEAKY_PREFIXES):
        return False
    if any(part in column for part in NON_FEATURE_SUBSTRINGS):
        return False
    if column.endswith("_id") or column.endswith("_key"):
        return False
    if column in {"coach_coach", "coach_coach_prior_last_team_name", "coach_coach_prior_last_conference"}:
        return False
    return True


def numeric_feature_columns(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    columns = rows[0].keys()
    features: list[str] = []
    for column in columns:
        if not feature_allowed(column):
            continue
        if prefixes and not column.startswith(prefixes):
            continue
        if any(as_float(row.get(column)) is not None for row in rows):
            features.append(column)
    return features


def select_feature_columns(
    rows: list[dict[str, Any]],
    columns: list[str],
    target_values: list[float],
    max_features: int | None,
) -> list[str]:
    if max_features is None or len(columns) <= max_features:
        return columns

    target_mean = sum(target_values) / len(target_values) if target_values else 0.0
    target_variance = sum((value - target_mean) ** 2 for value in target_values)
    if target_variance <= 0:
        return columns[:max_features]

    scores: list[tuple[float, str]] = []
    for column in columns:
        observed = [value for row in rows if (value := as_float(row.get(column))) is not None]
        med = median(observed)
        values = [as_float(row.get(column)) if as_float(row.get(column)) is not None else med for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values)
        if variance <= 0:
            continue
        covariance = sum((value - mean) * (target - target_mean) for value, target in zip(values, target_values))
        score = abs(covariance / math.sqrt(variance * target_variance))
        scores.append((score, column))

    selected = [column for _, column in sorted(scores, reverse=True)[:max_features]]
    return selected or columns[:max_features]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


@dataclass
class FeatureSpec:
    columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    conferences: list[str]


def fit_feature_spec(rows: list[dict[str, Any]], columns: list[str]) -> FeatureSpec:
    medians = {}
    means = {}
    scales = {}
    for column in columns:
        observed = [value for row in rows if (value := as_float(row.get(column))) is not None]
        med = median(observed)
        filled = [as_float(row.get(column)) if as_float(row.get(column)) is not None else med for row in rows]
        avg = sum(filled) / len(filled) if filled else 0.0
        variance = sum((value - avg) ** 2 for value in filled) / len(filled) if filled else 0.0
        medians[column] = med
        means[column] = avg
        scales[column] = math.sqrt(variance) or 1.0

    conferences = sorted({str(row.get("conference") or "") for row in rows if row.get("conference")})
    return FeatureSpec(columns=columns, medians=medians, means=means, scales=scales, conferences=conferences)


def transform_row(row: dict[str, Any], spec: FeatureSpec) -> list[float]:
    values = [1.0]
    for column in spec.columns:
        raw_value = as_float(row.get(column))
        missing = raw_value is None
        value = spec.medians[column] if missing else raw_value
        values.append((value - spec.means[column]) / spec.scales[column])
        values.append(1.0 if missing else 0.0)

    conference = str(row.get("conference") or "")
    for known_conference in spec.conferences:
        values.append(1.0 if conference == known_conference else 0.0)
    return values


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


@dataclass
class RidgeModel:
    spec: FeatureSpec
    weights: list[float]

    def predict(self, row: dict[str, Any]) -> float:
        x = transform_row(row, self.spec)
        return sum(weight * value for weight, value in zip(self.weights, x))


def fit_ridge(
    rows: list[dict[str, Any]],
    columns: list[str],
    target_values: list[float],
    alpha: float,
) -> RidgeModel:
    spec = fit_feature_spec(rows, columns)
    design = [transform_row(row, spec) for row in rows]
    width = len(design[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for x, y in zip(design, target_values):
        for i, xi in enumerate(x):
            xty[i] += xi * y
            for j, xj in enumerate(x):
                xtx[i][j] += xi * xj

    for i in range(1, width):
        xtx[i][i] += alpha

    return RidgeModel(spec=spec, weights=solve_linear_system(xtx, xty))


@dataclass
class TreeFeatureSpec:
    columns: list[str]
    medians: dict[str, float]


@dataclass
class TreeNode:
    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict_values(self, values: list[float]) -> float:
        if self.feature_index is None or self.threshold is None or self.left is None or self.right is None:
            return self.value
        if values[self.feature_index] <= self.threshold:
            return self.left.predict_values(values)
        return self.right.predict_values(values)


@dataclass
class BoostedTreesModel:
    spec: TreeFeatureSpec
    base_value: float
    trees: list[TreeNode]
    learning_rate: float

    def predict(self, row: dict[str, Any]) -> float:
        values = transform_tree_row(row, self.spec)
        return self.base_value + self.learning_rate * sum(tree.predict_values(values) for tree in self.trees)


def fit_tree_feature_spec(rows: list[dict[str, Any]], columns: list[str]) -> TreeFeatureSpec:
    medians = {}
    for column in columns:
        observed = [value for row in rows if (value := as_float(row.get(column))) is not None]
        medians[column] = median(observed)
    return TreeFeatureSpec(columns=columns, medians=medians)


def transform_tree_row(row: dict[str, Any], spec: TreeFeatureSpec) -> list[float]:
    values = []
    for column in spec.columns:
        value = as_float(row.get(column))
        values.append(spec.medians[column] if value is None else value)
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sse_from_sums(total: float, total_sq: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total_sq - (total * total / count)


def candidate_thresholds(values: list[float], bins: int) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return []
    if len(unique) <= bins + 1:
        return [(left + right) / 2 for left, right in zip(unique, unique[1:])]
    thresholds = []
    for index in range(1, bins + 1):
        position = int(index * (len(unique) - 1) / (bins + 1))
        thresholds.append(unique[position])
    return sorted(set(thresholds))


def fit_regression_tree(
    x_rows: list[list[float]],
    y_values: list[float],
    row_indices: list[int],
    *,
    depth_remaining: int,
    min_leaf: int,
    threshold_bins: int,
) -> TreeNode:
    node_values = [y_values[index] for index in row_indices]
    node_mean = mean(node_values)
    if depth_remaining <= 0 or len(row_indices) < min_leaf * 2:
        return TreeNode(value=node_mean)

    total = sum(node_values)
    total_sq = sum(value * value for value in node_values)
    base_sse = sse_from_sums(total, total_sq, len(row_indices))
    best_gain = 0.0
    best_feature: int | None = None
    best_threshold: float | None = None
    best_left: list[int] = []
    best_right: list[int] = []
    width = len(x_rows[0]) if x_rows else 0

    for feature_index in range(width):
        thresholds = candidate_thresholds([x_rows[index][feature_index] for index in row_indices], threshold_bins)
        for threshold in thresholds:
            left: list[int] = []
            right: list[int] = []
            left_sum = 0.0
            left_sq = 0.0
            right_sum = 0.0
            right_sq = 0.0
            for row_index in row_indices:
                value = y_values[row_index]
                if x_rows[row_index][feature_index] <= threshold:
                    left.append(row_index)
                    left_sum += value
                    left_sq += value * value
                else:
                    right.append(row_index)
                    right_sum += value
                    right_sq += value * value
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            split_sse = sse_from_sums(left_sum, left_sq, len(left)) + sse_from_sums(
                right_sum,
                right_sq,
                len(right),
            )
            gain = base_sse - split_sse
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
                best_left = left
                best_right = right

    if best_feature is None or best_threshold is None:
        return TreeNode(value=node_mean)

    return TreeNode(
        value=node_mean,
        feature_index=best_feature,
        threshold=best_threshold,
        left=fit_regression_tree(
            x_rows,
            y_values,
            best_left,
            depth_remaining=depth_remaining - 1,
            min_leaf=min_leaf,
            threshold_bins=threshold_bins,
        ),
        right=fit_regression_tree(
            x_rows,
            y_values,
            best_right,
            depth_remaining=depth_remaining - 1,
            min_leaf=min_leaf,
            threshold_bins=threshold_bins,
        ),
    )


def fit_boosted_trees(
    rows: list[dict[str, Any]],
    columns: list[str],
    target_values: list[float],
    *,
    estimators: int,
    learning_rate: float,
    max_depth: int,
    min_leaf: int,
    threshold_bins: int,
) -> BoostedTreesModel:
    spec = fit_tree_feature_spec(rows, columns)
    x_rows = [transform_tree_row(row, spec) for row in rows]
    base_value = mean(target_values)
    predictions = [base_value for _ in target_values]
    trees: list[TreeNode] = []
    row_indices = list(range(len(rows)))

    for _ in range(estimators):
        residuals = [target - prediction for target, prediction in zip(target_values, predictions)]
        tree = fit_regression_tree(
            x_rows,
            residuals,
            row_indices,
            depth_remaining=max_depth,
            min_leaf=min_leaf,
            threshold_bins=threshold_bins,
        )
        trees.append(tree)
        for index, values in enumerate(x_rows):
            predictions[index] += learning_rate * tree.predict_values(values)

    return BoostedTreesModel(
        spec=spec,
        base_value=base_value,
        trees=trees,
        learning_rate=learning_rate,
    )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    prefixes: tuple[str, ...]
    mode: str = "residual"
    algorithm: str = "ridge"
    alpha: float | None = None
    max_features: int | None = None
    excluded_substrings: tuple[str, ...] = ()
    forced_features: tuple[str, ...] = ()
    estimators: int = 24
    learning_rate: float = 0.06
    max_depth: int = 2
    min_leaf: int = 35
    threshold_bins: int = 6


def target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if as_float(row.get("target_net_rank")) is not None
        and as_float(row.get("target_net_percentile")) is not None
    ]


def config_feature_columns(rows: list[dict[str, Any]], config: ModelConfig) -> list[str]:
    columns = numeric_feature_columns(rows, config.prefixes)
    if not config.excluded_substrings:
        return columns
    return [
        column
        for column in columns
        if not any(excluded in column for excluded in config.excluded_substrings)
    ]


def with_forced_features(
    selected_columns: list[str],
    candidate_columns: list[str],
    forced_features: tuple[str, ...],
) -> list[str]:
    if not forced_features:
        return selected_columns
    output = list(selected_columns)
    for feature in forced_features:
        if feature in candidate_columns and feature not in output:
            output.append(feature)
    return output


def rank_from_percentile(percentile: float, teams_ranked: float) -> float:
    return 1 + (1 - percentile) * max(teams_ranked - 1, 1)


def percentile_from_rank(rank: float, teams_ranked: float) -> float:
    return 1 - ((rank - 1) / max(teams_ranked - 1, 1))


def prediction_row(
    row: dict[str, Any],
    *,
    model_name: str,
    predicted_rank: float,
    predicted_percentile: float,
    feature_count: int,
    train_rows: int,
) -> dict[str, Any]:
    actual_rank = as_float(row["target_net_rank"]) or 0.0
    teams_ranked = as_float(row["target_teams_ranked"]) or 0.0
    output = {
        "model": model_name,
        "season": row["season"],
        "team": row["team"],
        "team_key": row["team_key"],
        "conference": row["conference"],
        "actual_net_rank": actual_rank,
        "actual_net_percentile": as_float(row["target_net_percentile"]),
        "teams_ranked": teams_ranked,
        "predicted_net_rank": predicted_rank,
        "predicted_net_percentile": predicted_percentile,
        "absolute_rank_error": abs(predicted_rank - actual_rank),
        "feature_count": feature_count,
        "train_rows": train_rows,
    }
    for band in TARGET_BANDS:
        output[f"actual_top_{band}"] = actual_rank <= band
        output[f"predicted_top_{band}"] = predicted_rank <= band
    output["actual_schedule_band"] = exclusive_band_from_rank(actual_rank)
    output["predicted_schedule_band"] = exclusive_band_from_rank(predicted_rank)
    output["program_consistency_band"] = program_consistency_band(row)
    output["actual_below_200"] = actual_rank > 200
    output["predicted_below_200"] = predicted_rank > 200 if teams_ranked else None
    return output


def blended_predictions(
    predictions: list[dict[str, Any]],
    blends: dict[str, list[str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault((str(row["season"]), str(row["team_key"])), {})[str(row["model"])] = row

    outputs: list[dict[str, Any]] = []
    for models_by_team in grouped.values():
        for blend_name, component_models in blends.items():
            if not all(model in models_by_team for model in component_models):
                continue
            components = [models_by_team[model] for model in component_models]
            template = components[0]
            predicted_percentile = sum(
                as_float(row["predicted_net_percentile"]) or 0.0 for row in components
            ) / len(components)
            teams_ranked = as_float(template.get("teams_ranked")) or 0.0
            actual_rank = as_float(template["actual_net_rank"]) or 0.0
            predicted_rank = rank_from_percentile(predicted_percentile, teams_ranked)
            output = dict(template)
            output["model"] = blend_name
            output["predicted_net_rank"] = predicted_rank
            output["predicted_net_percentile"] = predicted_percentile
            output["absolute_rank_error"] = abs(predicted_rank - actual_rank)
            output["feature_count"] = sum(int(as_float(row.get("feature_count")) or 0) for row in components)
            output["train_rows"] = min(int(as_float(row.get("train_rows")) or 0) for row in components)
            for band in TARGET_BANDS:
                output[f"predicted_top_{band}"] = predicted_rank <= band
            output["predicted_schedule_band"] = exclusive_band_from_rank(predicted_rank)
            output["predicted_below_200"] = predicted_rank > 200 if teams_ranked else None
            outputs.append(output)

    return outputs


def band_confusion_counts(
    rows: list[dict[str, Any]],
    *,
    band: int,
    prediction_prefix: str,
) -> dict[str, int]:
    actual_key = f"actual_top_{band}"
    predicted_key = f"{prediction_prefix}_top_{band}"
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in rows:
        actual_positive = bool_value(row[actual_key])
        predicted_positive = bool_value(row[predicted_key])
        if actual_positive and predicted_positive:
            true_positive += 1
        elif not actual_positive and predicted_positive:
            false_positive += 1
        elif actual_positive and not predicted_positive:
            false_negative += 1
        else:
            true_negative += 1

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def band_scores(counts: dict[str, int]) -> dict[str, float | None]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    true_negative = counts["true_negative"]
    false_negative = counts["false_negative"]
    total = true_positive + false_positive + true_negative + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "accuracy": (true_positive + true_negative) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def exclusive_band_from_rank(rank: float | None) -> str:
    if rank is None:
        return "unknown"
    for band in TARGET_BANDS:
        if rank <= band:
            return EXCLUSIVE_BAND_LABELS[band]
    return "301_plus"


def exclusive_band_from_flags(row: dict[str, Any], prefix: str) -> str:
    for band in TARGET_BANDS:
        if bool_value(row.get(f"{prefix}_top_{band}")):
            return EXCLUSIVE_BAND_LABELS[band]
    return "301_plus"


def stronger_band(left: str | None, right: str | None) -> str:
    left = left or "301_plus"
    right = right or "301_plus"
    left_index = EXCLUSIVE_BAND_ORDER.index(left) if left in EXCLUSIVE_BAND_ORDER else len(EXCLUSIVE_BAND_ORDER) - 1
    right_index = EXCLUSIVE_BAND_ORDER.index(right) if right in EXCLUSIVE_BAND_ORDER else len(EXCLUSIVE_BAND_ORDER) - 1
    return EXCLUSIVE_BAND_ORDER[min(left_index, right_index)]


def program_consistency_band(row: dict[str, Any]) -> str:
    last_rank = as_float(row.get("program_prior_last_rank_adj_em"))
    top25_rate = as_float(row.get("program_prior_top25_rate")) or 0.0
    top50_rate = as_float(row.get("program_prior_top50_rate")) or 0.0
    top75_rate = as_float(row.get("program_prior_top75_rate")) or 0.0
    top100_rate = as_float(row.get("program_prior_top100_rate")) or 0.0
    top135_rate = as_float(row.get("program_prior_top135_rate")) or 0.0
    last3_adj_em = as_float(row.get("program_prior_last3_avg_adj_em"))

    if last_rank is None:
        return "301_plus"
    if last_rank <= 25 and (top25_rate >= 0.50 or (last3_adj_em is not None and last3_adj_em >= 25)):
        return "top_25"
    if last_rank <= 50 and (top50_rate >= 0.50 or (last3_adj_em is not None and last3_adj_em >= 18)):
        return "26_50"
    if last_rank <= 75 and top75_rate >= 0.40:
        return "51_75"
    if last_rank <= 100 and top100_rate >= 0.40:
        return "76_100"
    if last_rank <= 135 and top135_rate >= 0.40:
        return "101_135"
    if last_rank <= 160:
        return "136_160"
    if last_rank <= 200:
        return "161_200"
    if last_rank <= 250:
        return "201_250"
    if last_rank <= 300:
        return "251_300"
    return "301_plus"


def best_band_threshold(prior_rows: list[dict[str, Any]], band: int) -> float:
    if not prior_rows:
        return float(band)

    max_rank = max(
        int(as_float(row.get("teams_ranked")) or as_float(row.get("actual_net_rank")) or band)
        for row in prior_rows
    )
    candidates = range(1, max_rank + 1)

    best_threshold = float(band)
    best_score: tuple[float, float, float] | None = None
    for threshold in candidates:
        counts = {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }
        for row in prior_rows:
            predicted_rank = as_float(row.get("predicted_net_rank"))
            actual_rank = as_float(row.get("actual_net_rank"))
            if predicted_rank is None:
                continue
            actual_positive = actual_rank is not None and actual_rank <= band
            predicted_positive = predicted_rank <= threshold
            if actual_positive and predicted_positive:
                counts["true_positive"] += 1
            elif not actual_positive and predicted_positive:
                counts["false_positive"] += 1
            elif actual_positive and not predicted_positive:
                counts["false_negative"] += 1
            else:
                counts["true_negative"] += 1
        scores = band_scores(counts)
        f1 = scores["f1"] or 0.0
        precision = scores["precision"] or 0.0
        score = (f1, precision, -abs(threshold - band))
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def calibrated_band_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add prior-season calibrated top-band decisions for every prediction row."""
    outputs: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in predictions})
    for model in models:
        model_rows = [row for row in predictions if str(row["model"]) == model]
        prior_rows: list[dict[str, Any]] = []
        for season in sorted({int(row["season"]) for row in model_rows}):
            season_rows = [row for row in model_rows if int(row["season"]) == season]
            thresholds = {band: best_band_threshold(prior_rows, band) for band in TARGET_BANDS}
            for row in season_rows:
                output = dict(row)
                predicted_rank = as_float(row.get("predicted_net_rank"))
                for band, threshold in thresholds.items():
                    output[f"calibrated_threshold_top_{band}"] = threshold
                    output[f"calibrated_top_{band}"] = (
                        predicted_rank <= threshold if predicted_rank is not None else False
                    )
                output["calibrated_schedule_band"] = exclusive_band_from_flags(output, "calibrated")
                output["opponent_quality_tier"] = stronger_band(
                    output.get("calibrated_schedule_band"),
                    output.get("program_consistency_band"),
                )
                outputs.append(output)
            prior_rows.extend(season_rows)
    return outputs


def baseline_prediction(row: dict[str, Any]) -> tuple[float, float]:
    teams_ranked = as_float(row.get("target_teams_ranked")) or 0.0
    rank = as_float(row.get("kenpom_preseason_rank_adj_em"))
    if rank is None:
        rank = teams_ranked / 2 if teams_ranked else 175.0
    rank = min(max(rank, 1.0), teams_ranked or rank)
    return rank, percentile_from_rank(rank, teams_ranked)


def residual_target(row: dict[str, Any]) -> float:
    _, baseline_percentile = baseline_prediction(row)
    actual_percentile = as_float(row["target_net_percentile"]) or 0.0
    return actual_percentile - baseline_percentile


def model_target(row: dict[str, Any], mode: str) -> float:
    if mode == "direct":
        return as_float(row["target_net_percentile"]) or 0.0
    if mode == "residual":
        return residual_target(row)
    raise ValueError(f"Unsupported model mode: {mode}")


def model_prediction_percentile(row: dict[str, Any], model: RidgeModel, mode: str) -> float:
    prediction = model.predict(row)
    if mode == "residual":
        _, baseline_percentile = baseline_prediction(row)
        prediction += baseline_percentile
    elif mode != "direct":
        raise ValueError(f"Unsupported model mode: {mode}")
    return min(1.0, max(0.0, prediction))


def rolling_predictions(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[ModelConfig],
    alpha: float,
    max_features: int | None = None,
    first_test_season: int | None = None,
) -> list[dict[str, Any]]:
    labeled = target_rows(rows)
    seasons = sorted({int(row["season"]) for row in labeled})
    predictions: list[dict[str, Any]] = []

    for test_season in seasons:
        if first_test_season is not None and test_season < first_test_season:
            continue
        train_rows = [row for row in labeled if int(row["season"]) < test_season]
        test_rows = [row for row in labeled if int(row["season"]) == test_season]
        if not train_rows or not test_rows:
            continue

        for row in test_rows:
            predicted_rank, predicted_percentile = baseline_prediction(row)
            predictions.append(
                prediction_row(
                    row,
                    model_name="kenpom_preseason_baseline",
                    predicted_rank=predicted_rank,
                    predicted_percentile=predicted_percentile,
                    feature_count=1,
                    train_rows=len(train_rows),
                )
            )

        for config in model_configs:
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
                model = fit_ridge(
                    train_rows,
                    columns,
                    target_values,
                    config.alpha if config.alpha is not None else alpha,
                )
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
            for row in test_rows:
                predicted_percentile = model_prediction_percentile(row, model, config.mode)
                teams_ranked = as_float(row["target_teams_ranked"]) or 0.0
                predicted_rank = rank_from_percentile(predicted_percentile, teams_ranked)
                predictions.append(
                    prediction_row(
                        row,
                        model_name=config.name,
                        predicted_rank=predicted_rank,
                        predicted_percentile=predicted_percentile,
                        feature_count=len(columns),
                        train_rows=len(train_rows),
                    )
                )

    return predictions


def rolling_feature_selections(
    rows: list[dict[str, Any]],
    *,
    model_configs: list[ModelConfig],
    alpha: float,
    max_features: int | None = None,
    first_test_season: int | None = None,
) -> list[dict[str, Any]]:
    labeled = target_rows(rows)
    seasons = sorted({int(row["season"]) for row in labeled})
    selections: list[dict[str, Any]] = []

    for test_season in seasons:
        if first_test_season is not None and test_season < first_test_season:
            continue
        train_rows = [row for row in labeled if int(row["season"]) < test_season]
        test_rows = [row for row in labeled if int(row["season"]) == test_season]
        if not train_rows or not test_rows:
            continue

        for config in model_configs:
            target_values = [model_target(row, config.mode) for row in train_rows]
            feature_limit = config.max_features if config.max_features is not None else max_features
            candidate_columns = config_feature_columns(train_rows, config)
            selected_columns = select_feature_columns(
                train_rows,
                candidate_columns,
                target_values,
                feature_limit,
            )
            selected_columns = with_forced_features(
                selected_columns,
                candidate_columns,
                config.forced_features,
            )
            for rank, column in enumerate(selected_columns, start=1):
                selections.append(
                    {
                        "model": config.name,
                        "mode": config.mode,
                        "algorithm": config.algorithm,
                        "test_season": test_season,
                        "train_rows": len(train_rows),
                        "test_rows": len(test_rows),
                        "alpha": config.alpha if config.alpha is not None else alpha,
                        "candidate_features": len(candidate_columns),
                        "selected_features": len(selected_columns),
                        "feature_rank": rank,
                        "feature": column,
                    }
                )

    return selections


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def row_slice_flags(row: dict[str, Any]) -> dict[str, bool]:
    lost_minutes_pct = as_float(row.get("prior_roster_lost_or_uncertain_minutes_pct"))
    confirmed_unavailable_pct = as_float(row.get("prior_roster_confirmed_unavailable_minutes_pct"))
    returning_minutes_pct = as_float(row.get("prior_roster_expected_returning_minutes_pct"))
    returning_top_7_share = as_float(row.get("prior_roster_expected_returning_top_7_minutes_roster_share"))
    hs_percentile = as_float(row.get("incoming_on3_hs_rank_percentile"))
    transfer_percentile = as_float(row.get("incoming_on3_transfer_rank_percentile"))
    cbb_transfer_players = as_float(row.get("incoming_cbb_transfer_players"))
    cbb_transfer_adjusted_warp = as_float(row.get("incoming_cbb_transfer_source_adjusted_warp"))

    high_churn = (lost_minutes_pct is not None and lost_minutes_pct >= 0.35) or (
        confirmed_unavailable_pct is not None and confirmed_unavailable_pct >= 0.30
    )
    extreme_churn = (lost_minutes_pct is not None and lost_minutes_pct >= 0.50) or (
        confirmed_unavailable_pct is not None and confirmed_unavailable_pct >= 0.45
    )
    major_hs_class = hs_percentile is not None and hs_percentile >= 0.80
    major_transfer_class = (
        (transfer_percentile is not None and transfer_percentile >= 0.80)
        or (cbb_transfer_players is not None and cbb_transfer_players >= 3)
        or (cbb_transfer_adjusted_warp is not None and cbb_transfer_adjusted_warp >= 1.5)
    )
    major_incoming_class = major_hs_class or major_transfer_class

    return {
        "roster_data_available": row.get("prior_roster_source_season") not in (None, ""),
        "high_churn": high_churn,
        "extreme_churn": extreme_churn,
        "very_high_churn": lost_minutes_pct is not None and lost_minutes_pct >= 0.75,
        "low_returning_production": returning_minutes_pct is not None and returning_minutes_pct <= 0.35,
        "very_low_returning_core": returning_top_7_share is not None and returning_top_7_share <= 0.25,
        "stable_roster": returning_minutes_pct is not None and returning_minutes_pct >= 0.65,
        "major_hs_class": major_hs_class,
        "major_transfer_class": major_transfer_class,
        "major_incoming_class": major_incoming_class,
        "high_churn_or_major_incoming": high_churn or major_incoming_class,
    }


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["season"]), str(row["team_key"])): row for row in rows}


def metric_row(
    rows: list[dict[str, Any]],
    *,
    model: str,
    season: str | int,
    prediction_prefix: str = "predicted",
) -> dict[str, Any]:
    count = len(rows)
    mae = sum(as_float(row["absolute_rank_error"]) or 0.0 for row in rows) / count if count else None
    rmse = (
        math.sqrt(sum((as_float(row["absolute_rank_error"]) or 0.0) ** 2 for row in rows) / count)
        if count
        else None
    )
    output: dict[str, Any] = {
        "model": model,
        "season": season,
        "rows": count,
        "rank_mae": mae,
        "rank_rmse": rmse,
    }
    for band in TARGET_BANDS:
        counts = band_confusion_counts(rows, band=band, prediction_prefix=prediction_prefix)
        scores = band_scores(counts)
        output[f"top_{band}_accuracy"] = scores["accuracy"]
        output[f"top_{band}_precision"] = scores["precision"]
        output[f"top_{band}_recall"] = scores["recall"]
        output[f"top_{band}_f1"] = scores["f1"]
        output[f"top_{band}_true_positive"] = counts["true_positive"]
        output[f"top_{band}_false_positive"] = counts["false_positive"]
        output[f"top_{band}_true_negative"] = counts["true_negative"]
        output[f"top_{band}_false_negative"] = counts["false_negative"]
    return output


def backtest_slice_metrics(predictions: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_row_index = row_index(model_rows)
    annotated: list[tuple[dict[str, Any], dict[str, bool]]] = []
    for prediction in predictions:
        source = model_row_index.get((str(prediction["season"]), str(prediction["team_key"])))
        if source is None:
            continue
        annotated.append((prediction, row_slice_flags(source)))

    metrics: list[dict[str, Any]] = []
    models = sorted({prediction["model"] for prediction, _ in annotated})
    slice_names = sorted({slice_name for _, flags in annotated for slice_name in flags})
    for model in models:
        model_items = [(prediction, flags) for prediction, flags in annotated if prediction["model"] == model]
        for slice_name in slice_names:
            slice_rows = [prediction for prediction, flags in model_items if flags[slice_name]]
            if not slice_rows:
                continue
            row = metric_row(slice_rows, model=model, season="overall")
            row["slice"] = slice_name
            metrics.append(row)
            complement_rows = [prediction for prediction, flags in model_items if not flags[slice_name]]
            if complement_rows:
                complement = metric_row(complement_rows, model=model, season="overall")
                complement["slice"] = f"not_{slice_name}"
                metrics.append(complement)
    return metrics


def backtest_metrics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in predictions})
    for model in models:
        model_rows = [row for row in predictions if row["model"] == model]
        metrics.append(metric_row(model_rows, model=model, season="overall"))
        for season in sorted({row["season"] for row in model_rows}):
            season_rows = [row for row in model_rows if row["season"] == season]
            metrics.append(metric_row(season_rows, model=model, season=season))
    return metrics


def backtest_band_metrics(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    decision_types = [
        ("rank_threshold", "predicted"),
        ("calibrated_threshold", "calibrated"),
    ]
    models = sorted({row["model"] for row in predictions})
    for decision_type, prediction_prefix in decision_types:
        for model in models:
            model_rows = [row for row in predictions if row["model"] == model]
            row = metric_row(
                model_rows,
                model=model,
                season="overall",
                prediction_prefix=prediction_prefix,
            )
            row["decision_type"] = decision_type
            metrics.append(row)
            for season in sorted({row["season"] for row in model_rows}):
                season_rows = [row for row in model_rows if row["season"] == season]
                row = metric_row(
                    season_rows,
                    model=model,
                    season=season,
                    prediction_prefix=prediction_prefix,
                )
                row["decision_type"] = decision_type
                metrics.append(row)
    return metrics
