"""Guarantee-game upset risk modeling utilities."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key
from net_predictor.model_table import as_float, as_int, read_csv_rows, read_json_rows


HIGH_MAJOR_CONFERENCES = {"ACC", "B10", "B12", "BE", "P12", "SEC"}
EARLY_SEASON_MONTHS = {11, 12}
TARGET_SEASON = 2027

KENPOM_FEATURE_FILES = {
    "ratings": "ratings.json",
    "four_factors": "four_factors.json",
    "misc": "misc_stats.json",
    "height": "height.json",
    "pointdist": "pointdist.json",
}

BASE_NUMERIC_FEATURES = (
    "adj_em",
    "rank_adj_em",
    "adj_oe",
    "rank_adj_oe",
    "adj_de",
    "rank_adj_de",
    "adj_tempo",
    "rank_adj_tempo",
    "tempo",
    "luck",
    "sos",
    "ncsos",
    "apl_off",
    "apl_def",
    "efg_pct",
    "to_pct",
    "or_pct",
    "ft_rate",
    "def_efg_pct",
    "def_to_pct",
    "def_or_pct",
    "def_ft_rate",
    "fg3_pct",
    "fg2_pct",
    "ft_pct",
    "block_pct",
    "steal_rate",
    "non_steal_turnover_rate",
    "assist_rate",
    "three_point_attempt_rate",
    "opp_fg3_pct",
    "opp_fg2_pct",
    "opp_steal_rate",
    "opp_non_steal_turnover_rate",
    "opp_assist_rate",
    "opp_three_point_attempt_rate",
    "avg_height",
    "effective_height",
    "experience",
    "bench",
    "continuity",
    "off_points_from_three_pct",
    "def_points_from_three_pct",
)

MODEL_FEATURES = (
    "days_from_nov_1",
    "venue_capacity_log",
    "away_adj_em",
    "away_rank_adj_em",
    "away_adj_oe",
    "away_adj_de",
    "away_adj_tempo",
    "away_luck",
    "away_apl_off",
    "away_efg_pct",
    "away_to_pct",
    "away_or_pct",
    "away_ft_rate",
    "away_def_efg_pct",
    "away_def_to_pct",
    "away_def_or_pct",
    "away_fg3_pct",
    "away_three_point_attempt_rate",
    "away_opp_three_point_attempt_rate",
    "away_experience",
    "away_bench",
    "away_continuity",
    "away_off_points_from_three_pct",
    "home_adj_em",
    "home_rank_adj_em",
    "home_adj_oe",
    "home_adj_de",
    "home_adj_tempo",
    "home_def_efg_pct",
    "home_def_to_pct",
    "home_def_or_pct",
    "home_def_ft_rate",
    "home_opp_fg3_pct",
    "home_opp_three_point_attempt_rate",
    "home_experience",
    "home_continuity",
    "adj_em_gap",
    "adj_oe_gap",
    "adj_de_gap",
    "tempo_gap",
    "three_point_attempt_matchup",
    "three_point_make_matchup",
    "turnover_pressure_matchup",
    "off_rebound_matchup",
    "pace_shrink_signal",
    "away_quality_x_three_rate",
    "away_quality_x_experience",
    "home_vulnerability_index",
)


def write_json(rows: list[dict[str, Any]], output_path: Path) -> Path:
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


def value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_kenpom_team_features(kenpom_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    features: dict[tuple[int, str], dict[str, Any]] = {}
    for season_dir in sorted(path for path in kenpom_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        season = int(season_dir.name)
        for source_name, filename in KENPOM_FEATURE_FILES.items():
            path = season_dir / filename
            if not path.exists():
                continue
            for raw in read_json_rows(path):
                team = value(raw, "TeamName")
                if not team:
                    continue
                key = (season, canonical_team_key(team))
                row = features.setdefault(
                    key,
                    {
                        "season": season,
                        "team": team,
                        "team_key": key[1],
                    },
                )
                if source_name == "ratings":
                    row.update(
                        {
                            "conference": value(raw, "ConfShort"),
                            "coach": value(raw, "Coach"),
                            "adj_em": as_float(value(raw, "AdjEM")),
                            "rank_adj_em": as_float(value(raw, "RankAdjEM")),
                            "adj_oe": as_float(value(raw, "AdjOE")),
                            "rank_adj_oe": as_float(value(raw, "RankAdjOE")),
                            "adj_de": as_float(value(raw, "AdjDE")),
                            "rank_adj_de": as_float(value(raw, "RankAdjDE")),
                            "adj_tempo": as_float(value(raw, "AdjTempo")),
                            "rank_adj_tempo": as_float(value(raw, "RankAdjTempo")),
                            "tempo": as_float(value(raw, "Tempo")),
                            "luck": as_float(value(raw, "Luck")),
                            "sos": as_float(value(raw, "SOS")),
                            "ncsos": as_float(value(raw, "NCSOS")),
                            "apl_off": as_float(value(raw, "APL_Off")),
                            "apl_def": as_float(value(raw, "APL_Def")),
                        }
                    )
                elif source_name == "four_factors":
                    row.update(
                        {
                            "efg_pct": as_float(value(raw, "eFG_Pct")),
                            "to_pct": as_float(value(raw, "TO_Pct")),
                            "or_pct": as_float(value(raw, "OR_Pct")),
                            "ft_rate": as_float(value(raw, "FT_Rate")),
                            "def_efg_pct": as_float(value(raw, "DeFG_Pct")),
                            "def_to_pct": as_float(value(raw, "DTO_Pct")),
                            "def_or_pct": as_float(value(raw, "DOR_Pct")),
                            "def_ft_rate": as_float(value(raw, "DFT_Rate")),
                        }
                    )
                elif source_name == "misc":
                    row.update(
                        {
                            "fg3_pct": as_float(value(raw, "FG3Pct")),
                            "fg2_pct": as_float(value(raw, "FG2Pct")),
                            "ft_pct": as_float(value(raw, "FTPct")),
                            "block_pct": as_float(value(raw, "BlockPct")),
                            "steal_rate": as_float(value(raw, "StlRate")),
                            "non_steal_turnover_rate": as_float(value(raw, "NSTRate")),
                            "assist_rate": as_float(value(raw, "ARate")),
                            "three_point_attempt_rate": as_float(value(raw, "F3GRate")),
                            "opp_fg3_pct": as_float(value(raw, "OppFG3Pct")),
                            "opp_fg2_pct": as_float(value(raw, "OppFG2Pct")),
                            "opp_steal_rate": as_float(value(raw, "OppStlRate")),
                            "opp_non_steal_turnover_rate": as_float(value(raw, "OppNSTRate")),
                            "opp_assist_rate": as_float(value(raw, "OppARate")),
                            "opp_three_point_attempt_rate": as_float(value(raw, "OppF3GRate")),
                        }
                    )
                elif source_name == "height":
                    row.update(
                        {
                            "avg_height": as_float(value(raw, "AvgHgt")),
                            "effective_height": as_float(value(raw, "HgtEff")),
                            "experience": as_float(value(raw, "Exp")),
                            "bench": as_float(value(raw, "Bench")),
                            "continuity": as_float(value(raw, "Continuity")),
                        }
                    )
                elif source_name == "pointdist":
                    row.update(
                        {
                            "off_points_from_three_pct": as_float(value(raw, "OffFg3")),
                            "def_points_from_three_pct": as_float(value(raw, "DefFg3")),
                        }
                    )
    return features


def date_from_schedule_row(row: dict[str, Any]) -> datetime | None:
    raw = value(row, "date", "start_date", "game_date")
    if not raw:
        return None
    text = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(str(raw)[:10], "%Y-%m-%d")
        except ValueError:
            return None


def is_true(value_: Any) -> bool:
    return str(value_).strip().lower() in {"1", "true", "t", "yes", "y"}


def is_final(row: dict[str, Any]) -> bool:
    completed = str(value(row, "status_type_completed") or "").lower()
    description = str(value(row, "status_type_description", "status_type_name") or "").lower()
    return completed == "true" or "final" in description


def days_from_nov_1(game_date: datetime) -> int:
    nov_1 = datetime(game_date.year, 11, 1, tzinfo=game_date.tzinfo)
    return (game_date - nov_1).days


def prefixed_features(source: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{feature}": source.get(feature) for feature in BASE_NUMERIC_FEATURES}


def product(left: Any, right: Any) -> float | None:
    left_float = as_float(left)
    right_float = as_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float * right_float


def difference(left: Any, right: Any) -> float | None:
    left_float = as_float(left)
    right_float = as_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def average(values: list[Any]) -> float | None:
    observed = [as_float(item) for item in values if as_float(item) is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def enrich_matchup_features(row: dict[str, Any]) -> None:
    row["adj_em_gap"] = difference(row.get("away_adj_em"), row.get("home_adj_em"))
    row["adj_oe_gap"] = difference(row.get("away_adj_oe"), row.get("home_adj_oe"))
    row["adj_de_gap"] = difference(row.get("away_adj_de"), row.get("home_adj_de"))
    row["tempo_gap"] = difference(row.get("away_adj_tempo"), row.get("home_adj_tempo"))
    row["three_point_attempt_matchup"] = product(
        row.get("away_three_point_attempt_rate"),
        row.get("home_opp_three_point_attempt_rate"),
    )
    row["three_point_make_matchup"] = product(row.get("away_fg3_pct"), row.get("home_opp_fg3_pct"))
    row["turnover_pressure_matchup"] = product(row.get("away_to_pct"), row.get("home_def_to_pct"))
    row["off_rebound_matchup"] = product(row.get("away_or_pct"), row.get("home_def_or_pct"))
    row["pace_shrink_signal"] = -as_float(row.get("away_adj_tempo")) if row.get("away_adj_tempo") not in (None, "") else None
    row["away_quality_x_three_rate"] = product(row.get("away_adj_em"), row.get("away_three_point_attempt_rate"))
    row["away_quality_x_experience"] = product(row.get("away_adj_em"), row.get("away_experience"))
    row["home_vulnerability_index"] = average(
        [
            row.get("home_def_efg_pct"),
            row.get("home_opp_fg3_pct"),
            row.get("home_def_or_pct"),
            row.get("home_def_ft_rate"),
            -as_float(row.get("home_adj_em")) if row.get("home_adj_em") not in (None, "") else None,
        ]
    )


def build_training_rows(schedule_csv: Path, kenpom_dir: Path) -> list[dict[str, Any]]:
    team_features = load_kenpom_team_features(kenpom_dir)
    rows: list[dict[str, Any]] = []
    with schedule_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for game in reader:
            season = as_int(game.get("season"))
            if season is None or season <= min(s for s, _ in team_features):
                continue
            game_date = date_from_schedule_row(game)
            if game_date is None or game_date.month not in EARLY_SEASON_MONTHS:
                continue
            if not is_final(game):
                continue
            if is_true(game.get("neutral_site")):
                continue
            if is_true(game.get("conference_competition")):
                continue
            if str(game.get("season_type") or "") != "2":
                continue
            home_score = as_int(game.get("home_score"))
            away_score = as_int(game.get("away_score"))
            if home_score is None or away_score is None:
                continue
            home_team = value(game, "home_location", "home_short_display_name")
            away_team = value(game, "away_location", "away_short_display_name")
            home_key = canonical_team_key(home_team)
            away_key = canonical_team_key(away_team)
            home_current = team_features.get((season, home_key))
            away_current = team_features.get((season, away_key))
            home_prior = team_features.get((season - 1, home_key))
            away_prior = team_features.get((season - 1, away_key))
            if not home_current or not away_current or not home_prior or not away_prior:
                continue
            home_conf = home_current.get("conference")
            away_conf = away_current.get("conference")
            if home_conf not in HIGH_MAJOR_CONFERENCES:
                continue
            if away_conf in HIGH_MAJOR_CONFERENCES:
                continue
            if home_conf == away_conf:
                continue

            row = {
                "game_id": value(game, "game_id", "id"),
                "season": season,
                "game_date": game_date.date().isoformat(),
                "days_from_nov_1": days_from_nov_1(game_date),
                "home_team": home_current.get("team") or home_team,
                "home_team_key": home_key,
                "home_conference": home_conf,
                "away_team": away_current.get("team") or away_team,
                "away_team_key": away_key,
                "away_conference": away_conf,
                "home_score": home_score,
                "away_score": away_score,
                "score_margin_for_away": away_score - home_score,
                "upset": int(away_score > home_score),
                "venue_capacity": as_float(game.get("venue_capacity")),
                "venue_capacity_log": math.log1p(as_float(game.get("venue_capacity")) or 0.0),
                **prefixed_features(away_prior, "away"),
                **prefixed_features(home_prior, "home"),
            }
            enrich_matchup_features(row)
            rows.append(row)

    rows.sort(key=lambda item: (int(item["season"]), str(item["game_date"]), str(item["game_id"])))
    return rows


def median(values: list[float]) -> float | None:
    observed = sorted(value for value in values if value is not None and math.isfinite(value))
    if not observed:
        return None
    mid = len(observed) // 2
    if len(observed) % 2:
        return observed[mid]
    return (observed[mid - 1] + observed[mid]) / 2


def imputation_values(rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in features:
        feature_median = median([as_float(row.get(feature)) for row in rows])
        values[feature] = feature_median if feature_median is not None else 0.0
    return values


def transform_matrix(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    imputations: dict[str, float],
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    raw_matrix: list[list[float]] = []
    for row in rows:
        raw_matrix.append(
            [
                as_float(row.get(feature))
                if as_float(row.get(feature)) is not None
                else imputations[feature]
                for feature in features
            ]
        )

    if means is None:
        means = {
            feature: sum(row[index] for row in raw_matrix) / len(raw_matrix)
            for index, feature in enumerate(features)
        }
    if scales is None:
        scales = {}
        for index, feature in enumerate(features):
            mean = means[feature]
            variance = sum((row[index] - mean) ** 2 for row in raw_matrix) / max(len(raw_matrix), 1)
            scales[feature] = math.sqrt(variance) or 1.0

    matrix = []
    for row in raw_matrix:
        matrix.append(
            [
                (value_ - means[feature]) / scales[feature]
                for value_, feature in zip(row, features, strict=True)
            ]
        )
    return matrix, means, scales


def sigmoid(value_: float) -> float:
    if value_ >= 0:
        z = math.exp(-value_)
        return 1 / (1 + z)
    z = math.exp(value_)
    return z / (1 + z)


class LogisticRiskModel:
    def __init__(
        self,
        features: tuple[str, ...] = MODEL_FEATURES,
        learning_rate: float = 0.05,
        epochs: int = 1400,
        l2: float = 0.02,
        balance_classes: bool = False,
    ) -> None:
        self.features = features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.balance_classes = balance_classes
        self.intercept = 0.0
        self.weights = [0.0 for _ in features]
        self.imputations: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    def fit(self, rows: list[dict[str, Any]]) -> "LogisticRiskModel":
        if not rows:
            raise ValueError("Cannot fit upset risk model with no rows.")
        self.imputations = imputation_values(rows, self.features)
        matrix, self.means, self.scales = transform_matrix(rows, self.features, self.imputations)
        labels = [float(row["upset"]) for row in rows]
        positives = sum(labels)
        negatives = len(labels) - positives
        positive_weight = (negatives / positives) if self.balance_classes and positives else 1.0

        self.intercept = math.log((positives + 0.5) / (negatives + 0.5))
        self.weights = [0.0 for _ in self.features]

        for _ in range(self.epochs):
            grad_intercept = 0.0
            grad_weights = [0.0 for _ in self.features]
            total_weight = 0.0
            for features, label in zip(matrix, labels, strict=True):
                linear = self.intercept + sum(w * x for w, x in zip(self.weights, features, strict=True))
                prediction = sigmoid(linear)
                sample_weight = positive_weight if label == 1.0 else 1.0
                error = (prediction - label) * sample_weight
                grad_intercept += error
                for index, value_ in enumerate(features):
                    grad_weights[index] += error * value_
                total_weight += sample_weight

            total_weight = total_weight or 1.0
            self.intercept -= self.learning_rate * grad_intercept / total_weight
            for index in range(len(self.weights)):
                regularization = self.l2 * self.weights[index]
                self.weights[index] -= self.learning_rate * (
                    grad_weights[index] / total_weight + regularization
                )
        return self

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        matrix, _, _ = transform_matrix(
            rows,
            self.features,
            self.imputations,
            means=self.means,
            scales=self.scales,
        )
        return [
            sigmoid(self.intercept + sum(w * x for w, x in zip(self.weights, features, strict=True)))
            for features in matrix
        ]

    def coefficients(self) -> list[dict[str, Any]]:
        return [
            {
                "feature": feature,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            }
            for feature, coefficient in sorted(
                zip(self.features, self.weights, strict=True),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def log_loss(labels: list[int], scores: list[float]) -> float:
    total = 0.0
    for label, score in zip(labels, scores, strict=True):
        clipped = min(max(score, 1e-6), 1 - 1e-6)
        total += -(label * math.log(clipped) + (1 - label) * math.log(1 - clipped))
    return total / len(labels)


def brier_score(labels: list[int], scores: list[float]) -> float:
    return sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / len(labels)


def precision_at(labels: list[int], scores: list[float], share: float) -> float | None:
    if not labels:
        return None
    count = max(1, int(round(len(labels) * share)))
    top = sorted(zip(scores, labels, strict=True), reverse=True)[:count]
    return sum(label for _, label in top) / len(top)


def risk_bucket(probability: float) -> str:
    if probability >= 0.12:
        return "very_high"
    if probability >= 0.08:
        return "high"
    if probability >= 0.05:
        return "medium"
    if probability >= 0.03:
        return "low"
    return "very_low"


def schedule_value_from_band(band: str | None) -> int:
    values = {
        "top_25": 10,
        "26_50": 9,
        "51_75": 8,
        "76_100": 7,
        "101_135": 6,
        "136_160": 5,
        "161_200": 4,
        "201_250": 3,
        "251_300": 2,
        "301_plus": 1,
    }
    return values.get(str(band or ""), 0)


def recommendation(schedule_value: int, probability: float) -> str:
    if probability >= 0.10 and schedule_value <= 6:
        return "avoid_bad_risk_reward"
    if probability >= 0.08:
        return "avoid_unless_needed"
    if schedule_value >= 5 and probability < 0.05:
        return "strong_target"
    if schedule_value >= 4 and probability < 0.07:
        return "good_target"
    if schedule_value <= 2:
        return "low_value"
    return "monitor"


def rolling_backtest(rows: list[dict[str, Any]], min_train_seasons: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    seasons = sorted({int(row["season"]) for row in rows})
    for season in seasons:
        train = [row for row in rows if int(row["season"]) < season]
        test = [row for row in rows if int(row["season"]) == season]
        if len({int(row["season"]) for row in train}) < min_train_seasons or not test:
            continue
        model = LogisticRiskModel().fit(train)
        scores = model.predict_proba(test)
        labels = [int(row["upset"]) for row in test]
        for row, score in zip(test, scores, strict=True):
            predictions.append(
                {
                    **row,
                    "upset_probability": score,
                    "risk_bucket": risk_bucket(score),
                }
            )
        metrics.append(
            {
                "season": season,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_upsets": sum(labels),
                "test_upset_rate": sum(labels) / len(labels),
                "auc": auc_score(labels, scores),
                "log_loss": log_loss(labels, scores),
                "brier": brier_score(labels, scores),
                "upset_rate_top_10_pct_risk": precision_at(labels, scores, 0.10),
                "upset_rate_top_20_pct_risk": precision_at(labels, scores, 0.20),
            }
        )
    return predictions, metrics


def median_high_major_host(team_features: dict[tuple[int, str], dict[str, Any]], season: int) -> dict[str, Any]:
    high_major = [
        row
        for (feature_season, _), row in team_features.items()
        if feature_season == season and row.get("conference") in HIGH_MAJOR_CONFERENCES
    ]
    host: dict[str, Any] = {"team": "Median High-Major Host", "team_key": "median_high_major_host"}
    for feature in BASE_NUMERIC_FEATURES:
        host[feature] = median([as_float(row.get(feature)) for row in high_major])
    return host


def prediction_index(prediction_rows: list[dict[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    index = {}
    for row in prediction_rows:
        if row.get("model") == model:
            index[str(row.get("team_key"))] = row
    return index


def current_risk_board(
    model: LogisticRiskModel,
    kenpom_dir: Path,
    current_predictions_csv: Path,
    current_feature_season: int = 2026,
    prediction_model: str = "direct_ridge_schedule_building",
) -> list[dict[str, Any]]:
    team_features = load_kenpom_team_features(kenpom_dir)
    host = median_high_major_host(team_features, current_feature_season)
    prediction_rows = read_csv_rows(current_predictions_csv)
    schedule_index = prediction_index(prediction_rows, prediction_model)
    rows: list[dict[str, Any]] = []

    for (season, team_key), away in team_features.items():
        if season != current_feature_season:
            continue
        if away.get("conference") in HIGH_MAJOR_CONFERENCES:
            continue
        schedule_row = schedule_index.get(team_key)
        if not schedule_row:
            continue
        row = {
            "season": TARGET_SEASON,
            "team": schedule_row.get("team") or away.get("team"),
            "team_key": team_key,
            "conference": schedule_row.get("conference") or away.get("conference"),
            "schedule_score_rank": as_float(schedule_row.get("schedule_score_rank")),
            "schedule_score_percentile": as_float(schedule_row.get("schedule_score_percentile")),
            "opponent_quality_tier": schedule_row.get("opponent_quality_tier"),
            "program_consistency_band": schedule_row.get("program_consistency_band"),
            "incoming_on3_hs_rank": schedule_row.get("incoming_on3_hs_rank"),
            "incoming_on3_transfer_rank": schedule_row.get("incoming_on3_transfer_rank"),
            "days_from_nov_1": 20,
            "venue_capacity_log": math.log1p(13000),
            **prefixed_features(away, "away"),
            **prefixed_features(host, "home"),
        }
        enrich_matchup_features(row)
        rows.append(row)

    probabilities = model.predict_proba(rows) if rows else []
    for row, probability in zip(rows, probabilities, strict=True):
        schedule_value = schedule_value_from_band(str(row.get("opponent_quality_tier") or ""))
        row["upset_probability_vs_median_high_major"] = probability
        row["risk_bucket"] = risk_bucket(probability)
        row["schedule_value_score"] = schedule_value
        row["danger_index"] = probability / max(schedule_value, 1)
        row["safe_value_score"] = schedule_value - (probability * 40)
        row["recommendation"] = recommendation(schedule_value, probability)

    rows.sort(
        key=lambda row: (
            str(row.get("recommendation")) not in {"strong_target", "good_target"},
            -(as_float(row.get("safe_value_score")) or 0),
            as_float(row.get("schedule_score_rank")) or 999,
        )
    )
    return rows


def summarize_training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["upset"]) for row in rows]
    seasons = sorted({int(row["season"]) for row in rows})
    by_season = Counter(int(row["season"]) for row in rows)
    return {
        "rows": len(rows),
        "upsets": sum(labels),
        "upset_rate": sum(labels) / len(labels) if labels else None,
        "first_season": seasons[0] if seasons else None,
        "last_season": seasons[-1] if seasons else None,
        "rows_by_season": dict(sorted(by_season.items())),
    }
