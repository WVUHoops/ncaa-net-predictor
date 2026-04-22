"""Guarantee-game upset risk modeling utilities."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key, normalize_coach_name
from net_predictor.model_table import as_float, as_int, read_csv_rows, read_json_rows


HIGH_MAJOR_CONFERENCES = {"ACC", "B10", "B12", "BE", "P12", "SEC"}
EARLY_SEASON_MONTHS = {11, 12}
TARGET_SEASON = 2027
HOME_COURT_ADJ_EM = 3.5

TEAM_KEY_ALIASES = {
    "american university": "american",
    "iu indianapolis": "iupui",
    "iu indy": "iupui",
    "long island": "liu",
    "long island university": "liu",
    "queens nc": "queens",
    "queens university": "queens",
    "queens university of charlotte": "queens",
    "se louisiana": "southeastern louisiana",
    "s carolina upstate": "usc upstate",
    "sc upstate": "usc upstate",
    "south carolina upstate": "usc upstate",
    "saint thomas minnesota": "saint thomas",
    "st thomas minnesota": "saint thomas",
    "texas a and m corpus christi": "texas a and m corpus chris",
    "texas a and m cc": "texas a and m corpus chris",
}

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

ROSTER_TALENT_FEATURES = (
    "roster_talent_returning_production_pct_avg",
    "roster_talent_returning_quality_index",
    "roster_talent_returning_core_continuity",
    "roster_talent_known_roster_players",
    "roster_talent_returner_impact_share",
    "roster_talent_hs_newcomer_impact_share",
    "roster_talent_transfer_newcomer_impact_share",
    "roster_talent_newcomer_impact_share",
    "roster_talent_cbb_transfer_quality_index",
    "roster_talent_incoming_hs_score",
    "roster_talent_incoming_transfer_score",
    "roster_talent_incoming_hs_rank_percentile",
    "roster_talent_incoming_transfer_rank_percentile",
    "roster_talent_incoming_transfer_production_percentile",
    "roster_talent_weighted_returning_core_continuity",
    "roster_talent_weighted_hs_rank_percentile",
    "roster_talent_weighted_transfer_rank_percentile",
    "roster_talent_continuity_plus_incoming",
)

COACH_STYLE_BASE_FEATURES = (
    "tempo",
    "rank_tempo",
    "efg_pct",
    "to_pct",
    "or_pct",
    "ft_rate",
    "def_efg_pct",
    "def_to_pct",
    "def_or_pct",
    "def_ft_rate",
    "fg3_pct",
    "three_point_attempt_rate",
    "opp_fg3_pct",
    "opp_three_point_attempt_rate",
    "off_points_from_three_pct",
    "def_points_from_three_pct",
)


def coach_style_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for window in ("avg", "last3_avg", "last5_avg", "last"):
        for feature in COACH_STYLE_BASE_FEATURES:
            names.append(f"coach_prior_{window}_{feature}")
    return tuple(names)


COACH_NUMERIC_FEATURES = (
    "coach_prior_seasons",
    "coach_prior_program_count",
    "coach_prior_win_pct",
    "coach_prior_avg_adj_em",
    "coach_prior_last3_avg_adj_em",
    "coach_prior_last5_avg_adj_em",
    "coach_prior_avg_adj_em_over_expected",
    "coach_prior_last3_avg_adj_em_over_expected",
    "coach_prior_last5_avg_adj_em_over_expected",
    "coach_prior_positive_adj_em_over_expected_rate",
    "coach_prior_big_overperform_rate",
    "coach_prior_big_underperform_rate",
    "coach_prior_avg_rank_over_expected",
    "coach_prior_top100_rate",
    "coach_prior_ncaa_bid_rate",
    "coach_prior_same_school_seasons",
    "coach_prior_same_school_avg_adj_em_over_expected",
    "coach_first_year_at_school",
    *coach_style_feature_names(),
)

MODEL_FEATURES = (
    "days_from_nov_1",
    "venue_capacity_log",
    "away_roster_talent_continuity_plus_incoming",
    "away_roster_talent_returning_quality_index",
    "away_roster_talent_returning_core_continuity",
    "away_roster_talent_cbb_transfer_quality_index",
    "away_roster_talent_incoming_transfer_score",
    "away_roster_talent_incoming_transfer_production_percentile",
    "away_roster_talent_incoming_hs_score",
    "away_roster_talent_incoming_hs_rank_percentile",
    "away_roster_talent_weighted_returning_core_continuity",
    "away_roster_talent_weighted_transfer_rank_percentile",
    "away_roster_talent_weighted_hs_rank_percentile",
    "home_roster_talent_continuity_plus_incoming",
    "roster_talent_gap",
    "away_roster_talent_x_coach_three_rate",
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
    "away_coach_prior_seasons",
    "away_coach_prior_win_pct",
    "away_coach_prior_avg_adj_em_over_expected",
    "away_coach_prior_last3_avg_adj_em_over_expected",
    "away_coach_prior_positive_adj_em_over_expected_rate",
    "away_coach_prior_big_overperform_rate",
    "away_coach_prior_avg_tempo",
    "away_coach_prior_last3_avg_tempo",
    "away_coach_prior_avg_three_point_attempt_rate",
    "away_coach_prior_last3_avg_three_point_attempt_rate",
    "away_coach_prior_avg_off_points_from_three_pct",
    "away_coach_prior_avg_to_pct",
    "away_coach_prior_avg_or_pct",
    "away_coach_prior_avg_ft_rate",
    "away_coach_prior_avg_def_to_pct",
    "away_coach_prior_avg_def_or_pct",
    "home_coach_prior_seasons",
    "home_coach_prior_win_pct",
    "home_coach_prior_avg_adj_em_over_expected",
    "home_coach_prior_last3_avg_adj_em_over_expected",
    "home_coach_prior_big_underperform_rate",
    "home_coach_prior_avg_tempo",
    "home_coach_prior_avg_opp_three_point_attempt_rate",
    "home_coach_prior_avg_def_efg_pct",
    "home_coach_prior_avg_def_to_pct",
    "home_coach_prior_avg_def_or_pct",
    "home_coach_prior_avg_def_ft_rate",
    "home_coach_first_year_at_school",
    "coach_overperformance_gap",
    "coach_positive_over_expected_gap",
    "coach_experience_gap",
    "coach_style_tempo_gap",
    "away_coach_style_three_rate_x_quality",
    "away_coach_style_threes_vs_home_allow",
    "away_coach_overperformance_x_quality",
    "home_coach_underperformance_x_vulnerability",
    "away_coach_hm_guarantee_games_log",
    "away_coach_hm_guarantee_upset_rate",
    "away_coach_hm_guarantee_close_rate",
    "away_coach_hm_guarantee_over_expected_rate",
    "away_coach_hm_guarantee_avg_margin_over_expected",
    "away_coach_road_hm_games_log",
    "away_coach_road_hm_upset_rate",
    "away_coach_road_hm_close_rate",
    "away_coach_road_hm_over_expected_rate",
    "away_coach_road_hm_avg_margin_over_expected",
    "home_coach_hm_guarantee_games_log",
    "home_coach_hm_guarantee_upset_allowed_rate",
    "home_coach_hm_guarantee_close_allowed_rate",
    "home_coach_hm_guarantee_under_expected_rate",
    "home_coach_hm_guarantee_avg_margin_allowed_over_expected",
    "coach_guarantee_upset_rate_gap",
    "coach_guarantee_over_expected_gap",
    "away_coach_guarantee_pest_index",
    "home_coach_guarantee_vulnerability_index",
    "away_coach_road_hm_pest_index",
)

AWAY_COACH_UPSET_SIGNAL_FIELDS = (
    *(f"away_{feature}" for feature in COACH_NUMERIC_FEATURES),
    "away_coach_hm_guarantee_games",
    "away_coach_hm_guarantee_games_log",
    "away_coach_hm_guarantee_upset_rate",
    "away_coach_hm_guarantee_close_rate",
    "away_coach_hm_guarantee_over_expected_rate",
    "away_coach_hm_guarantee_avg_margin_over_expected",
    "away_coach_road_hm_games",
    "away_coach_road_hm_games_log",
    "away_coach_road_hm_upset_rate",
    "away_coach_road_hm_close_rate",
    "away_coach_road_hm_over_expected_rate",
    "away_coach_road_hm_avg_margin_over_expected",
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


def schedule_team_key(team: Any) -> str:
    key = canonical_team_key(team)
    return TEAM_KEY_ALIASES.get(key, key)


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


def load_historical_coach_features(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    features: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_csv_rows(path):
        season = as_int(row.get("season"))
        team_key = canonical_team_key(row.get("team_name"))
        if season is None or not team_key:
            continue
        features[(season, team_key)] = normalize_coach_features(row)
    return features


def load_latest_coach_features(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    features: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        coach_key = row.get("coach_key") or normalize_coach_name(row.get("coach"))
        if coach_key:
            features[str(coach_key)] = normalize_coach_features(row)
    return features


def normalize_roster_talent_features(row: dict[str, Any]) -> dict[str, Any]:
    output = {feature: as_float(row.get(feature)) for feature in ROSTER_TALENT_FEATURES}
    if output["roster_talent_continuity_plus_incoming"] is None:
        output["roster_talent_continuity_plus_incoming"] = as_float(
            row.get("composition_weighted_roster_talent")
        )
    if output["roster_talent_incoming_transfer_production_percentile"] is None:
        output["roster_talent_incoming_transfer_production_percentile"] = as_float(
            row.get("incoming_cbb_transfer_production_percentile")
        )
    if output["roster_talent_known_roster_players"] is None:
        output["roster_talent_known_roster_players"] = as_float(row.get("roster_known_players"))
    return output


def load_roster_talent_features(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    features: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_csv_rows(path):
        season = as_int(row.get("season"))
        team_key = schedule_team_key(row.get("team") or row.get("team_name"))
        if season is None or not team_key:
            continue
        features[(season, team_key)] = normalize_roster_talent_features(row)
    return features


def normalize_coach_features(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for feature in COACH_NUMERIC_FEATURES:
        value_ = row.get(feature)
        if feature == "coach_first_year_at_school":
            output[feature] = 1.0 if is_true(value_) else 0.0
        else:
            output[feature] = as_float(value_)
    return output


def prefixed_coach_features(source: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    source = source or {}
    return {f"{prefix}_{feature}": source.get(feature) for feature in COACH_NUMERIC_FEATURES}


def prefixed_roster_talent_features(source: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    source = normalize_roster_talent_features(source or {})
    return {f"{prefix}_{feature}": source.get(feature) for feature in ROSTER_TALENT_FEATURES}


def median_coach_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        feature: median([as_float(row.get(feature)) for row in rows])
        for feature in COACH_NUMERIC_FEATURES
    }


def median_roster_talent_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_roster_talent_features(row) for row in rows]
    return {
        feature: median([as_float(row.get(feature)) for row in normalized])
        for feature in ROSTER_TALENT_FEATURES
    }


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
    row["coach_overperformance_gap"] = difference(
        row.get("away_coach_prior_avg_adj_em_over_expected"),
        row.get("home_coach_prior_avg_adj_em_over_expected"),
    )
    row["coach_positive_over_expected_gap"] = difference(
        row.get("away_coach_prior_positive_adj_em_over_expected_rate"),
        row.get("home_coach_prior_positive_adj_em_over_expected_rate"),
    )
    row["coach_experience_gap"] = difference(
        row.get("away_coach_prior_seasons"),
        row.get("home_coach_prior_seasons"),
    )
    row["roster_talent_gap"] = difference(
        row.get("away_roster_talent_continuity_plus_incoming"),
        row.get("home_roster_talent_continuity_plus_incoming"),
    )
    row["away_roster_talent_x_coach_three_rate"] = product(
        row.get("away_roster_talent_continuity_plus_incoming"),
        row.get("away_coach_prior_avg_three_point_attempt_rate"),
    )
    row["coach_style_tempo_gap"] = difference(
        row.get("away_coach_prior_avg_tempo"),
        row.get("home_coach_prior_avg_tempo"),
    )
    row["away_coach_style_three_rate_x_quality"] = product(
        row.get("away_adj_em"),
        row.get("away_coach_prior_avg_three_point_attempt_rate"),
    )
    row["away_coach_style_threes_vs_home_allow"] = product(
        row.get("away_coach_prior_avg_three_point_attempt_rate"),
        row.get("home_coach_prior_avg_opp_three_point_attempt_rate"),
    )
    row["away_coach_overperformance_x_quality"] = product(
        row.get("away_adj_em"),
        row.get("away_coach_prior_avg_adj_em_over_expected"),
    )
    row["home_coach_underperformance_x_vulnerability"] = product(
        row.get("home_vulnerability_index"),
        row.get("home_coach_prior_big_underperform_rate"),
    )
    row["coach_guarantee_upset_rate_gap"] = difference(
        row.get("away_coach_hm_guarantee_upset_rate"),
        row.get("home_coach_hm_guarantee_upset_allowed_rate"),
    )
    row["coach_guarantee_over_expected_gap"] = difference(
        row.get("away_coach_hm_guarantee_avg_margin_over_expected"),
        row.get("home_coach_hm_guarantee_avg_margin_allowed_over_expected"),
    )
    row["away_coach_guarantee_pest_index"] = average(
        [
            row.get("away_coach_hm_guarantee_upset_rate"),
            row.get("away_coach_hm_guarantee_close_rate"),
            row.get("away_coach_hm_guarantee_over_expected_rate"),
        ]
    )
    row["away_coach_road_hm_pest_index"] = average(
        [
            row.get("away_coach_road_hm_upset_rate"),
            row.get("away_coach_road_hm_close_rate"),
            row.get("away_coach_road_hm_over_expected_rate"),
        ]
    )
    row["home_coach_guarantee_vulnerability_index"] = average(
        [
            row.get("home_coach_hm_guarantee_upset_allowed_rate"),
            row.get("home_coach_hm_guarantee_close_allowed_rate"),
            row.get("home_coach_hm_guarantee_under_expected_rate"),
        ]
    )


def empty_coach_road_hm_summary() -> dict[str, float | None]:
    return {
        "away_coach_road_hm_games": 0.0,
        "away_coach_road_hm_games_log": 0.0,
        "away_coach_road_hm_upset_rate": None,
        "away_coach_road_hm_close_rate": None,
        "away_coach_road_hm_over_expected_rate": None,
        "away_coach_road_hm_avg_margin_over_expected": None,
    }


def coach_road_hm_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return empty_coach_road_hm_summary()

    margins = [as_float(row.get("margin_over_expected_for_away")) for row in rows]
    observed_margins = [margin for margin in margins if margin is not None]
    games = float(len(rows))
    upsets = sum(1 for row in rows if int(row.get("upset") or 0) == 1)
    close_games = sum(1 for row in rows if (as_float(row.get("score_margin_for_away")) or -999) >= -10)
    over_expected = sum(1 for margin in observed_margins if margin > 0)
    return {
        "away_coach_road_hm_games": games,
        "away_coach_road_hm_games_log": math.log1p(games),
        "away_coach_road_hm_upset_rate": upsets / games,
        "away_coach_road_hm_close_rate": close_games / games,
        "away_coach_road_hm_over_expected_rate": (
            over_expected / len(observed_margins) if observed_margins else None
        ),
        "away_coach_road_hm_avg_margin_over_expected": (
            sum(observed_margins) / len(observed_margins) if observed_margins else None
        ),
    }


def attach_prior_coach_road_hm_history(
    rows: list[dict[str, Any]],
    road_hm_rows: list[dict[str, Any]],
) -> None:
    history_by_coach: dict[str, list[dict[str, Any]]] = {}
    road_by_season: dict[int, list[dict[str, Any]]] = {}
    for row in road_hm_rows:
        road_by_season.setdefault(int(row["season"]), []).append(row)

    for season in sorted({int(row["season"]) for row in rows}):
        for row in [item for item in rows if int(item["season"]) == season]:
            coach_key = str(row.get("away_coach_key") or "")
            row.update(coach_road_hm_summary(history_by_coach.get(coach_key, [])))
            enrich_matchup_features(row)

        for road_row in road_by_season.get(season, []):
            coach_key = str(road_row.get("away_coach_key") or "")
            if coach_key:
                history_by_coach.setdefault(coach_key, []).append(road_row)


def coach_road_hm_summary_index(road_hm_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in road_hm_rows:
        coach_key = str(row.get("away_coach_key") or "")
        if coach_key:
            grouped.setdefault(coach_key, []).append(row)
    return {coach_key: coach_road_hm_summary(rows) for coach_key, rows in grouped.items()}


def coach_guarantee_summary(rows: list[dict[str, Any]], *, role: str) -> dict[str, float | None]:
    if not rows:
        if role == "away":
            return {
                "away_coach_hm_guarantee_games": 0.0,
                "away_coach_hm_guarantee_games_log": 0.0,
                "away_coach_hm_guarantee_upset_rate": None,
                "away_coach_hm_guarantee_close_rate": None,
                "away_coach_hm_guarantee_over_expected_rate": None,
                "away_coach_hm_guarantee_avg_margin_over_expected": None,
            }
        return {
            "home_coach_hm_guarantee_games": 0.0,
            "home_coach_hm_guarantee_games_log": 0.0,
            "home_coach_hm_guarantee_upset_allowed_rate": None,
            "home_coach_hm_guarantee_close_allowed_rate": None,
            "home_coach_hm_guarantee_under_expected_rate": None,
            "home_coach_hm_guarantee_avg_margin_allowed_over_expected": None,
        }

    margins = [as_float(row.get("margin_over_expected_for_away")) for row in rows]
    observed_margins = [margin for margin in margins if margin is not None]
    games = float(len(rows))
    upsets = sum(1 for row in rows if int(row.get("upset") or 0) == 1)
    close_games = sum(1 for row in rows if (as_float(row.get("score_margin_for_away")) or -999) >= -10)
    over_expected = sum(1 for margin in observed_margins if margin > 0)
    avg_margin_over_expected = (
        sum(observed_margins) / len(observed_margins) if observed_margins else None
    )
    if role == "away":
        return {
            "away_coach_hm_guarantee_games": games,
            "away_coach_hm_guarantee_games_log": math.log1p(games),
            "away_coach_hm_guarantee_upset_rate": upsets / games,
            "away_coach_hm_guarantee_close_rate": close_games / games,
            "away_coach_hm_guarantee_over_expected_rate": (
                over_expected / len(observed_margins) if observed_margins else None
            ),
            "away_coach_hm_guarantee_avg_margin_over_expected": avg_margin_over_expected,
        }
    return {
        "home_coach_hm_guarantee_games": games,
        "home_coach_hm_guarantee_games_log": math.log1p(games),
        "home_coach_hm_guarantee_upset_allowed_rate": upsets / games,
        "home_coach_hm_guarantee_close_allowed_rate": close_games / games,
        "home_coach_hm_guarantee_under_expected_rate": (
            over_expected / len(observed_margins) if observed_margins else None
        ),
        "home_coach_hm_guarantee_avg_margin_allowed_over_expected": avg_margin_over_expected,
    }


def attach_prior_coach_guarantee_history(rows: list[dict[str, Any]]) -> None:
    away_history: dict[str, list[dict[str, Any]]] = {}
    home_history: dict[str, list[dict[str, Any]]] = {}
    seasons = sorted({int(row["season"]) for row in rows})
    for season in seasons:
        season_rows = [row for row in rows if int(row["season"]) == season]
        for row in season_rows:
            away_key = str(row.get("away_coach_key") or "")
            home_key = str(row.get("home_coach_key") or "")
            row.update(coach_guarantee_summary(away_history.get(away_key, []), role="away"))
            row.update(coach_guarantee_summary(home_history.get(home_key, []), role="home"))
            enrich_matchup_features(row)

        for row in season_rows:
            away_key = str(row.get("away_coach_key") or "")
            home_key = str(row.get("home_coach_key") or "")
            if away_key:
                away_history.setdefault(away_key, []).append(row)
            if home_key:
                home_history.setdefault(home_key, []).append(row)


def build_low_major_road_high_major_rows(schedule_csv: Path, kenpom_dir: Path) -> list[dict[str, Any]]:
    team_features = load_kenpom_team_features(kenpom_dir)
    rows: list[dict[str, Any]] = []
    with schedule_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for game in reader:
            season = as_int(game.get("season"))
            if season is None:
                continue
            game_date = date_from_schedule_row(game)
            if game_date is None:
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

            home_key = schedule_team_key(value(game, "home_location", "home_short_display_name"))
            away_key = schedule_team_key(value(game, "away_location", "away_short_display_name"))
            home_current = team_features.get((season, home_key))
            away_current = team_features.get((season, away_key))
            home_prior = team_features.get((season - 1, home_key))
            away_prior = team_features.get((season - 1, away_key))
            if not home_current or not away_current:
                continue
            if home_current.get("conference") not in HIGH_MAJOR_CONFERENCES:
                continue
            if away_current.get("conference") in HIGH_MAJOR_CONFERENCES:
                continue
            if home_current.get("conference") == away_current.get("conference"):
                continue

            row = {
                "game_id": value(game, "game_id", "id"),
                "season": season,
                "game_date": game_date.date().isoformat(),
                "home_team": home_current.get("team"),
                "home_team_key": home_key,
                "home_conference": home_current.get("conference"),
                "away_team": away_current.get("team"),
                "away_team_key": away_key,
                "away_conference": away_current.get("conference"),
                "away_coach": away_current.get("coach"),
                "away_coach_key": normalize_coach_name(away_current.get("coach")),
                "home_score": home_score,
                "away_score": away_score,
                "score_margin_for_away": away_score - home_score,
                "upset": int(away_score > home_score),
            }
            expected_margin_for_away = None
            if home_prior and away_prior:
                expected_margin_for_away = difference(away_prior.get("adj_em"), home_prior.get("adj_em"))
                if expected_margin_for_away is not None:
                    expected_margin_for_away -= HOME_COURT_ADJ_EM
            row["expected_margin_for_away"] = expected_margin_for_away
            row["margin_over_expected_for_away"] = difference(
                row.get("score_margin_for_away"),
                row.get("expected_margin_for_away"),
            )
            rows.append(row)
    rows.sort(key=lambda item: (int(item["season"]), str(item["game_date"]), str(item["game_id"])))
    return rows


def coach_guarantee_summary_indexes(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float | None]]]:
    away_history: dict[str, list[dict[str, Any]]] = {}
    home_history: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        away_key = str(row.get("away_coach_key") or "")
        home_key = str(row.get("home_coach_key") or "")
        if away_key:
            away_history.setdefault(away_key, []).append(row)
        if home_key:
            home_history.setdefault(home_key, []).append(row)
    return (
        {
            coach_key: coach_guarantee_summary(history, role="away")
            for coach_key, history in away_history.items()
        },
        {
            coach_key: coach_guarantee_summary(history, role="home")
            for coach_key, history in home_history.items()
        },
    )


def median_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in rows for key in row})
    return {key: median([as_float(row.get(key)) for row in rows]) for key in keys}


def build_training_rows(
    schedule_csv: Path,
    kenpom_dir: Path,
    coach_history_csv: Path | None = None,
    coach_road_hm_rows: list[dict[str, Any]] | None = None,
    modeling_table_csv: Path | None = None,
) -> list[dict[str, Any]]:
    team_features = load_kenpom_team_features(kenpom_dir)
    coach_features = load_historical_coach_features(coach_history_csv)
    roster_talent_features = load_roster_talent_features(modeling_table_csv)
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
            home_key = schedule_team_key(home_team)
            away_key = schedule_team_key(away_team)
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
                "home_coach": home_current.get("coach"),
                "home_coach_key": normalize_coach_name(home_current.get("coach")),
                "home_conference": home_conf,
                "away_team": away_current.get("team") or away_team,
                "away_team_key": away_key,
                "away_coach": away_current.get("coach"),
                "away_coach_key": normalize_coach_name(away_current.get("coach")),
                "away_conference": away_conf,
                "home_score": home_score,
                "away_score": away_score,
                "score_margin_for_away": away_score - home_score,
                "upset": int(away_score > home_score),
                "venue_capacity": as_float(game.get("venue_capacity")),
                "venue_capacity_log": math.log1p(as_float(game.get("venue_capacity")) or 0.0),
                **prefixed_features(away_prior, "away"),
                **prefixed_features(home_prior, "home"),
                **prefixed_roster_talent_features(roster_talent_features.get((season, away_key)), "away"),
                **prefixed_roster_talent_features(roster_talent_features.get((season, home_key)), "home"),
                **prefixed_coach_features(coach_features.get((season, away_key)), "away"),
                **prefixed_coach_features(coach_features.get((season, home_key)), "home"),
            }
            expected_margin_for_away = difference(away_prior.get("adj_em"), home_prior.get("adj_em"))
            if expected_margin_for_away is not None:
                expected_margin_for_away -= HOME_COURT_ADJ_EM
            row["expected_margin_for_away"] = expected_margin_for_away
            row["margin_over_expected_for_away"] = difference(
                row.get("score_margin_for_away"),
                row.get("expected_margin_for_away"),
            )
            enrich_matchup_features(row)
            rows.append(row)

    rows.sort(key=lambda item: (int(item["season"]), str(item["game_date"]), str(item["game_id"])))
    if coach_road_hm_rows is None:
        coach_road_hm_rows = build_low_major_road_high_major_rows(schedule_csv, kenpom_dir)
    attach_prior_coach_road_hm_history(rows, coach_road_hm_rows)
    attach_prior_coach_guarantee_history(rows)
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


def feature_signal_group(feature: str) -> str:
    if "_roster_talent_" in feature or feature.startswith("roster_talent_"):
        return "roster_talent"
    if "_coach_" in feature or feature.startswith("coach_"):
        return "coach_history_style"
    if feature in {"days_from_nov_1", "venue_capacity_log"}:
        return "game_context"
    return "program_history"


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

    def l2_multiplier(self, feature: str) -> float:
        if "_roster_talent_" in feature or feature.startswith("roster_talent_"):
            return 0.55
        if "_coach_" in feature or feature.startswith("coach_"):
            return 0.8
        if feature in {"days_from_nov_1", "venue_capacity_log"}:
            return 1.0
        return 1.25

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
                regularization = self.l2 * self.l2_multiplier(self.features[index]) * self.weights[index]
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
                "signal_group": feature_signal_group(feature),
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


def median_high_major_host(
    team_features: dict[tuple[int, str], dict[str, Any]],
    season: int,
    coach_feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    high_major = [
        row
        for (feature_season, _), row in team_features.items()
        if feature_season == season and row.get("conference") in HIGH_MAJOR_CONFERENCES
    ]
    host: dict[str, Any] = {"team": "Median High-Major Host", "team_key": "median_high_major_host"}
    for feature in BASE_NUMERIC_FEATURES:
        host[feature] = median([as_float(row.get(feature)) for row in high_major])
    if coach_feature_rows:
        host.update(median_coach_features(coach_feature_rows))
    return host


def prediction_index(prediction_rows: list[dict[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    index = {}
    for row in prediction_rows:
        if row.get("model") == model:
            index[str(row.get("team_key"))] = row
    return index


def prediction_coach_key(row: dict[str, Any]) -> str:
    return str(row.get("projected_coach_key") or normalize_coach_name(row.get("projected_coach")))


def current_risk_board(
    model: LogisticRiskModel,
    kenpom_dir: Path,
    current_predictions_csv: Path,
    coach_latest_summary_csv: Path | None = None,
    coach_guarantee_rows: list[dict[str, Any]] | None = None,
    coach_road_hm_rows: list[dict[str, Any]] | None = None,
    current_feature_season: int = 2026,
    prediction_model: str = "direct_ridge_schedule_building",
) -> list[dict[str, Any]]:
    team_features = load_kenpom_team_features(kenpom_dir)
    prediction_rows = read_csv_rows(current_predictions_csv)
    schedule_index = prediction_index(prediction_rows, prediction_model)
    coach_by_key = load_latest_coach_features(coach_latest_summary_csv)
    high_major_coach_rows = [
        coach_by_key[prediction_coach_key(row)]
        for row in prediction_rows
        if row.get("model") == prediction_model
        and row.get("conference") in HIGH_MAJOR_CONFERENCES
        and prediction_coach_key(row) in coach_by_key
    ]
    high_major_prediction_rows = [
        row
        for row in prediction_rows
        if row.get("model") == prediction_model and row.get("conference") in HIGH_MAJOR_CONFERENCES
    ]
    away_guarantee_by_coach, home_guarantee_by_coach = coach_guarantee_summary_indexes(
        coach_guarantee_rows or []
    )
    away_road_hm_by_coach = coach_road_hm_summary_index(coach_road_hm_rows or [])
    high_major_home_guarantee_rows = [
        home_guarantee_by_coach[prediction_coach_key(row)]
        for row in prediction_rows
        if row.get("model") == prediction_model
        and row.get("conference") in HIGH_MAJOR_CONFERENCES
        and prediction_coach_key(row) in home_guarantee_by_coach
    ]
    median_home_guarantee = median_numeric_rows(high_major_home_guarantee_rows)
    median_home_roster_talent = median_roster_talent_features(high_major_prediction_rows)
    host = median_high_major_host(team_features, current_feature_season, high_major_coach_rows)
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
            "projected_coach": schedule_row.get("projected_coach"),
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
            **prefixed_roster_talent_features(schedule_row, "away"),
            **prefixed_roster_talent_features(median_home_roster_talent, "home"),
            **prefixed_coach_features(
                coach_by_key.get(prediction_coach_key(schedule_row)),
                "away",
            ),
            **prefixed_coach_features(host, "home"),
            **away_guarantee_by_coach.get(
                prediction_coach_key(schedule_row),
                coach_guarantee_summary([], role="away"),
            ),
            **away_road_hm_by_coach.get(
                prediction_coach_key(schedule_row),
                empty_coach_road_hm_summary(),
            ),
            **median_home_guarantee,
        }
        enrich_matchup_features(row)
        rows.append(row)

    probabilities = model.predict_proba(rows) if rows else []
    median_away_coach_signal = {
        feature: median([as_float(row.get(feature)) for row in rows])
        for feature in AWAY_COACH_UPSET_SIGNAL_FIELDS
    }
    coach_neutral_rows = []
    for row in rows:
        coach_neutral = dict(row)
        coach_neutral.update(median_away_coach_signal)
        enrich_matchup_features(coach_neutral)
        coach_neutral_rows.append(coach_neutral)
    coach_neutral_probabilities = model.predict_proba(coach_neutral_rows) if coach_neutral_rows else []

    for row, probability, coach_neutral_probability in zip(
        rows,
        probabilities,
        coach_neutral_probabilities,
        strict=True,
    ):
        schedule_value = schedule_value_from_band(str(row.get("opponent_quality_tier") or ""))
        coach_lift = probability - coach_neutral_probability
        row["upset_probability_vs_median_high_major"] = probability
        row["coach_neutral_upset_probability"] = coach_neutral_probability
        row["coach_upset_lift"] = coach_lift
        row["coach_upset_lift_pp"] = coach_lift * 100
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
