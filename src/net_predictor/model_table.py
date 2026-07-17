"""Season-level modeling table assembly."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key

RETURNER_IMMEDIATE_IMPACT_WEIGHT = 1.0
HS_IMMEDIATE_IMPACT_WEIGHT = 0.65
TRANSFER_IMMEDIATE_IMPACT_WEIGHT = 1.15
EXPECTED_ROTATION_SIZE = 9.0
ROTATION_MINUTES_COMPLETENESS_TARGET = 0.75
DIVISION_I_TEAM_COUNT = 364.0
ROSTER_PROXY_COACH_WEIGHT = 0.6
ROSTER_PROXY_PROGRAM_WEIGHT = 0.4
POINT_GUARD_TARGET = 2.0
CENTER_TARGET = 2.0
BACKCOURT_TARGET = 5.0
FRONTCOURT_TARGET = 5.0
FRESHMAN_ROLE_FILL_REFERENCE_SHARE = 0.08
MIN_HS_CALIBRATION_SAMPLE = 12
MIN_TRANSFER_CALIBRATION_SAMPLE = 20


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def season_dirs(kenpom_dir: Path, min_season: int | None = None, max_season: int | None = None) -> list[Path]:
    dirs = []
    for path in kenpom_dir.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        season = int(path.name)
        if min_season is not None and season < min_season:
            continue
        if max_season is not None and season > max_season:
            continue
        dirs.append(path)
    return sorted(dirs, key=lambda path: int(path.name))


def kenpom_preseason_rows(kenpom_dir: Path, min_season: int | None = None, max_season: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season_dir in season_dirs(kenpom_dir, min_season, max_season):
        path = season_dir / "archive_preseason.json"
        if not path.exists():
            continue
        for row in read_json_rows(path):
            season = as_int(row.get("Season"))
            team = row.get("TeamName")
            if season is None or not team:
                continue
            rows.append(
                {
                    "season": season,
                    "team": team,
                    "team_key": canonical_team_key(team),
                    "conference": row.get("ConfShort"),
                    "kenpom_preseason_adj_em": row.get("AdjEM"),
                    "kenpom_preseason_rank_adj_em": row.get("RankAdjEM"),
                    "kenpom_preseason_adj_oe": row.get("AdjOE"),
                    "kenpom_preseason_rank_adj_oe": row.get("RankAdjOE"),
                    "kenpom_preseason_adj_de": row.get("AdjDE"),
                    "kenpom_preseason_rank_adj_de": row.get("RankAdjDE"),
                    "kenpom_preseason_tempo": row.get("AdjTempo"),
                    "kenpom_preseason_rank_tempo": row.get("RankAdjTempo"),
                }
            )
    rows.sort(key=lambda row: (row["season"], row["team"]))
    return rows


def index_by_season_team(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        season = as_int(row.get("season"))
        team_key = row.get("team_key") or canonical_team_key(row.get("team") or row.get("team_name"))
        if season is None or not team_key:
            continue
        index[(season, str(team_key))] = row
    return index


def prefixed(row: dict[str, Any] | None, prefix: str, exclude: set[str]) -> dict[str, Any]:
    if not row:
        return {}
    return {f"{prefix}{key}": value for key, value in row.items() if key not in exclude}


def add_roster_talent_features(
    row: dict[str, Any],
    *,
    prior_roster_players: list[dict[str, Any]] | None = None,
    incoming_transfer_players: list[dict[str, Any]] | None = None,
    incoming_hs_players: list[dict[str, Any]] | None = None,
    hs_recruit_calibration: dict[str, Any] | None = None,
    team_minutes_index: dict[tuple[int, str], float] | None = None,
    transfer_role_calibration: dict[str, Any] | None = None,
) -> None:
    returning_players = as_float(row.get("prior_roster_expected_returning_players"))
    returning_minutes_pct = as_float(row.get("prior_roster_expected_returning_minutes_pct"))
    returning_possessions_pct = as_float(row.get("prior_roster_expected_returning_possessions_pct"))
    returning_points_pct = as_float(row.get("prior_roster_expected_returning_points_pct"))
    lost_minutes_pct = as_float(row.get("prior_roster_lost_or_uncertain_minutes_pct"))
    returning_warp = as_float(row.get("prior_roster_expected_returning_warp"))
    returning_win_shares = as_float(row.get("prior_roster_expected_returning_win_shares"))
    returning_per = as_float(row.get("prior_roster_expected_returning_minutes_weighted_per"))
    returning_net_rating = as_float(row.get("prior_roster_expected_returning_minutes_weighted_net_rating"))
    top_7_minutes_share = as_float(row.get("prior_roster_expected_returning_top_7_minutes_roster_share"))
    hs_score = as_float(row.get("incoming_on3_hs_score"))
    hs_rank_percentile = as_float(row.get("incoming_on3_hs_rank_percentile"))
    transfer_score = as_float(row.get("incoming_on3_transfer_index_score"))
    transfer_raw_score_in = as_float(row.get("incoming_on3_transfer_raw_score_in"))
    transfer_rank_percentile = as_float(row.get("incoming_on3_transfer_rank_percentile"))
    hs_players = first_existing(
        as_float(row.get("incoming_on3_hs_applied_commits")),
        as_float(row.get("incoming_on3_hs_commits")),
    )
    cbb_transfer_players = as_float(row.get("incoming_cbb_transfer_players"))
    transfer_players = first_existing(
        cbb_transfer_players,
        as_float(row.get("incoming_on3_transfer_transfers_in")),
    )
    cbb_transfer_minutes = as_float(row.get("incoming_cbb_transfer_minutes"))
    cbb_transfer_warp = as_float(row.get("incoming_cbb_transfer_warp"))
    cbb_transfer_win_shares = as_float(row.get("incoming_cbb_transfer_win_shares"))
    cbb_transfer_adjusted_warp = as_float(row.get("incoming_cbb_transfer_source_adjusted_warp"))
    cbb_transfer_adjusted_win_shares = as_float(
        row.get("incoming_cbb_transfer_source_adjusted_win_shares")
    )
    cbb_transfer_net_rating = as_float(row.get("incoming_cbb_transfer_minutes_weighted_net_rating"))
    cbb_transfer_source_adj_em = as_float(row.get("incoming_cbb_transfer_minutes_weighted_source_adj_em"))
    cbb_transfer_production_percentile = as_float(
        row.get("incoming_cbb_transfer_production_percentile")
    )

    row["roster_talent_returning_production_pct_avg"] = average_existing(
        returning_minutes_pct,
        returning_possessions_pct,
        returning_points_pct,
    )
    row["roster_talent_returning_quality_index"] = average_existing(
        returning_warp,
        returning_win_shares,
        returning_per,
        returning_net_rating,
    )
    row["roster_talent_returning_core_continuity"] = average_existing(
        returning_minutes_pct,
        top_7_minutes_share,
    )
    composition = roster_composition_weights(
        returning_players=returning_players,
        hs_players=hs_players,
        transfer_players=transfer_players,
    )
    row["roster_talent_known_roster_players"] = composition["known_roster_players"]
    row["roster_talent_returner_roster_share"] = composition["returner_share"]
    row["roster_talent_hs_newcomer_roster_share"] = composition["hs_share"]
    row["roster_talent_transfer_newcomer_roster_share"] = composition["transfer_share"]
    row["roster_talent_newcomer_roster_share"] = composition["newcomer_share"]
    row["roster_talent_returner_impact_share"] = composition["returner_impact_share"]
    row["roster_talent_hs_newcomer_impact_share"] = composition["hs_impact_share"]
    row["roster_talent_transfer_newcomer_impact_share"] = composition["transfer_impact_share"]
    row["roster_talent_newcomer_impact_share"] = composition["newcomer_impact_share"]
    row["roster_talent_cbb_transfer_volume_index"] = average_existing(
        cbb_transfer_players,
        cbb_transfer_minutes,
    )
    row["roster_talent_cbb_transfer_quality_index"] = average_existing(
        cbb_transfer_production_percentile,
        cbb_transfer_warp,
        cbb_transfer_win_shares,
        cbb_transfer_adjusted_warp,
        cbb_transfer_adjusted_win_shares,
        cbb_transfer_net_rating,
        cbb_transfer_source_adj_em,
    )
    transfer_signal = first_existing(
        cbb_transfer_production_percentile,
        row.get("roster_talent_cbb_transfer_quality_index"),
        transfer_score,
        transfer_raw_score_in,
    )
    hs_player_features = hs_recruit_player_features(
        row,
        prior_roster_players or [],
        incoming_transfer_players or [],
        incoming_hs_players or [],
        hs_recruit_calibration,
        team_minutes_index,
        transfer_role_calibration,
    )
    role_composition = roster_composition_weights(
        returning_players=hs_player_features.get("returner_role_units"),
        hs_players=hs_player_features.get("hs_role_units"),
        transfer_players=hs_player_features.get("transfer_role_units"),
    )
    if role_composition.get("known_roster_players") is not None:
        composition = role_composition
    hs_player_talent_percentile = hs_player_features.get("player_talent_percentile")
    hs_player_impact_percentile = hs_player_features.get("need_adjusted_impact_percentile")
    hs_rank_percentile = weighted_average_existing(
        (hs_player_impact_percentile, 0.55),
        (hs_player_talent_percentile, 0.25),
        (hs_rank_percentile, 0.20),
    ) or hs_rank_percentile
    hs_score = weighted_average_existing(
        (hs_player_impact_percentile * 100.0 if hs_player_impact_percentile is not None else None, 0.55),
        (hs_player_talent_percentile * 100.0 if hs_player_talent_percentile is not None else None, 0.25),
        (hs_score, 0.20),
    ) or hs_score
    row["roster_talent_incoming_hs_score"] = hs_score
    row["roster_talent_incoming_transfer_score"] = transfer_signal
    row["roster_talent_incoming_hs_rank_percentile"] = hs_rank_percentile
    row["roster_talent_incoming_transfer_rank_percentile"] = transfer_rank_percentile
    row["roster_talent_incoming_transfer_production_percentile"] = (
        cbb_transfer_production_percentile
    )
    row["roster_talent_hs_need_fit"] = product_if_present(hs_score, lost_minutes_pct)
    row["roster_talent_transfer_need_fit"] = product_if_present(transfer_signal, lost_minutes_pct)
    row["roster_talent_weighted_returning_core_continuity"] = product_if_present(
        row.get("roster_talent_returning_core_continuity"),
        composition["returner_impact_share"],
    )
    row["roster_talent_weighted_hs_rank_percentile"] = product_if_present(
        hs_rank_percentile,
        composition["hs_impact_share"],
    )
    transfer_percentile_signal = first_existing(
        cbb_transfer_production_percentile,
        transfer_rank_percentile,
    )
    row["roster_talent_weighted_transfer_rank_percentile"] = product_if_present(
        transfer_percentile_signal,
        composition["transfer_impact_share"],
    )
    known_only_signal = sum_existing(
        row.get("roster_talent_weighted_returning_core_continuity"),
        row.get("roster_talent_weighted_hs_rank_percentile"),
        row.get("roster_talent_weighted_transfer_rank_percentile"),
    )
    row["roster_talent_continuity_plus_incoming_known_only"] = known_only_signal

    known_roster_players = composition["known_roster_players"]
    count_roster_completeness = roster_completeness_ratio(known_roster_players)
    missing_roster_players = missing_roster_slots(known_roster_players)
    known_projected_minutes_share = sum_existing(
        hs_player_features.get("returner_projected_minutes_share_total"),
        hs_player_features.get("transfer_projected_minutes_share_total"),
        hs_player_features.get("projected_minutes_share_total"),
    )
    minutes_roster_completeness = minutes_based_completeness_ratio(known_projected_minutes_share)
    roster_completeness = first_existing(minutes_roster_completeness, count_roster_completeness)
    missing_rotation_share = missing_rotation_minutes_share(known_projected_minutes_share)
    coach_proxy = coach_talent_proxy_percentile(row)
    program_proxy = program_talent_proxy_percentile(row)
    blended_proxy = weighted_average_existing(
        (coach_proxy, ROSTER_PROXY_COACH_WEIGHT),
        (program_proxy, ROSTER_PROXY_PROGRAM_WEIGHT),
    )

    row["roster_talent_expected_rotation_size"] = EXPECTED_ROTATION_SIZE
    row["roster_talent_rotation_minutes_target"] = ROTATION_MINUTES_COMPLETENESS_TARGET
    row["roster_talent_roster_completeness"] = roster_completeness
    row["roster_talent_count_based_completeness"] = count_roster_completeness
    row["roster_talent_minutes_based_completeness"] = minutes_roster_completeness
    row["roster_talent_missing_roster_players"] = missing_roster_players
    row["roster_talent_known_projected_rotation_minutes_share"] = known_projected_minutes_share
    row["roster_talent_missing_projected_rotation_minutes_share"] = missing_rotation_share
    row["roster_talent_known_roster_share_of_expected"] = count_roster_completeness
    row["roster_talent_proxy_player_share_of_expected"] = (
        missing_roster_players / EXPECTED_ROTATION_SIZE if missing_roster_players is not None else None
    )
    row["roster_talent_returner_expected_roster_share"] = share_of_expected_roster(returning_players)
    row["roster_talent_hs_newcomer_expected_roster_share"] = share_of_expected_roster(hs_players)
    row["roster_talent_transfer_newcomer_expected_roster_share"] = share_of_expected_roster(
        transfer_players
    )
    row["roster_talent_proxy_coach_percentile"] = coach_proxy
    row["roster_talent_proxy_program_percentile"] = program_proxy
    row["roster_talent_proxy_percentile"] = blended_proxy
    row["roster_talent_continuity_plus_incoming"] = blend_known_roster_with_proxy(
        known_signal=known_only_signal,
        roster_completeness=roster_completeness,
        proxy_signal=blended_proxy,
    )
    for key, value in hs_player_features.items():
        row[f"roster_talent_hs_{key}"] = value


def first_existing(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def average_existing(*values: float | None) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def sum_existing(*values: Any) -> float | None:
    observed = [as_float(value) for value in values if as_float(value) is not None]
    if not observed:
        return None
    return sum(observed)


def product_if_present(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left * right


def weighted_average_existing(*weighted_values: tuple[float | None, float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in weighted_values:
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def share_of_expected_roster(players: float | None) -> float | None:
    if players is None:
        return None
    return clip(players / EXPECTED_ROTATION_SIZE, 0.0, 1.0)


def roster_completeness_ratio(known_roster_players: float | None) -> float | None:
    return share_of_expected_roster(known_roster_players)


def missing_roster_slots(known_roster_players: float | None) -> float | None:
    if known_roster_players is None:
        return EXPECTED_ROTATION_SIZE
    return clip(EXPECTED_ROTATION_SIZE - known_roster_players, 0.0, EXPECTED_ROTATION_SIZE)


def minutes_based_completeness_ratio(projected_minutes_share: float | None) -> float | None:
    if projected_minutes_share is None:
        return None
    return clip(projected_minutes_share / ROTATION_MINUTES_COMPLETENESS_TARGET, 0.0, 1.0)


def missing_rotation_minutes_share(projected_minutes_share: float | None) -> float | None:
    if projected_minutes_share is None:
        return None
    return clip(ROTATION_MINUTES_COMPLETENESS_TARGET - projected_minutes_share, 0.0, ROTATION_MINUTES_COMPLETENESS_TARGET)


def rank_to_percentile(rank: float | None) -> float | None:
    if rank is None or rank <= 0:
        return None
    bounded_rank = clip(rank, 1.0, DIVISION_I_TEAM_COUNT)
    return (DIVISION_I_TEAM_COUNT + 1.0 - bounded_rank) / DIVISION_I_TEAM_COUNT


def percentile_share(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value >= threshold) / len(values)


def preseason_strength_proxy(row: dict[str, Any]) -> float | None:
    return weighted_average_existing(
        (as_float(row.get("roster_talent_continuity_plus_incoming")), 0.55),
        (rank_to_percentile(as_float(row.get("program_prior_last_rank_adj_em"))), 0.20),
        (rank_to_percentile(as_float(row.get("coach_coach_prior_last_rank_adj_em"))), 0.10),
        (as_float(row.get("program_prior_top50_rate")), 0.10),
        (as_float(row.get("coach_coach_prior_top50_rate")), 0.05),
    )


def add_conference_schedule_environment(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    national_by_season: dict[int, list[float]] = defaultdict(list)
    row_proxy_by_id: dict[int, float | None] = {}

    for row in rows:
        season = as_int(row.get("season"))
        conference = str(row.get("conference") or "").strip()
        if season is None or not conference:
            continue
        proxy = preseason_strength_proxy(row)
        row_proxy_by_id[id(row)] = proxy
        if proxy is not None:
            national_by_season[season].append(proxy)
        grouped[(season, conference)].append(row)

    national_mean_by_season = {
        season: average_existing(*values)
        for season, values in national_by_season.items()
        if values
    }

    for (season, _conference), conference_rows in grouped.items():
        national_mean = national_mean_by_season.get(season)
        proxies = [row_proxy_by_id.get(id(row)) for row in conference_rows]
        league_size = float(len(conference_rows))
        for row in conference_rows:
            own_proxy = row_proxy_by_id.get(id(row))
            peer_values = [
                proxy
                for peer_row, proxy in zip(conference_rows, proxies)
                if peer_row is not row and proxy is not None
            ]
            top_values = sorted(peer_values, reverse=True)
            peer_mean = average_existing(*peer_values)
            peer_median = median(peer_values) if peer_values else None
            peer_best = top_values[0] if top_values else None
            peer_top3_mean = average_existing(*top_values[:3])
            peer_top5_mean = average_existing(*top_values[:5])
            peer_top25_share = percentile_share(peer_values, rank_to_percentile(25.0) or 0.0)
            peer_top50_share = percentile_share(peer_values, rank_to_percentile(50.0) or 0.0)
            peer_top100_share = percentile_share(peer_values, rank_to_percentile(100.0) or 0.0)
            peer_top150_share = percentile_share(peer_values, rank_to_percentile(150.0) or 0.0)
            row["preseason_strength_proxy"] = own_proxy
            row["conference_schedule_env_league_size"] = league_size
            row["conference_schedule_env_peer_mean"] = peer_mean
            row["conference_schedule_env_peer_median"] = peer_median
            row["conference_schedule_env_peer_best"] = peer_best
            row["conference_schedule_env_peer_top3_mean"] = peer_top3_mean
            row["conference_schedule_env_peer_top5_mean"] = peer_top5_mean
            row["conference_schedule_env_peer_top25_share"] = peer_top25_share
            row["conference_schedule_env_peer_top50_share"] = peer_top50_share
            row["conference_schedule_env_peer_top100_share"] = peer_top100_share
            row["conference_schedule_env_peer_top150_share"] = peer_top150_share
            row["conference_schedule_env_peer_mean_minus_national"] = (
                peer_mean - national_mean if peer_mean is not None and national_mean is not None else None
            )
            row["conference_schedule_env_two_thirds_schedule_proxy"] = weighted_average_existing(
                (peer_mean, 0.67),
                (national_mean, 0.33),
            )
            row["conference_schedule_env_two_thirds_delta"] = (
                row["conference_schedule_env_two_thirds_schedule_proxy"] - national_mean
                if row.get("conference_schedule_env_two_thirds_schedule_proxy") is not None
                and national_mean is not None
                else None
            )


def classify_position(value: Any) -> str:
    token = str(value or "").upper().strip()
    if token in {"PG"}:
        return "pg"
    if token in {"SG", "CG", "G"}:
        return "guard"
    if token in {"C"}:
        return "big"
    if token in {"PF", "F"}:
        return "forward"
    return "wing"


def slot_need(known_count: float, target_count: float) -> float:
    return clip((target_count - known_count) / target_count, 0.0, 1.0)


def normalize_player_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def hs_rank_bucket(rank: float | None) -> str:
    if rank is None or rank <= 0:
        return "unranked"
    if rank <= 25:
        return "top25"
    if rank <= 50:
        return "26_50"
    if rank <= 100:
        return "51_100"
    if rank <= 150:
        return "101_150"
    if rank <= 250:
        return "151_250"
    return "251_plus"


def transfer_share_bucket(value: float | None) -> str:
    if value is None or value <= 0:
        return "0_05"
    if value <= 0.05:
        return "0_05"
    if value <= 0.10:
        return "05_10"
    if value <= 0.15:
        return "10_15"
    if value <= 0.20:
        return "15_20"
    if value <= 0.25:
        return "20_25"
    return "25_plus"


def adjusted_transfer_source_share(source_share: float | None, context_multiplier: float | None) -> float | None:
    if source_share is None:
        return None
    if context_multiplier is None:
        return clip(source_share, 0.0, 0.35)
    scale = clip(context_multiplier, 0.65, 1.35) ** 0.5
    return clip(source_share * scale, 0.0, 0.35)


def role_fill_from_minutes_share(minutes_share: float | None) -> float:
    if minutes_share is None:
        return 0.0
    return clip(minutes_share / FRESHMAN_ROLE_FILL_REFERENCE_SHARE, 0.0, 1.5)


def total_role_units_from_minutes_share(minutes_share: float | None) -> float | None:
    if minutes_share is None:
        return None
    return clip(minutes_share / FRESHMAN_ROLE_FILL_REFERENCE_SHARE, 0.0, EXPECTED_ROTATION_SIZE)


def recruit_percentile(player: dict[str, Any]) -> float | None:
    return first_existing(
        rank_to_percentile(as_float(player.get("industry_rank"))),
        as_float(player.get("industry_rating")) / 100.0 if as_float(player.get("industry_rating")) is not None else None,
        as_float(player.get("on3_rating")) / 100.0 if as_float(player.get("on3_rating")) is not None else None,
    )


def status_is_probable_returner(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip() == "probable_returner"


def summarize_hs_calibration(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"sample": 0, "avg_minutes_share": None}
    return {
        "sample": float(len(values)),
        "avg_minutes_share": sum(values) / len(values),
    }


def build_hs_recruit_role_calibration(
    roster_player_rows: list[dict[str, Any]],
    hs_recruit_player_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not roster_player_rows or not hs_recruit_player_rows:
        return {"by_bucket_position": {}, "by_bucket": {}, "by_position": {}, "overall": {"sample": 0, "avg_minutes_share": 0.06}}

    team_minutes: dict[tuple[int, str], float] = {}
    player_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for player in roster_player_rows:
        season = as_int(player.get("season"))
        team = player.get("team_market") or player.get("team_name")
        team_key = player.get("team_key") or canonical_team_key(team)
        if season is None or not team_key:
            continue
        key = (season, str(team_key))
        team_minutes[key] = team_minutes.get(key, 0.0) + max(as_float(player.get("minutes")) or 0.0, 0.0)
        name_key = normalize_player_name(player.get("player_name"))
        if not name_key:
            continue
        player_key = (season, str(team_key), name_key)
        existing = player_index.get(player_key)
        if existing is None or (as_float(player.get("minutes")) or 0.0) > (as_float(existing.get("minutes")) or 0.0):
            player_index[player_key] = player

    by_bucket_position: dict[tuple[str, str], list[float]] = {}
    by_bucket: dict[str, list[float]] = {}
    by_position: dict[str, list[float]] = {}
    overall: list[float] = []

    for recruit in hs_recruit_player_rows:
        season = as_int(recruit.get("season"))
        team_key = recruit.get("team_key") or canonical_team_key(recruit.get("team"))
        name_key = normalize_player_name(recruit.get("player_name"))
        if season is None or not team_key or not name_key:
            continue
        player = player_index.get((season, str(team_key), name_key))
        if not player:
            continue
        total_minutes = team_minutes.get((season, str(team_key))) or 0.0
        if total_minutes <= 0:
            continue
        minutes_share = clip((as_float(player.get("minutes")) or 0.0) / total_minutes, 0.0, 1.0)
        bucket = hs_rank_bucket(as_float(recruit.get("industry_rank")))
        position = classify_position(recruit.get("position_abbr") or recruit.get("position"))
        by_bucket_position.setdefault((bucket, position), []).append(minutes_share)
        by_bucket.setdefault(bucket, []).append(minutes_share)
        by_position.setdefault(position, []).append(minutes_share)
        overall.append(minutes_share)

    return {
        "by_bucket_position": {
            f"{bucket}:{position}": summarize_hs_calibration(values)
            for (bucket, position), values in by_bucket_position.items()
        },
        "by_bucket": {bucket: summarize_hs_calibration(values) for bucket, values in by_bucket.items()},
        "by_position": {position: summarize_hs_calibration(values) for position, values in by_position.items()},
        "overall": summarize_hs_calibration(overall),
    }


def lookup_hs_role_minutes_share(
    calibration: dict[str, Any] | None,
    *,
    recruit_rank: float | None,
    position_bucket: str,
) -> float | None:
    if not calibration:
        return 0.06
    bucket = hs_rank_bucket(recruit_rank)
    exact = (calibration.get("by_bucket_position") or {}).get(f"{bucket}:{position_bucket}") or {}
    if (as_float(exact.get("sample")) or 0.0) >= MIN_HS_CALIBRATION_SAMPLE:
        return as_float(exact.get("avg_minutes_share"))
    by_bucket = (calibration.get("by_bucket") or {}).get(bucket) or {}
    if (as_float(by_bucket.get("sample")) or 0.0) >= MIN_HS_CALIBRATION_SAMPLE:
        return as_float(by_bucket.get("avg_minutes_share"))
    by_position = (calibration.get("by_position") or {}).get(position_bucket) or {}
    if (as_float(by_position.get("sample")) or 0.0) >= MIN_HS_CALIBRATION_SAMPLE:
        return as_float(by_position.get("avg_minutes_share"))
    overall = calibration.get("overall") or {}
    return as_float(overall.get("avg_minutes_share")) or 0.06


def build_team_minutes_index(roster_player_rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    totals: dict[tuple[int, str], float] = {}
    for player in roster_player_rows:
        season = as_int(player.get("season"))
        team = player.get("team_market") or player.get("team_name")
        team_key = player.get("team_key") or canonical_team_key(team)
        if season is None or not team_key:
            continue
        totals[(season, str(team_key))] = totals.get((season, str(team_key)), 0.0) + max(
            as_float(player.get("minutes")) or 0.0,
            0.0,
        )
    return totals


def project_returner_minutes_share(
    player: dict[str, Any],
    team_minutes_index: dict[tuple[int, str], float] | None,
) -> float | None:
    if not team_minutes_index:
        return None
    season = as_int(player.get("season"))
    team = player.get("team_market") or player.get("team_name")
    team_key = player.get("team_key") or canonical_team_key(team)
    if season is None or not team_key:
        return None
    total = team_minutes_index.get((season, str(team_key))) or 0.0
    if total <= 0:
        return None
    return clip((as_float(player.get("minutes")) or 0.0) / total, 0.0, 0.35)


def summarize_transfer_calibration(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"sample": 0, "avg_minutes_share": None}
    return {
        "sample": float(len(values)),
        "avg_minutes_share": sum(values) / len(values),
    }


def build_transfer_role_calibration(
    roster_player_rows: list[dict[str, Any]],
    transfer_player_rows: list[dict[str, Any]],
    team_minutes_index: dict[tuple[int, str], float] | None = None,
) -> dict[str, Any]:
    if not roster_player_rows or not transfer_player_rows:
        return {"by_bucket_position": {}, "by_bucket": {}, "overall": {"sample": 0, "avg_minutes_share": 0.09}}
    team_minutes_index = team_minutes_index or build_team_minutes_index(roster_player_rows)
    target_index: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for player in roster_player_rows:
        season = as_int(player.get("season"))
        team = player.get("team_market") or player.get("team_name")
        team_key = player.get("team_key") or canonical_team_key(team)
        if season is None or not team_key:
            continue
        player_id = str(player.get("player_id") or "")
        name_key = normalize_player_name(player.get("player_name"))
        if not name_key:
            continue
        key = (season, str(team_key), player_id, name_key)
        existing = target_index.get(key)
        if existing is None or (as_float(player.get("minutes")) or 0.0) > (as_float(existing.get("minutes")) or 0.0):
            target_index[key] = player

    by_bucket_position: dict[tuple[str, str], list[float]] = {}
    by_bucket: dict[str, list[float]] = {}
    overall: list[float] = []
    for transfer in transfer_player_rows:
        season = as_int(transfer.get("season"))
        team_key = transfer.get("team_key") or canonical_team_key(transfer.get("team"))
        source_season = as_int(transfer.get("source_season"))
        source_team_key = transfer.get("source_team_key") or canonical_team_key(transfer.get("source_team"))
        if season is None or not team_key or source_season is None or not source_team_key:
            continue
        source_total = team_minutes_index.get((source_season, str(source_team_key))) or 0.0
        target_total = team_minutes_index.get((season, str(team_key))) or 0.0
        if source_total <= 0 or target_total <= 0:
            continue
        player_id = str(transfer.get("player_id") or "")
        name_key = normalize_player_name(transfer.get("player_name"))
        target_player = target_index.get((season, str(team_key), player_id, name_key))
        if target_player is None:
            target_player = target_index.get((season, str(team_key), "", name_key))
        if target_player is None:
            continue
        source_share = (as_float(transfer.get("minutes")) or 0.0) / source_total
        adjusted_share = adjusted_transfer_source_share(
            source_share,
            as_float(transfer.get("source_context_multiplier")),
        )
        target_share = clip((as_float(target_player.get("minutes")) or 0.0) / target_total, 0.0, 0.35)
        position = classify_position(target_player.get("position") or transfer.get("position"))
        bucket = transfer_share_bucket(adjusted_share)
        by_bucket_position.setdefault((bucket, position), []).append(target_share)
        by_bucket.setdefault(bucket, []).append(target_share)
        overall.append(target_share)

    return {
        "by_bucket_position": {
            f"{bucket}:{position}": summarize_transfer_calibration(values)
            for (bucket, position), values in by_bucket_position.items()
        },
        "by_bucket": {bucket: summarize_transfer_calibration(values) for bucket, values in by_bucket.items()},
        "overall": summarize_transfer_calibration(overall),
    }


def lookup_transfer_role_minutes_share(
    calibration: dict[str, Any] | None,
    *,
    adjusted_source_share: float | None,
    position_bucket: str,
) -> float | None:
    bucket = transfer_share_bucket(adjusted_source_share)
    if calibration:
        exact = (calibration.get("by_bucket_position") or {}).get(f"{bucket}:{position_bucket}") or {}
        if (as_float(exact.get("sample")) or 0.0) >= MIN_TRANSFER_CALIBRATION_SAMPLE:
            return as_float(exact.get("avg_minutes_share"))
        by_bucket = (calibration.get("by_bucket") or {}).get(bucket) or {}
        if (as_float(by_bucket.get("sample")) or 0.0) >= MIN_TRANSFER_CALIBRATION_SAMPLE:
            return as_float(by_bucket.get("avg_minutes_share"))
        overall = calibration.get("overall") or {}
        if (as_float(overall.get("sample")) or 0.0) > 0:
            return as_float(overall.get("avg_minutes_share"))
    return clip((adjusted_source_share or 0.0) * 0.9, 0.0, 0.22)


def project_transfer_minutes_share(
    player: dict[str, Any],
    team_minutes_index: dict[tuple[int, str], float] | None,
    calibration: dict[str, Any] | None,
) -> float | None:
    if not team_minutes_index:
        return None
    source_season = as_int(player.get("source_season"))
    source_team_key = player.get("source_team_key") or canonical_team_key(player.get("source_team"))
    if source_season is None or not source_team_key:
        return None
    source_total = team_minutes_index.get((source_season, str(source_team_key))) or 0.0
    if source_total <= 0:
        return None
    source_share = (as_float(player.get("minutes")) or 0.0) / source_total
    adjusted_share = adjusted_transfer_source_share(
        source_share,
        as_float(player.get("source_context_multiplier")),
    )
    return lookup_transfer_role_minutes_share(
        calibration,
        adjusted_source_share=adjusted_share,
        position_bucket=classify_position(player.get("position")),
    )


def hs_recruit_player_features(
    row: dict[str, Any],
    prior_roster_players: list[dict[str, Any]],
    incoming_transfer_players: list[dict[str, Any]],
    incoming_hs_players: list[dict[str, Any]],
    calibration: dict[str, Any] | None = None,
    team_minutes_index: dict[tuple[int, str], float] | None = None,
    transfer_role_calibration: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    if not incoming_hs_players:
        return {
            "player_talent_percentile": None,
            "need_adjusted_impact_percentile": None,
            "returner_role_units": None,
            "transfer_role_units": None,
            "hs_role_units": None,
            "returner_projected_minutes_share_total": None,
            "transfer_projected_minutes_share_total": None,
            "projected_minutes_share_total": None,
            "top_projected_minutes_share": None,
            "pg_need": None,
            "big_need": None,
            "backcourt_need": None,
            "frontcourt_need": None,
        }

    known_pg = 0.0
    known_backcourt = 0.0
    known_big = 0.0
    known_frontcourt = 0.0
    returner_projected_minutes_share_total = 0.0
    transfer_projected_minutes_share_total = 0.0
    for player in prior_roster_players:
        if not status_is_probable_returner(player):
            continue
        projected_minutes_share = project_returner_minutes_share(player, team_minutes_index)
        role_fill = role_fill_from_minutes_share(projected_minutes_share)
        returner_projected_minutes_share_total += projected_minutes_share or 0.0
        bucket = classify_position(player.get("position"))
        if bucket == "pg":
            known_pg += role_fill
            known_backcourt += role_fill
        elif bucket == "guard":
            known_backcourt += role_fill
        elif bucket == "big":
            known_big += role_fill
            known_frontcourt += role_fill
        else:
            known_frontcourt += role_fill

    for player in incoming_transfer_players:
        projected_minutes_share = project_transfer_minutes_share(
            player,
            team_minutes_index,
            transfer_role_calibration,
        )
        role_fill = role_fill_from_minutes_share(projected_minutes_share)
        transfer_projected_minutes_share_total += projected_minutes_share or 0.0
        bucket = classify_position(player.get("position"))
        if bucket == "pg":
            known_pg += role_fill
            known_backcourt += role_fill
        elif bucket == "guard":
            known_backcourt += role_fill
        elif bucket == "big":
            known_big += role_fill
            known_frontcourt += role_fill
        else:
            known_frontcourt += role_fill

    pg_need = slot_need(known_pg, POINT_GUARD_TARGET)
    big_need = slot_need(known_big, CENTER_TARGET)
    backcourt_need = slot_need(known_backcourt, BACKCOURT_TARGET)
    frontcourt_need = slot_need(known_frontcourt, FRONTCOURT_TARGET)

    talent_values: list[float] = []
    impact_pairs: list[tuple[float, float]] = []
    talent_pairs: list[tuple[float, float]] = []
    projected_minutes_shares: list[float] = []
    recruits_sorted = sorted(
        incoming_hs_players,
        key=lambda player: (-(recruit_percentile(player) or -1.0), str(player.get("player_name") or "")),
    )
    for index, player in enumerate(recruits_sorted):
        base = recruit_percentile(player)
        if base is None:
            continue
        bucket = classify_position(player.get("position_abbr") or player.get("position"))
        if bucket == "pg":
            need = average_existing(pg_need, backcourt_need)
        elif bucket == "guard":
            need = backcourt_need
        elif bucket == "big":
            need = average_existing(big_need, frontcourt_need)
        else:
            need = frontcourt_need
        historical_minutes_share = lookup_hs_role_minutes_share(
            calibration,
            recruit_rank=as_float(player.get("industry_rank")),
            position_bucket=bucket,
        ) or 0.06
        projected_minutes_share = clip(
            historical_minutes_share * (0.65 + 0.85 * (need or 0.0)),
            0.0,
            0.22,
        )
        role_fill = clip(projected_minutes_share / FRESHMAN_ROLE_FILL_REFERENCE_SHARE, 0.0, 1.0)
        talent_values.append(base)
        projected_minutes_shares.append(projected_minutes_share)
        talent_pairs.append((base, historical_minutes_share))
        impact_pairs.append((base, projected_minutes_share))

        if bucket == "pg":
            known_pg += role_fill
            known_backcourt += role_fill
        elif bucket == "guard":
            known_backcourt += role_fill
        elif bucket == "big":
            known_big += role_fill
            known_frontcourt += role_fill
        else:
            known_frontcourt += role_fill

        pg_need = slot_need(known_pg, POINT_GUARD_TARGET)
        big_need = slot_need(known_big, CENTER_TARGET)
        backcourt_need = slot_need(known_backcourt, BACKCOURT_TARGET)
        frontcourt_need = slot_need(known_frontcourt, FRONTCOURT_TARGET)

    return {
        "player_talent_percentile": weighted_average_existing(*talent_pairs) or average_existing(*talent_values),
        "need_adjusted_impact_percentile": weighted_average_existing(*impact_pairs),
        "returner_role_units": total_role_units_from_minutes_share(returner_projected_minutes_share_total),
        "transfer_role_units": total_role_units_from_minutes_share(transfer_projected_minutes_share_total),
        "hs_role_units": total_role_units_from_minutes_share(
            sum(projected_minutes_shares) if projected_minutes_shares else None
        ),
        "returner_projected_minutes_share_total": returner_projected_minutes_share_total,
        "transfer_projected_minutes_share_total": transfer_projected_minutes_share_total,
        "projected_minutes_share_total": sum(projected_minutes_shares) if projected_minutes_shares else None,
        "top_projected_minutes_share": max(projected_minutes_shares) if projected_minutes_shares else None,
        "pg_need": pg_need,
        "big_need": big_need,
        "backcourt_need": backcourt_need,
        "frontcourt_need": frontcourt_need,
    }


def coach_talent_proxy_percentile(row: dict[str, Any]) -> float | None:
    return average_existing(
        rank_to_percentile(as_float(row.get("coach_coach_prior_avg_rank_adj_em"))),
        rank_to_percentile(as_float(row.get("coach_coach_prior_last_rank_adj_em"))),
        rank_to_percentile(as_float(row.get("coach_coach_prior_best_rank_adj_em"))),
    )


def program_talent_proxy_percentile(row: dict[str, Any]) -> float | None:
    return average_existing(
        rank_to_percentile(as_float(row.get("program_prior_avg_rank_adj_em"))),
        rank_to_percentile(as_float(row.get("program_prior_last_rank_adj_em"))),
        rank_to_percentile(as_float(row.get("program_prior_best_rank_adj_em"))),
    )


def blend_known_roster_with_proxy(
    *,
    known_signal: float | None,
    roster_completeness: float | None,
    proxy_signal: float | None,
) -> float | None:
    if known_signal is None:
        return proxy_signal
    if roster_completeness is None:
        return known_signal
    completeness = clip(roster_completeness, 0.0, 1.0)
    if proxy_signal is None:
        return known_signal
    return (known_signal * completeness) + (proxy_signal * (1.0 - completeness))


def roster_composition_weights(
    *,
    returning_players: float | None,
    hs_players: float | None,
    transfer_players: float | None,
) -> dict[str, float | None]:
    buckets = {
        "returner": max(returning_players or 0.0, 0.0),
        "hs": max(hs_players or 0.0, 0.0),
        "transfer": max(transfer_players or 0.0, 0.0),
    }
    known_roster_players = sum(buckets.values())
    if known_roster_players <= 0:
        return {
            "known_roster_players": None,
            "returner_share": None,
            "hs_share": None,
            "transfer_share": None,
            "newcomer_share": None,
            "returner_impact_share": None,
            "hs_impact_share": None,
            "transfer_impact_share": None,
            "newcomer_impact_share": None,
        }

    returner_share = buckets["returner"] / known_roster_players
    hs_share = buckets["hs"] / known_roster_players
    transfer_share = buckets["transfer"] / known_roster_players
    impact_units = {
        "returner": returner_share * RETURNER_IMMEDIATE_IMPACT_WEIGHT,
        "hs": hs_share * HS_IMMEDIATE_IMPACT_WEIGHT,
        "transfer": transfer_share * TRANSFER_IMMEDIATE_IMPACT_WEIGHT,
    }
    total_impact_units = sum(impact_units.values())
    if total_impact_units > 0:
        returner_impact_share = impact_units["returner"] / total_impact_units
        hs_impact_share = impact_units["hs"] / total_impact_units
        transfer_impact_share = impact_units["transfer"] / total_impact_units
    else:
        returner_impact_share = returner_share
        hs_impact_share = hs_share
        transfer_impact_share = transfer_share
    return {
        "known_roster_players": known_roster_players,
        "returner_share": returner_share,
        "hs_share": hs_share,
        "transfer_share": transfer_share,
        "newcomer_share": hs_share + transfer_share,
        "returner_impact_share": returner_impact_share,
        "hs_impact_share": hs_impact_share,
        "transfer_impact_share": transfer_impact_share,
        "newcomer_impact_share": hs_impact_share + transfer_impact_share,
    }


def build_model_rows(
    preseason_rows: list[dict[str, Any]],
    coach_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    roster_rows: list[dict[str, Any]] | None = None,
    roster_player_rows: list[dict[str, Any]] | None = None,
    on3_rows: list[dict[str, Any]] | None = None,
    transfer_rows: list[dict[str, Any]] | None = None,
    transfer_player_rows: list[dict[str, Any]] | None = None,
    hs_recruit_player_rows: list[dict[str, Any]] | None = None,
    program_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    coach_index = index_by_season_team(coach_rows)
    target_index = index_by_season_team(target_rows)
    roster_index = index_by_season_team(roster_rows or [])
    roster_player_index = index_player_rows_by_season_team(roster_player_rows or [])
    on3_index = index_by_season_team(on3_rows or [])
    transfer_index = index_by_season_team(transfer_rows or [])
    transfer_player_index = index_player_rows_by_season_team(transfer_player_rows or [])
    hs_recruit_player_index = index_player_rows_by_season_team(hs_recruit_player_rows or [])
    program_index = index_by_season_team(program_rows or [])
    team_minutes_index = build_team_minutes_index(roster_player_rows or [])
    hs_recruit_calibration = build_hs_recruit_role_calibration(
        roster_player_rows or [],
        hs_recruit_player_rows or [],
    )
    transfer_role_calibration = build_transfer_role_calibration(
        roster_player_rows or [],
        transfer_player_rows or [],
        team_minutes_index,
    )
    rows: list[dict[str, Any]] = []

    for base in preseason_rows:
        key = (int(base["season"]), str(base["team_key"]))
        prior_roster_key = (int(base["season"]) - 1, str(base["team_key"]))
        coach = coach_index.get(key)
        target = target_index.get(key)
        prior_roster = roster_index.get(prior_roster_key)
        on3 = on3_index.get(key)
        transfer = transfer_index.get(key)
        incoming_transfer_players = transfer_player_index.get(key, [])
        incoming_hs_players = hs_recruit_player_index.get(key, [])
        program = program_index.get(key)
        prior_roster_players = roster_player_index.get(prior_roster_key, [])
        prior_roster_features = prefixed(
            prior_roster,
            "prior_roster_",
            {"season", "team_key", "competition_id", "team_id", "team_market", "team_name", "conference_id"},
        )
        if prior_roster:
            prior_roster_features["prior_roster_source_season"] = prior_roster.get("season")
            prior_roster_features["prior_roster_source_competition_id"] = prior_roster.get("competition_id")
        row = {
            **base,
            **prefixed(coach, "coach_", {"season", "team_id", "team_name", "team_key", "conference", "coach_key"}),
            **prior_roster_features,
            **prefixed(on3, "incoming_", {"season", "team", "team_key"}),
            **prefixed(transfer, "incoming_", {"season", "team", "team_key"}),
            **prefixed(program, "program_", {"season", "team", "team_key", "conference"}),
            **prefixed(target, "target_", {"season", "team", "team_key", "conference"}),
        }
        add_roster_talent_features(
            row,
            prior_roster_players=prior_roster_players,
            incoming_transfer_players=incoming_transfer_players,
            incoming_hs_players=incoming_hs_players,
            hs_recruit_calibration=hs_recruit_calibration,
            team_minutes_index=team_minutes_index,
            transfer_role_calibration=transfer_role_calibration,
        )
        rows.append(row)

    add_conference_schedule_environment(rows)
    rows.sort(key=lambda row: (row["season"], row["team"]))
    return rows


def index_player_rows_by_season_team(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    index: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        season = as_int(row.get("season"))
        team = row.get("team") or row.get("team_name") or row.get("team_market")
        team_key = row.get("team_key") or canonical_team_key(team)
        if season is None or not team_key:
            continue
        index.setdefault((season, str(team_key)), []).append(row)
    return index


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
