"""Season-level modeling table assembly."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key

RETURNER_IMMEDIATE_IMPACT_WEIGHT = 1.0
HS_IMMEDIATE_IMPACT_WEIGHT = 0.65
TRANSFER_IMMEDIATE_IMPACT_WEIGHT = 1.15


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


def add_roster_talent_features(row: dict[str, Any]) -> None:
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
    row["roster_talent_continuity_plus_incoming"] = sum_existing(
        row.get("roster_talent_weighted_returning_core_continuity"),
        row.get("roster_talent_weighted_hs_rank_percentile"),
        row.get("roster_talent_weighted_transfer_rank_percentile"),
    )


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


def sum_existing(*values: Any) -> float | None:
    observed = [as_float(value) for value in values if as_float(value) is not None]
    if not observed:
        return None
    return sum(observed)


def product_if_present(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left * right


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
    on3_rows: list[dict[str, Any]] | None = None,
    transfer_rows: list[dict[str, Any]] | None = None,
    program_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    coach_index = index_by_season_team(coach_rows)
    target_index = index_by_season_team(target_rows)
    roster_index = index_by_season_team(roster_rows or [])
    on3_index = index_by_season_team(on3_rows or [])
    transfer_index = index_by_season_team(transfer_rows or [])
    program_index = index_by_season_team(program_rows or [])
    rows: list[dict[str, Any]] = []

    for base in preseason_rows:
        key = (int(base["season"]), str(base["team_key"]))
        prior_roster_key = (int(base["season"]) - 1, str(base["team_key"]))
        coach = coach_index.get(key)
        target = target_index.get(key)
        prior_roster = roster_index.get(prior_roster_key)
        on3 = on3_index.get(key)
        transfer = transfer_index.get(key)
        program = program_index.get(key)
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
        add_roster_talent_features(row)
        rows.append(row)

    rows.sort(key=lambda row: (row["season"], row["team"]))
    return rows


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
