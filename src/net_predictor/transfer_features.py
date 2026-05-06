"""Incoming transfer production features from CBB Analytics player aggregates."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from net_predictor.coach_factor import canonical_team_key, normalize_text


COUNTING_STATS = (
    "minutes",
    "possessions",
    "points",
    "field_goal_attempts",
    "free_throw_attempts",
    "assists",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "steals",
    "blocks",
    "turnovers",
    "warp",
    "win_shares",
    "offensive_win_shares",
    "defensive_win_shares",
)
RATE_STATS = (
    "usage_pct",
    "ortg_player",
    "drtg_player",
    "per",
    "warp_per_40",
    "win_shares_per_40",
    "effective_fg_pct",
    "true_shooting_pct",
    "assist_pct",
    "rebound_pct",
    "turnover_pct",
    "rapm",
    "offensive_rapm",
    "defensive_rapm",
)
CONTEXT_ADJUSTED_STATS = ("points", "assists", "rebounds", "warp", "win_shares")
HIGH_MAJOR_CONFERENCES = {"ACC", "B10", "B12", "BE", "P12", "SEC"}
PERCENTILE_STATS = (
    "cbb_transfer_players",
    "cbb_transfer_500_minute_players",
    "cbb_transfer_200_minute_players",
    "cbb_transfer_high_usage_players",
    "cbb_transfer_positive_warp_players",
    "cbb_transfer_plus_one_warp_players",
    "cbb_transfer_minutes",
    "cbb_transfer_warp",
    "cbb_transfer_win_shares",
    "cbb_transfer_source_adjusted_warp",
    "cbb_transfer_source_adjusted_win_shares",
    "cbb_transfer_minutes_weighted_per",
    "cbb_transfer_minutes_weighted_net_rating",
    "cbb_transfer_minutes_weighted_source_adj_em",
)
PLAYER_ID_FIELDS = {"playerid", "player id", "player_id"}
PLAYER_NAME_FIELDS = {"player", "name", "player name", "player_name", "full name", "full_name"}
SOURCE_TEAM_FIELDS = {
    "source team",
    "source_team",
    "from",
    "from team",
    "from_team",
    "prior team",
    "prior_team",
    "previous team",
    "previous_team",
    "old team",
    "old_team",
}
DESTINATION_TEAM_FIELDS = {
    "destination team",
    "destination_team",
    "to",
    "to team",
    "to_team",
    "new team",
    "new_team",
    "current team",
    "current_team",
    "committed team",
    "committed_team",
    "school",
}
PORTAL_STATUS_FIELDS = {"portalstatus", "portal_status", "status", "rawstatus", "raw_status"}
SOURCE_DIVISION_FIELDS = {"divisionid", "division_id", "source_division", "division"}
DESTINATION_DIVISION_FIELDS = {
    "divisionidto",
    "division_id_to",
    "destination_division",
    "division_to",
}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def numeric(value: Any) -> float:
    return as_float(value) or 0.0


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


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


def kenpom_context_by_team(kenpom_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    context: dict[tuple[int, str], dict[str, Any]] = {}
    for ratings_path in sorted(kenpom_dir.glob("*/ratings.json")):
        rows = read_json_rows(ratings_path)
        for row in rows:
            season = int(row.get("Season") or 0)
            team = row.get("TeamName")
            if not season or not team:
                continue
            context[(season, canonical_team_key(team))] = {
                "source_kenpom_adj_em": as_float(row.get("AdjEM")),
                "source_kenpom_rank_adj_em": as_float(row.get("RankAdjEM")),
                "source_kenpom_sos": as_float(row.get("SOS")),
                "source_kenpom_rank_sos": as_float(row.get("RankSOS")),
                "source_kenpom_conference": row.get("ConfShort"),
            }
    return context


def source_context_multiplier(context: dict[str, Any] | None) -> float:
    if not context:
        return 1.0
    adj_em = as_float(context.get("source_kenpom_adj_em"))
    if adj_em is None:
        return 1.0
    rank = as_float(context.get("source_kenpom_rank_adj_em"))
    conference = context.get("source_kenpom_conference")

    multiplier = 1 + (adj_em / 60)
    if conference in HIGH_MAJOR_CONFERENCES:
        multiplier += 0.05
    if rank is not None:
        if rank <= 25:
            multiplier += 0.10
        elif rank <= 50:
            multiplier += 0.06
        elif rank <= 100:
            multiplier += 0.03
        elif rank >= 300:
            multiplier -= 0.12
        elif rank >= 250:
            multiplier -= 0.08
        elif rank >= 200:
            multiplier -= 0.04
    return min(1.45, max(0.55, multiplier))


def canonical_player_key(value: Any) -> str:
    words = [word for word in normalize_text(value).split() if word not in NAME_SUFFIXES]
    return " ".join(words)


def detect_ledger_column(fieldnames: list[str], candidates: set[str]) -> str | None:
    normalized = {normalize_text(name): name for name in fieldnames}
    for candidate in candidates:
        match = normalized.get(normalize_text(candidate))
        if match:
            return match
    return None


def player_row_indexes(
    player_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_name_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in player_rows:
        player_id = str(row.get("player_id") or "").strip()
        player_key = canonical_player_key(row.get("player_name"))
        source_team_key = canonical_team_key(row.get("team_market") or row.get("team_name"))
        if player_id:
            by_id[player_id] = row
        if player_key:
            by_name[player_key].append(row)
            if source_team_key:
                by_name_source[(player_key, source_team_key)].append(row)
    return by_id, by_name_source, by_name


def supplement_player_rows_with_roster_rows(
    player_rows: list[dict[str, Any]],
    roster_rows: list[dict[str, Any]],
    *,
    season: int | None,
) -> list[dict[str, Any]]:
    if not roster_rows:
        return player_rows

    combined = [dict(row) for row in player_rows]
    by_id, by_name_source, _ = player_row_indexes(combined)
    for roster_row in roster_rows:
        player_id = str(roster_row.get("playerId") or roster_row.get("player_id") or "").strip()
        player_name = roster_row.get("fullName") or roster_row.get("player_name") or roster_row.get("playerName")
        team_market = roster_row.get("teamMarket") or roster_row.get("team_market")
        team_name = roster_row.get("teamName") or roster_row.get("team_name")
        if not player_name or not team_market:
            continue

        player_key = canonical_player_key(player_name)
        source_team_key = canonical_team_key(team_market)
        if player_id and player_id in by_id:
            continue
        if player_key and source_team_key and by_name_source.get((player_key, source_team_key)):
            continue

        placeholder = {
            "season": season,
            "competition_id": roster_row.get("competitionId") or roster_row.get("competition_id"),
            "team_id": roster_row.get("teamId") or roster_row.get("team_id"),
            "team_market": team_market,
            "team_name": team_name,
            "conference_id": roster_row.get("conferenceId") or roster_row.get("conference_id"),
            "player_id": player_id or None,
            "player_name": player_name,
            "class_year": roster_row.get("classYr") or roster_row.get("class_year"),
            "position": roster_row.get("position"),
            "height": roster_row.get("height"),
            "games_started": 0.0,
            "minutes": 0.0,
            "possessions": 0.0,
            "points": 0.0,
            "field_goal_attempts": 0.0,
            "free_throw_attempts": 0.0,
            "assists": 0.0,
            "offensive_rebounds": 0.0,
            "defensive_rebounds": 0.0,
            "rebounds": 0.0,
            "steals": 0.0,
            "blocks": 0.0,
            "turnovers": 0.0,
            "personal_fouls": 0.0,
            "warp": 0.0,
            "win_shares": 0.0,
            "offensive_win_shares": 0.0,
            "defensive_win_shares": 0.0,
            "usage_pct": None,
            "ortg_player": None,
            "drtg_player": None,
            "per": None,
            "warp_per_40": None,
            "win_shares_per_40": None,
            "effective_fg_pct": None,
            "true_shooting_pct": None,
            "assist_pct": None,
            "rebound_pct": None,
            "turnover_pct": None,
            "rapm": None,
            "offensive_rapm": None,
            "defensive_rapm": None,
            "source_row_type": "competition_team_players_placeholder",
        }
        combined.append(placeholder)
        if player_id:
            by_id[player_id] = placeholder
        by_name_source[(player_key, source_team_key)].append(placeholder)
    return combined


def transfer_rows_from_ledger(
    ledger_rows: list[dict[str, Any]],
    player_rows: list[dict[str, Any]],
    kenpom_context: dict[tuple[int, str], dict[str, Any]],
    *,
    destination_season: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not ledger_rows:
        return [], []

    fieldnames = list(ledger_rows[0].keys())
    id_column = detect_ledger_column(fieldnames, PLAYER_ID_FIELDS)
    player_column = detect_ledger_column(fieldnames, PLAYER_NAME_FIELDS)
    source_column = detect_ledger_column(fieldnames, SOURCE_TEAM_FIELDS)
    destination_column = detect_ledger_column(fieldnames, DESTINATION_TEAM_FIELDS)
    portal_status_column = detect_ledger_column(fieldnames, PORTAL_STATUS_FIELDS)
    source_division_column = detect_ledger_column(fieldnames, SOURCE_DIVISION_FIELDS)
    destination_division_column = detect_ledger_column(fieldnames, DESTINATION_DIVISION_FIELDS)
    if not player_column or not source_column or not destination_column:
        raise ValueError(
            "Could not detect transfer ledger columns. Need player/source team/destination team columns."
        )

    by_id, by_name_source, by_name = player_row_indexes(player_rows)
    transfers: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for ledger_row in ledger_rows:
        ledger_player = ledger_row.get(player_column)
        source_team = ledger_row.get(source_column)
        destination_team = ledger_row.get(destination_column)
        if not ledger_player or not source_team or not destination_team:
            continue
        portal_status = str(ledger_row.get(portal_status_column) or "").strip() if portal_status_column else ""
        source_division = str(ledger_row.get(source_division_column) or "").strip() if source_division_column else ""
        destination_division = (
            str(ledger_row.get(destination_division_column) or "").strip()
            if destination_division_column
            else ""
        )

        if portal_status and normalize_text(portal_status) != "transferred":
            continue

        player_id = str(ledger_row.get(id_column) or "").strip() if id_column else ""
        player_key = canonical_player_key(ledger_player)
        source_team_key = canonical_team_key(source_team)
        destination_team_key = canonical_team_key(destination_team)
        if source_team_key == destination_team_key:
            continue

        matched_row: dict[str, Any] | None = None
        match_method = ""
        if player_id and player_id in by_id:
            matched_row = by_id[player_id]
            match_method = "player_id"
        else:
            exact_candidates = by_name_source.get((player_key, source_team_key), [])
            if len(exact_candidates) == 1:
                matched_row = exact_candidates[0]
                match_method = "player_name_source_team"
            else:
                name_candidates = by_name.get(player_key, [])
                if len(name_candidates) == 1:
                    matched_row = name_candidates[0]
                    match_method = "player_name_only"

        if matched_row is None:
            unmatched.append(
                {
                    "ledger_player_name": ledger_player,
                    "ledger_source_team": source_team,
                    "ledger_destination_team": destination_team,
                    "ledger_player_id": player_id or None,
                    "portal_status": portal_status or None,
                    "source_division": source_division or None,
                    "destination_division": destination_division or None,
                    "reason": "no_match",
                }
            )
            continue

        source_season = int(matched_row.get("season") or 0)
        if not source_season:
            unmatched.append(
                {
                    "ledger_player_name": ledger_player,
                    "ledger_source_team": source_team,
                    "ledger_destination_team": destination_team,
                    "ledger_player_id": player_id or None,
                    "portal_status": portal_status or None,
                    "source_division": source_division or None,
                    "destination_division": destination_division or None,
                    "reason": "matched_row_missing_season",
                }
            )
            continue

        context = kenpom_context.get((source_season, source_team_key), {})
        multiplier = source_context_multiplier(context)
        item: dict[str, Any] = {
            "source_season": source_season,
            "season": destination_season or (source_season + 1),
            "team": destination_team,
            "team_key": destination_team_key,
            "source_team": source_team,
            "source_team_key": source_team_key,
            "player_id": matched_row.get("player_id"),
            "player_name": matched_row.get("player_name"),
            "class_year": matched_row.get("class_year"),
            "position": matched_row.get("position"),
            "height": matched_row.get("height"),
            "source_context_multiplier": multiplier,
            "transfer_match_method": match_method,
            "ledger_player_name": ledger_player,
            "ledger_source_team": source_team,
            "ledger_destination_team": destination_team,
            **context,
        }
        for column in COUNTING_STATS + RATE_STATS:
            item[column] = as_float(matched_row.get(column))
        for column in CONTEXT_ADJUSTED_STATS:
            item[f"context_adjusted_{column}"] = numeric(matched_row.get(column)) * multiplier
        transfers.append(item)

    transfers.sort(key=lambda item: (item["season"], item["team_key"], str(item["player_name"])))
    unmatched.sort(key=lambda item: (str(item["ledger_destination_team"]), str(item["ledger_player_name"])))
    return transfers, unmatched


def player_transfer_rows(
    player_rows: list[dict[str, Any]],
    kenpom_context: dict[tuple[int, str], dict[str, Any]],
    *,
    current_roster_transfers: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in player_rows:
        next_team_id = row.get("next_team_id")
        team_id = row.get("team_id")
        next_team_market = row.get("next_team_market")
        source_season = int(row.get("season") or 0)
        if not source_season:
            continue

        if next_team_id not in (None, "") and str(next_team_id) != str(team_id):
            destination_team = next_team_market
            source_team = row.get("team_market")
            source_team_key = str(row.get("team_key") or canonical_team_key(source_team))
        elif current_roster_transfers and is_true(row.get("is_transfer")):
            prior_team = row.get("prior_team_market")
            current_team = row.get("team_market")
            if prior_team in (None, "") or current_team in (None, ""):
                continue
            if canonical_team_key(prior_team) == canonical_team_key(current_team):
                continue
            destination_team = current_team
            source_team = prior_team
            source_team_key = canonical_team_key(prior_team)
        else:
            continue

        if destination_team in (None, ""):
            continue

        destination_team_key = canonical_team_key(destination_team)
        context = kenpom_context.get((source_season, source_team_key), {})
        multiplier = source_context_multiplier(context)

        item: dict[str, Any] = {
            "source_season": source_season,
            "season": source_season + 1,
            "team": destination_team,
            "team_key": destination_team_key,
            "source_team": source_team,
            "source_team_key": source_team_key,
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "class_year": row.get("class_year"),
            "position": row.get("position"),
            "height": row.get("height"),
            "source_context_multiplier": multiplier,
            **context,
        }
        for column in COUNTING_STATS + RATE_STATS:
            item[column] = as_float(row.get(column))
        for column in CONTEXT_ADJUSTED_STATS:
            item[f"context_adjusted_{column}"] = numeric(row.get(column)) * multiplier
        rows.append(item)

    rows.sort(key=lambda item: (item["season"], item["team_key"], str(item["player_name"])))
    return rows


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def weighted_average(rows: list[dict[str, Any]], column: str, weight_column: str = "minutes") -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = as_float(row.get(column))
        weight = numeric(row.get(weight_column))
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def sum_column(rows: list[dict[str, Any]], column: str) -> float:
    return sum(numeric(row.get(column)) for row in rows)


def transfer_summary_rows(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in player_rows:
        grouped[(int(row["season"]), str(row["team_key"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (season, team_key), rows in grouped.items():
        first = rows[0]
        summary: dict[str, Any] = {
            "season": season,
            "team": first["team"],
            "team_key": team_key,
            "cbb_transfer_players": len(rows),
            "cbb_transfer_source_teams": len({row["source_team_key"] for row in rows}),
            "cbb_transfer_500_minute_players": sum(1 for row in rows if numeric(row.get("minutes")) >= 500),
            "cbb_transfer_200_minute_players": sum(1 for row in rows if numeric(row.get("minutes")) >= 200),
            "cbb_transfer_high_usage_players": sum(
                1 for row in rows if numeric(row.get("minutes")) >= 200 and numeric(row.get("usage_pct")) >= 0.20
            ),
            "cbb_transfer_positive_warp_players": sum(1 for row in rows if numeric(row.get("warp")) > 0),
            "cbb_transfer_plus_one_warp_players": sum(1 for row in rows if numeric(row.get("warp")) >= 1.0),
            "cbb_transfer_top_50_source_players": sum(
                1 for row in rows if 0 < numeric(row.get("source_kenpom_rank_adj_em")) <= 50
            ),
            "cbb_transfer_top_100_source_players": sum(
                1 for row in rows if 0 < numeric(row.get("source_kenpom_rank_adj_em")) <= 100
            ),
        }
        for column in COUNTING_STATS:
            summary[f"cbb_transfer_{column}"] = sum_column(rows, column)
        for column in RATE_STATS:
            summary[f"cbb_transfer_minutes_weighted_{column}"] = weighted_average(rows, column)
        for column in CONTEXT_ADJUSTED_STATS:
            summary[f"cbb_transfer_source_adjusted_{column}"] = sum_column(rows, f"context_adjusted_{column}")

        summary["cbb_transfer_minutes_weighted_source_adj_em"] = weighted_average(rows, "source_kenpom_adj_em")
        summary["cbb_transfer_minutes_weighted_source_rank_adj_em"] = weighted_average(rows, "source_kenpom_rank_adj_em")
        summary["cbb_transfer_minutes_weighted_source_sos"] = weighted_average(rows, "source_kenpom_sos")
        summary["cbb_transfer_minutes_weighted_source_context_multiplier"] = weighted_average(
            rows,
            "source_context_multiplier",
        )
        ortg = summary.get("cbb_transfer_minutes_weighted_ortg_player")
        drtg = summary.get("cbb_transfer_minutes_weighted_drtg_player")
        summary["cbb_transfer_minutes_weighted_net_rating"] = (
            ortg - drtg if isinstance(ortg, (int, float)) and isinstance(drtg, (int, float)) else None
        )
        summaries.append(summary)

    summaries.sort(key=lambda item: (item["season"], item["team_key"]))
    add_transfer_percentiles(summaries)
    return summaries


def add_transfer_percentiles(rows: list[dict[str, Any]]) -> None:
    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_season[int(row["season"])].append(row)

    for season_rows in by_season.values():
        for column in PERCENTILE_STATS:
            observed = sorted(
                (
                    numeric(row.get(column)),
                    str(row.get("team_key") or ""),
                    row,
                )
                for row in season_rows
                if row.get(column) not in (None, "")
            )
            if not observed:
                continue
            denominator = max(len(observed) - 1, 1)
            for index, (_, __, row) in enumerate(observed):
                row[f"{column}_percentile"] = index / denominator

        for row in season_rows:
            row["cbb_transfer_production_percentile"] = average_existing(
                as_float(row.get("cbb_transfer_source_adjusted_warp_percentile")),
                as_float(row.get("cbb_transfer_source_adjusted_win_shares_percentile")),
                as_float(row.get("cbb_transfer_minutes_percentile")),
                as_float(row.get("cbb_transfer_500_minute_players_percentile")),
                as_float(row.get("cbb_transfer_minutes_weighted_per_percentile")),
                as_float(row.get("cbb_transfer_minutes_weighted_net_rating_percentile")),
                as_float(row.get("cbb_transfer_minutes_weighted_source_adj_em_percentile")),
            )


def average_existing(*values: float | None) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)
