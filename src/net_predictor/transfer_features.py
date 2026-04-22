"""Incoming transfer production features from CBB Analytics player aggregates."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key


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
