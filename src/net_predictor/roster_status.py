"""Roster status features from CBB Analytics player aggregates."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SENIOR_CLASS_LABELS = {"senior", "graduate", "grad", "gr"}
STATUS_LABELS = (
    "probable_returner",
    "confirmed_returning_same_team",
    "transferred_out_committed",
    "in_portal_uncommitted_or_pending",
    "draft_prospect_review",
    "senior_eligibility_review",
)
RETURNING_STATUSES = {"probable_returner", "confirmed_returning_same_team"}
LOST_OR_UNCERTAIN_STATUSES = set(STATUS_LABELS) - RETURNING_STATUSES

COUNTING_STATS = (
    "games_started",
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
    "personal_fouls",
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
    "offensive_rebound_pct",
    "defensive_rebound_pct",
    "rebound_pct",
    "steal_pct",
    "block_pct",
    "turnover_pct",
    "rapm",
    "offensive_rapm",
    "defensive_rapm",
)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_senior_class(class_year: Any) -> bool:
    if not isinstance(class_year, str):
        return False
    normalized = class_year.strip().lower()
    return normalized in SENIOR_CLASS_LABELS or "senior" in normalized or "grad" in normalized


def classify_player(row: dict[str, Any]) -> str:
    team_id = row.get("teamId")
    next_team_id = row.get("nextTeamId")

    if next_team_id not in (None, "") and str(next_team_id) == str(team_id):
        return "confirmed_returning_same_team"

    if next_team_id not in (None, "") and str(next_team_id) != str(team_id):
        return "transferred_out_committed"

    if as_bool(row.get("inPortalAfterSeason")) or as_bool(row.get("willTransfer")):
        return "in_portal_uncommitted_or_pending"

    if as_bool(row.get("isDraftProspect")):
        return "draft_prospect_review"

    if is_senior_class(row.get("classYr")):
        return "senior_eligibility_review"

    return "probable_returner"


def player_status_rows(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for row in player_rows:
        status = classify_player(row)
        mins = as_float(row.get("mins"))
        poss = as_float(row.get("poss"))
        pts = as_float(row.get("ptsScored"))

        statuses.append(
            {
                "competition_id": row.get("competitionId"),
                "team_id": row.get("teamId"),
                "team_market": row.get("teamMarket"),
                "team_name": row.get("teamName"),
                "conference_id": row.get("conferenceId"),
                "player_id": row.get("playerId"),
                "player_name": row.get("fullName"),
                "class_year": row.get("classYr"),
                "position": row.get("position"),
                "height": row.get("height"),
                "status": status,
                "next_team_id": row.get("nextTeamId"),
                "next_team_market": row.get("nextTeamMarket"),
                "prior_team_id": row.get("priorTeamId"),
                "prior_team_market": row.get("priorTeamMarket"),
                "is_transfer": row.get("isTransfer"),
                "is_draft_prospect": row.get("isDraftProspect"),
                "will_transfer": row.get("willTransfer"),
                "in_portal_after_season": row.get("inPortalAfterSeason"),
                "games_played": row.get("gp"),
                "games_started": row.get("gs"),
                "minutes": mins,
                "possessions": poss,
                "points": pts,
                "field_goal_attempts": row.get("fga"),
                "free_throw_attempts": row.get("fta"),
                "assists": row.get("ast"),
                "offensive_rebounds": row.get("orb"),
                "defensive_rebounds": row.get("drb"),
                "rebounds": row.get("reb"),
                "steals": row.get("stl"),
                "blocks": row.get("blk"),
                "turnovers": row.get("tov"),
                "personal_fouls": row.get("pf"),
                "usage_pct": row.get("usagePct"),
                "ortg_player": row.get("ortgPlayer"),
                "drtg_player": row.get("drtgPlayer"),
                "effective_fg_pct": row.get("efgPct"),
                "true_shooting_pct": row.get("tsPct"),
                "assist_pct": row.get("astPct"),
                "offensive_rebound_pct": row.get("orbPct"),
                "defensive_rebound_pct": row.get("drbPct"),
                "rebound_pct": row.get("rebPct"),
                "steal_pct": row.get("stlPct"),
                "block_pct": row.get("blkPct"),
                "turnover_pct": row.get("tovPct"),
                "per": row.get("per"),
                "warp": row.get("warp"),
                "warp_per_40": row.get("warpP40"),
                "win_shares": row.get("ws"),
                "offensive_win_shares": row.get("ows"),
                "defensive_win_shares": row.get("dws"),
                "win_shares_per_40": row.get("wsP40"),
                "rapm": row.get("rapm"),
                "offensive_rapm": row.get("orapm"),
                "defensive_rapm": row.get("drapm"),
            }
        )
    return statuses


def pct(part: float, total: float) -> float | None:
    if total <= 0:
        return None
    return part / total


def sum_stat(rows: list[dict[str, Any]], column: str) -> float:
    return sum(as_float(row.get(column)) for row in rows)


def weighted_average(rows: list[dict[str, Any]], column: str, weight_column: str = "minutes") -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for row in rows:
        value = as_float(row.get(column))
        weight = as_float(row.get(weight_column))
        if weight <= 0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return weighted_sum / weight_sum


def count_players_at_least(rows: list[dict[str, Any]], column: str, threshold: float) -> int:
    return sum(1 for row in rows if as_float(row.get(column)) >= threshold)


def add_group_summary(
    summary: dict[str, Any],
    *,
    prefix: str,
    rows: list[dict[str, Any]],
    totals: dict[str, float],
) -> None:
    summary[f"{prefix}_players"] = len(rows)
    for column in COUNTING_STATS:
        value = sum_stat(rows, column)
        summary[f"{prefix}_{column}"] = value
        summary[f"{prefix}_{column}_pct"] = pct(value, totals.get(column, 0.0))

    for column in RATE_STATS:
        summary[f"{prefix}_minutes_weighted_{column}"] = weighted_average(rows, column)

    summary[f"{prefix}_high_usage_players"] = sum(
        1
        for row in rows
        if as_float(row.get("minutes")) >= 200 and as_float(row.get("usage_pct")) >= 0.20
    )
    summary[f"{prefix}_double_digit_per_players"] = sum(
        1 for row in rows if as_float(row.get("minutes")) >= 200 and as_float(row.get("per")) >= 10
    )
    summary[f"{prefix}_positive_warp_players"] = count_players_at_least(rows, "warp", 0.01)
    summary[f"{prefix}_plus_one_warp_players"] = count_players_at_least(rows, "warp", 1.0)

    ortg = summary.get(f"{prefix}_minutes_weighted_ortg_player")
    drtg = summary.get(f"{prefix}_minutes_weighted_drtg_player")
    summary[f"{prefix}_minutes_weighted_net_rating"] = (
        ortg - drtg if isinstance(ortg, (int, float)) and isinstance(drtg, (int, float)) else None
    )


def add_core_returning_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    total_minutes = sum_stat(rows, "minutes")
    returning_rows = [row for row in rows if row["status"] in RETURNING_STATUSES]

    for sort_column in ("minutes", "possessions", "points", "warp", "win_shares"):
        ordered = sorted(rows, key=lambda row: as_float(row.get(sort_column)), reverse=True)
        for cutoff in (3, 5, 7):
            core = ordered[:cutoff]
            core_total = sum_stat(core, sort_column)
            returning_core = [row for row in core if row["status"] in RETURNING_STATUSES]
            returning_value = sum_stat(returning_core, sort_column)
            prefix = f"returning_top_{cutoff}_{sort_column}"
            summary[f"{prefix}_players"] = len(returning_core)
            summary[f"{prefix}_share"] = pct(returning_value, core_total)

    summary["expected_returning_top_7_minutes_roster_share"] = pct(
        sum_stat(
            [row for row in sorted(rows, key=lambda row: as_float(row.get("minutes")), reverse=True)[:7] if row in returning_rows],
            "minutes",
        ),
        total_minutes,
    )


def team_summary_rows(status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        grouped[(row["competition_id"], row["team_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (_, _), rows in grouped.items():
        first = rows[0]
        totals = {column: sum_stat(rows, column) for column in COUNTING_STATS}
        total_minutes = totals["minutes"]
        total_possessions = totals["possessions"]
        total_points = totals["points"]

        summary: dict[str, Any] = {
            "competition_id": first["competition_id"],
            "team_id": first["team_id"],
            "team_market": first["team_market"],
            "team_name": first["team_name"],
            "conference_id": first["conference_id"],
            "players": len(rows),
            "total_minutes": total_minutes,
            "total_possessions": total_possessions,
            "total_points": total_points,
            "total_games_started": totals["games_started"],
            "total_field_goal_attempts": totals["field_goal_attempts"],
            "total_free_throw_attempts": totals["free_throw_attempts"],
            "total_assists": totals["assists"],
            "total_rebounds": totals["rebounds"],
            "total_steals": totals["steals"],
            "total_blocks": totals["blocks"],
            "total_turnovers": totals["turnovers"],
            "total_warp": totals["warp"],
            "total_win_shares": totals["win_shares"],
        }

        for status in STATUS_LABELS:
            status_rows_for_team = [row for row in rows if row["status"] == status]
            minutes = sum(as_float(row["minutes"]) for row in status_rows_for_team)
            possessions = sum(as_float(row["possessions"]) for row in status_rows_for_team)
            points = sum(as_float(row["points"]) for row in status_rows_for_team)

            prefix = status
            summary[f"{prefix}_players"] = len(status_rows_for_team)
            summary[f"{prefix}_minutes"] = minutes
            summary[f"{prefix}_minutes_pct"] = pct(minutes, total_minutes)
            summary[f"{prefix}_possessions"] = possessions
            summary[f"{prefix}_possessions_pct"] = pct(possessions, total_possessions)
            summary[f"{prefix}_points"] = points
            summary[f"{prefix}_points_pct"] = pct(points, total_points)

        add_group_summary(
            summary,
            prefix="expected_returning",
            rows=[row for row in rows if row["status"] in RETURNING_STATUSES],
            totals=totals,
        )
        add_group_summary(
            summary,
            prefix="lost_or_uncertain",
            rows=[row for row in rows if row["status"] in LOST_OR_UNCERTAIN_STATUSES],
            totals=totals,
        )
        add_group_summary(
            summary,
            prefix="confirmed_unavailable",
            rows=[
                row
                for row in rows
                if row["status"]
                in {"transferred_out_committed", "draft_prospect_review", "senior_eligibility_review"}
            ],
            totals=totals,
        )
        add_group_summary(
            summary,
            prefix="portal_pending",
            rows=[row for row in rows if row["status"] == "in_portal_uncommitted_or_pending"],
            totals=totals,
        )
        add_core_returning_summary(summary, rows)

        summaries.append(summary)

    summaries.sort(key=lambda row: (str(row["team_market"]), str(row["team_name"])))
    return summaries


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

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path
