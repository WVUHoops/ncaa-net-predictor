"""Historical coach metrics from KenPom team and rating seasons."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key, normalize_coach_name


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def season_dirs(kenpom_dir: Path) -> list[Path]:
    dirs = [path for path in kenpom_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(dirs, key=lambda path: int(path.name))


def load_kenpom_team_rows(kenpom_dir: Path, min_season: int | None = None, max_season: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season_dir in season_dirs(kenpom_dir):
        season = int(season_dir.name)
        if min_season is not None and season < min_season:
            continue
        if max_season is not None and season > max_season:
            continue

        teams_path = season_dir / "teams.json"
        if teams_path.exists():
            rows.extend(read_json_rows(teams_path))
    return rows


def load_kenpom_rating_rows(kenpom_dir: Path, min_season: int | None = None, max_season: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season_dir in season_dirs(kenpom_dir):
        season = int(season_dir.name)
        if min_season is not None and season < min_season:
            continue
        if max_season is not None and season > max_season:
            continue

        ratings_path = season_dir / "ratings.json"
        if ratings_path.exists():
            rows.extend(read_json_rows(ratings_path))
    return rows


def load_kenpom_preseason_rows(kenpom_dir: Path, min_season: int | None = None, max_season: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season_dir in season_dirs(kenpom_dir):
        season = int(season_dir.name)
        if min_season is not None and season < min_season:
            continue
        if max_season is not None and season > max_season:
            continue

        preseason_path = season_dir / "archive_preseason.json"
        if preseason_path.exists():
            rows.extend(read_json_rows(preseason_path))
    return rows


def team_index(team_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in team_rows:
        season = as_int(row.get("Season"))
        team_name = row.get("TeamName")
        if season is None or not team_name:
            continue
        index[(season, canonical_team_key(team_name))] = row
    return index


def preseason_index(preseason_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in preseason_rows:
        season = as_int(row.get("Season"))
        team_name = row.get("TeamName")
        if season is None or not team_name:
            continue
        index[(season, canonical_team_key(team_name))] = row
    return index


def delta(final: float | int | None, preseason: float | int | None) -> float | None:
    if final is None or preseason is None:
        return None
    return float(final) - float(preseason)


def rank_delta(final_rank: int | None, preseason_rank: int | None) -> int | None:
    if final_rank is None or preseason_rank is None:
        return None
    return preseason_rank - final_rank


def coach_season_rows(
    team_rows: list[dict[str, Any]],
    rating_rows: list[dict[str, Any]],
    preseason_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    teams_by_season_name = team_index(team_rows)
    preseason_by_season_name = preseason_index(preseason_rows or [])
    rows: list[dict[str, Any]] = []

    for rating in rating_rows:
        season = as_int(rating.get("Season"))
        team_name = rating.get("TeamName")
        coach = rating.get("Coach")
        if season is None or not team_name or not coach:
            continue

        team = teams_by_season_name.get((season, canonical_team_key(team_name)), {})
        preseason = preseason_by_season_name.get((season, canonical_team_key(team_name)), {})
        wins = as_int(rating.get("Wins"))
        losses = as_int(rating.get("Losses"))
        games = (wins or 0) + (losses or 0)
        preseason_adj_em = as_float(preseason.get("AdjEM"))
        preseason_rank_adj_em = as_int(preseason.get("RankAdjEM"))
        final_adj_em = as_float(preseason.get("AdjEMFinal"))
        if final_adj_em is None:
            final_adj_em = as_float(rating.get("AdjEM"))
        final_rank_adj_em = as_int(preseason.get("RankAdjEMFinal"))
        if final_rank_adj_em is None:
            final_rank_adj_em = as_int(rating.get("RankAdjEM"))

        rows.append(
            {
                "season": season,
                "team_id": team.get("TeamID"),
                "team_name": team_name,
                "team_key": canonical_team_key(team_name),
                "conference": rating.get("ConfShort") or team.get("ConfShort"),
                "coach": coach,
                "coach_key": normalize_coach_name(coach),
                "wins": wins,
                "losses": losses,
                "win_pct": (wins / games) if games else None,
                "preseason_adj_em": preseason_adj_em,
                "preseason_rank_adj_em": preseason_rank_adj_em,
                "final_adj_em": final_adj_em,
                "final_rank_adj_em": final_rank_adj_em,
                "adj_em_over_expected": as_float(preseason.get("AdjEMChg"))
                if preseason.get("AdjEMChg") is not None
                else delta(final_adj_em, preseason_adj_em),
                "rank_over_expected": as_int(preseason.get("RankChg"))
                if preseason.get("RankChg") is not None
                else rank_delta(final_rank_adj_em, preseason_rank_adj_em),
                "preseason_adj_oe": as_float(preseason.get("AdjOE")),
                "final_adj_oe": as_float(preseason.get("AdjOEFinal")) or as_float(rating.get("AdjOE")),
                "adj_oe_over_expected": delta(
                    as_float(preseason.get("AdjOEFinal")) or as_float(rating.get("AdjOE")),
                    as_float(preseason.get("AdjOE")),
                ),
                "preseason_adj_de": as_float(preseason.get("AdjDE")),
                "final_adj_de": as_float(preseason.get("AdjDEFinal")) or as_float(rating.get("AdjDE")),
                "adj_de_over_expected": delta(
                    as_float(preseason.get("AdjDE")),
                    as_float(preseason.get("AdjDEFinal")) or as_float(rating.get("AdjDE")),
                ),
                "adj_em": final_adj_em,
                "rank_adj_em": final_rank_adj_em,
                "adj_oe": as_float(rating.get("AdjOE")),
                "rank_adj_oe": as_int(rating.get("RankAdjOE")),
                "adj_de": as_float(rating.get("AdjDE")),
                "rank_adj_de": as_int(rating.get("RankAdjDE")),
                "tempo": as_float(rating.get("AdjTempo")),
                "rank_tempo": as_int(rating.get("RankAdjTempo")),
                "sos": as_float(rating.get("SOS")),
                "rank_sos": as_int(rating.get("RankSOS")),
                "ncsos": as_float(rating.get("NCSOS")),
                "rank_ncsos": as_int(rating.get("RankNCSOS")),
                "seed": as_int(rating.get("Seed")),
                "event": rating.get("Event"),
                "data_through": rating.get("DataThrough"),
            }
        )

    rows.sort(key=lambda row: (row["season"], row["team_name"]))
    return rows


def prior_rows(rows: list[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    current_season = int(current["season"])
    current_coach = current["coach_key"]
    return [
        row
        for row in rows
        if row["coach_key"] == current_coach and int(row["season"]) < current_season
    ]


def same_school_rows(prior: list[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    if current.get("team_id") is not None:
        same = [row for row in prior if row.get("team_id") is not None and str(row["team_id"]) == str(current["team_id"])]
        if same:
            return same
    return [row for row in prior if row["team_key"] == current["team_key"]]


def distinct_programs(rows: list[dict[str, Any]]) -> int:
    programs = {
        str(row.get("team_id")) if row.get("team_id") is not None else row["team_key"]
        for row in rows
    }
    return len(programs)


def summary_from_history(history: list[dict[str, Any]], *, current: dict[str, Any] | None = None) -> dict[str, Any]:
    history = sorted(history, key=lambda row: (int(row["season"]), row["team_name"]))
    last = history[-1] if history else None
    last_3 = history[-3:]
    last_5 = history[-5:]
    wins = sum(row["wins"] or 0 for row in history)
    losses = sum(row["losses"] or 0 for row in history)
    games = wins + losses
    ranks = [row["rank_adj_em"] for row in history if row["rank_adj_em"] is not None]
    seeds = [row["seed"] for row in history if row["seed"] is not None]
    over_expected = [row["adj_em_over_expected"] for row in history if row["adj_em_over_expected"] is not None]
    rank_over_expected = [row["rank_over_expected"] for row in history if row["rank_over_expected"] is not None]

    same_school = same_school_rows(history, current) if current is not None else []
    current_season = int(current["season"]) if current is not None else None

    return {
        "coach_prior_seasons": len(history),
        "coach_prior_program_count": distinct_programs(history),
        "coach_prior_win_pct": (wins / games) if games else None,
        "coach_prior_avg_adj_em": mean([row["adj_em"] for row in history]),
        "coach_prior_last3_avg_adj_em": mean([row["adj_em"] for row in last_3]),
        "coach_prior_last5_avg_adj_em": mean([row["adj_em"] for row in last_5]),
        "coach_prior_best_adj_em": max((row["adj_em"] for row in history if row["adj_em"] is not None), default=None),
        "coach_prior_avg_adj_em_over_expected": mean(over_expected),
        "coach_prior_last3_avg_adj_em_over_expected": mean([row["adj_em_over_expected"] for row in last_3]),
        "coach_prior_last5_avg_adj_em_over_expected": mean([row["adj_em_over_expected"] for row in last_5]),
        "coach_prior_best_adj_em_over_expected": max(over_expected, default=None),
        "coach_prior_worst_adj_em_over_expected": min(over_expected, default=None),
        "coach_prior_positive_adj_em_over_expected_rate": rate(
            [row["adj_em_over_expected"] > 0 for row in history if row["adj_em_over_expected"] is not None]
        ),
        "coach_prior_big_overperform_rate": rate(
            [row["adj_em_over_expected"] >= 5 for row in history if row["adj_em_over_expected"] is not None]
        ),
        "coach_prior_big_underperform_rate": rate(
            [row["adj_em_over_expected"] <= -5 for row in history if row["adj_em_over_expected"] is not None]
        ),
        "coach_prior_avg_rank_over_expected": mean(rank_over_expected),
        "coach_prior_avg_rank_adj_em": mean(ranks),
        "coach_prior_best_rank_adj_em": min(ranks) if ranks else None,
        "coach_prior_top25_rate": rate([row["rank_adj_em"] <= 25 for row in history if row["rank_adj_em"] is not None]),
        "coach_prior_top50_rate": rate([row["rank_adj_em"] <= 50 for row in history if row["rank_adj_em"] is not None]),
        "coach_prior_top100_rate": rate([row["rank_adj_em"] <= 100 for row in history if row["rank_adj_em"] is not None]),
        "coach_prior_ncaa_bid_rate": rate([row["event"] == "NCAA" for row in history]),
        "coach_prior_avg_seed": mean(seeds),
        "coach_prior_avg_sos": mean([row["sos"] for row in history]),
        "coach_prior_avg_ncsos": mean([row["ncsos"] for row in history]),
        "coach_prior_same_school_seasons": len(same_school),
        "coach_prior_same_school_avg_adj_em": mean([row["adj_em"] for row in same_school]),
        "coach_prior_same_school_avg_adj_em_over_expected": mean(
            [row["adj_em_over_expected"] for row in same_school]
        ),
        "coach_prior_same_school_last_adj_em": same_school[-1]["adj_em"] if same_school else None,
        "coach_prior_same_school_last_adj_em_over_expected": (
            same_school[-1]["adj_em_over_expected"] if same_school else None
        ),
        "coach_first_year_at_school": current is not None and not same_school,
        "coach_prior_last_season": last["season"] if last else None,
        "coach_prior_last_season_gap": (current_season - int(last["season"])) if current_season is not None and last else None,
        "coach_prior_last_team_id": last["team_id"] if last else None,
        "coach_prior_last_team_name": last["team_name"] if last else None,
        "coach_prior_last_conference": last["conference"] if last else None,
        "coach_prior_last_adj_em": last["adj_em"] if last else None,
        "coach_prior_last_rank_adj_em": last["rank_adj_em"] if last else None,
        "coach_prior_last_adj_em_over_expected": last["adj_em_over_expected"] if last else None,
        "coach_prior_last_rank_over_expected": last["rank_over_expected"] if last else None,
    }


def coach_history_feature_rows(coach_seasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in coach_seasons:
        history = prior_rows(coach_seasons, row)
        rows.append(
            {
                "season": row["season"],
                "team_id": row["team_id"],
                "team_name": row["team_name"],
                "conference": row["conference"],
                "coach": row["coach"],
                "coach_key": row["coach_key"],
                **summary_from_history(history, current=row),
                "observed_preseason_adj_em": row["preseason_adj_em"],
                "observed_preseason_rank_adj_em": row["preseason_rank_adj_em"],
                "observed_final_adj_em": row["adj_em"],
                "observed_final_rank_adj_em": row["rank_adj_em"],
                "observed_adj_em_over_expected": row["adj_em_over_expected"],
                "observed_rank_over_expected": row["rank_over_expected"],
                "observed_adj_oe_over_expected": row["adj_oe_over_expected"],
                "observed_adj_de_over_expected": row["adj_de_over_expected"],
                "observed_final_wins": row["wins"],
                "observed_final_losses": row["losses"],
                "observed_final_event": row["event"],
                "observed_final_seed": row["seed"],
            }
        )

    rows.sort(key=lambda row: (int(row["season"]), str(row["team_name"])))
    return rows


def coach_latest_summary_rows(coach_seasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in coach_seasons:
        grouped.setdefault(row["coach_key"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for coach_key, history in grouped.items():
        history = sorted(history, key=lambda row: (int(row["season"]), row["team_name"]))
        last = history[-1]
        summaries.append(
            {
                "coach": last["coach"],
                "coach_key": coach_key,
                "last_seen_season": last["season"],
                "last_seen_team_id": last["team_id"],
                "last_seen_team_name": last["team_name"],
                "last_seen_conference": last["conference"],
                **summary_from_history(history),
            }
        )

    summaries.sort(key=lambda row: str(row["coach"]))
    return summaries


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
