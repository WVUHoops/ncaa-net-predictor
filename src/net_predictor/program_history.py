"""Program-history features from prior final KenPom seasons."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key


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


def load_kenpom_rating_rows(
    kenpom_dir: Path,
    min_season: int | None = None,
    max_season: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season_dir in season_dirs(kenpom_dir, min_season, max_season):
        path = season_dir / "ratings.json"
        if path.exists():
            rows.extend(read_json_rows(path))
    return rows


def normalized_rating_row(row: dict[str, Any]) -> dict[str, Any] | None:
    season = as_int(row.get("Season"))
    team = row.get("TeamName")
    if season is None or not team:
        return None
    return {
        "season": season,
        "team": team,
        "team_key": canonical_team_key(team),
        "conference": row.get("ConfShort"),
        "adj_em": as_float(row.get("AdjEM")),
        "rank_adj_em": as_int(row.get("RankAdjEM")),
        "adj_oe": as_float(row.get("AdjOE")),
        "rank_adj_oe": as_int(row.get("RankAdjOE")),
        "adj_de": as_float(row.get("AdjDE")),
        "rank_adj_de": as_int(row.get("RankAdjDE")),
        "tempo": as_float(row.get("AdjTempo")),
        "rank_tempo": as_int(row.get("RankAdjTempo")),
        "sos": as_float(row.get("SOS")),
        "rank_sos": as_int(row.get("RankSOS")),
        "ncsos": as_float(row.get("NCSOS")),
        "rank_ncsos": as_int(row.get("RankNCSOS")),
        "wins": as_int(row.get("Wins")),
        "losses": as_int(row.get("Losses")),
        "seed": as_int(row.get("Seed")),
        "event": row.get("Event"),
    }


def history_features(history: list[dict[str, Any]]) -> dict[str, Any]:
    history = sorted(history, key=lambda row: int(row["season"]))
    last = history[-1] if history else None
    previous = history[-2] if len(history) >= 2 else None
    last_3 = history[-3:]
    last_5 = history[-5:]
    ranks = [row["rank_adj_em"] for row in history if row["rank_adj_em"] is not None]
    seeds = [row["seed"] for row in history if row["seed"] is not None]
    wins = sum(row["wins"] or 0 for row in history)
    losses = sum(row["losses"] or 0 for row in history)
    games = wins + losses
    last_adj_em = last["adj_em"] if last else None
    previous_adj_em = previous["adj_em"] if previous else None
    last_rank = last["rank_adj_em"] if last else None
    previous_rank = previous["rank_adj_em"] if previous else None

    return {
        "prior_seasons": len(history),
        "prior_win_pct": wins / games if games else None,
        "prior_last_adj_em": last_adj_em,
        "prior_last_rank_adj_em": last_rank,
        "prior_last_adj_oe": last["adj_oe"] if last else None,
        "prior_last_rank_adj_oe": last["rank_adj_oe"] if last else None,
        "prior_last_adj_de": last["adj_de"] if last else None,
        "prior_last_rank_adj_de": last["rank_adj_de"] if last else None,
        "prior_last_sos": last["sos"] if last else None,
        "prior_last_rank_sos": last["rank_sos"] if last else None,
        "prior_avg_adj_em": mean([row["adj_em"] for row in history]),
        "prior_last3_avg_adj_em": mean([row["adj_em"] for row in last_3]),
        "prior_last5_avg_adj_em": mean([row["adj_em"] for row in last_5]),
        "prior_best_adj_em": max((row["adj_em"] for row in history if row["adj_em"] is not None), default=None),
        "prior_worst_adj_em": min((row["adj_em"] for row in history if row["adj_em"] is not None), default=None),
        "prior_avg_rank_adj_em": mean(ranks),
        "prior_best_rank_adj_em": min(ranks) if ranks else None,
        "prior_top25_rate": rate([row["rank_adj_em"] <= 25 for row in history if row["rank_adj_em"] is not None]),
        "prior_top50_rate": rate([row["rank_adj_em"] <= 50 for row in history if row["rank_adj_em"] is not None]),
        "prior_top75_rate": rate([row["rank_adj_em"] <= 75 for row in history if row["rank_adj_em"] is not None]),
        "prior_top100_rate": rate([row["rank_adj_em"] <= 100 for row in history if row["rank_adj_em"] is not None]),
        "prior_top135_rate": rate([row["rank_adj_em"] <= 135 for row in history if row["rank_adj_em"] is not None]),
        "prior_ncaa_bid_rate": rate([row["event"] == "NCAA" for row in history]),
        "prior_avg_seed": mean(seeds),
        "prior_avg_sos": mean([row["sos"] for row in history]),
        "prior_avg_ncsos": mean([row["ncsos"] for row in history]),
        "prior_adj_em_one_year_change": (
            last_adj_em - previous_adj_em if last_adj_em is not None and previous_adj_em is not None else None
        ),
        "prior_rank_adj_em_one_year_change": (
            previous_rank - last_rank if last_rank is not None and previous_rank is not None else None
        ),
    }


def program_history_feature_rows(rating_rows: list[dict[str, Any]], *, through_season: int | None = None) -> list[dict[str, Any]]:
    normalized = [row for source in rating_rows if (row := normalized_rating_row(source)) is not None]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        grouped.setdefault(str(row["team_key"]), []).append(row)

    observed_seasons = sorted({int(row["season"]) for row in normalized})
    if not observed_seasons:
        return []
    max_output_season = through_season or max(observed_seasons) + 1
    output_seasons = range(min(observed_seasons), max_output_season + 1)
    rows: list[dict[str, Any]] = []
    for season in output_seasons:
        for team_key, history in grouped.items():
            prior = [row for row in history if int(row["season"]) < season]
            if not prior:
                continue
            last = sorted(prior, key=lambda row: int(row["season"]))[-1]
            rows.append(
                {
                    "season": season,
                    "team": last["team"],
                    "team_key": team_key,
                    "conference": last["conference"],
                    **history_features(prior),
                }
            )

    rows.sort(key=lambda row: (int(row["season"]), str(row["team"])))
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

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path
