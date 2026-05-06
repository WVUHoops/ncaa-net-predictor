"""Feature builders for On3 recruiting and transfer team rankings."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key


HS_FEATURE_FIELDS = (
    "rank",
    "score",
    "on3_score",
    "commits",
    "applied_commits",
    "avg_rating",
    "on3_avg_rating",
    "total_rating",
    "on3_total_rating",
    "five_stars",
    "on3_five_stars",
    "four_stars",
    "on3_four_stars",
    "three_stars",
    "on3_three_stars",
    "avg_nil_value",
    "conference_rank",
)

TRANSFER_FEATURE_FIELDS = (
    "rank",
    "index_score",
    "raw_score",
    "transfers_in",
    "transfers_in_avg_rating",
    "raw_score_in",
    "transfers_out",
    "transfers_out_avg_rating",
    "raw_score_out",
    "five_stars_net",
    "five_stars_in",
    "five_stars_out",
    "four_stars_net",
    "four_stars_in",
    "four_stars_out",
    "three_stars_net",
    "three_stars_in",
    "three_stars_out",
    "original_nil_valuation",
    "adjusted_nil_valuation",
    "nil_valuation_change",
)


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    value_float = as_float(value)
    if value_float is None:
        return None
    return int(value_float)


def year_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{4}", part):
            return int(part)
    match = re.search(r"on3_(?:hs|transfer)_(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def latest_files_by_source_year(paths: list[Path]) -> dict[tuple[str, int], Path]:
    latest: dict[tuple[str, int], Path] = {}
    for path in sorted(paths):
        if path.suffix != ".json":
            continue
        if "/hs/" in path.as_posix():
            source = "hs"
        elif "/transfer/" in path.as_posix():
            source = "transfer"
        else:
            continue
        year = year_from_path(path)
        if year is None:
            continue
        latest[(source, year)] = path
    return latest


def rank_percentile(rank: int | None, teams_ranked: int) -> float | None:
    if rank is None or teams_ranked <= 1:
        return None
    return 1 - ((rank - 1) / (teams_ranked - 1))


def numeric_features(row: dict[str, Any], fields: tuple[str, ...], prefix: str) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for field in fields:
        features[f"{prefix}{field}"] = as_float(row.get(field))
    return features


def feature_row(
    row: dict[str, Any],
    *,
    source: str,
    ranking_year: int,
    teams_ranked: int,
    source_file: Path,
) -> dict[str, Any]:
    feature_season = ranking_year + 1
    team = row.get("team") or row.get("team_full_name")
    rank = as_int(row.get("rank"))
    prefix = "on3_hs_" if source == "hs" else "on3_transfer_"
    fields = HS_FEATURE_FIELDS if source == "hs" else TRANSFER_FEATURE_FIELDS

    features: dict[str, Any] = {
        "season": feature_season,
        "ranking_year": ranking_year,
        "team": team,
        "team_key": canonical_team_key(team),
        f"{prefix}teams_ranked": teams_ranked,
        f"{prefix}rank_percentile": rank_percentile(rank, teams_ranked),
        f"{prefix}source_file": source_file.as_posix(),
        f"{prefix}source_url": row.get("source_url"),
        f"{prefix}captured_at": row.get("captured_at"),
        f"{prefix}ranking_date_updated": row.get("ranking_date_updated"),
    }
    features.update(numeric_features(row, fields, prefix))
    return features


def build_on3_feature_rows(paths: list[Path]) -> list[dict[str, Any]]:
    latest_files = latest_files_by_source_year(paths)
    combined: dict[tuple[int, str], dict[str, Any]] = {}

    for (source, ranking_year), path in sorted(latest_files.items(), key=lambda item: item[0]):
        rows = read_json_rows(path)
        teams_ranked = len(rows)
        for row in rows:
            features = feature_row(
                row,
                source=source,
                ranking_year=ranking_year,
                teams_ranked=teams_ranked,
                source_file=path,
            )
            key = (int(features["season"]), str(features["team_key"]))
            target = combined.setdefault(
                key,
                {
                    "season": features["season"],
                    "team": features["team"],
                    "team_key": features["team_key"],
                },
            )
            target.update({k: v for k, v in features.items() if k not in {"season", "team", "team_key"}})

    return sorted(combined.values(), key=lambda row: (row["season"], str(row["team"])))


def latest_hs_commit_files_by_year(paths: list[Path]) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    for path in sorted(paths):
        if path.suffix != ".json":
            continue
        year = year_from_path(path)
        if year is None:
            continue
        latest[year] = path
    return latest


def read_hs_commit_player_rows(paths: list[Path]) -> list[dict[str, Any]]:
    latest_files = latest_hs_commit_files_by_year(paths)
    rows: list[dict[str, Any]] = []
    for year, path in sorted(latest_files.items()):
        for row in read_json_rows(path):
            if int(as_int(row.get("class_year")) or 0) != year:
                continue
            entry = dict(row)
            entry["team_key"] = canonical_team_key(entry.get("team"))
            entry["season"] = int(as_int(entry.get("season")) or (year + 1))
            entry["ranking_year"] = year
            entry["source_file"] = path.as_posix()
            rows.append(entry)
    rows.sort(key=lambda row: (row["season"], str(row.get("team") or ""), str(row.get("player_name") or "")))
    return rows


def recruit_percentile(row: dict[str, Any]) -> float | None:
    rank = as_float(row.get("industry_rank"))
    if rank is not None and rank > 0:
        return max(0.0, min(1.0, (401.0 - rank) / 400.0))
    rating = as_float(row.get("industry_rating"))
    if rating is None:
        rating = as_float(row.get("on3_rating"))
    if rating is None:
        return None
    return max(0.0, min(1.0, rating / 100.0))


def position_bucket(position: Any) -> str:
    token = str(position or "").upper().strip()
    if token in {"PG"}:
        return "pg"
    if token in {"SG", "CG", "G"}:
        return "guard"
    if token in {"C"}:
        return "big"
    if token in {"PF", "F"}:
        return "forward"
    return "wing"


def aggregate_hs_commit_rows(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in player_rows:
        key = (int(row.get("season") or 0), str(row.get("team_key") or ""))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (season, team_key), recruits in grouped.items():
        recruits_sorted = sorted(
            recruits,
            key=lambda row: (
                -(recruit_percentile(row) or -1.0),
                as_float(row.get("industry_rank")) or 9999.0,
                str(row.get("player_name") or ""),
            ),
        )
        percentiles = [value for value in (recruit_percentile(row) for row in recruits_sorted) if value is not None]
        if not percentiles:
            continue
        weights = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3]
        weighted_total = 0.0
        weight_sum = 0.0
        for index, percentile in enumerate(percentiles):
            weight = weights[index] if index < len(weights) else 0.25
            weighted_total += percentile * weight
            weight_sum += weight

        position_counts = {"pg": 0, "guard": 0, "wing": 0, "forward": 0, "big": 0}
        for recruit in recruits_sorted:
            position_counts[position_bucket(recruit.get("position_abbr") or recruit.get("position"))] += 1

        first = recruits_sorted[0]
        rows.append(
            {
                "season": season,
                "team": first.get("team"),
                "team_key": team_key,
                "on3_hs_player_count": len(recruits_sorted),
                "on3_hs_player_top_25_count": sum(
                    1 for recruit in recruits_sorted if 0 < (as_int(recruit.get("industry_rank")) or 0) <= 25
                ),
                "on3_hs_player_top_50_count": sum(
                    1 for recruit in recruits_sorted if 0 < (as_int(recruit.get("industry_rank")) or 0) <= 50
                ),
                "on3_hs_player_top_100_count": sum(
                    1 for recruit in recruits_sorted if 0 < (as_int(recruit.get("industry_rank")) or 0) <= 100
                ),
                "on3_hs_player_best_rank": as_float(first.get("industry_rank")),
                "on3_hs_player_top_percentile": percentiles[0],
                "on3_hs_player_avg_percentile": sum(percentiles) / len(percentiles),
                "on3_hs_player_weighted_percentile": weighted_total / weight_sum if weight_sum else None,
                "on3_hs_player_avg_rating": average_existing(
                    *[as_float(recruit.get("industry_rating")) for recruit in recruits_sorted]
                ),
                "on3_hs_player_pg_count": position_counts["pg"],
                "on3_hs_player_guard_count": position_counts["guard"] + position_counts["pg"],
                "on3_hs_player_forward_count": position_counts["forward"] + position_counts["wing"],
                "on3_hs_player_big_count": position_counts["big"],
                "on3_hs_player_source_file": first.get("source_file"),
            }
        )
    return sorted(rows, key=lambda row: (row["season"], str(row["team"])))


def average_existing(*values: float | None) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


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
