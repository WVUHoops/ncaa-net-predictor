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
