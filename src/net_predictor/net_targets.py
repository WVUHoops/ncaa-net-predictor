"""NET target construction for model training and schedule-building bands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from net_predictor.coach_factor import canonical_team_key


TARGET_BANDS = (50, 75, 100, 135)


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


def season_from_row(row: dict[str, Any]) -> int | None:
    season = as_int(row.get("season"))
    if season is not None:
        return season

    date_value = row.get("selection_thru_games") or row.get("through_games")
    if isinstance(date_value, str) and len(date_value) >= 4 and date_value[:4].isdigit():
        return int(date_value[:4])
    return None


def target_rows_from_net_rows(rows: list[dict[str, Any]], *, source_file: str | None = None) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in rows
        if as_int(row.get("net_rank") if row.get("net_rank") is not None else row.get("rank")) is not None
    ]
    team_count = len(ranked)
    targets: list[dict[str, Any]] = []

    for row in ranked:
        net_rank = as_int(row.get("net_rank") if row.get("net_rank") is not None else row.get("rank"))
        if net_rank is None:
            continue
        percentile = None
        if team_count > 1:
            percentile = 1 - ((net_rank - 1) / (team_count - 1))

        team_name = row.get("team") or row.get("school")
        conference = row.get("conference") or row.get("conf")
        target: dict[str, Any] = {
            "season": season_from_row(row),
            "season_label": row.get("season_label"),
            "team": team_name,
            "team_key": canonical_team_key(team_name),
            "conference": conference,
            "selection_thru_games": row.get("selection_thru_games") or row.get("through_games") or row.get("data_through"),
            "net_rank": net_rank,
            "net_percentile": percentile,
            "teams_ranked": team_count,
            "below_200": net_rank > 200,
            "source_url": row.get("source_url"),
            "source_file": source_file,
        }

        for band in TARGET_BANDS:
            target[f"top_{band}"] = net_rank <= band
        targets.append(target)

    targets.sort(key=lambda row: (row["season"] or 0, row["net_rank"]))
    return targets


def load_target_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source_rows = read_json_rows(path) if path.suffix == ".json" else read_csv_rows(path)
        rows.extend(target_rows_from_net_rows(source_rows, source_file=path.as_posix()))
    rows.sort(key=lambda row: (row["season"] or 0, row["net_rank"]))
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
