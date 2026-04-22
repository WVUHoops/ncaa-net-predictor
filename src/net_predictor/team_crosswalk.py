"""Team-name crosswalk helpers across KenPom, NCAA NET, CBB Analytics, On3, and HoopDirt."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from net_predictor.coach_factor import canonical_team_key


KNOWN_UNMATCHED_TEAM_KEYS = {
    "augusta",
    "concordia university",
    "hampden sydney",
    "lane college",
    "lincoln memorial",
    "north central college",
    "professional",
    "rollins college",
    "saint john fisher",
    "saint thomas university",
    "wayne state college",
    "west florida",
}


@dataclass(frozen=True)
class SourceTeam:
    source: str
    team_name: str
    team_id: str | None = None
    season: int | None = None
    extra: str | None = None


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


def unique_source_teams(rows: Iterable[SourceTeam]) -> list[SourceTeam]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[SourceTeam] = []
    for row in rows:
        key = (row.source, canonical_team_key(row.team_name), row.team_id)
        if not row.team_name or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_kenpom_teams(kenpom_dir: Path, max_season: int | None = None) -> list[SourceTeam]:
    rows: list[SourceTeam] = []
    for path in sorted(kenpom_dir.glob("*/teams.json")):
        season = as_int(path.parent.name)
        if season is None:
            continue
        if max_season is not None and season > max_season:
            continue
        for item in read_json_rows(path):
            rows.append(
                SourceTeam(
                    source="kenpom",
                    team_id=str(item.get("TeamID")) if item.get("TeamID") is not None else None,
                    team_name=str(item.get("TeamName") or ""),
                    season=season,
                    extra=str(item.get("ConfShort") or ""),
                )
            )
    return rows


def latest_kenpom_by_team_id(rows: list[SourceTeam]) -> list[SourceTeam]:
    latest: dict[str, SourceTeam] = {}
    no_id: list[SourceTeam] = []
    for row in rows:
        if row.team_id is None:
            no_id.append(row)
            continue
        current = latest.get(row.team_id)
        if current is None or (row.season or 0) > (current.season or 0):
            latest[row.team_id] = row
    return sorted([*latest.values(), *no_id], key=lambda row: (row.team_name, row.team_id or ""))


def load_cbb_teams(paths: list[Path]) -> list[SourceTeam]:
    rows: list[SourceTeam] = []
    for path in paths:
        data = read_json_rows(path) if path.suffix == ".json" else read_csv_rows(path)
        for item in data:
            team_name = item.get("team_market") or item.get("teamMarket") or item.get("team")
            if not team_name:
                continue
            rows.append(
                SourceTeam(
                    source="cbb_analytics",
                    team_id=str(item.get("team_id") or item.get("teamId") or ""),
                    team_name=str(team_name),
                    season=as_int(item.get("season") or item.get("competition_id") or item.get("competitionId")),
                    extra=str(item.get("team_name") or item.get("teamName") or ""),
                )
            )
    return unique_source_teams(rows)


def load_net_teams(paths: list[Path]) -> list[SourceTeam]:
    rows: list[SourceTeam] = []
    for path in paths:
        data = read_json_rows(path) if path.suffix == ".json" else read_csv_rows(path)
        for item in data:
            team_name = item.get("team") or item.get("Team")
            if not team_name:
                continue
            rows.append(
                SourceTeam(
                    source="ncaa_net",
                    team_name=str(team_name),
                    season=as_int(item.get("season")),
                    extra=str(item.get("conference") or ""),
                )
            )
    return unique_source_teams(rows)


def load_on3_teams(paths: list[Path]) -> list[SourceTeam]:
    rows: list[SourceTeam] = []
    for path in paths:
        data = read_json_rows(path) if path.suffix == ".json" else read_csv_rows(path)
        for item in data:
            team_name = item.get("team") or item.get("team_full_name")
            if not team_name:
                continue
            rows.append(
                SourceTeam(
                    source=str(item.get("source") or "on3"),
                    team_id=str(item.get("organization_key") or ""),
                    team_name=str(team_name),
                    season=as_int(item.get("ranking_year")),
                    extra=str(item.get("team_slug") or ""),
                )
            )
    return unique_source_teams(rows)


def load_hoopdirt_teams(paths: list[Path]) -> list[SourceTeam]:
    rows: list[SourceTeam] = []
    for path in paths:
        data = read_json_rows(path) if path.suffix == ".json" else read_csv_rows(path)
        for item in data:
            team_name = item.get("school")
            if not team_name:
                continue
            rows.append(
                SourceTeam(
                    source="hoopdirt",
                    team_id=str(item.get("hoopdirt_row_id") or ""),
                    team_name=str(team_name),
                    season=as_int(item.get("season")),
                    extra=str(item.get("conference") or ""),
                )
            )
    return unique_source_teams(rows)


def score_match(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def best_kenpom_match(source_team: SourceTeam, kenpom_rows: list[SourceTeam]) -> tuple[SourceTeam | None, str, float]:
    source_key = canonical_team_key(source_team.team_name)
    best: SourceTeam | None = None
    best_score = 0.0
    for candidate in kenpom_rows:
        candidate_key = canonical_team_key(candidate.team_name)
        score = score_match(source_key, candidate_key)
        if score > best_score:
            best = candidate
            best_score = score

    if best is None:
        return None, "no_match", 0.0
    if best_score == 1.0:
        return best, "canonical_exact", best_score
    if best_score >= 0.94:
        return best, "fuzzy_high", best_score
    if best_score >= 0.86:
        return best, "fuzzy_review", best_score
    return best, "low_confidence", best_score


def build_crosswalk_rows(
    kenpom_rows: list[SourceTeam],
    source_rows: list[SourceTeam],
    review_threshold: float = 0.94,
) -> list[dict[str, Any]]:
    latest_kenpom = latest_kenpom_by_team_id(kenpom_rows)
    rows: list[dict[str, Any]] = []

    for source_team in sorted(source_rows, key=lambda row: (row.source, row.team_name, row.team_id or "")):
        source_key = canonical_team_key(source_team.team_name)
        if source_key in KNOWN_UNMATCHED_TEAM_KEYS:
            match = None
            match_type = "known_unmatched"
            match_score = 0.0
        else:
            match, match_type, match_score = best_kenpom_match(source_team, latest_kenpom)
        rows.append(
            {
                "source": source_team.source,
                "source_team_id": source_team.team_id,
                "source_team_name": source_team.team_name,
                "source_team_key": source_key,
                "source_season": source_team.season,
                "source_extra": source_team.extra,
                "kenpom_team_id": match.team_id if match else None,
                "kenpom_team_name": match.team_name if match else None,
                "kenpom_team_key": canonical_team_key(match.team_name) if match else None,
                "kenpom_season": match.season if match else None,
                "kenpom_extra": match.extra if match else None,
                "match_type": match_type,
                "match_score": match_score,
                "needs_review": (
                    match_type != "known_unmatched"
                    and (
                        match_score < review_threshold
                        or match_type in {"fuzzy_review", "low_confidence", "no_match"}
                    )
                ),
            }
        )
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
