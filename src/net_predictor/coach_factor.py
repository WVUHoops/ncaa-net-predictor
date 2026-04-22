"""Coach-change features from KenPom team history and HoopDirt tracker rows."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


TEAM_ALIASES = {
    "a and m corpus christi": "texas a and m corpus chris",
    "alcorn": "alcorn state",
    "app state": "appalachian state",
    "arkansas little rock": "little rock",
    "ark pine bluff": "arkansas pine bluff",
    "army west point": "army",
    "belmont university": "belmont",
    "binghamton university": "binghamton",
    "boston u": "boston university",
    "bryant university": "bryant",
    "california baptist": "cal baptist",
    "california baptist university": "cal baptist",
    "csun": "cal state northridge",
    "charleston so": "charleston southern",
    "california state university fullerton": "cal state fullerton",
    "california state university long beach": "long beach state",
    "california state university northridge": "cal state northridge",
    "canisius university": "canisius",
    "cleveland state university": "cleveland state",
    "college of charleston": "charleston",
    "col of charleston": "charleston",
    "central ark": "central arkansas",
    "central conn state": "central connecticut",
    "central mich": "central michigan",
    "denver university": "denver",
    "eastern ill": "eastern illinois",
    "eastern ky": "eastern kentucky",
    "eastern mich": "eastern michigan",
    "eastern wash": "eastern washington",
    "etsu": "east tennessee state",
    "detroit mercy": "detroit",
    "east texas a and m": "texas a and m commerce",
    "east texas a and m commerce": "texas a and m commerce",
    "fairfield university": "fairfield",
    "fdu": "fairleigh dickinson",
    "fgcu": "florida gulf coast",
    "fla atlantic": "florida atlantic",
    "florida gulf coast university": "florida gulf coast",
    "ga southern": "georgia southern",
    "george washington university": "george washington",
    "gonzaga university": "gonzaga",
    "grand canyon university": "grand canyon",
    "grambling": "grambling state",
    "houston christian": "houston baptist",
    "indiana university indianapolis": "iupui",
    "iu indy": "iupui",
    "kansas city": "umkc",
    "lamar university": "lamar",
    "le moyne college": "le moyne",
    "lindenwood university": "lindenwood",
    "lmu ca": "loyola marymount",
    "loyola chi": "loyola chicago",
    "loyola maryland": "loyola md",
    "manhattan college": "manhattan",
    "mcneese": "mcneese state",
    "miami": "miami fl",
    "middle tenn": "middle tennessee",
    "middle tennessee state": "middle tennessee",
    "mississippi val": "mississippi valley state",
    "missouri kansas city": "umkc",
    "missouri state university": "missouri state",
    "n c a and t": "north carolina a and t",
    "n c central": "north carolina central",
    "new jersey institute of technology": "njit",
    "niagara university": "niagara",
    "njit": "njit",
    "niu": "northern illinois",
    "northern ariz": "northern arizona",
    "northern colo": "northern colorado",
    "northern ky": "northern kentucky",
    "nicholls": "nicholls state",
    "ole miss": "mississippi",
    "omaha": "nebraska omaha",
    "pennsylvania": "penn",
    "prairie view": "prairie view a and m",
    "purdue university fort wayne": "purdue fort wayne",
    "queens nc": "queens",
    "queens university of charlotte": "queens",
    "saint marys ca": "saint marys",
    "sam houston": "sam houston state",
    "seattle u": "seattle",
    "sfa": "stephen f austin",
    "siue": "siu edwardsville",
    "southeast missouri": "southeast missouri state",
    "southeast mo state": "southeast missouri state",
    "southeastern la": "southeastern louisiana",
    "south fla": "south florida",
    "southern ca": "usc",
    "southern california": "usc",
    "southern ill": "southern illinois",
    "southern ind": "southern indiana",
    "southern u": "southern",
    "st johns ny": "saint johns",
    "saint johns ny": "saint johns",
    "saint josephs university": "saint josephs",
    "saint bonaventure university": "saint bonaventure",
    "saint marys college of california": "saint marys",
    "st francis brooklyn": "saint francis ny",
    "saint francis brooklyn": "saint francis ny",
    "saint francis": "saint francis pa",
    "st thomas mn": "saint thomas",
    "saint thomas mn": "saint thomas",
    "santa clara university": "santa clara",
    "southern university": "southern",
    "north carolina state": "nc state",
    "north ala": "north alabama",
    "n c state": "nc state",
    "nc state": "nc state",
    "uconn": "connecticut",
    "ualbany": "albany",
    "uic": "illinois chicago",
    "uiw": "incarnate word",
    "ulm": "louisiana monroe",
    "umes": "maryland eastern shore",
    "uncw": "unc wilmington",
    "uni": "northern iowa",
    "university of california irvine": "uc irvine",
    "university of california riverside": "uc riverside",
    "university of california san diego": "uc san diego",
    "university of california santa barbara": "uc santa barbara",
    "university of maryland baltimore county": "umbc",
    "university of nebraska at omaha": "nebraska omaha",
    "university of north carolina asheville": "unc asheville",
    "university of north carolina wilmington": "unc wilmington",
    "university of san francisco": "san francisco",
    "university of south carolina upstate": "usc upstate",
    "utah tech": "dixie state",
    "ut martin": "tennessee martin",
    "utrgv": "ut rio grande valley",
    "the university of texas at arlington": "ut arlington",
    "the university of texas rio grande valley": "ut rio grande valley",
    "usf": "south florida",
    "utah valley university": "utah valley",
    "virginia military institute": "vmi",
    "west ga": "west georgia",
    "western caro": "western carolina",
    "western ill": "western illinois",
    "western ky": "western kentucky",
    "western mich": "western michigan",
    "wisconsin milwaukee": "milwaukee",
    "wisconsin green bay": "green bay",
    "cal state bakersfield": "cal state bakersfield",
    "csu bakersfield": "cal state bakersfield",
    "st marys": "saint marys",
    "saint marys": "saint marys",
    "texas a and m university corpus christi": "texas a and m corpus chris",
}

COACH_TITLE_WORDS = {"interim", "acting", "head", "coach"}
UNKNOWN_COACH_VALUES = {"", "tbd", "vacant", "open"}
FIRST_NAME_ALIASES = {
    "bill": {"billy", "william"},
    "billy": {"bill", "william"},
    "bob": {"bobby", "robert"},
    "bobby": {"bob", "robert"},
    "chris": {"christopher"},
    "dave": {"david"},
    "david": {"dave"},
    "jim": {"james"},
    "james": {"jim"},
    "joe": {"joseph"},
    "john": {"jon", "johnny"},
    "jon": {"john", "johnny"},
    "mike": {"michael"},
    "michael": {"mike"},
    "rick": {"richard"},
    "richard": {"rick"},
    "steve": {"steven"},
    "steven": {"steve"},
}


def strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = strip_accents(str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_team_key(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\baq\b$", "", text).strip()
    text = re.sub(r"^st\s+", "saint ", text)
    text = re.sub(r"\bcal st\b", "cal state", text)
    text = re.sub(r"\bst$", "state", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def normalize_coach_name(value: Any) -> str:
    text = normalize_text(re.sub(r"\([^)]*\)", "", str(value or "")))
    words = [word for word in text.split() if word not in COACH_TITLE_WORDS]
    return " ".join(words)


def coach_keys(value: Any) -> set[str]:
    normalized = normalize_coach_name(value)
    if not normalized:
        return set()

    keys = {normalized}
    words = normalized.split()
    if len(words) >= 2:
        first = words[0]
        last = words[-1]
        keys.add(f"{first} {last}")
        for alias in FIRST_NAME_ALIASES.get(first, set()):
            keys.add(f"{alias} {last}")
    return keys


def is_unknown_coach(value: Any) -> bool:
    return normalize_coach_name(value) in UNKNOWN_COACH_VALUES


def coach_match_type(left: Any, right: Any) -> str:
    left_normalized = normalize_coach_name(left)
    right_normalized = normalize_coach_name(right)
    if not left_normalized or not right_normalized:
        return "missing"
    if left_normalized == right_normalized:
        return "exact_normalized"
    if coach_keys(left) & coach_keys(right):
        return "first_last"
    if SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.9:
        return "fuzzy"
    return "no_match"


def coach_match_bool(match_type: str) -> bool:
    return match_type in {"exact_normalized", "first_last", "fuzzy"}


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def latest_team_rows(kenpom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in kenpom_rows:
        team_id = str(row.get("TeamID") or row.get("team_id") or row.get("TeamName") or "")
        if not team_id:
            continue

        current = latest.get(team_id)
        season = int(row.get("Season") or row.get("season") or 0)
        current_season = int((current or {}).get("Season") or (current or {}).get("season") or 0)
        if current is None or season > current_season:
            latest[team_id] = row

    return sorted(latest.values(), key=lambda row: str(row.get("TeamName") or row.get("team_name") or ""))


def consecutive_team_coach_tenure(row: dict[str, Any], all_rows: list[dict[str, Any]]) -> int | None:
    team_id = str(row.get("TeamID") or "")
    coach = normalize_coach_name(row.get("Coach"))
    latest_season = int(row.get("Season") or 0)
    if not team_id or not coach or not latest_season:
        return None

    seasons_by_year = {
        int(candidate.get("Season") or 0): candidate
        for candidate in all_rows
        if str(candidate.get("TeamID") or "") == team_id
    }
    tenure = 0
    year = latest_season
    while year in seasons_by_year:
        candidate = seasons_by_year[year]
        if normalize_coach_name(candidate.get("Coach")) != coach:
            break
        tenure += 1
        year -= 1

    return tenure or None


def build_change_index(change_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in change_rows:
        key = canonical_team_key(row.get("school"))
        if key:
            index[key].append(row)
    return index


def best_fuzzy_change_match(team_key: str, change_index: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any] | None, str, float | None]:
    best_key = None
    best_score = 0.0
    for change_key in change_index:
        score = SequenceMatcher(None, team_key, change_key).ratio()
        if score > best_score:
            best_key = change_key
            best_score = score

    if best_key is not None and best_score >= 0.94:
        return change_index[best_key][0], "fuzzy_team_name", best_score
    return None, "no_match", None


def find_change_for_team(
    team_name: Any,
    change_index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str, float | None]:
    team_key = canonical_team_key(team_name)
    if team_key in change_index:
        return change_index[team_key][0], "canonical_team_name", 1.0
    return best_fuzzy_change_match(team_key, change_index)


def latest_coach_job(
    coach: Any,
    all_rows: list[dict[str, Any]],
    *,
    exclude_team_id: Any = None,
) -> dict[str, Any] | None:
    keys = coach_keys(coach)
    if not keys:
        return None

    candidates: list[dict[str, Any]] = []
    for row in all_rows:
        if exclude_team_id is not None and str(row.get("TeamID") or "") == str(exclude_team_id):
            continue
        if keys & coach_keys(row.get("Coach")):
            candidates.append(row)

    if not candidates:
        return None

    return max(candidates, key=lambda row: int(row.get("Season") or 0))


def coach_feature_rows(
    kenpom_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest_rows = latest_team_rows(kenpom_rows)
    change_index = build_change_index(change_rows)
    matched_change_ids: set[str] = set()
    feature_rows: list[dict[str, Any]] = []

    for team in latest_rows:
        change, team_match_type, team_match_score = find_change_for_team(team.get("TeamName"), change_index)
        change_id = None if change is None else str(change.get("hoopdirt_row_id") or "")
        if change_id:
            matched_change_ids.add(change_id)

        former_coach = (change or {}).get("former_coach")
        new_coach = (change or {}).get("new_coach")
        former_match_type = coach_match_type(team.get("Coach"), former_coach) if change else "not_applicable"
        new_coach_job = latest_coach_job(new_coach, kenpom_rows, exclude_team_id=team.get("TeamID")) if change else None
        new_coach_known = bool(change and not is_unknown_coach(new_coach))

        feature_rows.append(
            {
                "team_id": team.get("TeamID"),
                "team_name": team.get("TeamName"),
                "season": team.get("Season"),
                "conference": team.get("ConfShort"),
                "kenpom_coach": team.get("Coach"),
                "kenpom_coach_tenure_years_available": consecutive_team_coach_tenure(team, kenpom_rows),
                "coach_changed": bool(change),
                "coach_change_status": (
                    "changed_tbd"
                    if change and not new_coach_known
                    else "changed_known_new_coach"
                    if change
                    else "no_change_listed"
                ),
                "team_match_type": team_match_type,
                "team_match_score": team_match_score,
                "hoopdirt_school": (change or {}).get("school"),
                "hoopdirt_conference": (change or {}).get("conference"),
                "former_coach": former_coach,
                "new_coach": new_coach,
                "former_coach_matches_kenpom": coach_match_bool(former_match_type),
                "former_coach_match_type": former_match_type,
                "new_coach_known": new_coach_known,
                "new_coach_seen_in_kenpom_history": new_coach_job is not None,
                "new_coach_last_seen_team_id": (new_coach_job or {}).get("TeamID"),
                "new_coach_last_seen_team_name": (new_coach_job or {}).get("TeamName"),
                "new_coach_last_seen_conference": (new_coach_job or {}).get("ConfShort"),
                "new_coach_last_seen_season": (new_coach_job or {}).get("Season"),
                "hoopdirt_row_id": (change or {}).get("hoopdirt_row_id"),
                "source_url": (change or {}).get("source_url"),
                "captured_at": (change or {}).get("captured_at"),
            }
        )

    unmatched = []
    latest_team_keys = {
        canonical_team_key(row.get("TeamName")): row
        for row in latest_rows
        if canonical_team_key(row.get("TeamName"))
    }
    for change in change_rows:
        change_id = str(change.get("hoopdirt_row_id") or "")
        if change_id and change_id in matched_change_ids:
            continue
        change_key = canonical_team_key(change.get("school"))
        best_team = None
        best_score = 0.0
        for team_key, team in latest_team_keys.items():
            score = SequenceMatcher(None, change_key, team_key).ratio()
            if score > best_score:
                best_team = team
                best_score = score

        unmatched.append(
            {
                **change,
                "canonical_school_key": change_key,
                "best_kenpom_team_name": (best_team or {}).get("TeamName"),
                "best_kenpom_team_score": best_score if best_team else None,
            }
        )

    feature_rows.sort(key=lambda row: str(row.get("team_name") or ""))
    unmatched.sort(key=lambda row: str(row.get("school") or ""))
    return feature_rows, unmatched


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
