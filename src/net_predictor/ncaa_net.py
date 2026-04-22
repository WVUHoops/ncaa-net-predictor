"""NCAA NET rankings ingestion helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NCAA_NET_RANKINGS_URL = (
    "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings"
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class NCAANETError(RuntimeError):
    """Raised when NCAA NET rankings cannot be fetched or parsed."""


class RankingsTableParser(HTMLParser):
    """Small table parser tuned for the NCAA.com rankings page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_last_updated = False
        self._last_updated_parts: list[str] = []
        self._table_depth = 0
        self._in_cell = False
        self._current_cell_parts: list[str] = []
        self._current_row: list[str] = []
        self._headers: list[str] = []
        self._rows: list[list[str]] = []
        self._seen_rankings_table = False

    @property
    def through_games_label(self) -> str | None:
        text = " ".join(part.strip() for part in self._last_updated_parts if part.strip())
        return re.sub(r"\s+", " ", text).strip() or None

    @property
    def headers(self) -> list[str]:
        return self._headers

    @property
    def rows(self) -> list[list[str]]:
        return self._rows

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = set((attr_map.get("class") or "").split())

        if tag == "figure" and "rankings-last-updated" in classes:
            self._in_last_updated = True

        if tag == "table" and "sticky" in classes:
            self._table_depth += 1
            self._seen_rankings_table = True

        if self._table_depth and tag == "tr":
            self._current_row = []

        if self._table_depth and tag in {"th", "td"}:
            self._in_cell = True
            self._current_cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_last_updated:
            self._last_updated_parts.append(data)
        if self._in_cell:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure" and self._in_last_updated:
            self._in_last_updated = False

        if self._table_depth and tag in {"th", "td"} and self._in_cell:
            text = unescape(" ".join(self._current_cell_parts))
            text = re.sub(r"\s+", " ", text).strip()
            self._current_row.append(text)
            self._in_cell = False

        if self._table_depth and tag == "tr" and self._current_row:
            if not self._headers:
                self._headers = self._current_row
            else:
                self._rows.append(self._current_row)
            self._current_row = []

        if tag == "table" and self._table_depth:
            self._table_depth -= 1


@dataclass(frozen=True)
class ParsedRecord:
    wins: int | None
    losses: int | None


def fetch_ncaa_net_html(url: str = NCAA_NET_RANKINGS_URL) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise NCAANETError(f"NCAA.com returned HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise NCAANETError(f"Could not reach NCAA.com: {exc.reason}") from exc


def parse_record(value: str) -> ParsedRecord:
    match = re.fullmatch(r"\s*(\d+)-(\d+)\s*", value)
    if not match:
        return ParsedRecord(None, None)
    return ParsedRecord(int(match.group(1)), int(match.group(2)))


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value in {"-", "--"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_through_games_date(label: str | None) -> str | None:
    if not label:
        return None

    match = re.search(r"([A-Za-z]+)\.?\s+(\d{1,2})\s+(\d{4})", label)
    if not match:
        return None

    month_name, day, year = match.groups()
    month = MONTHS.get(month_name.lower().rstrip("."))
    if not month:
        return None
    return date(int(year), month, int(day)).isoformat()


def normalize_header(value: str) -> str:
    normalized = value.lower().replace("-", " ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if normalized == "non_div_i":
        return "non_division_i"
    if normalized == "quad_1":
        return "q1"
    if normalized == "quad_2":
        return "q2"
    if normalized == "quad_3":
        return "q3"
    if normalized == "quad_4":
        return "q4"
    return normalized


def add_record_parts(row: dict[str, object], key: str) -> None:
    value = row.get(key)
    if not isinstance(value, str):
        return
    parsed = parse_record(value)
    row[f"{key}_wins"] = parsed.wins
    row[f"{key}_losses"] = parsed.losses


def parse_ncaa_net_html(
    html: str,
    source_url: str = NCAA_NET_RANKINGS_URL,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    parser = RankingsTableParser()
    parser.feed(html)

    if not parser.headers or not parser.rows:
        if "verify that you're not a robot" in html.lower():
            raise NCAANETError("NCAA.com returned a bot verification page instead of rankings.")
        raise NCAANETError("Could not find the NCAA NET rankings table in the HTML.")

    headers = [normalize_header(header) for header in parser.headers]
    through_games_label = parser.through_games_label
    through_games_date = parse_through_games_date(through_games_label)

    rows: list[dict[str, object]] = []
    for cells in parser.rows:
        if len(cells) != len(headers):
            continue

        row: dict[str, object] = dict(zip(headers, cells, strict=True))
        row["rank"] = parse_int(str(row.get("rank", "")))
        row["previous_rank"] = parse_int(str(row.pop("prev", "")))
        row["through_games"] = through_games_date
        row["through_games_label"] = through_games_label
        row["source_url"] = source_url

        for record_key in (
            "record",
            "road",
            "neutral",
            "home",
            "non_division_i",
            "q1",
            "q2",
            "q3",
            "q4",
        ):
            add_record_parts(row, record_key)

        rows.append(row)

    metadata = {
        "source_url": source_url,
        "through_games": through_games_date,
        "through_games_label": through_games_label,
        "row_count": len(rows),
    }
    return rows, metadata


def write_json(rows: Iterable[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
