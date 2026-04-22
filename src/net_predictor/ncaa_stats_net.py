"""NCAA Statistics NET archive ingestion helpers."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from net_predictor.ncaa_net import NCAANETError, parse_int, parse_record, write_csv, write_json


NCAA_STATS_NET_ARCHIVE_URL = (
    "https://stats.ncaa.org/selection_rankings/season_divisions/17783/nitty_gritties"
)
NCAA_STATS_BASE_URL = "https://stats.ncaa.org"


@dataclass(frozen=True)
class SelectionLink:
    season: int
    season_label: str
    thru_games: str
    label: str
    url: str


class LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._current_href: str | None = None
        self._current_parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self._current_href = urljoin(self.base_url, href)
            self._current_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href:
            text = clean_text(" ".join(self._current_parts))
            self.links.append((text, self._current_href))
            self._current_href = None
            self._current_parts = []


class GenericTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._table_depth = 0
        self._in_cell = False
        self._current_cell_parts: list[str] = []
        self._current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
        if self._table_depth and tag == "tr":
            self._current_row = []
        if self._table_depth and tag in {"th", "td"}:
            self._in_cell = True
            self._current_cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth and tag in {"th", "td"} and self._in_cell:
            self._current_row.append(clean_text(" ".join(self._current_cell_parts)))
            self._in_cell = False
        if self._table_depth and tag == "tr" and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = []
        if tag == "table" and self._table_depth:
            self._table_depth -= 1


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def fetch_html(url: str) -> str:
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.ncaa.com/",
            "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
        },
    )
    try:
        with opener.open(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
        if is_akamai_interstitial(html):
            return solve_akamai_interstitial(opener, url, html)
        return html
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise NCAANETError(f"NCAA Statistics returned HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise NCAANETError(f"Could not reach NCAA Statistics: {exc.reason}") from exc


def is_akamai_interstitial(html: str) -> bool:
    return "/_sec/verify?provider=interstitial" in html and "bm-verify" in html


def akamai_pow_value(html: str) -> int | None:
    base_match = re.search(r"var\s+i\s*=\s*(\d+)\s*;", html)
    suffix_match = re.search(r"var\s+j\s*=\s*i\s*\+\s*Number\((.*?)\)\s*;", html)
    if not base_match or not suffix_match:
        return None

    suffix_digits = "".join(re.findall(r'"(\d+)"', suffix_match.group(1)))
    if not suffix_digits:
        return None
    return int(base_match.group(1)) + int(suffix_digits)


def solve_akamai_interstitial(opener, url: str, html: str) -> str:
    token_match = re.search(r'"bm-verify"\s*:\s*"([^"]+)"', html)
    pow_value = akamai_pow_value(html)
    if not token_match or pow_value is None:
        raise NCAANETError("NCAA Statistics returned an Akamai challenge that could not be parsed.")

    verify_url = urljoin(url, "/_sec/verify?provider=interstitial")
    payload = json.dumps({"bm-verify": token_match.group(1), "pow": pow_value}).encode("utf-8")
    verify_request = Request(
        verify_url,
        data=payload,
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/json",
            "Origin": NCAA_STATS_BASE_URL,
            "Referer": url,
            "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
            "X-Requested-With": "XMLHttpRequest",
        },
        method="POST",
    )
    with opener.open(verify_request, timeout=30) as response:
        response_body = response.read().decode("utf-8", errors="replace")

    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise NCAANETError(
            f"NCAA Statistics Akamai verification returned non-JSON: {response_body[:300]}"
        ) from exc

    next_url = response_data.get("location") if isinstance(response_data, dict) else None
    request_url = urljoin(url, next_url) if isinstance(next_url, str) else url
    retry_request = Request(
        request_url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": url,
            "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
        },
    )
    with opener.open(retry_request, timeout=30) as response:
        retry_html = response.read().decode("utf-8", errors="replace")

    if is_akamai_interstitial(retry_html):
        raise NCAANETError("NCAA Statistics still returned the Akamai challenge after verification.")
    return retry_html


def parse_mmddyyyy(value: str) -> str | None:
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", value)
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    return date(year, month, day).isoformat()


def season_label_for_ending_year(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def discover_selection_links(html: str, base_url: str = NCAA_STATS_BASE_URL) -> list[SelectionLink]:
    parser = LinkParser(base_url)
    parser.feed(html)

    selections: list[SelectionLink] = []
    for label, url in parser.links:
        if "selection" not in label.lower():
            continue

        thru_games = parse_mmddyyyy(label)
        if not thru_games:
            continue

        season = int(thru_games[:4])
        selections.append(
            SelectionLink(
                season=season,
                season_label=season_label_for_ending_year(season),
                thru_games=thru_games,
                label=label,
                url=url,
            )
        )

    for value, label_html in re.findall(
        r'<option[^>]*\bvalue="([^"]+)"[^>]*>(.*?)</option>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label = clean_text(label_html)
        if "selection" not in label.lower():
            continue

        thru_games = parse_mmddyyyy(label)
        if not thru_games:
            continue

        season = int(thru_games[:4])
        selections.append(
            SelectionLink(
                season=season,
                season_label=season_label_for_ending_year(season),
                thru_games=thru_games,
                label=label,
                url=urljoin(base_url, f"/selection_rankings/nitty_gritties/{value}"),
            )
        )

    selections = list({item.url: item for item in selections}.values())
    selections.sort(key=lambda item: item.season)
    return selections


def parse_selection_metadata(html: str) -> dict[str, object]:
    text = clean_text(re.sub(r"<[^>]+>", " ", html))
    thru_games = None
    season_label = None

    ranking_match = re.search(
        r"(\d{4}-\d{2})\s+D-I Men's Basketball NET Ranking thru games\s+(\d{2}/\d{2}/\d{4})",
        text,
    )
    if ranking_match:
        season_label = ranking_match.group(1)
        thru_games = parse_mmddyyyy(ranking_match.group(2))
    else:
        thru_games = parse_mmddyyyy(text)

    season = int(thru_games[:4]) if thru_games else None
    return {
        "season": season,
        "season_label": season_label or (season_label_for_ending_year(season) if season else None),
        "thru_games": thru_games,
    }


def add_record_parts(row: dict[str, object], key: str) -> None:
    value = row.get(key)
    if not isinstance(value, str):
        return
    parsed = parse_record(value)
    row[f"{key}_wins"] = parsed.wins
    row[f"{key}_losses"] = parsed.losses


def parse_selection_snapshot(
    html: str,
    source_url: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    parser = GenericTableParser()
    parser.feed(html)
    metadata = parse_selection_metadata(html)

    rows: list[dict[str, object]] = []
    for cells in parser.rows:
        if len(cells) < 24:
            continue

        rank_index = 2
        net_rank = parse_int(cells[rank_index])
        if net_rank is None and len(cells) > 24:
            rank_index = 3
            net_rank = parse_int(cells[rank_index])
        if net_rank is None:
            continue

        row: dict[str, object] = {
            "season": metadata["season"],
            "season_label": metadata["season_label"],
            "selection_thru_games": metadata["thru_games"],
            "team": cells[0],
            "conference": cells[1],
            "net_rank": net_rank,
            "previous_net_rank": parse_int(cells[rank_index + 1]),
            "net_avg_opponent_rank": parse_int(cells[rank_index + 2]),
            "avg_opponent_net_rank": parse_int(cells[rank_index + 3]),
            "wins": parse_int(cells[rank_index + 4]),
            "overall_record": cells[rank_index + 5],
            "conference_wins": parse_int(cells[rank_index + 6]),
            "conference_record": cells[rank_index + 7],
            "nonconference_wins": parse_int(cells[rank_index + 8]),
            "nonconference_record": cells[rank_index + 9],
            "road_wins": parse_int(cells[rank_index + 10]),
            "road_record": cells[rank_index + 11],
            "net_sos": parse_int(cells[rank_index + 12]),
            "net_nonconference_sos": parse_int(cells[rank_index + 13]),
            "q1_wins": parse_int(cells[rank_index + 14]),
            "q1_record": cells[rank_index + 15],
            "q2_wins": parse_int(cells[rank_index + 16]),
            "q2_record": cells[rank_index + 17],
            "q3_wins": parse_int(cells[rank_index + 18]),
            "q3_record": cells[rank_index + 19],
            "q4_wins": parse_int(cells[rank_index + 20]),
            "q4_record": cells[rank_index + 21],
            "source_url": source_url,
        }

        for record_key in (
            "overall_record",
            "conference_record",
            "nonconference_record",
            "road_record",
            "q1_record",
            "q2_record",
            "q3_record",
            "q4_record",
        ):
            add_record_parts(row, record_key)

        rows.append(row)

    if not rows:
        raise NCAANETError("Could not find NET selection rows in the NCAA Statistics HTML.")

    metadata["row_count"] = len(rows)
    metadata["source_url"] = source_url
    return rows, metadata


def save_selection_snapshot(
    rows: list[dict[str, object]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    json_path = write_json(rows, output_dir / f"{stem}.json")
    csv_path = write_csv(rows, output_dir / f"{stem}.csv")
    return json_path, csv_path
