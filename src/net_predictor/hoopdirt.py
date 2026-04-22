"""HoopDirt coaching-change tracker ingestion helpers."""

from __future__ import annotations

import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOOPDIRT_2026_TRACKER_URL = "https://hoopdirt.com/2026-coaching-changes-tracker/"
DEFAULT_D1_TABLE_ID = "1064755"


class HoopDirtError(RuntimeError):
    """Raised when HoopDirt tracker data cannot be fetched or parsed."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def fetch_text(url: str, *, accept: str = "text/html,application/xhtml+xml", referer: str | None = None) -> str:
    headers = {
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
    }
    if referer:
        headers["Referer"] = referer

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HoopDirtError(f"HoopDirt returned HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise HoopDirtError(f"Could not reach HoopDirt: {exc.reason}") from exc


def extract_d1_ajax_url(html_text: str, table_id: str = DEFAULT_D1_TABLE_ID) -> str:
    table_pattern = re.compile(
        rf'"data_request_url"\s*:\s*"(?P<url>[^"]*table_id={re.escape(table_id)}[^"]*)"',
        flags=re.DOTALL,
    )
    match = table_pattern.search(html_text)
    if not match:
        raise HoopDirtError(f"Could not find HoopDirt Ninja Table AJAX URL for table_id={table_id}.")

    url = match.group("url").replace(r"\/", "/")
    return html.unescape(url)


def parse_ajax_rows(payload: str | list[Any], *, season: int, source_url: str, captured_at: str) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HoopDirtError("Could not parse HoopDirt AJAX JSON.") from exc
    else:
        data = payload

    if not isinstance(data, list):
        raise HoopDirtError("Expected HoopDirt AJAX payload to be a JSON list.")

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        value = item.get("value") or {}
        if not isinstance(value, dict):
            continue

        school = clean_text(value.get("school"))
        if not school:
            continue

        rows.append(
            {
                "source": "hoopdirt_coaching_changes",
                "source_url": source_url,
                "captured_at": captured_at,
                "season": season,
                "division": "D-I",
                "school": school,
                "conference": clean_text(value.get("conference")),
                "former_coach": clean_text(value.get("old_coach")),
                "new_coach": clean_text(value.get("new_coach")),
                "hoopdirt_row_id": value.get("___id___"),
            }
        )

    return rows


def fetch_d1_coaching_changes(
    *,
    season: int,
    tracker_url: str = HOOPDIRT_2026_TRACKER_URL,
    table_id: str = DEFAULT_D1_TABLE_ID,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captured_at = utc_now_iso()
    html_text = fetch_text(tracker_url)
    ajax_url = extract_d1_ajax_url(html_text, table_id=table_id)
    ajax_payload = fetch_text(ajax_url, accept="application/json,text/javascript,*/*;q=0.1", referer=tracker_url)
    rows = parse_ajax_rows(ajax_payload, season=season, source_url=tracker_url, captured_at=captured_at)
    metadata = {
        "source_url": tracker_url,
        "ajax_url": ajax_url,
        "captured_at": captured_at,
        "season": season,
        "table_id": table_id,
        "row_count": len(rows),
    }
    return rows, metadata


def parse_ajax_file(
    path: Path,
    *,
    season: int,
    source_url: str = HOOPDIRT_2026_TRACKER_URL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captured_at = utc_now_iso()
    rows = parse_ajax_rows(path.read_text(encoding="utf-8"), season=season, source_url=source_url, captured_at=captured_at)
    return rows, {
        "source_url": source_url,
        "ajax_file": path.as_posix(),
        "captured_at": captured_at,
        "season": season,
        "row_count": len(rows),
    }


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
