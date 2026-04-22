"""On3 recruiting and transfer ranking ingestion helpers."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


ON3_HS_RECRUITING_URL_TEMPLATE = (
    "https://www.on3.com/rivals/rankings/industry-team/basketball/{year}/"
)
ON3_TRANSFER_PORTAL_URL_TEMPLATE = (
    "https://www.on3.com/transfer-portal/team-rankings/basketball/{year}/"
)


class On3Error(RuntimeError):
    """Raised when On3 rankings cannot be fetched or parsed."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def epoch_to_iso(value: int | float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(microsecond=0).isoformat()


def page_url(url: str, page: int) -> str:
    if page <= 1:
        return url

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def source_url(source: str, year: int) -> str:
    if source == "hs":
        return ON3_HS_RECRUITING_URL_TEMPLATE.format(year=year)
    if source == "transfer":
        return ON3_TRANSFER_PORTAL_URL_TEMPLATE.format(year=year)
    raise On3Error(f"Unknown On3 source: {source}")


def fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 ncaa-net-predictor/0.1",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise On3Error(f"On3 returned HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise On3Error(f"Could not reach On3: {exc.reason}") from exc


def extract_next_data(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise On3Error("Could not find __NEXT_DATA__ in the On3 HTML.")

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise On3Error("Could not parse On3 __NEXT_DATA__ JSON.") from exc


def organization_fields(item: dict[str, Any]) -> dict[str, Any]:
    org = item.get("organization") or {}
    return {
        "organization_key": org.get("key") or item.get("organizationKey"),
        "team": org.get("name"),
        "team_full_name": org.get("fullName"),
        "team_slug": org.get("slug") or org.get("urlSlug"),
        "team_abbreviation": org.get("abbreviation"),
    }


def hs_rows_from_page(data: dict[str, Any], source_url_value: str, captured_at: str) -> list[dict[str, Any]]:
    props = data["props"]["pageProps"]
    team_data = props["teamData"]
    ranking = team_data["relatedModel"]
    year = ranking.get("year")

    rows: list[dict[str, Any]] = []
    for item in team_data["list"]:
        row = {
            "source": "on3_hs_recruiting",
            "source_url": source_url_value,
            "captured_at": captured_at,
            "ranking_year": year,
            "ranking_key": ranking.get("key"),
            "ranking_date_updated": epoch_to_iso(ranking.get("dateUpdated")),
            "ranking_status": ranking.get("status"),
            "average_commits_used": ranking.get("averageCommits"),
            **organization_fields(item),
            "rank": item.get("overallConsensusRank") or item.get("overallRank"),
            "on3_rank": item.get("overallRank"),
            "industry_rank": item.get("overallConsensusRank"),
            "score": item.get("dispayConsensusScore"),
            "on3_score": item.get("dispayOn3Score"),
            "commits": item.get("commits"),
            "applied_commits": item.get("appliedCommits"),
            "avg_rating": item.get("appliedAverageConsensusRating"),
            "on3_avg_rating": item.get("appliedAverageRating"),
            "total_rating": item.get("appliedTotalConsensusRating"),
            "on3_total_rating": item.get("appliedTotalRating"),
            "five_stars": item.get("consensusFiveStars"),
            "on3_five_stars": item.get("fiveStars"),
            "four_stars": item.get("consensusFourStars"),
            "on3_four_stars": item.get("fourStars"),
            "three_stars": item.get("consensusThreeStars"),
            "on3_three_stars": item.get("threeStars"),
            "avg_nil_value": item.get("averageNilValue"),
            "conference_rank": item.get("conferenceConsensusRank"),
        }
        rows.append(row)
    return rows


def transfer_rows_from_page(
    data: dict[str, Any],
    source_url_value: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    props = data["props"]["pageProps"]
    team_rankings = props["teamRankings"]
    ranking = team_rankings["relatedModel"]
    year = ranking.get("year")

    rows: list[dict[str, Any]] = []
    for item in team_rankings["list"]:
        row = {
            "source": "on3_transfer_portal",
            "source_url": source_url_value,
            "captured_at": captured_at,
            "ranking_year": year,
            "ranking_key": ranking.get("key"),
            "ranking_date_updated": epoch_to_iso(ranking.get("dateUpdated")),
            "ranking_date_scheduled": epoch_to_iso(ranking.get("dateScheduled")),
            "ranking_status": ranking.get("status"),
            "ranking_staleness": ranking.get("staleness"),
            **organization_fields(item),
            "rank": item.get("overallRank"),
            "index_score": item.get("overallScore"),
            "raw_score": item.get("rawScore"),
            "date": item.get("date"),
            "date_modified": epoch_to_iso(item.get("dateModified")),
            "transfers_in": item.get("totalIn"),
            "transfers_in_avg_rating": item.get("totalInAverageRating"),
            "raw_score_in": item.get("rawScoreIn"),
            "transfers_out": item.get("totalOut"),
            "transfers_out_avg_rating": item.get("totalOutAverageRating"),
            "raw_score_out": item.get("rawScoreOut"),
            "five_stars_net": item.get("fiveStarsNet"),
            "five_stars_in": item.get("fiveStarsIn"),
            "five_stars_out": item.get("fiveStarsOut"),
            "four_stars_net": item.get("fourStarsNet"),
            "four_stars_in": item.get("fourStarsIn"),
            "four_stars_out": item.get("fourStarsOut"),
            "three_stars_net": item.get("threeStarsNet"),
            "three_stars_in": item.get("threeStarsIn"),
            "three_stars_out": item.get("threeStarsOut"),
            "original_nil_valuation": item.get("originalNilValuation"),
            "adjusted_nil_valuation": item.get("adjustedNilValuation"),
            "nil_valuation_change": item.get("nilValuationChange"),
        }
        rows.append(row)
    return rows


def page_payload(source: str, data: dict[str, Any]) -> dict[str, Any]:
    props = data["props"]["pageProps"]
    if source == "hs":
        payload = props.get("teamData")
    if source == "transfer":
        payload = props.get("teamRankings")
    if source not in {"hs", "transfer"}:
        raise On3Error(f"Unknown On3 source: {source}")
    if not isinstance(payload, dict):
        raise On3Error(f"Could not find On3 {source} rankings payload in the page.")
    return payload


def parse_page(source: str, html: str, url: str, captured_at: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = extract_next_data(html)
    payload = page_payload(source, data)
    pagination = payload.get("pagination") or {}

    if source == "hs":
        rows = hs_rows_from_page(data, url, captured_at)
    elif source == "transfer":
        rows = transfer_rows_from_page(data, url, captured_at)
    else:
        raise On3Error(f"Unknown On3 source: {source}")

    metadata = {
        "source": source,
        "source_url": url,
        "captured_at": captured_at,
        "count": pagination.get("count"),
        "current_page": pagination.get("currentPage"),
        "page_count": pagination.get("pageCount"),
        "items_per_page": pagination.get("itemsPerPage"),
        "row_count": len(rows),
    }
    return rows, metadata


def fetch_rankings(source: str, year: int, all_pages: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captured_at = utc_now_iso()
    base_url = source_url(source, year)
    first_html = fetch_html(base_url)
    rows, metadata = parse_page(source, first_html, base_url, captured_at)

    page_count = int(metadata.get("page_count") or 1)
    if all_pages:
        for page in range(2, page_count + 1):
            url = page_url(base_url, page)
            page_rows, _ = parse_page(source, fetch_html(url), url, captured_at)
            rows.extend(page_rows)

    metadata["row_count"] = len(rows)
    return rows, metadata


def parse_html_file(source: str, html_path: Path, url: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captured_at = utc_now_iso()
    source_url_value = url or html_path.as_posix()
    return parse_page(source, html_path.read_text(encoding="utf-8"), source_url_value, captured_at)


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
