#!/usr/bin/env python3
"""Discover NCAA Stats season-division pages for D-I men's basketball NET."""

from __future__ import annotations

import argparse
import html
import re
import sys
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.ncaa_stats_net import (  # noqa: E402
    NCAA_STATS_BASE_URL,
    NCAANETError,
    discover_selection_links,
    is_akamai_interstitial,
    solve_akamai_interstitial,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-url",
        default=(
            "https://stats.ncaa.org/selection_rankings/season_divisions/18703/nitty_gritties"
        ),
        help="Known D-I men's basketball NET season page containing the academic-year selector.",
    )
    parser.add_argument(
        "--list-selections",
        action="store_true",
        help="Fetch each discovered season page and print Selection Sunday snapshot links.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Probe a numeric season-division ID range instead of using the seed page selector.",
    )
    parser.add_argument("--start-id", type=int, default=17_000)
    parser.add_argument("--end-id", type=int, default=19_000)
    parser.add_argument("--max-bytes", type=int, default=160_000)
    return parser.parse_args()


def fetch_partial(opener, url: str, max_bytes: int) -> str | None:
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
        with opener.open(request, timeout=20) as response:
            html = response.read(max_bytes).decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        detail = exc.read(300).decode("utf-8", errors="replace")
        raise NCAANETError(f"NCAA Statistics returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise NCAANETError(f"Could not reach NCAA Statistics: {exc.reason}") from exc

    if is_akamai_interstitial(html):
        try:
            html = solve_akamai_interstitial(opener, url, html)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
    return html


def is_mens_basketball_net_page(html: str) -> bool:
    return (
        ("D-I Men&#39;s Basketball" in html or "D-I Men's Basketball" in html)
        and "NET Ranking" in html
    )


def academic_year_options(page_html: str) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value, label_html in re.findall(
        r'<option[^>]*\bvalue="([^"]+)"[^>]*>(.*?)</option>',
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label = html.unescape(re.sub(r"<[^>]+>", " ", label_html)).strip()
        if not re.fullmatch(r"\d{4}-\d{2}", label):
            continue
        if value in seen:
            continue
        options.append((value, label))
        seen.add(value)
    return options


def discover_from_seed(opener, seed_url: str, *, list_selections: bool, max_bytes: int) -> None:
    page_html = fetch_partial(opener, seed_url, max_bytes)
    if not page_html:
        return

    for season_division_id, label in academic_year_options(page_html):
        season_url = (
            f"{NCAA_STATS_BASE_URL}/selection_rankings/season_divisions/"
            f"{season_division_id}/nitty_gritties"
        )
        if not list_selections:
            print(f"{label} {season_division_id} {season_url}")
            continue

        season_html = page_html if season_url == seed_url else fetch_partial(opener, season_url, max_bytes)
        if not season_html:
            print(f"{label} {season_division_id}")
            continue
        selection_links = discover_selection_links(season_html, base_url=season_url)
        selections = ", ".join(
            f"{link.season}:{link.thru_games}:{link.url}" for link in selection_links
        )
        print(f"{label} {season_division_id} {selections}".rstrip())


def main() -> int:
    args = parse_args()
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    if not args.scan:
        discover_from_seed(
            opener,
            args.seed_url,
            list_selections=args.list_selections,
            max_bytes=args.max_bytes,
        )
        return 0

    for season_division_id in range(args.start_id, args.end_id + 1):
        url = (
            f"{NCAA_STATS_BASE_URL}/selection_rankings/season_divisions/"
            f"{season_division_id}/nitty_gritties"
        )
        html = fetch_partial(opener, url, args.max_bytes)
        if not html or not is_mens_basketball_net_page(html):
            continue

        selection_links = discover_selection_links(html, base_url=url)
        selections = ", ".join(
            f"{link.season}:{link.thru_games}:{link.url.rsplit('/', 1)[-1]}"
            for link in selection_links
        )
        print(f"{season_division_id} {selections}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NCAANETError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
