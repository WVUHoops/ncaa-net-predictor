#!/usr/bin/env python3
"""Fetch player-level On3 high-school commit pages for ranked basketball classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from net_predictor.coach_factor import canonical_team_key  # noqa: E402
from net_predictor.on3 import (  # noqa: E402
    On3Error,
    extract_next_data,
    fetch_html,
    hs_commit_rows_from_page,
    hs_commits_url,
    utc_now_iso,
    write_csv,
    write_json,
)
from net_predictor.on3_features import latest_files_by_source_year, read_json_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--on3-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "on3",
        help="Base On3 raw directory containing hs team-ranking snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "on3" / "hs_commits",
        help="Directory where dated player-level commit snapshots are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latest_files = latest_files_by_source_year(list(args.on3_dir.glob("hs/*/*.json")))
    ranking_path = latest_files.get(("hs", args.year))
    if ranking_path is None:
        raise SystemExit(f"no hs team-ranking snapshot found for {args.year}")

    ranking_rows = read_json_rows(ranking_path)
    captured_at = utc_now_iso()
    all_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    seen_urls: set[str] = set()
    for ranking_row in ranking_rows:
        team_slug = ranking_row.get("team_slug")
        team = ranking_row.get("team")
        commits = ranking_row.get("commits")
        if not team_slug or not team:
            continue
        if not commits:
            continue
        url = hs_commits_url(str(team_slug), args.year)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            html = fetch_html(url)
            data = extract_next_data(html)
            rows = hs_commit_rows_from_page(
                data,
                url,
                captured_at,
                team_slug=str(team_slug),
                team_name=str(team),
                ranking_year=args.year,
            )
            for row in rows:
                row["team_key"] = canonical_team_key(row.get("team"))
            all_rows.extend(rows)
        except On3Error as exc:
            failures.append((str(team), str(exc)))

    all_rows.sort(key=lambda row: (int(row.get("season") or 0), str(row.get("team") or ""), str(row.get("player_name") or "")))
    stamp = captured_at[:10]
    output_dir = args.output_dir / str(args.year)
    json_path = write_json(all_rows, output_dir / f"on3_hs_commits_{args.year}_{stamp}.json")
    csv_path = write_csv(all_rows, output_dir / f"on3_hs_commits_{args.year}_{stamp}.csv")

    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"player rows: {len(all_rows)}")
    print(f"teams attempted: {len(seen_urls)}")
    print(f"teams failed: {len(failures)}")
    if failures:
        for team, error in failures[:20]:
            print(f"failed: {team}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
