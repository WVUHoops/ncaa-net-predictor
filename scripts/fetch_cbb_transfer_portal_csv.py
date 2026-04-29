#!/usr/bin/env python3
"""Fetch the CBB Analytics transfer portal ledger and save it as CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_BASE_URL = "https://api.cbbanalytics.com/api"
GS_API_BASE_URL = f"{API_BASE_URL}/gs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition-id",
        type=int,
        default=41097,
        help="CBB Analytics competition ID for the current men's D-I season.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "cbb_analytics" / "transfer_portal" / "current",
    )
    parser.add_argument(
        "--output-name-prefix",
        default="cbb_transfer_portal_41097",
        help="Prefix for the downloaded CSV filename.",
    )
    parser.add_argument(
        "--include-raw-json",
        action="store_true",
        help="Also persist the raw API JSON response next to the CSV.",
    )
    return parser.parse_args()


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_project_env_fallbacks() -> dict[str, str]:
    fallbacks: dict[str, str] = {}
    plain_path = PROJECT_ROOT / ".env.cbb"
    if plain_path.exists():
        fallbacks.update(parse_env_text(plain_path.read_text(encoding="utf-8")))

    rtf_path = PROJECT_ROOT / ".env.cbb.rtf"
    if rtf_path.exists() and shutil.which("textutil"):
        try:
            text = subprocess.check_output(
                ["textutil", "-convert", "txt", "-stdout", str(rtf_path)],
                text=True,
            )
            fallbacks.update(parse_env_text(text))
        except (OSError, subprocess.SubprocessError):
            pass
    return fallbacks


def env_value(name: str, fallbacks: dict[str, str]) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return fallbacks.get(name, "").strip()


def fetch_bytes(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed with HTTP {exc.code}: {detail[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach CBB Analytics: {exc.reason}") from exc


def post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            data = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Login failed with HTTP {exc.code}: {detail[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach CBB Analytics login: {exc.reason}") from exc

    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise RuntimeError("Login response was not a JSON object.")
    return parsed


def login_token(fallbacks: dict[str, str]) -> str | None:
    token = env_value("CBB_AUTH_TOKEN", fallbacks)
    if token:
        return token

    email = env_value("CBB_EMAIL", fallbacks)
    password = env_value("CBB_PASSWORD", fallbacks)
    if not email or not password:
        return None

    response = post_json(
        f"{API_BASE_URL}/users/login",
        {"email": email, "password": password},
        {
            "User-Agent": "ncaa-net-predictor/0.1",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    token = str(response.get("token") or "").strip()
    if not token:
        raise RuntimeError("CBB login succeeded but no token was returned.")
    return token


def endpoint_rows(competition_id: int, token: str | None) -> list[dict[str, object]]:
    params = {"competitionId": competition_id}
    url = f"{GS_API_BASE_URL}/vc-transfer-portal?{urlencode(params)}"
    headers = {
        "User-Agent": "ncaa-net-predictor/0.1",
        "Accept": "application/json",
    }
    if token:
        headers["x-auth-token"] = token

    body, _ = fetch_bytes(url, headers)
    parsed = json.loads(body.decode("utf-8", errors="replace"))
    if not isinstance(parsed, list):
        raise RuntimeError("Transfer portal endpoint did not return a JSON list.")
    return [row for row in parsed if isinstance(row, dict)]


def direct_csv_bytes(fallbacks: dict[str, str]) -> tuple[bytes, str] | None:
    url = env_value("CBB_TRANSFER_CSV_URL", fallbacks)
    if not url:
        return None

    headers = {
        "User-Agent": "ncaa-net-predictor/0.1",
        "Accept": "text/csv,application/csv,text/plain,*/*",
    }
    authorization = env_value("CBB_TRANSFER_AUTHORIZATION", fallbacks)
    cookie = env_value("CBB_TRANSFER_COOKIE", fallbacks)
    referer = env_value("CBB_TRANSFER_REFERER", fallbacks)
    if authorization:
        headers["Authorization"] = authorization
    if cookie:
        headers["Cookie"] = cookie
    if referer:
        headers["Referer"] = referer
    return fetch_bytes(url, headers)


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        first_name = str(row.get("firstName") or "").strip()
        last_name = str(row.get("lastName") or "").strip()
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        normalized.append(
            {
                "updatedWhen": row.get("updatedWhen"),
                "portalStatus": row.get("status"),
                "playerId": row.get("playerId"),
                "fullName": full_name or row.get("fullName"),
                "player_name": full_name or row.get("fullName"),
                "competitionId": row.get("competitionId"),
                "divisionId": row.get("divisionIdFrom") or row.get("divisionId"),
                "conferenceId": row.get("conferenceIdFrom"),
                "teamId": row.get("teamIdFrom"),
                "teamMarket": row.get("teamMarketFrom"),
                "source_team": row.get("teamMarketFrom"),
                "divisionIdTo": row.get("divisionIdTo"),
                "conferenceIdTo": row.get("conferenceIdTo"),
                "teamIdTo": row.get("teamIdTo"),
                "teamMarketTo": row.get("teamMarketTo"),
                "destination_team": row.get("teamMarketTo"),
                "eligibilityYear": row.get("eligibilityYear"),
                "createdWhen": row.get("createdWhen"),
                "updated": row.get("updated"),
                "activeFlag": row.get("activeFlag"),
                "redshirtFlag": row.get("redshirtFlag"),
                "portalRecency": row.get("portalRecency"),
                "vcPlayerId": row.get("vcPlayerId"),
                "vcFirstName": row.get("vcFirstName"),
                "vcLastName": row.get("vcLastName"),
                "vcTeamMarketFrom": row.get("vcTeamMarketFrom"),
                "vcTeamMarketTo": row.get("vcTeamMarketTo"),
                "noMatch": row.get("noMatch"),
                "rawStatus": row.get("status"),
            }
        )
    return normalized


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No transfer portal rows were returned.")
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


def main() -> int:
    args = parse_args()
    fallbacks = load_project_env_fallbacks()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.output_name_prefix}_{stamp}.csv"

    direct_result = direct_csv_bytes(fallbacks)
    if direct_result is not None:
        body, content_type = direct_result
        if not body:
            raise RuntimeError("CBB transfer CSV download returned an empty response.")
        preview = body[:512].decode("utf-8", errors="replace").lower()
        if "<html" in preview or "enable javascript" in preview:
            raise RuntimeError("Configured CBB transfer CSV URL returned HTML instead of CSV.")
        output_path.write_bytes(body)
        print(f"saved {output_path}")
        print(f"bytes: {len(body)}")
        print(f"content_type: {content_type}")
        return 0

    token = login_token(fallbacks)
    rows = endpoint_rows(args.competition_id, token)
    normalized = normalize_rows(rows)
    write_csv(normalized, output_path)
    print(f"saved {output_path}")
    print(f"rows: {len(normalized)}")
    print(f"competition_id: {args.competition_id}")
    print("source: gs/vc-transfer-portal")

    if args.include_raw_json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
