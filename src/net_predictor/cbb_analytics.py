"""CBB Analytics API client."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from net_predictor.config import get_cbb_analytics_api_key


CBB_ANALYTICS_BASE_URL = "https://rest.cbbanalytics.com"
DEFAULT_LIMIT = 1000


class CBBAnalyticsError(RuntimeError):
    """Raised when CBB Analytics cannot be reached or returns an error."""


class CBBAnalyticsClient:
    def __init__(
        self,
        api_key: str | None = None,
        version: str = "v1",
        base_url: str = CBB_ANALYTICS_BASE_URL,
        retries: int = 3,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key or get_cbb_analytics_api_key()
        self.version = version.strip("/")
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def endpoint_url(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        query = {
            key: value
            for key, value in (params or {}).items()
            if value is not None and value != ""
        }
        url = f"{self.base_url}/{self.version}{endpoint}"
        if query:
            url = f"{url}?{urlencode(query, doseq=True)}"
        return url

    def get(self, endpoint: str, **params: Any) -> Any:
        url = self.endpoint_url(endpoint, params)
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ncaa-net-predictor/0.1",
                "X-API-Key": self.api_key,
            },
        )

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=60) as response:
                    body = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise CBBAnalyticsError(
                    f"CBB Analytics returned HTTP {exc.code}: {detail[:500]}"
                ) from exc
            except URLError as exc:
                if attempt >= self.retries:
                    raise CBBAnalyticsError(f"Could not reach CBB Analytics: {exc.reason}") from exc
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CBBAnalyticsError("CBB Analytics returned non-JSON response.") from exc

    def get_all(
        self,
        endpoint: str,
        *,
        limit: int = DEFAULT_LIMIT,
        max_pages: int | None = None,
        **params: Any,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = int(params.pop("offset", 0) or 0)
        pages = 0

        while True:
            page = self.get(endpoint, limit=limit, offset=offset, **params)
            page_rows = normalize_rows(page)
            rows.extend(page_rows)
            pages += 1

            if len(page_rows) < limit:
                break
            if max_pages is not None and pages >= max_pages:
                break
            offset += limit

        return rows


def normalize_rows(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    if isinstance(response, dict):
        for key in ("data", "results", "items", "rows"):
            value = response.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [response]
    raise CBBAnalyticsError(f"Unexpected CBB Analytics response type: {type(response).__name__}")


def compact_endpoint_name(endpoint: str) -> str:
    return endpoint.strip("/").replace("/", "_").replace("-", "_")


def write_json(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

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
    return output_path
