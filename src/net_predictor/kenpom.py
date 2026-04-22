"""KenPom API client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from net_predictor.config import get_kenpom_api_token


KENPOM_API_URL = "https://kenpom.com/api.php"


class KenPomAPIError(RuntimeError):
    """Raised when the KenPom API cannot be reached or returns an error."""


class KenPomClient:
    def __init__(self, token: str | None = None, base_url: str = KENPOM_API_URL) -> None:
        self.token = token or get_kenpom_api_token()
        self.base_url = base_url

    def get(self, endpoint: str, **params: Any) -> Any:
        query = {"endpoint": endpoint}
        query.update({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}?{urlencode(query)}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "ncaa-net-predictor/0.1",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KenPomAPIError(f"KenPom API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise KenPomAPIError(f"Could not reach KenPom API: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise KenPomAPIError("KenPom API returned non-JSON response.") from exc

    def save_json(self, endpoint: str, output_path: Path, **params: Any) -> Path:
        data = self.get(endpoint, **params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output_path
