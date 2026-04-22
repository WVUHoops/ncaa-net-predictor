"""Local configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs into the process environment."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    load_env_file()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to {DEFAULT_ENV_FILE} or export it in your shell."
        )
    return value


def get_kenpom_api_token() -> str:
    return get_required_env("KENPOM_API_TOKEN")


def get_cbb_analytics_api_key() -> str:
    return get_required_env("CBB_ANALYTICS_API_KEY")
