"""Minimal dotenv loader (no external dependency).

Parses ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` without
overwriting values already set in the environment — so shell exports and the
SDKs' own env resolution keep priority over the file. Lines starting with ``#``
and blank lines are ignored; surrounding quotes on values are stripped.

Providers read these directly when ``api_key``/``base_url`` are left ``None``
(``ANTHROPIC_API_KEY``/``ANTHROPIC_BASE_URL``/``OPENAI_API_KEY``/``OPENAI_BASE_URL``/
``DEEPSEEK_API_KEY``), so loading the file is all that's needed for the env
system to take effect.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Load ``.env`` into ``os.environ`` (without overriding existing values).

    Returns the mapping of keys that were actually set (i.e. absent from the
    environment beforehand). Missing file -> no-op, returns ``{}``.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
