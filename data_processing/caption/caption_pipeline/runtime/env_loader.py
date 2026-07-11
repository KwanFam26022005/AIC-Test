"""Small dependency-free loader for caption-stage environment files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE entries and return the effective values."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Environment file not found: {target}")
    loaded: dict[str, str] = {}
    for line_no, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {target}:{line_no}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty environment key at {target}:{line_no}")
        value = _strip_quotes(value.strip())
        value = _expand_string(value)
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)
    return loaded


def expand_env_values(value: Any) -> Any:
    """Recursively expand ${NAME} references in config values."""
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, str):
        return _expand_string(value)
    return value


def _expand_string(value: str) -> str:
    previous = None
    current = value
    for _ in range(10):
        if current == previous:
            break
        previous = current
        current = _ENV_REF.sub(lambda match: os.environ.get(match.group(1), match.group(0)), current)
        current = os.path.expandvars(current)
    return current


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
