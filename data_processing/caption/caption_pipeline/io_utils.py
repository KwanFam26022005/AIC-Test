"""Low-level JSON-Lines and file IO helpers.

Follows the same atomic-write pattern used by the audio pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_parent(path: str | Path) -> Path:
    """Create parent directories if needed and return a Path object."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# JSONL readers
# ---------------------------------------------------------------------------

def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield dicts from a JSONL file, skipping blank lines."""
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {target}:{line_no}: {exc}"
                ) from exc
            if isinstance(value, dict):
                yield value


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a complete JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


# ---------------------------------------------------------------------------
# JSONL writers (atomic)
# ---------------------------------------------------------------------------

def _tmp_path_near(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    os.close(fd)
    return Path(name)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    """Atomically write *rows* to a JSONL file."""
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
                f.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """Atomically write a JSON file."""
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def write_text(path: str | Path, text: str) -> Path:
    """Write a text file (e.g. markdown report)."""
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, target)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def round_sec(value: float | int | None, digits: int = 3) -> float:
    """Round a timestamp to *digits* decimal places (default 3)."""
    if value is None:
        return 0.0
    return round(float(value), digits)
