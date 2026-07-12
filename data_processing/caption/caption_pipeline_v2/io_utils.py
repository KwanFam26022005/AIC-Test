"""Small JSON, JSONL, and text IO helpers for caption pipeline v2."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {target}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
                handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def write_json(path: str | Path, value: Any, indent: int = 2) -> Path:
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = ensure_parent(path)
    tmp = _tmp_path_near(target)
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tmp_path_near(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    return Path(name)
