"""Resumable JSONL checkpoints for frame, shot, and event generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl, write_jsonl


class JsonlCheckpoint:
    def __init__(
        self,
        path: str | Path,
        key_field: str,
        generation_signature: str,
        resume: bool = True,
        write_every: int = 10,
    ) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.generation_signature = generation_signature
        self.write_every = max(int(write_every), 1)
        self.rows: dict[str, dict] = {}
        self.order: list[str] = []
        self.pending = 0
        self.num_reused = 0
        if resume and self.path.exists():
            for row in read_jsonl(self.path):
                key = str(row.get(self.key_field, ""))
                if key and key not in self.rows:
                    self.order.append(key)
                    self.rows[key] = row

    def get(self, key: str, input_signature: str = "") -> dict | None:
        row = self.rows.get(str(key))
        if not row:
            return None
        if row.get("generation_signature") != self.generation_signature:
            return None
        if input_signature and row.get("input_signature") != input_signature:
            return None
        if row.get("fallback_used"):
            return None
        self.num_reused += 1
        return dict(row)

    def put(self, row: dict) -> None:
        key = str(row.get(self.key_field, ""))
        if not key:
            raise ValueError(f"Checkpoint row is missing key field '{self.key_field}'")
        item = dict(row)
        item["generation_signature"] = self.generation_signature
        if key not in self.rows:
            self.order.append(key)
        self.rows[key] = item
        self.pending += 1
        if self.pending >= self.write_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending and self.path.exists():
            return
        write_jsonl(self.path, [self.rows[key] for key in self.order])
        self.pending = 0


def stable_input_signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
