from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .io_utils import ensure_dir
from .shape_utils import json_safe


class VLMCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vlm_cache (
                cache_key TEXT PRIMARY KEY,
                raw_text TEXT,
                corrected_text TEXT,
                payload_json TEXT,
                model_id TEXT,
                prompt_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def get(self, cache_key: str) -> dict | None:
        row = self.conn.execute("SELECT payload_json FROM vlm_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, cache_key: str, raw_text: str, corrected_text: str, payload: dict, model_id: str, prompt_version: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO vlm_cache(cache_key, raw_text, corrected_text, payload_json, model_id, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cache_key, raw_text, corrected_text, json.dumps(json_safe(payload), ensure_ascii=False), model_id, prompt_version),
        )
        self.conn.commit()
