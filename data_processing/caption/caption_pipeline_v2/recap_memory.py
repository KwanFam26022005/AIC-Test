"""Recap memory — compact memory object for sequential shot processing.

Phase 5 prerequisite (§11.3): bounded memory that flows between shots.

Rules:
  - Maximum serialized size: 600 characters (configurable).
  - Remove stale details instead of accumulating transcript text.
  - Reset at hard scene boundary.
  - Memory is context, not proof that an action continues.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .schemas import RECAP_MEMORY_SCHEMA

logger = logging.getLogger(__name__)


def empty_memory() -> dict[str, Any]:
    """Return a fresh, empty memory object."""
    return {
        "active_entities": [],
        "setting": "",
        "ongoing_topic": "",
        "ongoing_action": "",
        "last_event": "",
    }


def serialize_memory(memory: dict[str, Any]) -> str:
    """Serialize memory to a compact JSON string."""
    return json.dumps(memory, ensure_ascii=False, separators=(",", ":"))


def truncate_memory(memory: dict[str, Any], max_chars: int = 600) -> dict[str, Any]:
    """Ensure memory does not exceed *max_chars* when serialized.

    Truncation priority (least important first):
      1. ongoing_action
      2. ongoing_topic
      3. last_event
      4. setting
      5. active_entities (truncate list)
    """
    result = dict(memory)
    serialized = serialize_memory(result)

    if len(serialized) <= max_chars:
        return result

    # Truncation cascade
    for field in ("ongoing_action", "ongoing_topic", "last_event", "setting"):
        if len(serialized) <= max_chars:
            break
        value = result.get(field, "")
        if len(value) > 50:
            result[field] = value[:50].rsplit(" ", 1)[0] + "..."
            serialized = serialize_memory(result)

    # Truncate entities list
    while len(serialized) > max_chars and result.get("active_entities"):
        result["active_entities"] = result["active_entities"][:-1]
        serialized = serialize_memory(result)

    if len(serialized) > max_chars:
        logger.warning(
            "Memory still exceeds %d chars after truncation: %d",
            max_chars, len(serialized),
        )

    return result


def should_reset_memory(
    prev_shot: dict | None,
    curr_shot: dict,
    scene_change_threshold_sec: float = 10.0,
) -> tuple[bool, str]:
    """Detect whether memory should be reset at a hard scene boundary.

    Returns ``(should_reset, reason)``.
    """
    if prev_shot is None:
        return True, "first_shot"

    # Large time gap between shots
    gap = curr_shot["start_sec"] - prev_shot["end_sec"]
    if gap > scene_change_threshold_sec:
        return True, f"time_gap_{gap:.1f}s"

    return False, ""


def build_memory_ledger_row(
    video_id: str,
    shot_id: str,
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
    reset_applied: bool,
    reset_reason: str,
    source_generation_mode: str,
    input_signature: str = "",
) -> dict[str, Any]:
    """Build a memory ledger row for state/recap_memory.jsonl."""
    return {
        "schema_version": RECAP_MEMORY_SCHEMA,
        "video_id": video_id,
        "shot_id": shot_id,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "reset_applied": reset_applied,
        "reset_reason": reset_reason,
        "source_generation_mode": source_generation_mode,
        "input_signature": input_signature,
    }
