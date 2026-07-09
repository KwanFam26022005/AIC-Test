"""Event-step index builder for Phase 4 TRAKE outputs."""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import round_sec
from .schemas import EVENT_STEP_INDEX_SCHEMA
from .text_utils import join_texts, normalize_whitespace, remove_accents

logger = logging.getLogger(__name__)


def build_event_step_index(
    shot_index: list[dict],
    shot_evidence: list[dict],
    event_records: list[dict],
) -> list[dict]:
    """Build ``event_step_index.jsonl`` rows from TRAKE event records."""
    shot_map = {
        row.get("shot_id"): row
        for row in shot_index
        if row.get("shot_id")
    }
    evidence_map = {
        row.get("shot_id"): row
        for row in shot_evidence
        if row.get("shot_id")
    }

    results: list[dict] = []
    for rec in event_records:
        shot_id = rec["shot_id"]
        shot = shot_map.get(shot_id, {})
        evidence = evidence_map.get(shot_id, {})
        source = shot.get("evidence_text") or {}

        trake_text = rec.get("trake_text", "") or ""
        trake_text_search = _make_search_text(trake_text)
        trake_terms = _extract_terms(trake_text_search)

        source_text = {
            "caption_text": shot.get("caption_text", "") or "",
            "temporal_caption": shot.get("temporal_caption", "") or "",
            "merged_ocr_text": source.get("merged_ocr_text", "") or evidence.get("merged_ocr_text", "") or "",
            "merged_audio_text": source.get("merged_audio_text", "") or evidence.get("merged_audio_text", "") or "",
            "object_text": source.get("object_text", "") or "",
            "scene_text": source.get("scene_text", "") or "",
        }

        quality: dict[str, Any] = {
            "has_event_caption": bool(rec.get("event_caption")),
            "has_trake_text": bool(trake_text),
            "event_mode": rec.get("event_mode", "template"),
            "event_model": rec.get("event_model", ""),
            "prompt_version": rec.get("prompt_version", ""),
            "fallback_used": rec.get("fallback_used", False),
            "warnings": rec.get("warnings") or [],
        }

        doc = {
            "schema_version": EVENT_STEP_INDEX_SCHEMA,
            "document_id": f"event_step:{shot_id}",
            "doc_type": "event_step",
            "unit_type": "event_step",
            "video_id": shot.get("video_id", ""),
            "event_id": rec["event_id"],
            "shot_id": shot_id,
            "step_order": rec["step_order"],
            "start_sec": round_sec(shot.get("start_sec")),
            "end_sec": round_sec(shot.get("end_sec")),
            "representative_frame_ids": shot.get("representative_frame_ids") or [],
            "event_caption": rec.get("event_caption", "") or "",
            "current_observation": rec.get("current_observation", "") or "",
            "before_context": rec.get("before_context", "") or "",
            "after_context": rec.get("after_context", "") or "",
            "action_state": rec.get("action_state", "unknown"),
            "temporal_role": rec.get("temporal_role", "unknown"),
            "actors": rec.get("actors") or [],
            "actions": rec.get("actions") or [],
            "objects_involved": rec.get("objects_involved") or [],
            "scene": rec.get("scene", "") or "",
            "trake_text": trake_text,
            "trake_text_search": trake_text_search,
            "trake_terms": trake_terms,
            "source_text": source_text,
            "quality": quality,
        }
        results.append(doc)

    logger.info("Event-step index: %d records built", len(results))
    return results


def rebuild_compact_with_trake_text(
    compact_docs: list[dict],
    event_records: list[dict],
) -> list[dict]:
    """Set ``trake_text`` on shot docs in the compact search index."""
    trake_map = {
        row.get("shot_id"): row.get("trake_text", "") or ""
        for row in event_records
        if row.get("shot_id")
    }

    updated = 0
    results: list[dict] = []
    for doc in compact_docs:
        doc = dict(doc)
        if doc.get("unit_type") == "shot":
            trake_text = trake_map.get(doc.get("unit_id", ""), "")
            if trake_text:
                doc["trake_text"] = trake_text
                doc["all_text"] = join_texts(
                    doc.get("ocr_text", ""),
                    doc.get("audio_text", ""),
                    doc.get("object_text", ""),
                    doc.get("scene_text", ""),
                    doc.get("caption_text", ""),
                    doc.get("trake_text", ""),
                )
                updated += 1
        results.append(doc)

    logger.info("Compact index rebuild: %d shot docs updated with trake_text", updated)
    return results


def _make_search_text(text: str) -> str:
    if not text:
        return ""
    return normalize_whitespace(remove_accents(text.lower()))


def _extract_terms(search_text: str) -> list[str]:
    if not search_text:
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for tok in search_text.split():
        clean = tok.strip(".,;:!?\"'()[]{}")
        if clean and clean not in seen and len(clean) >= 2:
            seen.add(clean)
            terms.append(clean)
    return terms
