"""Caption index builder - build ``frame_index.jsonl`` and optionally
rebuild ``compact_search_index.jsonl`` with caption text.

Implements section 3 and section 4.2 of the Phase 2 plan.
"""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import round_sec
from .schemas import COMPACT_SEARCH_SCHEMA, FRAME_INDEX_SCHEMA
from .text_utils import join_texts, normalize_whitespace, remove_accents

logger = logging.getLogger(__name__)


def build_frame_index(
    frame_evidence: list[dict],
    caption_records: list[dict],
) -> list[dict]:
    """Build ``frame_index.jsonl`` rows from frame evidence and caption records.

    Returns a list of dicts following the ``caption_frame_index_v1`` schema.
    """
    # Index caption records by canonical_frame_id for O(1) lookup
    caption_map: dict[str, dict] = {}
    for cr in caption_records:
        cfid = cr["canonical_frame_id"]
        caption_map[cfid] = cr

    results: list[dict] = []
    for fe in frame_evidence:
        cfid = fe["canonical_frame_id"]
        cr = caption_map.get(cfid, {})

        caption_text = cr.get("caption_text", "") or ""
        caption_text_search = _make_search_text(caption_text)
        caption_terms = _extract_terms(caption_text_search)

        # Evidence text block
        ocr_ev = fe.get("ocr_evidence") or {}
        aud_ev = fe.get("audio_evidence") or {}
        obj_ev = fe.get("object_evidence") or {}

        ocr_text = ocr_ev.get("ocr_text", "") or ""
        audio_text = aud_ev.get("audio_text", "") or ""

        # Build concise object text from important_objects
        important = obj_ev.get("important_objects") or []
        object_text = ", ".join(important) if important else (obj_ev.get("object_text", "") or "")
        scene_text = obj_ev.get("scene_text", "") or ""

        evidence_text: dict[str, str] = {
            "ocr_text": ocr_text,
            "audio_text": audio_text,
            "object_text": object_text,
            "scene_text": scene_text,
        }

        # Search fields
        all_text = join_texts(
            ocr_text, audio_text, object_text, scene_text, caption_text,
        )
        visual_text = join_texts(
            object_text, scene_text,
            " ".join(obj_ev.get("ram_tags") or []),
        )
        search_fields: dict[str, str] = {
            "all_text": all_text,
            "caption_boost_text": caption_text_search,
            "visual_text": visual_text,
            "spoken_text": audio_text,
            "onscreen_text": ocr_text,
        }

        # Quality block
        quality: dict[str, Any] = {
            "has_caption": bool(caption_text),
            "caption_mode": cr.get("caption_mode", "template"),
            "caption_model": cr.get("caption_model", ""),
            "provider": cr.get("provider", ""),
            "prompt_version": cr.get("prompt_version", ""),
            "prompt_hash": cr.get("prompt_hash", ""),
            "generation_attempts": cr.get("generation_attempts", 0),
            "initial_caption_used": cr.get("initial_caption_used", False),
            "used_ocr": cr.get("used_ocr", False),
            "used_audio": cr.get("used_audio", False),
            "used_objects": cr.get("used_objects", False),
            "used_scene": cr.get("used_scene", False),
            "fallback_used": cr.get("fallback_used", False),
            "warnings": cr.get("warnings") or [],
        }

        ts = round_sec(fe.get("timestamp_sec"))

        doc: dict[str, Any] = {
            "schema_version": FRAME_INDEX_SCHEMA,
            "document_id": f"caption_frame:{cfid}",
            "unit_type": "frame",
            "video_id": fe.get("video_id", ""),
            "frame_id": fe.get("frame_id", cfid),
            "canonical_frame_id": cfid,
            "timestamp_sec": ts,
            "image_relpath": fe.get("image_relpath", ""),
            "caption_text": caption_text,
            "caption_text_search": caption_text_search,
            "caption_terms": caption_terms,
            "evidence_text": evidence_text,
            "search_fields": search_fields,
            "quality": quality,
        }
        results.append(doc)

    logger.info("Frame index: %d records built", len(results))
    return results


def rebuild_compact_with_captions(
    compact_docs: list[dict],
    caption_records: list[dict],
) -> list[dict]:
    """Rebuild compact search docs with ``caption_text`` from Phase 2.

    Only frame docs (``unit_type=frame``) get their ``caption_text`` and
    ``all_text`` updated.  Shot docs are left unchanged.
    """
    caption_map: dict[str, str] = {}
    for cr in caption_records:
        cfid = cr["canonical_frame_id"]
        caption_map[cfid] = cr.get("caption_text", "") or ""

    updated = 0
    result: list[dict] = []
    for doc in compact_docs:
        doc = dict(doc)  # shallow copy
        if doc.get("unit_type") == "frame":
            cfid = doc.get("unit_id", "")
            caption_text = caption_map.get(cfid, "")
            if caption_text:
                doc["caption_text"] = caption_text
                # Rebuild all_text to include caption
                doc["all_text"] = join_texts(
                    doc.get("ocr_text", ""),
                    doc.get("audio_text", ""),
                    doc.get("object_text", ""),
                    doc.get("scene_text", ""),
                    caption_text,
                )
                updated += 1
        result.append(doc)

    logger.info(
        "Compact index rebuild: %d/%d frame docs updated with caption_text",
        updated, sum(1 for d in result if d.get("unit_type") == "frame"),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_text(text: str) -> str:
    """Normalize caption text for search: lowercase + remove accents."""
    if not text:
        return ""
    return normalize_whitespace(remove_accents(text.lower()))


def _extract_terms(search_text: str) -> list[str]:
    """Extract unique search terms from normalized search text."""
    if not search_text:
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for tok in search_text.split():
        # Strip punctuation
        clean = tok.strip(".,;:!?\"'()[]{}")
        if clean and clean not in seen and len(clean) >= 2:
            seen.add(clean)
            terms.append(clean)
    return terms
