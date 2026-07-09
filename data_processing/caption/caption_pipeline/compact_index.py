"""Compact search index builder.

Implements §11 of the code plan — produces ``CompactSearchDoc`` rows for
both frames and shots, suitable for direct Elasticsearch ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import round_sec
from .schemas import COMPACT_SEARCH_SCHEMA
from .text_utils import join_texts

logger = logging.getLogger(__name__)


def build_compact_index(
    frame_evidence: list[dict],
    shot_evidence: list[dict],
) -> list[dict]:
    """Build compact search docs for all frames and shots.

    Returns a list of dicts following the ``caption_compact_search_v1`` schema,
    ordered: all frame docs first (by timestamp), then shot docs (by start_sec).
    """
    docs: list[dict] = []

    # --- Frame docs ---
    for fe in frame_evidence:
        ocr_ev = fe.get("ocr_evidence") or {}
        aud_ev = fe.get("audio_evidence") or {}
        obj_ev = fe.get("object_evidence") or {}

        ocr_text = ocr_ev.get("ocr_text", "") or ""
        audio_text = aud_ev.get("audio_text", "") or ""
        object_text = obj_ev.get("object_text", "") or ""
        scene_text = obj_ev.get("scene_text", "") or ""

        all_text = join_texts(ocr_text, audio_text, object_text, scene_text)

        ts = round_sec(fe.get("timestamp_sec"))

        doc: dict[str, Any] = {
            "schema_version": COMPACT_SEARCH_SCHEMA,
            "unit_type": "frame",
            "unit_id": fe["canonical_frame_id"],
            "video_id": fe.get("video_id", ""),
            "frame_id": fe.get("canonical_frame_id", ""),
            "timestamp_sec": ts,
            "start_sec": ts,
            "end_sec": ts,
            "ocr_text": ocr_text,
            "audio_text": audio_text,
            "object_text": object_text,
            "scene_text": scene_text,
            "caption_text": "",   # filled in Phase 2
            "trake_text": "",     # filled in Phase 4
            "all_text": all_text,
        }
        docs.append(doc)

    # --- Shot docs ---
    for se in shot_evidence:
        ocr_text = se.get("merged_ocr_text", "") or ""
        audio_text = se.get("merged_audio_text", "") or ""

        # Build object text from merged counts
        obj_parts: list[str] = []
        for label, cnt in sorted(
            (se.get("merged_object_counts") or {}).items(),
            key=lambda x: (-x[1], x[0]),
        ):
            obj_parts.append(f"{label}")
        object_text = " ".join(obj_parts)
        scene_text = " ".join(se.get("merged_scene_tags") or [])

        all_text = join_texts(ocr_text, audio_text, object_text, scene_text)

        doc = {
            "schema_version": COMPACT_SEARCH_SCHEMA,
            "unit_type": "shot",
            "unit_id": se["shot_id"],
            "video_id": se.get("video_id", ""),
            "frame_id": "",
            "timestamp_sec": round_sec(se.get("start_sec")),
            "start_sec": round_sec(se.get("start_sec")),
            "end_sec": round_sec(se.get("end_sec")),
            "ocr_text": ocr_text,
            "audio_text": audio_text,
            "object_text": object_text,
            "scene_text": scene_text,
            "caption_text": "",
            "trake_text": "",
            "all_text": all_text,
        }
        docs.append(doc)

    logger.info(
        "Compact index: %d docs (%d frame + %d shot)",
        len(docs), len(frame_evidence), len(shot_evidence),
    )
    return docs
