"""Shot index builder for Phase 3 shot captions."""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import round_sec
from .schemas import SHOT_INDEX_SCHEMA
from .text_utils import join_texts, normalize_whitespace, remove_accents

logger = logging.getLogger(__name__)


def build_shot_index(
    shot_evidence: list[dict],
    frame_index: list[dict],
    shot_caption_records: list[dict],
) -> list[dict]:
    """Build ``shot_index.jsonl`` rows from shot evidence and caption records."""
    frame_map = {
        row.get("canonical_frame_id"): row
        for row in frame_index
        if row.get("canonical_frame_id")
    }
    caption_map = {
        row.get("shot_id"): row
        for row in shot_caption_records
        if row.get("shot_id")
    }

    results: list[dict] = []
    for idx, shot in enumerate(shot_evidence):
        shot_id = shot["shot_id"]
        cr = caption_map.get(shot_id, {})

        caption_text = cr.get("caption_text", "") or ""
        temporal_caption = cr.get("temporal_caption", "") or ""
        caption_text_search = _make_search_text(caption_text)
        temporal_caption_search = _make_search_text(temporal_caption)
        caption_terms = _extract_terms(join_texts(caption_text_search, temporal_caption_search))

        rep_ids = shot.get("representative_frame_ids") or []
        rep_caption_texts = []
        for frame_id in rep_ids:
            frame_doc = frame_map.get(frame_id)
            caption = (frame_doc or {}).get("caption_text", "") or ""
            if caption:
                rep_caption_texts.append(caption)

        merged_ocr = shot.get("merged_ocr_text", "") or ""
        merged_audio = shot.get("merged_audio_text", "") or ""
        object_text = _object_counts_to_text(shot.get("merged_object_counts") or {}, max_labels=12)
        scene_text = " ".join(shot.get("merged_scene_tags") or [])
        rep_caption_text = " || ".join(rep_caption_texts)

        evidence_text = {
            "merged_ocr_text": merged_ocr,
            "merged_audio_text": merged_audio,
            "object_text": object_text,
            "scene_text": scene_text,
            "representative_frame_caption_text": rep_caption_text,
        }

        memory = {
            "memory_before": cr.get("memory_before", "") or "",
            "memory_after": cr.get("memory_after", "") or "",
            "previous_shot_id": shot_evidence[idx - 1]["shot_id"] if idx > 0 else "",
            "next_shot_id": shot_evidence[idx + 1]["shot_id"] if idx < len(shot_evidence) - 1 else "",
        }

        all_text = join_texts(
            merged_ocr,
            merged_audio,
            object_text,
            scene_text,
            rep_caption_text,
            caption_text,
            temporal_caption,
        )
        caption_boost_text = join_texts(caption_text_search, temporal_caption_search)
        temporal_text = join_texts(
            memory["memory_before"], temporal_caption, memory["memory_after"],
        )
        search_fields = {
            "all_text": all_text,
            "caption_boost_text": caption_boost_text,
            "frame_caption_text": rep_caption_text,
            "visual_text": join_texts(object_text, scene_text),
            "spoken_text": merged_audio,
            "onscreen_text": merged_ocr,
            "temporal_text": temporal_text,
        }

        quality: dict[str, Any] = {
            "has_caption": bool(caption_text),
            "caption_mode": cr.get("caption_mode", "template"),
            "caption_model": cr.get("caption_model", ""),
            "prompt_version": cr.get("prompt_version", ""),
            "used_ocr": cr.get("used_ocr", False),
            "used_audio": cr.get("used_audio", False),
            "used_objects": cr.get("used_objects", False),
            "used_scene": cr.get("used_scene", False),
            "used_frame_captions": cr.get("used_frame_captions", False),
            "fallback_used": cr.get("fallback_used", False),
            "num_representative_frames": len(rep_ids),
            "num_representative_frame_captions": len(rep_caption_texts),
            "warnings": cr.get("warnings") or [],
        }

        doc: dict[str, Any] = {
            "schema_version": SHOT_INDEX_SCHEMA,
            "document_id": f"caption_shot:{shot_id}",
            "unit_type": "shot",
            "video_id": shot.get("video_id", ""),
            "shot_id": shot_id,
            "start_sec": round_sec(shot.get("start_sec")),
            "end_sec": round_sec(shot.get("end_sec")),
            "representative_frame_ids": rep_ids,
            "frame_count": shot.get("frame_count", 0),
            "caption_text": caption_text,
            "temporal_caption": temporal_caption,
            "caption_text_search": caption_text_search,
            "temporal_caption_search": temporal_caption_search,
            "caption_terms": caption_terms,
            "evidence_text": evidence_text,
            "memory": memory,
            "search_fields": search_fields,
            "quality": quality,
        }
        results.append(doc)

    logger.info("Shot index: %d records built", len(results))
    return results


def rebuild_compact_with_frame_and_shot_captions(
    compact_docs: list[dict],
    frame_caption_records: list[dict] | None = None,
    shot_caption_records: list[dict] | None = None,
) -> list[dict]:
    """Rebuild compact docs with available frame and shot captions."""
    frame_caption_map = {
        row.get("canonical_frame_id"): row.get("caption_text", "") or ""
        for row in (frame_caption_records or [])
        if row.get("canonical_frame_id")
    }
    shot_caption_map = {
        row.get("shot_id"): row.get("caption_text", "") or ""
        for row in (shot_caption_records or [])
        if row.get("shot_id")
    }

    frame_updated = 0
    shot_updated = 0
    results: list[dict] = []
    for doc in compact_docs:
        doc = dict(doc)
        unit_type = doc.get("unit_type")
        unit_id = doc.get("unit_id", "")

        if unit_type == "frame":
            caption_text = frame_caption_map.get(unit_id, "")
            if caption_text:
                doc["caption_text"] = caption_text
                frame_updated += 1
        elif unit_type == "shot":
            caption_text = shot_caption_map.get(unit_id, "")
            if caption_text:
                doc["caption_text"] = caption_text
                shot_updated += 1

        if doc.get("caption_text"):
            doc["all_text"] = join_texts(
                doc.get("ocr_text", ""),
                doc.get("audio_text", ""),
                doc.get("object_text", ""),
                doc.get("scene_text", ""),
                doc.get("caption_text", ""),
                doc.get("trake_text", ""),
            )
        results.append(doc)

    logger.info(
        "Compact index rebuild: %d frame docs and %d shot docs updated with caption_text",
        frame_updated, shot_updated,
    )
    return results


def _object_counts_to_text(counts: dict[str, int], max_labels: int) -> str:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_labels]
    return ", ".join(f"{cnt} {label}" for label, cnt in items if cnt > 0)


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
