"""Loaders for OCR, object-detection, and audio feature JSONL files.

Each loader reads raw pipeline output and returns a dict keyed by the
appropriate join key (``canonical_frame_id`` for frame-level, or a list
sorted by ``start_sec`` for audio).

Loaders also apply the guardrails specified in the code plan:
- OCR: use ``ocr_text_search`` as primary text.
- Object: remove scene labels that leaked into object counts.
- Audio: only keep rows with ``usable_for_caption=true``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, round_sec
from .schemas import EXPECTED_AUDIO_SCHEMA, EXPECTED_OBJECT_SCHEMA, EXPECTED_OCR_SCHEMA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR loader  (§6.1)
# ---------------------------------------------------------------------------

def load_ocr(path: str | Path) -> dict[str, dict]:
    """Load OCR ES docs keyed by ``canonical_frame_id``.

    Returns a dict  ``{canonical_frame_id: ocr_record}``.
    """
    result: dict[str, dict] = {}
    path = Path(path)
    if not path.exists():
        logger.warning("OCR file not found: %s", path)
        return result

    for row in iter_jsonl(path):
        sv = row.get("schema_version", "")
        if sv and sv != EXPECTED_OCR_SCHEMA:
            logger.debug("OCR schema %s (expected %s)", sv, EXPECTED_OCR_SCHEMA)

        video_id = row.get("video_id", "")
        canonical = _resolve_canonical(row, video_id)
        if not canonical:
            logger.warning("OCR row missing canonical_frame_id: %s", row.get("frame_id"))
            continue

        # Quality guard: reject primary text if review lines leaked into
        # the primary search field.
        quality = row.get("quality") or {}
        need_review_in_primary = quality.get("num_need_review_lines_in_primary_search", 0)

        ocr_text = row.get("ocr_text_search", "") or ""
        if need_review_in_primary and need_review_in_primary > 0:
            logger.debug(
                "OCR frame %s: rejecting primary text (%d review lines in search)",
                canonical, need_review_in_primary,
            )
            ocr_text = ""

        record: dict[str, Any] = {
            "video_id": video_id,
            "canonical_frame_id": canonical,
            "frame_id": row.get("frame_id", canonical),
            "frame_name": row.get("frame_name", ""),
            "keyframe_idx": row.get("keyframe_idx"),
            "source_frame_idx": row.get("source_frame_idx"),
            "timestamp_sec": round_sec(row.get("timestamp_sec")),
            "image_path": row.get("image_path", ""),
            "image_relpath": row.get("image_relpath", ""),
            "ocr_text": ocr_text,
            "ocr_text_clean": row.get("ocr_text_clean", "") or "",
            "ocr_terms": row.get("ocr_terms") or [],
            "ocr_text_review": row.get("ocr_text_review", "") or "",
            "has_clean_text": bool(quality.get("has_clean_text", False)),
        }
        result[canonical] = record

    logger.info("OCR loader: loaded %d frames from %s", len(result), path.name)
    return result


# ---------------------------------------------------------------------------
# Object detection loader  (§6.2)
# ---------------------------------------------------------------------------

def load_objects(path: str | Path, return_stats: bool = False) -> dict[str, dict] | tuple[dict[str, dict], dict[str, int]]:
    """Load object-detection rows keyed by ``canonical_frame_id``.

    Applies the scene-label guardrail: any label in ``scene_tags`` is
    removed from ``object_counts_normalized``.
    """
    result: dict[str, dict] = {}
    stats = {"num_scene_labels_removed_from_counts": 0}
    path = Path(path)
    if not path.exists():
        logger.warning("Object JSONL not found: %s", path)
        return (result, stats) if return_stats else result

    scene_labels_removed = 0

    for row in iter_jsonl(path):
        sv = row.get("schema_version", "")
        if sv and sv != EXPECTED_OBJECT_SCHEMA:
            logger.debug("Object schema %s (expected %s)", sv, EXPECTED_OBJECT_SCHEMA)

        video_id = row.get("video_id", "")
        canonical = _resolve_canonical(row, video_id)
        if not canonical:
            logger.warning("Object row missing canonical_frame_id: %s", row.get("frame_id"))
            continue

        # --- scene-label guardrail ---
        scene_tags: list[str] = row.get("scene_tags") or []
        scene_set = {t.lower() for t in scene_tags}

        raw_counts: dict[str, int] = row.get("object_counts_normalized") or {}
        filtered_counts: dict[str, int] = {}
        removed_for_frame = 0
        for label, cnt in raw_counts.items():
            if cnt <= 0:
                continue
            if label.lower() in scene_set:
                scene_labels_removed += 1
                removed_for_frame += 1
                continue
            filtered_counts[label] = cnt

        # Rebuild object_count_items from filtered counts (sorted by count desc)
        count_items = sorted(
            [{"label": k, "count": v} for k, v in filtered_counts.items()],
            key=lambda x: (-x["count"], x["label"]),
        )

        # Rebuild important_objects from filtered count_items
        important = []
        for item in count_items:
            label, cnt = item["label"], item["count"]
            if cnt > 1:
                important.append(f"{cnt} {label}s")
            else:
                important.append(f"1 {label}")

        object_text = _object_counts_to_text(count_items)
        scene_text = (row.get("scene_text") or "").strip() or " ".join(scene_tags)
        ram_tags = row.get("ram_tags") or []
        ram_tag_text = (row.get("ram_tag_text") or "").strip() or " ".join(ram_tags)
        all_object_text = " ".join(
            part for part in (object_text, scene_text, ram_tag_text) if part
        )

        record: dict[str, Any] = {
            "video_id": video_id,
            "canonical_frame_id": canonical,
            "timestamp_sec": round_sec(row.get("timestamp_sec")),
            "image_path": row.get("image_path", ""),
            "image_relpath": row.get("image_relpath", ""),
            "frame_name": row.get("frame_name", ""),
            "keyframe_idx": row.get("keyframe_idx"),
            "source_frame_idx": row.get("source_frame_idx"),
            "object_counts": filtered_counts,
            "object_count_items": count_items,
            "object_tags": row.get("object_tags") or [],
            "scene_tags": scene_tags,
            "ram_tags": ram_tags,
            "important_objects": important,
            "object_text": object_text,
            "scene_text": scene_text,
            "ram_tag_text": ram_tag_text,
            "all_object_text": all_object_text,
            "scene_labels_removed_from_counts": removed_for_frame,
            "objects": row.get("objects") or [],
        }
        result[canonical] = record

    stats["num_scene_labels_removed_from_counts"] = scene_labels_removed
    logger.info(
        "Object loader: loaded %d frames from %s (scene labels removed from counts: %d)",
        len(result), path.name, scene_labels_removed,
    )
    return (result, stats) if return_stats else result


# ---------------------------------------------------------------------------
# Audio loader  (§6.3)
# ---------------------------------------------------------------------------

def load_audio_features(path: str | Path, return_stats: bool = False) -> list[dict] | tuple[list[dict], dict[str, int]]:
    """Load audio features, returning a sorted list of usable segments.

    Only rows with ``usable_for_caption=True`` are kept.
    Text is ``caption_text`` if available, else ``clean_transcript``.
    """
    usable: list[dict] = []
    total = 0
    stats = {"num_audio_features": 0, "num_audio_features_usable": 0}
    path = Path(path)
    if not path.exists():
        logger.warning("Audio features not found: %s", path)
        return (usable, stats) if return_stats else usable

    for row in iter_jsonl(path):
        total += 1
        sv = row.get("schema_version", "")
        if sv and sv != EXPECTED_AUDIO_SCHEMA:
            logger.debug("Audio schema %s (expected %s)", sv, EXPECTED_AUDIO_SCHEMA)

        if not row.get("usable_for_caption", False):
            continue

        caption_text = (row.get("caption_text") or "").strip()
        if not caption_text:
            caption_text = (row.get("clean_transcript") or "").strip()

        search_text = (row.get("search_text") or "").strip()

        usable.append({
            "feature_id": row.get("feature_id", ""),
            "video_id": row.get("video_id", ""),
            "start_sec": round_sec(row.get("start_sec")),
            "end_sec": round_sec(row.get("end_sec")),
            "caption_text": caption_text,
            "search_text": search_text,
            "quality_level": row.get("quality_level", ""),
        })

    # Ensure timeline order
    usable.sort(key=lambda s: (s["start_sec"], s["end_sec"], s["feature_id"]))

    logger.info(
        "Audio loader: %d usable / %d total from %s",
        len(usable), total, path.name,
    )
    stats["num_audio_features"] = total
    stats["num_audio_features_usable"] = len(usable)
    return (usable, stats) if return_stats else usable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_canonical(row: dict, video_id: str) -> str:
    """Resolve a canonical_frame_id from a row, applying fallbacks.

    Fallback rules from §2:
    - If ``canonical_frame_id`` present → use it.
    - If ``frame_id`` starts with ``{video_id}_`` → use it as canonical.
    - Otherwise, synthesize ``{video_id}_{frame_id}``.
    """
    canonical = (row.get("canonical_frame_id") or "").strip()
    if canonical:
        return canonical

    frame_id = (row.get("frame_id") or "").strip()
    if not frame_id:
        return ""

    if video_id and frame_id.startswith(f"{video_id}_"):
        return frame_id

    if video_id:
        return f"{video_id}_{frame_id}"

    return frame_id


def _object_counts_to_text(count_items: list[dict[str, Any]]) -> str:
    """Build search text from filtered object counts."""
    tokens: list[str] = []
    for item in count_items:
        label = str(item.get("label") or "").strip()
        count = int(item.get("count") or 0)
        if not label or count <= 0:
            continue
        tokens.extend([label] * count)
    return " ".join(tokens)
