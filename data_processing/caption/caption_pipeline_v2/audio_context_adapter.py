"""Audio context adapter — read-only ASR alignment by shot.

Phase 4 (§10): Align existing ASR segments to shots without modifying
upstream audio output.

Rules:
  - Only usable_for_caption rows are eligible.
  - Sort by start, end, feature ID.
  - Deduplicate normalized repeated transcripts.
  - Preserve Vietnamese diacritics, names, and numbers.
  - Truncate after complete segments.
  - Do not summarize ASR with another model before ReCap.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import AudioContextConfig
from .schemas import SHOT_AUDIO_CONTEXT_SCHEMA

logger = logging.getLogger(__name__)


def align_audio_to_shots(
    shots: list[dict],
    audio_segments: list[dict],
    cfg: AudioContextConfig,
    video_id: str = "",
) -> list[dict[str, Any]]:
    """Align ASR segments to each shot.

    Returns a list of shot_audio_context records (one per shot).
    """
    results: list[dict[str, Any]] = []

    for shot in shots:
        ctx = _build_shot_audio_context(shot, audio_segments, cfg, video_id)
        results.append(ctx)

    shots_with_audio = sum(1 for r in results if r["has_usable_audio"])
    logger.info(
        "Audio context: %d/%d shots have usable audio",
        shots_with_audio, len(results),
    )
    return results


def _build_shot_audio_context(
    shot: dict,
    audio_segments: list[dict],
    cfg: AudioContextConfig,
    video_id: str,
) -> dict[str, Any]:
    """Build audio context for a single shot."""
    shot_id = shot["shot_id"]
    shot_start = shot["start_sec"]
    shot_end = shot["end_sec"]

    # Alignment window with padding
    win_start = shot_start - cfg.padding_before_sec
    win_end = shot_end + cfg.padding_after_sec

    # Find overlapping segments  (§10.1)
    overlapping: list[dict] = []
    for seg in audio_segments:
        seg_start = seg.get("start_sec", 0.0)
        seg_end = seg.get("end_sec", 0.0)

        if seg_end >= win_start and seg_start <= win_end:
            overlapping.append(seg)

    # Sort by start, end, feature_id
    overlapping.sort(
        key=lambda s: (s.get("start_sec", 0), s.get("end_sec", 0), s.get("feature_id", "")),
    )

    # Deduplicate normalized repeated transcripts  (§10.2)
    deduped: list[dict] = []
    seen_texts: set[str] = set()
    for seg in overlapping:
        text = (seg.get("caption_text") or "").strip()
        normalized = _normalize_text(text)
        if normalized and normalized not in seen_texts:
            seen_texts.add(normalized)
            deduped.append(seg)

    # Build text — truncate after complete segments
    feature_ids: list[str] = []
    quality_levels: list[str] = []
    text_parts: list[str] = []
    total_chars = 0
    truncated = False

    for seg in deduped:
        feature_ids.append(seg.get("feature_id", ""))
        ql = seg.get("quality_level", "")
        if ql and ql not in quality_levels:
            quality_levels.append(ql)

        text = (seg.get("caption_text") or "").strip()
        if not text:
            continue

        if total_chars + len(text) > cfg.max_audio_context_chars and text_parts:
            truncated = True
            break

        text_parts.append(text)
        total_chars += len(text)

    combined_text = " ".join(text_parts)
    has_usable = bool(combined_text)

    # Detect language — default to Vietnamese per design
    language = "vi" if has_usable else ""

    warnings: list[str] = []
    if truncated:
        warnings.append("text_truncated_at_segment_boundary")

    return {
        "schema_version": SHOT_AUDIO_CONTEXT_SCHEMA,
        "video_id": video_id or shot.get("video_id", ""),
        "shot_id": shot_id,
        "shot_start_sec": shot_start,
        "shot_end_sec": shot_end,
        "alignment_window": {
            "start_sec": round(win_start, 3),
            "end_sec": round(win_end, 3),
        },
        "feature_ids": feature_ids,
        "text": combined_text,
        "language": language,
        "quality_levels": quality_levels,
        "has_usable_audio": has_usable,
        "text_truncated": truncated,
        "warnings": warnings,
    }


def _normalize_text(text: str) -> str:
    """Normalize text for deduplication — preserves Vietnamese diacritics."""
    # Collapse whitespace, lowercase for comparison
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized
