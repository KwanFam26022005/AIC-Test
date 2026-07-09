"""Shot grouping — group sequential frames into shots.

Implements §9 of the code plan (timestamp-gap method) and produces
ShotEvidence records (§5.2).
"""

from __future__ import annotations

import logging
from typing import Any

from .config import PipelineConfig
from .io_utils import round_sec
from .schemas import SHOT_EVIDENCE_SCHEMA
from .text_utils import dedup_lines, join_texts, truncate

logger = logging.getLogger(__name__)


def group_shots(
    frame_evidence: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """Group *frame_evidence* records into shots.

    Returns a list of ShotEvidence dicts (§5.2 schema).
    """
    sg = cfg.shot_grouping

    if not frame_evidence:
        return []

    # Ensure sorted by timestamp
    frames = sorted(frame_evidence, key=lambda f: (f["timestamp_sec"], f["canonical_frame_id"]))

    shots: list[list[dict]] = []
    current: list[dict] = [frames[0]]

    for f in frames[1:]:
        prev_ts = current[-1]["timestamp_sec"]
        curr_ts = f["timestamp_sec"]
        gap = curr_ts - prev_ts
        shot_duration = curr_ts - current[0]["timestamp_sec"]

        if gap > sg.max_gap_sec or shot_duration > sg.max_shot_duration_sec:
            shots.append(current)
            current = [f]
        else:
            current.append(f)

    if current:
        shots.append(current)

    # Build ShotEvidence for each shot
    results: list[dict] = []
    for idx, shot_frames in enumerate(shots, start=1):
        shot_id = f"{cfg.video_id}_shot_{idx:04d}"
        shot_ev = _build_shot_evidence(shot_id, shot_frames, cfg)
        results.append(shot_ev)

    logger.info(
        "Shot grouper: %d shots from %d frames (method=%s)",
        len(results), len(frame_evidence), sg.method,
    )
    return results


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_shot_evidence(
    shot_id: str,
    frames: list[dict],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Build a single ShotEvidence dict from a list of FrameEvidence records."""
    sg = cfg.shot_grouping
    eb = cfg.evidence_builder

    start_sec = round_sec(frames[0]["timestamp_sec"])
    end_sec = round_sec(frames[-1]["timestamp_sec"])

    # Representative frames  (§9)
    rep_ids = _pick_representatives(
        [f["canonical_frame_id"] for f in frames],
        sg.max_representative_frames,
        sg.representative_strategy,
    )

    # Merged object counts
    merged_counts: dict[str, int] = {}
    merged_scene_set: set[str] = set()
    for f in frames:
        obj_ev = f.get("object_evidence") or {}
        for label, cnt in (obj_ev.get("object_counts") or {}).items():
            merged_counts[label] = merged_counts.get(label, 0) + cnt
        for tag in (obj_ev.get("scene_tags") or []):
            merged_scene_set.add(tag)
    merged_scene_tags = sorted(merged_scene_set)

    # Merged OCR text (deduped, chronological)
    ocr_snippets: list[str] = []
    for f in frames:
        ocr_ev = f.get("ocr_evidence") or {}
        t = (ocr_ev.get("ocr_text") or "").strip()
        if t:
            ocr_snippets.append(t)
    merged_ocr = truncate(" ".join(dedup_lines(ocr_snippets)), eb.max_ocr_chars)

    # Merged audio text (deduped, chronological)
    audio_snippets: list[str] = []
    for f in frames:
        aud_ev = f.get("audio_evidence") or {}
        t = (aud_ev.get("audio_text") or "").strip()
        if t:
            audio_snippets.append(t)
    merged_audio = truncate(" ".join(dedup_lines(audio_snippets)), eb.max_audio_chars_shot)

    # Shot search text
    merged_obj_text = " ".join(
        f"{cnt} {label}" for label, cnt in sorted(
            merged_counts.items(), key=lambda x: (-x[1], x[0]),
        )
    )
    merged_scene_text = " ".join(merged_scene_tags)
    shot_search_text = join_texts(
        merged_ocr, merged_audio, merged_obj_text, merged_scene_text,
    )

    return {
        "schema_version": SHOT_EVIDENCE_SCHEMA,
        "video_id": cfg.video_id,
        "shot_id": shot_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "representative_frame_ids": rep_ids,
        "frame_count": len(frames),
        "merged_object_counts": merged_counts,
        "merged_scene_tags": merged_scene_tags,
        "merged_ocr_text": merged_ocr,
        "merged_audio_text": merged_audio,
        "shot_search_text": shot_search_text,
    }


# ---------------------------------------------------------------------------
# Representative frame picker
# ---------------------------------------------------------------------------

def _pick_representatives(
    frame_ids: list[str],
    max_rep: int,
    strategy: str,
) -> list[str]:
    """Pick representative frame IDs from a list.

    Strategy ``first_middle_last``:
    - 1 frame → first
    - 2 frames → first, last
    - >=3 frames → first, middle, last
    """
    n = len(frame_ids)
    if n == 0:
        return []
    if n == 1 or max_rep == 1:
        return [frame_ids[0]]
    if n == 2 or max_rep == 2:
        return [frame_ids[0], frame_ids[-1]]
    # >=3
    mid = n // 2
    reps = [frame_ids[0], frame_ids[mid], frame_ids[-1]]
    return reps[:max_rep]
