"""Evidence alignment — join OCR, object, and audio per frame.

Implements §7 of the code plan:
- Primary frame list from object detection.
- Left join OCR by canonical_frame_id.
- Attach audio segments by timestamp window.
- Timestamp validation between OCR and object.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import PipelineConfig
from .io_utils import round_sec

logger = logging.getLogger(__name__)


def align_frames(
    object_frames: dict[str, dict],
    ocr_frames: dict[str, dict],
    audio_segments: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], dict[str, Any]]:
    """Align evidence sources into a list of joined frame records.

    Returns ``(aligned_frames, alignment_stats)`` where *aligned_frames*
    is sorted by ``timestamp_sec`` and *alignment_stats* is a dict of
    counters for the validation report.
    """
    stats: dict[str, Any] = {
        "num_object_frames": len(object_frames),
        "num_ocr_frames": len(ocr_frames),
        "num_audio_features_usable": len(audio_segments),
        "num_joined_frames": 0,
        "num_frames_missing_ocr": 0,
        "num_frames_missing_object": 0,
        "num_timestamp_mismatch": 0,
        "warnings": [],
    }

    # --- build primary frame list from object detection ---
    joined: list[dict] = []
    ocr_used: set[str] = set()

    for cfid, obj in object_frames.items():
        ocr = ocr_frames.get(cfid)
        if ocr is not None:
            ocr_used.add(cfid)
        else:
            stats["num_frames_missing_ocr"] += 1

        # Timestamp validation  (§7.2)
        ts_obj = round_sec(obj.get("timestamp_sec"))
        ts_ocr = round_sec(ocr.get("timestamp_sec")) if ocr else ts_obj
        ts_mismatch = abs(ts_obj - ts_ocr) > cfg.timestamp_tolerance
        if ts_mismatch and ocr is not None:
            stats["num_timestamp_mismatch"] += 1
            stats["warnings"].append(
                f"Timestamp mismatch at {cfid}: obj={ts_obj} ocr={ts_ocr}"
            )

        # Canonical timestamp = object
        timestamp_sec = ts_obj

        # Audio window  (§7.3)
        audio_window, audio_segs = _find_audio_window(
            timestamp_sec, audio_segments, cfg,
        )

        frame = {
            "canonical_frame_id": cfid,
            "video_id": obj.get("video_id", cfg.video_id),
            "frame_id": obj.get("frame_id", cfid),
            "frame_name": obj.get("frame_name", ""),
            "keyframe_idx": obj.get("keyframe_idx"),
            "source_frame_idx": obj.get("source_frame_idx"),
            "timestamp_sec": timestamp_sec,
            "image_path": obj.get("image_path", ""),
            "image_relpath": obj.get("image_relpath", ""),
            "object": obj,
            "ocr": ocr,
            "audio_window": audio_window,
            "audio_segments": audio_segs,
            "timestamp_mismatch": ts_mismatch,
        }
        joined.append(frame)

    # Optionally include OCR-only frames
    if cfg.include_ocr_only_frames:
        for cfid, ocr in ocr_frames.items():
            if cfid not in ocr_used:
                stats["num_frames_missing_object"] += 1
                ts = round_sec(ocr.get("timestamp_sec"))
                audio_window, audio_segs = _find_audio_window(
                    ts, audio_segments, cfg,
                )
                frame = {
                    "canonical_frame_id": cfid,
                    "video_id": ocr.get("video_id", cfg.video_id),
                    "frame_id": ocr.get("frame_id", cfid),
                    "frame_name": ocr.get("frame_name", ""),
                    "keyframe_idx": ocr.get("keyframe_idx"),
                    "source_frame_idx": ocr.get("source_frame_idx"),
                    "timestamp_sec": ts,
                    "image_path": ocr.get("image_path", ""),
                    "image_relpath": ocr.get("image_relpath", ""),
                    "object": None,
                    "ocr": ocr,
                    "audio_window": audio_window,
                    "audio_segments": audio_segs,
                    "timestamp_mismatch": False,
                }
                joined.append(frame)
                stats["warnings"].append(f"OCR-only frame included: {cfid}")

    # Sort by timestamp
    joined.sort(key=lambda f: (f["timestamp_sec"], f["canonical_frame_id"]))
    stats["num_joined_frames"] = len(joined)

    logger.info(
        "Alignment: %d joined frames (obj=%d, ocr=%d, ts_mismatch=%d)",
        len(joined),
        stats["num_object_frames"],
        stats["num_ocr_frames"],
        stats["num_timestamp_mismatch"],
    )
    return joined, stats


# ---------------------------------------------------------------------------
# Audio window helper  (§7.3)
# ---------------------------------------------------------------------------

def _find_audio_window(
    frame_ts: float,
    audio_segments: list[dict],
    cfg: PipelineConfig,
) -> tuple[list[float], list[dict]]:
    """Find audio segments overlapping the frame's audio window.

    Window:  ``[frame_ts - before, frame_ts + after]``
    Overlap: ``audio.end_sec >= win_start and audio.start_sec <= win_end``

    Returns ``(window_bounds, selected_segments)`` sorted by start_sec.
    """
    before = cfg.audio_alignment.frame_window_sec_before
    after = cfg.audio_alignment.frame_window_sec_after
    max_chars = cfg.audio_alignment.max_audio_chars_frame
    max_segs = cfg.audio_alignment.max_audio_segments_frame

    win_start = frame_ts - before
    win_end = frame_ts + after

    selected: list[dict] = []
    total_chars = 0

    for seg in audio_segments:
        seg_start = seg.get("start_sec", 0.0)
        seg_end = seg.get("end_sec", 0.0)

        # Overlap check
        if seg_end >= win_start and seg_start <= win_end:
            text = seg.get("caption_text", "")
            if total_chars + len(text) > max_chars and selected:
                break
            selected.append(seg)
            total_chars += len(text)
            if len(selected) >= max_segs:
                break

    return [round_sec(win_start), round_sec(win_end)], selected
