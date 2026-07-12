"""Shot builder — group sequential frames into shots by timestamp gap.

Phase 1 prerequisite: each shot gets a stable shot_id and ordered frame list
before frame selection runs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_shots(
    frames: list[dict],
    video_id: str,
    max_gap_sec: float = 5.0,
    max_duration_sec: float = 20.0,
) -> list[dict[str, Any]]:
    """Group sorted *frames* into shots by timestamp gap.

    Each shot dict contains:
        video_id, shot_id, start_sec, end_sec, frame_ids, frames
    """
    if not frames:
        return []

    # Ensure sorted
    sorted_frames = sorted(
        frames,
        key=lambda f: (f["timestamp_sec"], f["canonical_frame_id"]),
    )

    groups: list[list[dict]] = []
    current: list[dict] = [sorted_frames[0]]

    for f in sorted_frames[1:]:
        prev_ts = current[-1]["timestamp_sec"]
        curr_ts = f["timestamp_sec"]
        gap = curr_ts - prev_ts
        duration = curr_ts - current[0]["timestamp_sec"]

        if gap > max_gap_sec or duration > max_duration_sec:
            groups.append(current)
            current = [f]
        else:
            current.append(f)

    if current:
        groups.append(current)

    # Build shot dicts
    shots: list[dict[str, Any]] = []
    for idx, shot_frames in enumerate(groups):
        shot_id = f"{video_id}_shot_{idx:04d}"
        start_sec = round(shot_frames[0]["timestamp_sec"], 3)
        end_sec = round(shot_frames[-1]["timestamp_sec"], 3)
        frame_ids = [f["canonical_frame_id"] for f in shot_frames]

        shots.append({
            "video_id": video_id,
            "shot_id": shot_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frame_ids": frame_ids,
            "frames": shot_frames,
        })

    logger.info(
        "Shot builder: %d shots from %d frames (max_gap=%.1fs, max_dur=%.1fs)",
        len(shots), len(frames), max_gap_sec, max_duration_sec,
    )
    return shots
