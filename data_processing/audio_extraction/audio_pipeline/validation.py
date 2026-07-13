"""Phase 6: Post-run validation and quality summary."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl

logger = logging.getLogger(__name__)


def build_audio_quality_summary(
    asr_rows: list[dict],
    quality_rows: list[dict],
    feature_rows: list[dict],
    video_id: str,
) -> dict[str, Any]:
    """Build a summary dict for quick post-run audit.

    Checks timeline ordering, quality distribution, boilerplate/duplicate
    counts, and row counts across outputs.
    """
    # Count quality levels
    quality_counts: Counter[str] = Counter()
    num_boilerplate = 0
    num_duplicate = 0
    for qr in quality_rows:
        quality = qr.get("quality", {})
        quality_counts[quality.get("quality_level", "unknown")] += 1
        if quality.get("is_boilerplate"):
            num_boilerplate += 1
        if quality.get("is_duplicate_transcript"):
            num_duplicate += 1

    # Check timeline ordering
    num_out_of_order = 0
    prev_start = -1.0
    for row in asr_rows:
        start = float(row.get("start_sec") or 0.0)
        vid = row.get("video_id", "")
        if vid == video_id:
            if start < prev_start:
                num_out_of_order += 1
            prev_start = start

    timeline_sorted = num_out_of_order == 0
    if not timeline_sorted:
        logger.warning(
            "ASR segments for %s are NOT sorted by timeline: %d out-of-order segments",
            video_id,
            num_out_of_order,
        )

    num_caption_usable = sum(
        1 for f in feature_rows if f.get("usable_for_caption")
    )

    return {
        "video_id": video_id,
        "num_asr_segments": len(asr_rows),
        "num_quality_reports": len(quality_rows),
        "num_audio_features": len(feature_rows),
        "num_caption_usable": num_caption_usable,
        "num_boilerplate_filtered": num_boilerplate,
        "num_duplicate_filtered": num_duplicate,
        "timeline_sorted": timeline_sorted,
        "num_out_of_order": num_out_of_order,
        "quality_counts": dict(quality_counts),
    }


def write_quality_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write the quality summary as a pretty-printed JSON file."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info("Wrote audio quality summary to %s", target)
    return target
