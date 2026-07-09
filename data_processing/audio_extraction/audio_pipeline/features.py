from __future__ import annotations

from collections import Counter
from typing import Any

from .io_utils import round_sec
from .quality import score_quality
from .text import (
    clean_transcript,
    extract_keywords,
    is_boilerplate_transcript,
    normalize_for_quality,
    summarize_transcript,
)


def build_quality_reports(asr_rows: list[dict], cfg: dict[str, Any]) -> list[dict]:
    """Build quality reports for each ASR segment.

    Phase 2: output is sorted by timeline.
    Phase 3: includes boilerplate and duplicate transcript detection.
    """
    text_cfg = cfg.get("text_cleaning", {})
    quality_cfg = cfg.get("quality_gate", {})
    boilerplate_patterns = quality_cfg.get("boilerplate_patterns", [])
    dup_min_count = int(quality_cfg.get("duplicate_transcript_min_count") or 3)

    # Phase 3: compute per-video duplicate transcript counts
    dup_counts = _compute_duplicate_counts(asr_rows, text_cfg)

    reports: list[dict] = []
    for row in asr_rows:
        clean_text = clean_transcript(row.get("raw_text") or "", text_cfg)
        normalized = normalize_for_quality(clean_text)
        video_id = row.get("video_id", "")

        boilerplate = is_boilerplate_transcript(clean_text, boilerplate_patterns)
        dup_count = dup_counts.get((video_id, normalized), 0)
        is_dup = dup_count >= dup_min_count

        quality = score_quality(
            clean_text,
            float(row.get("duration_sec") or 0.0),
            quality_cfg,
            is_boilerplate=boilerplate,
            is_duplicate_transcript=is_dup,
        )
        reports.append(
            {
                "schema_version": "asr_quality_v1",
                "asr_segment_id": row.get("asr_segment_id"),
                "job_id": row.get("job_id"),
                "video_id": video_id,
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
                "quality": quality,
            }
        )
    # Phase 2: sort by timeline
    reports.sort(key=lambda r: (
        str(r.get("video_id", "")),
        float(r.get("start_sec") or 0.0),
        float(r.get("end_sec") or 0.0),
    ))
    return reports


def build_audio_features(asr_rows: list[dict], cfg: dict[str, Any]) -> list[dict]:
    """Build audio feature records for downstream caption pipeline.

    Phase 2: output sorted by timeline.
    Phase 3: boilerplate and duplicate detection.
    Phase 5: caption-ready fields (time_window, caption_text, search_text,
             exclude_from_caption_reason).
    """
    text_cfg = cfg.get("text_cleaning", {})
    quality_cfg = cfg.get("quality_gate", {})
    schema_version = cfg.get("pipeline", {}).get("schema_version", "audio_feature_v1")
    include_empty = bool(quality_cfg.get("include_empty_features", False))
    max_keywords = int(text_cfg.get("max_keywords") or 8)
    boilerplate_patterns = quality_cfg.get("boilerplate_patterns", [])
    dup_min_count = int(quality_cfg.get("duplicate_transcript_min_count") or 3)

    # Phase 3: compute per-video duplicate transcript counts
    dup_counts = _compute_duplicate_counts(asr_rows, text_cfg)

    features: list[dict] = []
    seen_ids: set[str] = set()
    for row in asr_rows:
        if row.get("status") == "failed":
            continue
        asr_segment_id = str(row.get("asr_segment_id") or "")
        if not asr_segment_id or asr_segment_id in seen_ids:
            continue
        seen_ids.add(asr_segment_id)

        raw_text = row.get("raw_text") or ""
        clean_text = clean_transcript(raw_text, text_cfg)
        normalized = normalize_for_quality(clean_text)
        video_id = row.get("video_id", "")

        boilerplate = is_boilerplate_transcript(clean_text, boilerplate_patterns)
        dup_count = dup_counts.get((video_id, normalized), 0)
        is_dup = dup_count >= dup_min_count

        quality = score_quality(
            clean_text,
            float(row.get("duration_sec") or 0.0),
            quality_cfg,
            is_boilerplate=boilerplate,
            is_duplicate_transcript=is_dup,
        )
        if quality["quality_level"] == "empty" and not include_empty:
            continue

        summary = summarize_transcript(clean_text, text_cfg)
        keywords = extract_keywords(clean_text, max_keywords=max_keywords)
        asr_info = row.get("asr_info") or {}
        runtime = row.get("runtime") or {}
        start_sec = round_sec(row.get("start_sec"))
        end_sec = round_sec(row.get("end_sec"))
        feature_id = asr_segment_id.replace("asrseg_", "audfeat_", 1)
        if feature_id == asr_segment_id:
            feature_id = f"audfeat_{asr_segment_id}"

        # Phase 5: caption-ready fields
        usable = quality["usable_for_caption"]
        caption_text = clean_text if usable else ""
        search_text = clean_text  # always keep for audit/search
        reasons = quality.get("reasons", [])
        exclude_reason = "; ".join(reasons) if reasons else ""

        features.append(
            {
                "schema_version": schema_version,
                "feature_id": feature_id,
                "modality": "audio",
                "video_id": video_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round_sec(end_sec - start_sec),
                "time_window": [start_sec, end_sec],
                "raw_transcript": raw_text,
                "clean_transcript": clean_text,
                "caption_text": caption_text,
                "search_text": search_text,
                "summary": summary,
                "keywords": keywords,
                "usable_for_caption": usable,
                "exclude_from_caption_reason": exclude_reason,
                "quality_level": quality["quality_level"],
                "source_model": asr_info.get("model"),
                "source": {
                    "asr_segment_id": asr_segment_id,
                    "job_id": row.get("job_id"),
                    "backend": asr_info.get("backend"),
                    "language": row.get("language"),
                },
                "runtime": runtime,
            }
        )
    features.sort(key=lambda item: (str(item.get("video_id")), float(item.get("start_sec") or 0.0)))
    return features


def _compute_duplicate_counts(
    asr_rows: list[dict], text_cfg: dict[str, Any]
) -> dict[tuple[str, str], int]:
    """Count how many times each normalized transcript appears per video."""
    counter: Counter[tuple[str, str]] = Counter()
    for row in asr_rows:
        if row.get("status") == "failed":
            continue
        raw_text = row.get("raw_text") or ""
        clean_text = clean_transcript(raw_text, text_cfg)
        normalized = normalize_for_quality(clean_text)
        if normalized:
            video_id = row.get("video_id", "")
            counter[(video_id, normalized)] += 1
    return dict(counter)
