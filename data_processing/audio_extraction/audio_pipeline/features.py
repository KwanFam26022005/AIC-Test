from __future__ import annotations

from typing import Any

from .io_utils import round_sec
from .quality import score_quality
from .text import clean_transcript, extract_keywords, summarize_transcript


def build_quality_reports(asr_rows: list[dict], cfg: dict[str, Any]) -> list[dict]:
    text_cfg = cfg.get("text_cleaning", {})
    quality_cfg = cfg.get("quality_gate", {})
    reports: list[dict] = []
    for row in asr_rows:
        clean_text = clean_transcript(row.get("raw_text") or "", text_cfg)
        quality = score_quality(clean_text, float(row.get("duration_sec") or 0.0), quality_cfg)
        reports.append(
            {
                "schema_version": "asr_quality_v1",
                "asr_segment_id": row.get("asr_segment_id"),
                "job_id": row.get("job_id"),
                "video_id": row.get("video_id"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
                "quality": quality,
            }
        )
    return reports


def build_audio_features(asr_rows: list[dict], cfg: dict[str, Any]) -> list[dict]:
    text_cfg = cfg.get("text_cleaning", {})
    quality_cfg = cfg.get("quality_gate", {})
    schema_version = cfg.get("pipeline", {}).get("schema_version", "audio_feature_v1")
    include_empty = bool(quality_cfg.get("include_empty_features", False))
    max_keywords = int(text_cfg.get("max_keywords") or 8)

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
        quality = score_quality(clean_text, float(row.get("duration_sec") or 0.0), quality_cfg)
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

        features.append(
            {
                "schema_version": schema_version,
                "feature_id": feature_id,
                "modality": "audio",
                "video_id": row.get("video_id"),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round_sec(end_sec - start_sec),
                "raw_transcript": raw_text,
                "clean_transcript": clean_text,
                "summary": summary,
                "keywords": keywords,
                "usable_for_caption": quality["usable_for_caption"],
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

