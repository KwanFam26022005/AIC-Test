from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .asr import run_asr_jobs
from .features import build_audio_features, build_quality_reports
from .io_utils import read_jsonl, round_sec, safe_stem, utc_now_iso, write_jsonl
from .media import extract_audio, probe_video
from .paths import AudioOutputPaths
from .vad import build_asr_jobs
from .validation import build_audio_quality_summary, write_quality_summary

logger = logging.getLogger(__name__)


def run_pipeline(
    videos: list[Path],
    output_dir: str | Path,
    cfg: dict[str, Any],
    video_id_override: str | None = None,
    force: bool = False,
    skip_asr: bool = False,
) -> dict[str, Any]:
    if not videos:
        raise ValueError("No input videos were provided.")
    if video_id_override and len(videos) != 1:
        raise ValueError("--video_id can only be used with a single --video input.")

    paths = AudioOutputPaths.from_root(output_dir)
    paths.mkdirs()

    logger.info("Output root: %s", paths.output_root)
    video_rows = create_video_manifest(videos, video_id_override)
    write_jsonl(paths.video_manifest, video_rows)

    audio_rows = extract_audio_stage(video_rows, paths, cfg, force=force)
    write_jsonl(paths.audio_manifest, audio_rows)

    jobs = build_asr_jobs(audio_rows, cfg)
    write_jsonl(paths.asr_job_manifest, jobs)

    if force:
        write_jsonl(paths.asr_segments, [])
        write_jsonl(paths.asr_failed, [])

    if not skip_asr:
        run_asr_jobs(jobs, paths, cfg, force=force)
        asr_rows = read_jsonl(paths.asr_segments)
        quality_rows = build_quality_reports(asr_rows, cfg)
        feature_rows = build_audio_features(asr_rows, cfg)
        write_jsonl(paths.asr_quality_report, quality_rows)
        write_jsonl(paths.audio_features, feature_rows)

        # Phase 6: build and write validation summary per video
        video_ids = sorted({row.get("video_id", "") for row in asr_rows if row.get("video_id")})
        for vid in video_ids:
            vid_asr = [r for r in asr_rows if r.get("video_id") == vid]
            vid_quality = [r for r in quality_rows if r.get("video_id") == vid]
            vid_features = [r for r in feature_rows if r.get("video_id") == vid]
            summary_data = build_audio_quality_summary(vid_asr, vid_quality, vid_features, vid)
            write_quality_summary(summary_data, paths.audio_quality_summary)
    else:
        asr_rows = read_jsonl(paths.asr_segments)
        quality_rows = []
        feature_rows = []

    return {
        "output_root": str(paths.output_root),
        "video_manifest": str(paths.video_manifest),
        "audio_manifest": str(paths.audio_manifest),
        "asr_job_manifest": str(paths.asr_job_manifest),
        "asr_segments": str(paths.asr_segments),
        "asr_failed": str(paths.asr_failed),
        "asr_quality_report": str(paths.asr_quality_report),
        "audio_features": str(paths.audio_features),
        "num_videos": len(video_rows),
        "num_audio_success": sum(1 for row in audio_rows if row.get("has_audio") and row.get("extract_status") != "failed"),
        "num_asr_jobs": len(jobs),
        "num_asr_rows": len(asr_rows),
        "num_quality_rows": len(quality_rows),
        "num_audio_features": len(feature_rows),
    }


def create_video_manifest(videos: list[Path], video_id_override: str | None = None) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for video in videos:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_id = safe_stem(video_id_override or video_path.stem)
        if video_id in seen:
            raise ValueError(f"Duplicate video_id detected: {video_id}")
        seen.add(video_id)

        logger.info("Probing video: %s", video_path)
        try:
            info = probe_video(video_path)
        except Exception as exc:
            logger.exception("Probe failed for video_id=%s", video_id)
            rows.append(
                {
                    "schema_version": "video_manifest_v1",
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "duration_sec": None,
                    "fps": None,
                    "width": None,
                    "height": None,
                    "has_audio": False,
                    "audio_codec": None,
                    "audio_sample_rate": None,
                    "audio_channels": None,
                    "status": "probe_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now_iso(),
                }
            )
            continue
        rows.append(
            {
                "schema_version": "video_manifest_v1",
                "video_id": video_id,
                "video_path": str(video_path),
                "duration_sec": info.get("duration_sec"),
                "fps": info.get("fps"),
                "width": info.get("width"),
                "height": info.get("height"),
                "has_audio": info.get("has_audio"),
                "audio_codec": info.get("audio_codec"),
                "audio_sample_rate": info.get("audio_sample_rate"),
                "audio_channels": info.get("audio_channels"),
                "status": "probed" if info.get("has_audio") else "no_audio",
                "created_at": utc_now_iso(),
            }
        )
    return rows


def extract_audio_stage(
    video_rows: list[dict],
    paths: AudioOutputPaths,
    cfg: dict[str, Any],
    force: bool = False,
) -> list[dict]:
    audio_cfg = cfg.get("audio_extraction", {})
    rows: list[dict] = []
    for video_row in video_rows:
        video_id = video_row["video_id"]
        audio_path = paths.wav_dir / f"{video_id}.wav"
        if not video_row.get("has_audio"):
            rows.append(
                {
                    "schema_version": "audio_manifest_v1",
                    "video_id": video_id,
                    "video_path": video_row.get("video_path"),
                    "audio_path": str(audio_path),
                    "duration_sec": video_row.get("duration_sec"),
                    "sample_rate": audio_cfg.get("sample_rate"),
                    "channels": audio_cfg.get("channels"),
                    "format": audio_cfg.get("format", "wav"),
                    "has_audio": False,
                    "extract_status": "no_audio",
                    "created_at": utc_now_iso(),
                }
            )
            continue

        try:
            logger.info("Extracting audio: %s -> %s", video_row["video_path"], audio_path)
            extracted = extract_audio(
                video_row["video_path"],
                audio_path,
                sample_rate=int(audio_cfg.get("sample_rate") or 16000),
                channels=int(audio_cfg.get("channels") or 1),
                codec=str(audio_cfg.get("codec") or "pcm_s16le"),
                overwrite=bool(force or audio_cfg.get("overwrite", False)),
            )
            rows.append(
                {
                    "schema_version": "audio_manifest_v1",
                    "video_id": video_id,
                    "video_path": video_row.get("video_path"),
                    "audio_path": extracted["audio_path"],
                    "duration_sec": extracted.get("duration_sec"),
                    "sample_rate": extracted.get("sample_rate"),
                    "channels": extracted.get("channels"),
                    "sample_width": extracted.get("sample_width"),
                    "format": extracted.get("format", "wav"),
                    "has_audio": True,
                    "extract_status": extracted.get("extract_status"),
                    "created_at": utc_now_iso(),
                }
            )
        except Exception as exc:
            logger.exception("Audio extraction failed for video_id=%s", video_id)
            rows.append(
                {
                    "schema_version": "audio_manifest_v1",
                    "video_id": video_id,
                    "video_path": video_row.get("video_path"),
                    "audio_path": str(audio_path),
                    "duration_sec": round_sec(video_row.get("duration_sec")),
                    "sample_rate": audio_cfg.get("sample_rate"),
                    "channels": audio_cfg.get("channels"),
                    "format": audio_cfg.get("format", "wav"),
                    "has_audio": True,
                    "extract_status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now_iso(),
                }
            )
    return rows

