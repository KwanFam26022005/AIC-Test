from __future__ import annotations

import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .io_utils import append_jsonl, append_jsonl_many, read_jsonl, round_sec, utc_now_iso
from .media import write_wav_chunk
from .paths import AudioOutputPaths

logger = logging.getLogger(__name__)


def run_asr_jobs(
    jobs: list[dict],
    paths: AudioOutputPaths,
    cfg: dict[str, Any],
    force: bool = False,
) -> list[dict]:
    asr_cfg = cfg.get("asr", {})
    if not asr_cfg.get("enabled", True):
        logger.info("ASR disabled; skipping transcription")
        return read_jsonl(paths.asr_segments)

    if asr_cfg.get("backend") != "faster-whisper":
        raise ValueError("Only backend='faster-whisper' is implemented in this pipeline.")

    existing_rows = [] if force else read_jsonl(paths.asr_segments)
    done_job_ids = {
        row.get("job_id")
        for row in existing_rows
        if row.get("status") in {"success", "empty"} and row.get("job_id")
    }
    pending_jobs = [job for job in jobs if force or job["job_id"] not in done_job_ids]
    if asr_cfg.get("sort_jobs_by_duration", True):
        pending_jobs.sort(key=lambda job: float(job.get("duration_sec") or 0.0))

    logger.info("ASR jobs: total=%d done=%d pending=%d", len(jobs), len(done_job_ids), len(pending_jobs))
    if not pending_jobs:
        return existing_rows

    model = _load_faster_whisper_model(asr_cfg)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="asr_chunks_", dir=str(paths.temp_dir)) as temp_dir:
        temp_root = Path(temp_dir)
        for idx, job in enumerate(pending_jobs, start=1):
            logger.info(
                "ASR [%d/%d] %s %.3f-%.3fs",
                idx,
                len(pending_jobs),
                job["video_id"],
                float(job["start_sec"]),
                float(job["end_sec"]),
            )
            try:
                rows = _transcribe_job(model, job, asr_cfg, temp_root)
                append_jsonl_many(paths.asr_segments, rows)
            except Exception as exc:  # Keep batch resume-friendly.
                logger.exception("ASR failed for job=%s", job.get("job_id"))
                failed = {
                    "job_id": job.get("job_id"),
                    "video_id": job.get("video_id"),
                    "audio_path": job.get("audio_path"),
                    "start_sec": job.get("start_sec"),
                    "end_sec": job.get("end_sec"),
                    "duration_sec": job.get("duration_sec"),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "created_at": utc_now_iso(),
                }
                if asr_cfg.get("debug_save_failed_chunks", True):
                    failed["debug_chunk_path"] = _save_failed_chunk(job, paths)
                append_jsonl(paths.asr_failed, failed)

    return read_jsonl(paths.asr_segments)


def _load_faster_whisper_model(asr_cfg: dict[str, Any]):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install with "
            "`pip install -r data_processing/audio_extraction/requirements.txt`."
        ) from exc

    model_name = asr_cfg.get("model_path") or asr_cfg.get("model") or "large-v3"
    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model_name,
        asr_cfg.get("device"),
        asr_cfg.get("compute_type"),
    )
    return WhisperModel(
        model_name,
        device=asr_cfg.get("device", "cuda"),
        compute_type=asr_cfg.get("compute_type", "float16"),
        cpu_threads=int(asr_cfg.get("cpu_threads") or 4),
        num_workers=int(asr_cfg.get("num_workers") or 1),
    )


def _transcribe_job(model, job: dict, asr_cfg: dict[str, Any], temp_root: Path) -> list[dict]:
    chunk_path = temp_root / f"{job['job_id']}.wav"
    write_wav_chunk(job["audio_path"], float(job["start_sec"]), float(job["end_sec"]), chunk_path)

    start_time = time.perf_counter()
    kwargs = {
        "language": asr_cfg.get("language") or "vi",
        "beam_size": int(asr_cfg.get("beam_size") or 1),
        "word_timestamps": bool(asr_cfg.get("word_timestamps", False)),
        "condition_on_previous_text": bool(asr_cfg.get("condition_on_previous_text", False)),
        "vad_filter": bool(asr_cfg.get("vad_filter", False)),
    }
    if asr_cfg.get("temperature") is not None:
        kwargs["temperature"] = float(asr_cfg.get("temperature"))

    segments_iter, info = model.transcribe(str(chunk_path), **kwargs)
    segments = list(segments_iter)
    elapsed = time.perf_counter() - start_time
    duration = max(0.001, float(job.get("duration_sec") or 0.0))
    runtime = {
        "inference_time_sec": round_sec(elapsed),
        "rtf": round_sec(elapsed / duration, 4),
    }
    language = getattr(info, "language", None) or asr_cfg.get("language") or "vi"
    language_probability = getattr(info, "language_probability", None)
    if language_probability is not None:
        language_probability = float(language_probability)

    rows: list[dict] = []
    for sub_idx, segment in enumerate(segments, start=1):
        raw_text = (getattr(segment, "text", "") or "").strip()
        local_start = float(getattr(segment, "start", 0.0) or 0.0)
        local_end = float(getattr(segment, "end", duration) or duration)
        abs_start = max(float(job["start_sec"]), float(job["start_sec"]) + local_start)
        abs_end = min(float(job["end_sec"]), float(job["start_sec"]) + local_end)
        if abs_end <= abs_start:
            abs_end = min(float(job["end_sec"]), abs_start + 0.001)
        rows.append(
            _asr_row(
                job=job,
                sub_idx=sub_idx,
                start_sec=abs_start,
                end_sec=abs_end,
                raw_text=raw_text,
                language=language,
                language_probability=language_probability,
                asr_cfg=asr_cfg,
                runtime=runtime,
                status="success" if raw_text else "empty",
            )
        )

    if not rows:
        rows.append(
            _asr_row(
                job=job,
                sub_idx=0,
                start_sec=float(job["start_sec"]),
                end_sec=float(job["end_sec"]),
                raw_text="",
                language=language,
                language_probability=language_probability,
                asr_cfg=asr_cfg,
                runtime=runtime,
                status="empty",
            )
        )
    return rows


def _asr_row(
    job: dict,
    sub_idx: int,
    start_sec: float,
    end_sec: float,
    raw_text: str,
    language: str,
    language_probability: float | None,
    asr_cfg: dict[str, Any],
    runtime: dict[str, Any],
    status: str,
) -> dict:
    segment_id = f"asrseg_{job['video_id']}_{_job_index(job['job_id']):06d}_{sub_idx:03d}"
    return {
        "schema_version": "asr_segment_v1",
        "asr_segment_id": segment_id,
        "job_id": job["job_id"],
        "video_id": job["video_id"],
        "start_sec": round_sec(start_sec),
        "end_sec": round_sec(end_sec),
        "duration_sec": round_sec(end_sec - start_sec),
        "raw_text": raw_text,
        "language": language,
        "language_probability": language_probability,
        "asr_info": {
            "model": asr_cfg.get("model_path") or asr_cfg.get("model"),
            "backend": "faster-whisper",
            "device": asr_cfg.get("device"),
            "compute_type": asr_cfg.get("compute_type"),
            "beam_size": asr_cfg.get("beam_size"),
        },
        "runtime": runtime,
        "status": status,
        "created_at": utc_now_iso(),
    }


def _job_index(job_id: str) -> int:
    try:
        return int(str(job_id).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _save_failed_chunk(job: dict, paths: AudioOutputPaths) -> str | None:
    try:
        paths.failed_chunks_dir.mkdir(parents=True, exist_ok=True)
        target = paths.failed_chunks_dir / f"{job['job_id']}.wav"
        write_wav_chunk(job["audio_path"], float(job["start_sec"]), float(job["end_sec"]), target)
        return str(target)
    except Exception:
        logger.exception("Could not save failed debug chunk for job=%s", job.get("job_id"))
        return None
