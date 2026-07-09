from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

from .io_utils import round_sec
from .media import probe_wav, read_wav_mono_pcm16

logger = logging.getLogger(__name__)


def compute_job_fingerprint(job_data: dict, asr_cfg: dict[str, Any]) -> str:
    """Create a stable SHA-1 fingerprint from job params + ASR config.

    If any of these values change, the fingerprint changes and the pipeline
    will re-run the job instead of using stale cached results.
    """
    parts = [
        str(job_data.get("video_id", "")),
        str(job_data.get("audio_path", "")),
        str(job_data.get("start_sec", "")),
        str(job_data.get("end_sec", "")),
        str(job_data.get("duration_sec", "")),
        str(job_data.get("segment_type", "")),
        str(asr_cfg.get("model_path") or asr_cfg.get("model") or ""),
        str(asr_cfg.get("language") or ""),
        str(asr_cfg.get("beam_size") or ""),
        str(asr_cfg.get("temperature") or ""),
        str(asr_cfg.get("condition_on_previous_text") or ""),
        str(asr_cfg.get("vad_filter") or ""),
    ]
    raw = "|".join(parts)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_asr_jobs(audio_rows: list[dict], cfg: dict[str, Any]) -> list[dict]:
    vad_cfg = cfg.get("vad", {})
    asr_cfg = cfg.get("asr", {})
    jobs: list[dict] = []
    for audio_row in audio_rows:
        if not audio_row.get("has_audio") or audio_row.get("extract_status") in {"failed", "no_audio"}:
            continue
        audio_path = Path(audio_row["audio_path"])
        video_id = audio_row["video_id"]
        segments = segment_audio(audio_path, vad_cfg)
        for idx, segment in enumerate(segments, start=1):
            start_sec = round_sec(segment["start_sec"])
            end_sec = round_sec(segment["end_sec"])
            job_data = {
                "job_id": f"asrjob_{video_id}_{idx:06d}",
                "video_id": video_id,
                "audio_path": str(audio_path),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round_sec(end_sec - start_sec),
                "segment_type": segment.get("segment_type", "speech"),
                "priority": 1,
                "status": "planned",
            }
            job_data["job_fingerprint"] = compute_job_fingerprint(job_data, asr_cfg)
            jobs.append(job_data)
    return jobs


def segment_audio(audio_path: str | Path, vad_cfg: dict[str, Any]) -> list[dict]:
    info = probe_wav(audio_path)
    duration = float(info.get("duration_sec") or 0.0)
    max_duration = float(vad_cfg.get("max_speech_duration_sec") or 30.0)

    if duration <= 0:
        return []
    if not vad_cfg.get("enabled", True) or vad_cfg.get("method") in {"none", "fixed_window"}:
        return _split_interval(0.0, duration, max_duration, "fixed_window")

    method = vad_cfg.get("method", "energy_vad")
    if method != "energy_vad":
        logger.warning("Unsupported VAD method %s; falling back to energy_vad", method)
    segments = energy_vad(audio_path, vad_cfg)
    if not segments and vad_cfg.get("fallback_full_audio_when_no_speech", True):
        logger.warning("No speech detected in %s; using fixed windows as fallback", audio_path)
        return _split_interval(0.0, duration, max_duration, "fallback_window")
    return segments


def energy_vad(audio_path: str | Path, vad_cfg: dict[str, Any]) -> list[dict]:
    sample_rate, samples = read_wav_mono_pcm16(audio_path)
    if not samples:
        return []

    frame_ms = int(vad_cfg.get("frame_ms") or 30)
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    rms_values = _frame_rms(samples, frame_size)
    if not rms_values:
        return []

    threshold = vad_cfg.get("energy_threshold")
    if threshold is None:
        threshold = _dynamic_threshold(
            rms_values,
            min_rms=float(vad_cfg.get("min_rms") or 120.0),
            multiplier=float(vad_cfg.get("energy_threshold_multiplier") or 2.5),
            peak_ratio=float(vad_cfg.get("energy_peak_ratio") or 0.08),
        )

    voiced = [value >= float(threshold) for value in rms_values]
    raw_segments = _voiced_frames_to_segments(
        voiced,
        frame_ms=frame_ms,
        min_speech_ms=int(vad_cfg.get("min_speech_duration_ms") or 250),
        min_silence_ms=int(vad_cfg.get("min_silence_duration_ms") or 700),
    )

    duration = len(samples) / float(sample_rate)
    padded = _pad_and_merge(
        raw_segments,
        duration=duration,
        pad_sec=float(vad_cfg.get("speech_pad_ms") or 300) / 1000.0,
        merge_gap_sec=float(vad_cfg.get("min_silence_duration_ms") or 700) / 1000.0,
    )
    max_duration = float(vad_cfg.get("max_speech_duration_sec") or 30.0)
    output: list[dict] = []
    for start, end in padded:
        output.extend(_split_interval(start, end, max_duration, "speech"))
    logger.info("VAD %s: %d segments, threshold=%.1f", audio_path, len(output), float(threshold))
    return output


def _frame_rms(samples, frame_size: int) -> list[float]:
    values: list[float] = []
    for offset in range(0, len(samples), frame_size):
        frame = samples[offset : offset + frame_size]
        if not frame:
            continue
        total = 0
        for sample in frame:
            total += int(sample) * int(sample)
        values.append(math.sqrt(total / len(frame)))
    return values


def _dynamic_threshold(
    rms_values: list[float],
    min_rms: float,
    multiplier: float,
    peak_ratio: float,
) -> float:
    sorted_values = sorted(rms_values)
    noise = _percentile(sorted_values, 35)
    peak = _percentile(sorted_values, 95)
    return max(min_rms, noise * multiplier, peak * peak_ratio)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * percentile / 100.0))
    return sorted_values[max(0, min(len(sorted_values) - 1, idx))]


def _voiced_frames_to_segments(
    voiced: list[bool],
    frame_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[tuple[float, float]]:
    min_speech_frames = max(1, math.ceil(min_speech_ms / frame_ms))
    min_silence_frames = max(1, math.ceil(min_silence_ms / frame_ms))
    segments: list[tuple[int, int]] = []
    start: int | None = None
    silence_start: int | None = None

    for idx, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = idx
            silence_start = None
            continue

        if start is None:
            continue
        if silence_start is None:
            silence_start = idx
        if idx - silence_start + 1 >= min_silence_frames:
            end = silence_start
            if end - start >= min_speech_frames:
                segments.append((start, end))
            start = None
            silence_start = None

    if start is not None:
        end = len(voiced)
        if end - start >= min_speech_frames:
            segments.append((start, end))

    frame_sec = frame_ms / 1000.0
    return [(start * frame_sec, end * frame_sec) for start, end in segments]


def _pad_and_merge(
    segments: list[tuple[float, float]],
    duration: float,
    pad_sec: float,
    merge_gap_sec: float,
) -> list[tuple[float, float]]:
    if not segments:
        return []
    padded = [(max(0.0, s - pad_sec), min(duration, e + pad_sec)) for s, e in segments]
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = padded[0]
    for start, end in padded[1:]:
        if start - cur_end <= merge_gap_sec:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _split_interval(start: float, end: float, max_duration: float, segment_type: str) -> list[dict]:
    if end <= start:
        return []
    if max_duration <= 0 or end - start <= max_duration:
        return [{"start_sec": round_sec(start), "end_sec": round_sec(end), "segment_type": segment_type}]
    output: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + max_duration)
        output.append(
            {
                "start_sec": round_sec(cursor),
                "end_sec": round_sec(chunk_end),
                "segment_type": segment_type,
            }
        )
        cursor = chunk_end
    return output

