from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
import wave
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent, round_sec

logger = logging.getLogger(__name__)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found on PATH: {name}")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    logger.debug("Running command: %s", " ".join(args))
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(result.stdout or "{}")


def probe_video(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = ffprobe_json(target)
    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _first_float(
        data.get("format", {}).get("duration"),
        video_stream.get("duration") if video_stream else None,
        audio_stream.get("duration") if audio_stream else None,
    )
    fps = _parse_fps(video_stream.get("avg_frame_rate") if video_stream else None)
    return {
        "video_path": str(target),
        "duration_sec": round_sec(duration),
        "fps": fps,
        "width": _maybe_int(video_stream.get("width") if video_stream else None),
        "height": _maybe_int(video_stream.get("height") if video_stream else None),
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_sample_rate": _maybe_int(audio_stream.get("sample_rate") if audio_stream else None),
        "audio_channels": _maybe_int(audio_stream.get("channels") if audio_stream else None),
    }


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    sample_rate: int,
    channels: int,
    codec: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    require_binary("ffmpeg")
    target = ensure_parent(audio_path)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        info = probe_wav(target)
        return {
            "audio_path": str(target),
            "extract_status": "skipped_existing",
            **info,
        }
    should_overwrite = overwrite or (target.exists() and target.stat().st_size == 0)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if should_overwrite else "-n",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        codec,
        str(target),
    ]
    run_command(command)
    info = probe_wav(target)
    return {
        "audio_path": str(target),
        "extract_status": "success",
        **info,
    }


def probe_wav(path: str | Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        sample_width = wav.getsampwidth()
    duration = frames / float(sample_rate) if sample_rate else 0.0
    return {
        "duration_sec": round_sec(duration),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "format": "wav",
    }


def read_wav_mono_pcm16(path: str | Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample_width={sample_width}: {path}")
        raw = wav.readframes(wav.getnframes())

    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if channels == 1:
        return sample_rate, samples

    mono = array("h")
    for idx in range(0, len(samples), channels):
        frame = samples[idx : idx + channels]
        if frame:
            mono.append(int(sum(frame) / len(frame)))
    return sample_rate, mono


def write_wav_chunk(
    source_path: str | Path,
    start_sec: float,
    end_sec: float,
    out_path: str | Path,
) -> Path:
    target = ensure_parent(out_path)
    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        sample_rate = src.getframerate()
        start_frame = max(0, int(math.floor(float(start_sec) * sample_rate)))
        end_frame = min(src.getnframes(), int(math.ceil(float(end_sec) * sample_rate)))
        frame_count = max(0, end_frame - start_frame)
        src.setpos(start_frame)
        raw = src.readframes(frame_count)

    with wave.open(str(target), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(raw)
    return target


def _first_float(*values: object) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return round(float(Fraction(value)), 3)
    except (ValueError, ZeroDivisionError):
        return None


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
