from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "name": "audio_feature_extraction_for_video_retrieval",
        "version": "0.1.0",
        "schema_version": "audio_feature_v1",
    },
    "audio_extraction": {
        "sample_rate": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
        "format": "wav",
        "overwrite": False,
    },
    "vad": {
        "enabled": True,
        "method": "energy_vad",
        "frame_ms": 30,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 700,
        "speech_pad_ms": 300,
        "max_speech_duration_sec": 30.0,
        "energy_threshold": None,
        "energy_threshold_multiplier": 2.5,
        "energy_peak_ratio": 0.08,
        "min_rms": 120.0,
        "fallback_full_audio_when_no_speech": True,
    },
    "asr": {
        "enabled": True,
        "backend": "faster-whisper",
        "model": "large-v3",
        "model_path": None,
        "language": "vi",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 1,
        "word_timestamps": False,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "temperature": 0.0,
        "cpu_threads": 4,
        "num_workers": 1,
        "sort_jobs_by_duration": True,
        "debug_save_failed_chunks": True,
    },
    "text_cleaning": {
        "normalize_whitespace": True,
        "fix_simple_repetition": True,
        "add_basic_punctuation": True,
        "summary_mode": "passthrough",
        "max_keywords": 8,
    },
    "quality_gate": {
        "enabled": True,
        "min_text_length": 5,
        "max_repetition_ratio": 0.35,
        "min_duration_sec": 0.5,
        "include_empty_features": False,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {target}")
    return deep_update(cfg, loaded)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        cursor = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return cfg

