"""Pipeline v2 configuration — dataclass-based with sensible defaults.

All thresholds and model settings are configurable; defaults match the
design document for the L22_V012 reference run (Experiment A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FrameSelectionConfig:
    """Phase 1 — shot-aware frame selection."""

    similarity_method: str = "phash"
    novelty_threshold: float = 0.30
    min_selected_per_shot: int = 1
    max_selected_per_shot: int = 3
    force_first_last_for_long_shots: bool = True
    long_shot_min_frames: int = 6


@dataclass
class VLMConfig:
    """Phase 2 — Qwen2.5-VL visual-only captioning."""

    provider: str = "transformers"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_revision: str = ""
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    device_map: str = "auto"
    batch_size: int = 2
    max_visual_tokens: int = 1280
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    prompt_version: str = "visual_frame_caption_v2"
    prompt_file: str = ""
    max_caption_chars: int = 320
    min_caption_words: int = 5
    max_caption_words: int = 60
    max_retries: int = 1
    unload_after_stage: bool = True


@dataclass
class AudioContextConfig:
    """Phase 4 — ASR alignment to shots."""

    padding_before_sec: float = 1.5
    padding_after_sec: float = 1.5
    max_audio_context_chars: int = 1200


@dataclass
class ShotRecapConfig:
    """Phase 5 — text ReCap model."""

    provider: str = "transformers"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str = ""
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    device_map: str = "auto"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    prompt_version: str = "shot_recap_audio_context_v1"
    prompt_file: str = ""
    max_input_tokens: int = 4096
    max_memory_chars: int = 600
    max_retries: int = 1
    unload_after_stage: bool = True


@dataclass
class EscalationConfig:
    """Phase 6 — optional LVLM escalation (disabled by default)."""

    enabled: bool = False
    target_escalation_rate: float = 0.10
    min_escalation_rate: float = 0.05
    max_escalation_rate: float = 0.15
    max_images_per_shot: int = 3


@dataclass
class CheckpointConfig:
    """Phase 7 — checkpoint and resume."""

    resume: bool = True
    write_every: int = 10
    allow_fallback_reuse: bool = False


@dataclass
class PipelineV2Config:
    """Top-level configuration aggregating all sub-configs."""

    # Identity
    video_id: str = ""
    run_id: str = ""

    # Input paths
    keyframes_root: str = ""
    keyframe_map: str = ""
    audio_features_jsonl: str = ""

    # Output root
    output_dir: str = ""
    prompt_dir: str = ""

    # Shot grouping (reused from v1 logic)
    shot_max_gap_sec: float = 5.0
    shot_max_duration_sec: float = 20.0

    # Sub-configs
    frame_selection: FrameSelectionConfig = field(
        default_factory=FrameSelectionConfig,
    )
    vlm: VLMConfig = field(default_factory=VLMConfig)
    audio_context: AudioContextConfig = field(
        default_factory=AudioContextConfig,
    )
    shot_recap: ShotRecapConfig = field(default_factory=ShotRecapConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Strict mode
    strict: bool = False

    # ---- derived paths (populated by resolve_paths) ----
    manifest_dir: str = ""
    selection_dir: str = ""
    evidence_dir: str = ""
    captions_dir: str = ""
    state_dir: str = ""
    runtime_dir: str = ""
    checkpoints_dir: str = ""
    reports_dir: str = ""

    def resolve_paths(self) -> None:
        """Set derived output directories based on *output_dir* and *video_id*."""
        base = Path(self.output_dir) / self.video_id
        self.manifest_dir = str(base / "manifest")
        self.selection_dir = str(base / "selection")
        self.evidence_dir = str(base / "evidence")
        self.captions_dir = str(base / "captions")
        self.state_dir = str(base / "state")
        self.runtime_dir = str(base / "runtime")
        self.checkpoints_dir = str(base / "checkpoints")
        self.reports_dir = str(base / "reports")

    def ensure_dirs(self) -> None:
        """Create all output directories."""
        for attr in (
            "manifest_dir", "selection_dir", "evidence_dir",
            "captions_dir", "state_dir", "runtime_dir",
            "checkpoints_dir", "reports_dir",
        ):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


def load_pipeline_config(path: str | Path) -> PipelineV2Config:
    """Load a v2 YAML config into ``PipelineV2Config``."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load caption v2 configs") from exc

    from caption_pipeline.runtime import expand_env_values

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Caption v2 config must be a YAML mapping")
    raw = expand_env_values(raw)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> PipelineV2Config:
    """Build config from a YAML-compatible mapping."""
    pipeline = raw.get("pipeline") or {}
    inputs = raw.get("inputs") or {}
    paths = raw.get("paths") or {}
    models = raw.get("models") or {}
    generation = raw.get("generation") or {}

    cfg = PipelineV2Config(
        video_id=str(pipeline.get("video_id") or ""),
        run_id=str(pipeline.get("run_id") or ""),
        keyframes_root=str(inputs.get("keyframes_root") or ""),
        keyframe_map=str(inputs.get("keyframe_map") or inputs.get("keyframes_root") or ""),
        audio_features_jsonl=str(inputs.get("audio_features") or ""),
        output_dir=str(paths.get("output_root") or paths.get("output_dir") or ""),
        prompt_dir=str(paths.get("prompt_dir") or ""),
        shot_max_gap_sec=float(generation.get("shot_max_gap_sec", raw.get("shot_max_gap_sec", 5.0))),
        shot_max_duration_sec=float(
            generation.get("shot_max_duration_sec", raw.get("shot_max_duration_sec", 20.0))
        ),
        strict=bool(raw.get("strict", False)),
    )

    _update_dataclass(cfg.frame_selection, raw.get("frame_selection") or {})
    _update_dataclass(cfg.audio_context, raw.get("audio_context") or {})
    _update_dataclass(cfg.escalation, raw.get("escalation") or {})
    _update_dataclass(cfg.checkpoint, raw.get("checkpoint") or {})

    vlm = models.get("vlm") or raw.get("vlm") or {}
    _update_model_config(cfg.vlm, vlm)
    _update_dataclass(cfg.vlm, generation.get("vlm") or {})

    text_llm = models.get("text_llm") or models.get("shot_recap") or raw.get("shot_recap") or {}
    _update_model_config(cfg.shot_recap, text_llm)
    _update_dataclass(cfg.shot_recap, generation.get("shot_recap") or {})

    if cfg.prompt_dir:
        if not cfg.vlm.prompt_file:
            cfg.vlm.prompt_file = str(Path(cfg.prompt_dir) / "visual_frame_caption_prompt.txt")
        if not cfg.shot_recap.prompt_file:
            cfg.shot_recap.prompt_file = str(Path(cfg.prompt_dir) / "shot_recap_prompt.txt")

    _validate_config(cfg)
    cfg.resolve_paths()
    return cfg


def _update_model_config(target: Any, values: dict[str, Any]) -> None:
    mapped = dict(values)
    if "model_name" in mapped and "model_id" not in mapped:
        mapped["model_id"] = mapped.pop("model_name")
    _update_dataclass(target, mapped)


def _update_dataclass(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _validate_config(cfg: PipelineV2Config) -> None:
    missing = []
    if not cfg.video_id:
        missing.append("pipeline.video_id")
    if not cfg.keyframes_root:
        missing.append("inputs.keyframes_root")
    if not cfg.audio_features_jsonl:
        missing.append("inputs.audio_features")
    if not cfg.output_dir:
        missing.append("paths.output_root")
    if missing:
        raise ValueError("Caption v2 config is missing: " + ", ".join(missing))
