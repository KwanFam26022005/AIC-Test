"""Pipeline configuration with sensible defaults from the code plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioAlignmentConfig:
    """How audio segments are matched to frames."""

    frame_window_sec_before: float = 5.0
    frame_window_sec_after: float = 5.0
    max_audio_chars_frame: int = 700
    max_audio_segments_frame: int = 5


@dataclass
class EvidenceBuilderConfig:
    """Limits and toggles for evidence construction."""

    max_objects: int = 10
    max_object_tags: int = 20
    max_scene_tags: int = 20
    max_ocr_chars: int = 600
    max_audio_chars_frame: int = 700
    max_audio_chars_shot: int = 1500
    keep_object_positions: bool = True
    include_ocr_review: bool = False
    include_audio_search_text_for_caption: bool = False


@dataclass
class ShotGroupingConfig:
    """Parameters for the timestamp-gap shot grouper."""

    method: str = "timestamp_gap"
    max_gap_sec: float = 5.0
    max_shot_duration_sec: float = 20.0
    max_representative_frames: int = 3
    representative_strategy: str = "first_middle_last"


@dataclass
class FrameCaptionConfig:
    """Phase 2 frame caption fuser configuration."""

    caption_mode: str = "template"  # "template" or "llm"
    rebuild_compact_with_captions: bool = False
    max_caption_chars: int = 320


@dataclass
class ShotCaptionConfig:
    """Phase 3 shot captioner configuration."""

    caption_mode: str = "template"  # "template" or "llm"
    max_ocr_chars: int = 220
    max_audio_chars: int = 360
    max_frame_caption_chars: int = 360
    max_caption_chars: int = 520
    max_temporal_caption_chars: int = 620
    max_memory_chars: int = 500


@dataclass
class TrakeEventConfig:
    """Phase 4 TRAKE event-step builder configuration."""

    event_mode: str = "template"  # "template" or "llm"
    max_before_context_chars: int = 300
    max_current_observation_chars: int = 420
    max_after_context_chars: int = 300
    max_trake_text_chars: int = 900


@dataclass
class LLMRuntimeConfig:
    """Shared text-model runtime for frame, shot, and TRAKE prompts."""

    provider: str = "transformers"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "auto"
    prompt_dir: str = ""
    prompt_version: str = "caption_qwen25_official_v1"
    max_input_tokens: int = 4096
    frame_max_new_tokens: int = 128
    shot_max_new_tokens: int = 512
    trake_max_new_tokens: int = 384
    do_sample: bool = False
    temperature: float = 0.0
    max_retries: int = 2
    resume: bool = True
    checkpoint_every: int = 10
    max_fallback_rate: float = 1.0


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    video_id: str = ""
    ocr_jsonl: str = ""
    object_jsonl: str = ""
    audio_features_jsonl: str = ""
    keyframe_map: str = ""
    initial_caption_overrides: str = ""
    output_dir: str = ""
    include_ocr_only_frames: bool = False
    timestamp_tolerance: float = 0.001
    enable_frame_captions: bool = False
    enable_shot_captions: bool = False
    enable_trake_events: bool = False

    audio_alignment: AudioAlignmentConfig = field(
        default_factory=AudioAlignmentConfig,
    )
    evidence_builder: EvidenceBuilderConfig = field(
        default_factory=EvidenceBuilderConfig,
    )
    shot_grouping: ShotGroupingConfig = field(
        default_factory=ShotGroupingConfig,
    )
    frame_caption: FrameCaptionConfig = field(
        default_factory=FrameCaptionConfig,
    )
    shot_caption: ShotCaptionConfig = field(
        default_factory=ShotCaptionConfig,
    )
    trake_event: TrakeEventConfig = field(
        default_factory=TrakeEventConfig,
    )
    llm: LLMRuntimeConfig = field(
        default_factory=LLMRuntimeConfig,
    )

    # ---- derived paths (populated by resolve_paths) ----
    evidence_dir: str = ""
    captions_dir: str = ""
    indexes_dir: str = ""
    reports_dir: str = ""
    checkpoints_dir: str = ""

    def resolve_paths(self) -> None:
        """Set derived output directories based on *output_dir* and *video_id*."""
        base = Path(self.output_dir) / self.video_id
        self.evidence_dir = str(base / "evidence")
        self.captions_dir = str(base / "captions")
        self.indexes_dir = str(base / "indexes")
        self.reports_dir = str(base / "reports")
        self.checkpoints_dir = str(base / "checkpoints")
