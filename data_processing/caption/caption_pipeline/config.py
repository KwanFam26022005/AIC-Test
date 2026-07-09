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


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""

    video_id: str = ""
    ocr_jsonl: str = ""
    object_jsonl: str = ""
    audio_features_jsonl: str = ""
    keyframe_map: str = ""
    output_dir: str = ""
    include_ocr_only_frames: bool = False
    timestamp_tolerance: float = 0.001
    enable_frame_captions: bool = False

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

    # ---- derived paths (populated by resolve_paths) ----
    evidence_dir: str = ""
    captions_dir: str = ""
    indexes_dir: str = ""
    reports_dir: str = ""

    def resolve_paths(self) -> None:
        """Set derived output directories based on *output_dir* and *video_id*."""
        base = Path(self.output_dir) / self.video_id
        self.evidence_dir = str(base / "evidence")
        self.captions_dir = str(base / "captions")
        self.indexes_dir = str(base / "indexes")
        self.reports_dir = str(base / "reports")
