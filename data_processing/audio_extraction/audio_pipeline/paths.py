from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioOutputPaths:
    output_root: Path
    wav_dir: Path
    manifests_dir: Path
    asr_dir: Path
    features_dir: Path
    temp_dir: Path
    failed_chunks_dir: Path
    video_manifest: Path
    audio_manifest: Path
    asr_job_manifest: Path
    asr_segments: Path
    asr_failed: Path
    asr_quality_report: Path
    audio_features: Path
    audio_quality_summary: Path

    @classmethod
    def from_root(cls, output_root: str | Path) -> "AudioOutputPaths":
        root = Path(output_root)
        wav_dir = root / "wav"
        manifests_dir = root / "manifests"
        asr_dir = root / "asr"
        features_dir = root / "features"
        temp_dir = root / "tmp"
        failed_chunks_dir = root / "debug" / "failed_chunks"
        return cls(
            output_root=root,
            wav_dir=wav_dir,
            manifests_dir=manifests_dir,
            asr_dir=asr_dir,
            features_dir=features_dir,
            temp_dir=temp_dir,
            failed_chunks_dir=failed_chunks_dir,
            video_manifest=manifests_dir / "video_manifest.jsonl",
            audio_manifest=manifests_dir / "audio_manifest.jsonl",
            asr_job_manifest=manifests_dir / "asr_job_manifest.jsonl",
            asr_segments=asr_dir / "asr_segments.jsonl",
            asr_failed=asr_dir / "asr_failed.jsonl",
            asr_quality_report=asr_dir / "asr_quality_report.jsonl",
            audio_features=features_dir / "audio_features.jsonl",
            audio_quality_summary=features_dir / "audio_quality_summary.json",
        )

    def mkdirs(self) -> None:
        for path in [
            self.wav_dir,
            self.manifests_dir,
            self.asr_dir,
            self.features_dir,
            self.temp_dir,
            self.failed_chunks_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

