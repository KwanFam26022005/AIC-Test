"""Run manifest and markdown report helpers for caption pipeline v2."""

from __future__ import annotations

from typing import Any

from .io_utils import utc_now_iso
from .schemas import PIPELINE_VERSION, RUN_MANIFEST_SCHEMA


def build_run_manifest(
    *,
    video_id: str,
    run_id: str,
    config_path: str,
    stats: dict[str, Any],
    issues: list[str],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "video_id": video_id,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "config_path": config_path,
        "models": model_info,
        "stats": stats,
        "issues": issues,
        "status": "pass" if not issues else "warning",
    }


def build_markdown_report(manifest: dict[str, Any]) -> str:
    stats = manifest.get("stats") or {}
    issues = manifest.get("issues") or []
    models = manifest.get("models") or {}
    lines = [
        f"# Caption Pipeline V2 Report - {manifest.get('video_id', '')}",
        "",
        f"- Run ID: `{manifest.get('run_id', '')}`",
        f"- Pipeline: `{manifest.get('pipeline_version', '')}`",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Created at: `{manifest.get('created_at', '')}`",
        "",
        "## Models",
        "",
        f"- Frame VLM: `{models.get('vlm_model_id', '')}`",
        f"- Shot ReCap LLM: `{models.get('shot_recap_model_id', '')}`",
        "",
        "## Counts",
        "",
        f"- Frames: {stats.get('num_frames', 0)}",
        f"- Shots: {stats.get('num_shots', 0)}",
        f"- Selected for VLM: {stats.get('num_selected_for_vlm', 0)}",
        f"- Frame captions: {stats.get('num_frame_captions', 0)}",
        f"- Shot audio contexts: {stats.get('num_shot_audio_contexts', 0)}",
        f"- Shot captions: {stats.get('num_shot_captions', 0)}",
        f"- Shots with audio: {stats.get('num_shots_with_audio', 0)}",
        f"- Frame fallback rate: {stats.get('frame_fallback_rate', 0)}",
        "",
        "## Acceptance",
        "",
    ]
    if issues:
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("All caption v2 acceptance checks passed.")
    lines.append("")
    return "\n".join(lines)
