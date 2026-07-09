"""Validation report generator.

Implements section 10 of the code plan - produces both a JSON report and a
human-readable Markdown summary.
"""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import utc_now_iso

logger = logging.getLogger(__name__)


def build_report(
    video_id: str,
    frame_evidence: list[dict],
    shot_evidence: list[dict],
    alignment_stats: dict[str, Any],
    audio_segments: list[dict],
) -> dict[str, Any]:
    """Build the JSON validation report.

    Returns a dict suitable for writing as ``evidence_alignment_report.json``.
    """
    # Count evidence quality flags
    num_with_ocr = sum(1 for f in frame_evidence if f.get("quality", {}).get("has_ocr"))
    num_with_audio = sum(1 for f in frame_evidence if f.get("quality", {}).get("has_audio"))
    num_with_objects = sum(1 for f in frame_evidence if f.get("quality", {}).get("has_object"))

    report: dict[str, Any] = {
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "num_object_frames": alignment_stats.get("num_object_frames", 0),
        "num_ocr_frames": alignment_stats.get("num_ocr_frames", 0),
        "num_joined_frames": alignment_stats.get("num_joined_frames", 0),
        "num_frames_missing_ocr": alignment_stats.get("num_frames_missing_ocr", 0),
        "num_frames_missing_object": alignment_stats.get("num_frames_missing_object", 0),
        "num_timestamp_mismatch": alignment_stats.get("num_timestamp_mismatch", 0),
        "num_audio_features": alignment_stats.get("num_audio_features", len(audio_segments)),
        "num_audio_features_usable": alignment_stats.get(
            "num_audio_features_usable", len(audio_segments),
        ),
        "num_scene_labels_removed_from_counts": alignment_stats.get(
            "num_scene_labels_removed_from_counts", 0,
        ),
        "num_frames_with_ocr": num_with_ocr,
        "num_frames_with_audio": num_with_audio,
        "num_frames_with_objects": num_with_objects,
        "num_frame_evidence": len(frame_evidence),
        "num_shots": len(shot_evidence),
        "warnings": alignment_stats.get("warnings", []),
    }

    return report


def render_report_markdown(report: dict[str, Any]) -> str:
    """Render the validation report as a Markdown string."""
    lines: list[str] = []
    lines.append(f"# Evidence Alignment Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Object frames", "num_object_frames"),
        ("OCR frames", "num_ocr_frames"),
        ("Joined frames", "num_joined_frames"),
        ("Frames missing OCR", "num_frames_missing_ocr"),
        ("Frames missing object", "num_frames_missing_object"),
        ("Timestamp mismatches", "num_timestamp_mismatch"),
        ("Audio features (total)", "num_audio_features"),
        ("Audio features (usable)", "num_audio_features_usable"),
        ("Scene labels removed from object counts", "num_scene_labels_removed_from_counts"),
        ("Frames with OCR text", "num_frames_with_ocr"),
        ("Frames with audio text", "num_frames_with_audio"),
        ("Frames with objects", "num_frames_with_objects"),
        ("Frame evidence records", "num_frame_evidence"),
        ("Shots", "num_shots"),
    ]
    for label, key in metrics:
        val = report.get(key, 0)
        lines.append(f"| {label} | {val} |")

    lines.append("")

    # --- Warnings ---
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## Status")
        lines.append("")
        lines.append("OK: No warnings. All frames aligned successfully.")
        lines.append("")

    # --- Acceptance checks ---
    lines.append("## Acceptance Checks")
    lines.append("")
    checks = [
        (
            report.get("num_joined_frames", 0) == report.get("num_object_frames", 0),
            f"All object frames joined: {report.get('num_joined_frames')}/{report.get('num_object_frames')}",
        ),
        (
            report.get("num_timestamp_mismatch", 0) == 0,
            f"Timestamp mismatches: {report.get('num_timestamp_mismatch')}",
        ),
        (
            report.get("num_frames_missing_ocr", 0) == 0,
            f"Frames missing OCR: {report.get('num_frames_missing_ocr')}",
        ),
        (
            report.get("num_frame_evidence", 0) == report.get("num_joined_frames", 0),
            f"Frame evidence = joined: {report.get('num_frame_evidence')}/{report.get('num_joined_frames')}",
        ),
    ]
    for passed, desc in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        lines.append(f"- {icon} {desc}")

    lines.append("")
    return "\n".join(lines)
