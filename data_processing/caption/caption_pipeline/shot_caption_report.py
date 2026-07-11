"""Phase 3 shot caption report generator."""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import utc_now_iso

logger = logging.getLogger(__name__)


def build_shot_caption_report(
    video_id: str,
    shot_index: list[dict],
    shot_caption_records: list[dict],
) -> dict[str, Any]:
    """Build the Phase 3 shot caption validation report."""
    num_shots = len(shot_index)
    num_caption_generated = sum(1 for r in shot_index if r.get("caption_text"))
    num_caption_empty = sum(1 for r in shot_index if not r.get("caption_text"))
    num_temporal_generated = sum(1 for r in shot_index if r.get("temporal_caption"))
    num_temporal_empty = sum(1 for r in shot_index if not r.get("temporal_caption"))

    num_fallback = sum(1 for r in shot_caption_records if r.get("fallback_used"))
    num_used_ocr = sum(1 for r in shot_caption_records if r.get("used_ocr"))
    num_used_audio = sum(1 for r in shot_caption_records if r.get("used_audio"))
    num_used_objects = sum(1 for r in shot_caption_records if r.get("used_objects"))
    num_used_scene = sum(1 for r in shot_caption_records if r.get("used_scene"))
    num_used_frame_captions = sum(
        1 for r in shot_caption_records if r.get("used_frame_captions")
    )

    mode_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    for r in shot_caption_records:
        mode = r.get("caption_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        model = r.get("caption_model", "") or "none"
        prompt = r.get("prompt_version", "") or "none"
        model_counts[model] = model_counts.get(model, 0) + 1
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

    frame_warnings: list[dict[str, Any]] = []
    summary_warnings: list[str] = []
    for r in shot_caption_records:
        ws = r.get("warnings") or []
        if ws:
            frame_warnings.append({
                "shot_id": r.get("shot_id", ""),
                "warnings": ws,
            })
            for w in ws:
                summary_warnings.append(f"{r.get('shot_id', '')}: {w}")

    seen_doc_ids: set[str] = set()
    seen_shot_ids: set[str] = set()
    duplicate_doc_ids = 0
    duplicate_shot_ids = 0
    missing_rep_captions = 0
    for row in shot_index:
        doc_id = row.get("document_id", "")
        shot_id = row.get("shot_id", "")
        if doc_id in seen_doc_ids:
            duplicate_doc_ids += 1
        seen_doc_ids.add(doc_id)
        if shot_id in seen_shot_ids:
            duplicate_shot_ids += 1
        seen_shot_ids.add(shot_id)
        quality = row.get("quality") or {}
        missing_rep_captions += max(
            0,
            int(quality.get("num_representative_frames", 0))
            - int(quality.get("num_representative_frame_captions", 0)),
        )

    return {
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "num_shots": num_shots,
        "num_caption_generated": num_caption_generated,
        "num_caption_empty": num_caption_empty,
        "num_temporal_caption_generated": num_temporal_generated,
        "num_temporal_caption_empty": num_temporal_empty,
        "num_fallback_used": num_fallback,
        "coverage": {
            "ocr": num_used_ocr,
            "audio": num_used_audio,
            "objects": num_used_objects,
            "scene": num_used_scene,
            "frame_captions": num_used_frame_captions,
        },
        "caption_mode_counts": mode_counts,
        "caption_model_counts": model_counts,
        "prompt_version_counts": prompt_counts,
        "duplicate_document_ids": duplicate_doc_ids,
        "duplicate_shot_ids": duplicate_shot_ids,
        "missing_representative_frame_captions": missing_rep_captions,
        "num_shot_warnings": len(frame_warnings),
        "shot_warnings": frame_warnings[:50],
        "summary_warnings": summary_warnings[:50],
    }


def render_shot_caption_report_markdown(report: dict[str, Any]) -> str:
    """Render the Phase 3 shot caption report as Markdown."""
    lines: list[str] = []
    lines.append(f"# Shot Caption Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Total shots", "num_shots"),
        ("Captions generated", "num_caption_generated"),
        ("Captions empty", "num_caption_empty"),
        ("Temporal captions generated", "num_temporal_caption_generated"),
        ("Temporal captions empty", "num_temporal_caption_empty"),
        ("Fallbacks used", "num_fallback_used"),
        ("Duplicate document IDs", "duplicate_document_ids"),
        ("Duplicate shot IDs", "duplicate_shot_ids"),
        ("Missing representative frame captions", "missing_representative_frame_captions"),
        ("Shots with warnings", "num_shot_warnings"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    lines.append("")

    coverage = report.get("coverage") or {}
    num_shots = report.get("num_shots", 1) or 1
    lines.append("## Evidence Coverage")
    lines.append("")
    lines.append("| Evidence | Shots | % |")
    lines.append("|----------|------:|--:|")
    for ev_name in ["ocr", "audio", "objects", "scene", "frame_captions"]:
        count = coverage.get(ev_name, 0)
        pct = round(100 * count / num_shots, 1) if num_shots else 0
        lines.append(f"| {ev_name} | {count} | {pct}% |")
    lines.append("")

    mode_counts = report.get("caption_mode_counts") or {}
    if mode_counts:
        lines.append("## Caption Mode Distribution")
        lines.append("")
        lines.append("| Mode | Count |")
        lines.append("|------|------:|")
        for mode, count in sorted(mode_counts.items()):
            lines.append(f"| {mode} | {count} |")
        lines.append("")

    _append_counts_table(lines, "Caption Model Distribution", report.get("caption_model_counts") or {})
    _append_counts_table(lines, "Prompt Version Distribution", report.get("prompt_version_counts") or {})

    warnings = report.get("summary_warnings") or []
    if warnings:
        lines.append("## Warnings (first 50)")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## Status")
        lines.append("")
        lines.append("OK: No warnings. All shot captions generated successfully.")
        lines.append("")

    lines.append("## Acceptance Checks")
    lines.append("")
    checks = [
        (
            report.get("num_caption_generated", 0) == report.get("num_shots", 0),
            f"All shots captioned: {report.get('num_caption_generated')}/{report.get('num_shots')}",
        ),
        (
            report.get("num_caption_empty", 0) == 0,
            f"Empty captions: {report.get('num_caption_empty')}",
        ),
        (
            report.get("num_temporal_caption_empty", 0) == 0,
            f"Empty temporal captions: {report.get('num_temporal_caption_empty')}",
        ),
        (
            report.get("duplicate_document_ids", 0) == 0,
            f"Duplicate document_ids: {report.get('duplicate_document_ids')}",
        ),
        (
            report.get("duplicate_shot_ids", 0) == 0,
            f"Duplicate shot_ids: {report.get('duplicate_shot_ids')}",
        ),
    ]
    for passed, desc in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        lines.append(f"- {icon} {desc}")

    lines.append("")
    return "\n".join(lines)


def _append_counts_table(lines: list[str], title: str, counts: dict[str, int]) -> None:
    if not counts:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Value | Count |")
    lines.append("|-------|------:|")
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")
