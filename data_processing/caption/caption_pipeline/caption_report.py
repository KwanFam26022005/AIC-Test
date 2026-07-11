"""Phase 2 caption report generator.

Implements section 4.3 of the Phase 2 plan - produces both a JSON report
and a human-readable Markdown summary for frame caption quality.
"""

from __future__ import annotations

import logging
from typing import Any

from .io_utils import utc_now_iso

logger = logging.getLogger(__name__)


def build_caption_report(
    video_id: str,
    frame_index: list[dict],
    caption_records: list[dict],
) -> dict[str, Any]:
    """Build the Phase 2 validation report.

    Returns a dict suitable for writing as ``frame_caption_report.json``.
    """
    num_frames = len(frame_index)
    num_caption_generated = sum(1 for r in frame_index if r.get("caption_text"))
    num_caption_empty = sum(1 for r in frame_index if not r.get("caption_text"))

    num_fallback = sum(1 for r in caption_records if r.get("fallback_used"))
    num_used_ocr = sum(1 for r in caption_records if r.get("used_ocr"))
    num_used_audio = sum(1 for r in caption_records if r.get("used_audio"))
    num_used_objects = sum(1 for r in caption_records if r.get("used_objects"))
    num_used_scene = sum(1 for r in caption_records if r.get("used_scene"))

    # Caption mode distribution
    mode_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    for r in caption_records:
        mode = r.get("caption_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        model = r.get("caption_model", "") or "none"
        prompt = r.get("prompt_version", "") or "none"
        model_counts[model] = model_counts.get(model, 0) + 1
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

    # Collect all warnings
    all_warnings: list[str] = []
    frame_warnings: list[dict[str, Any]] = []
    for r in caption_records:
        ws = r.get("warnings") or []
        if ws:
            frame_warnings.append({
                "canonical_frame_id": r["canonical_frame_id"],
                "warnings": ws,
            })
            for w in ws:
                all_warnings.append(f"{r['canonical_frame_id']}: {w}")

    # Check for duplicate IDs
    seen_doc_ids: set[str] = set()
    seen_cfids: set[str] = set()
    duplicate_doc_ids = 0
    duplicate_cfids = 0
    for fi in frame_index:
        doc_id = fi.get("document_id", "")
        cfid = fi.get("canonical_frame_id", "")
        if doc_id in seen_doc_ids:
            duplicate_doc_ids += 1
        seen_doc_ids.add(doc_id)
        if cfid in seen_cfids:
            duplicate_cfids += 1
        seen_cfids.add(cfid)

    report: dict[str, Any] = {
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "num_frames": num_frames,
        "num_caption_generated": num_caption_generated,
        "num_caption_empty": num_caption_empty,
        "num_fallback_used": num_fallback,
        "coverage": {
            "ocr": num_used_ocr,
            "audio": num_used_audio,
            "objects": num_used_objects,
            "scene": num_used_scene,
        },
        "caption_mode_counts": mode_counts,
        "caption_model_counts": model_counts,
        "prompt_version_counts": prompt_counts,
        "duplicate_document_ids": duplicate_doc_ids,
        "duplicate_canonical_frame_ids": duplicate_cfids,
        "num_frame_warnings": len(frame_warnings),
        "frame_warnings": frame_warnings[:50],  # cap for readability
        "summary_warnings": all_warnings[:50],
    }

    return report


def render_caption_report_markdown(report: dict[str, Any]) -> str:
    """Render the Phase 2 caption report as Markdown."""
    lines: list[str] = []
    lines.append(f"# Frame Caption Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Total frames", "num_frames"),
        ("Captions generated", "num_caption_generated"),
        ("Captions empty", "num_caption_empty"),
        ("Fallbacks used", "num_fallback_used"),
        ("Duplicate document IDs", "duplicate_document_ids"),
        ("Duplicate canonical frame IDs", "duplicate_canonical_frame_ids"),
        ("Frames with warnings", "num_frame_warnings"),
    ]
    for label, key in metrics:
        val = report.get(key, 0)
        lines.append(f"| {label} | {val} |")
    lines.append("")

    # --- Evidence coverage ---
    coverage = report.get("coverage") or {}
    num_frames = report.get("num_frames", 1) or 1
    lines.append("## Evidence Coverage")
    lines.append("")
    lines.append("| Evidence | Frames | % |")
    lines.append("|----------|-------:|--:|")
    for ev_name in ["ocr", "audio", "objects", "scene"]:
        count = coverage.get(ev_name, 0)
        pct = round(100 * count / num_frames, 1) if num_frames else 0
        lines.append(f"| {ev_name.capitalize()} | {count} | {pct}% |")
    lines.append("")

    # --- Caption mode distribution ---
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

    # --- Warnings ---
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
        lines.append("OK: No warnings. All frame captions generated successfully.")
        lines.append("")

    # --- Acceptance checks ---
    lines.append("## Acceptance Checks")
    lines.append("")
    checks = [
        (
            report.get("num_caption_generated", 0) == report.get("num_frames", 0),
            f"All frames captioned: {report.get('num_caption_generated')}/{report.get('num_frames')}",
        ),
        (
            report.get("num_caption_empty", 0) == 0,
            f"Empty captions: {report.get('num_caption_empty')}",
        ),
        (
            report.get("duplicate_document_ids", 0) == 0,
            f"Duplicate document_ids: {report.get('duplicate_document_ids')}",
        ),
        (
            report.get("duplicate_canonical_frame_ids", 0) == 0,
            f"Duplicate canonical_frame_ids: {report.get('duplicate_canonical_frame_ids')}",
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
