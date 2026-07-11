"""Phase 4 TRAKE event-step report generator."""

from __future__ import annotations

from typing import Any

from .io_utils import utc_now_iso


def build_event_step_report(
    video_id: str,
    event_step_index: list[dict],
    event_records: list[dict],
) -> dict[str, Any]:
    """Build the Phase 4 event-step validation report."""
    num_steps = len(event_step_index)
    num_event_caption = sum(1 for row in event_step_index if row.get("event_caption"))
    num_trake_text = sum(1 for row in event_step_index if row.get("trake_text"))
    num_trake_empty = sum(1 for row in event_step_index if not row.get("trake_text"))
    num_fallback = sum(1 for row in event_records if row.get("fallback_used"))

    event_mode_counts: dict[str, int] = {}
    event_model_counts: dict[str, int] = {}
    prompt_version_counts: dict[str, int] = {}
    action_state_counts: dict[str, int] = {}
    temporal_role_counts: dict[str, int] = {}
    for row in event_step_index:
        quality = row.get("quality") or {}
        event_mode = quality.get("event_mode", "unknown")
        action_state = row.get("action_state", "unknown")
        temporal_role = row.get("temporal_role", "unknown")
        event_mode_counts[event_mode] = event_mode_counts.get(event_mode, 0) + 1
        event_model = quality.get("event_model", "") or "none"
        prompt_version = quality.get("prompt_version", "") or "none"
        event_model_counts[event_model] = event_model_counts.get(event_model, 0) + 1
        prompt_version_counts[prompt_version] = prompt_version_counts.get(prompt_version, 0) + 1
        action_state_counts[action_state] = action_state_counts.get(action_state, 0) + 1
        temporal_role_counts[temporal_role] = temporal_role_counts.get(temporal_role, 0) + 1

    duplicate_document_ids = _count_duplicates(event_step_index, "document_id")
    duplicate_event_ids = _count_duplicates(event_step_index, "event_id")
    duplicate_shot_ids = _count_duplicates(event_step_index, "shot_id")

    event_warnings: list[dict[str, Any]] = []
    summary_warnings: list[str] = []
    for row in event_records:
        warnings = row.get("warnings") or []
        if warnings:
            event_warnings.append({
                "event_id": row.get("event_id", ""),
                "shot_id": row.get("shot_id", ""),
                "warnings": warnings,
            })
            for warning in warnings:
                summary_warnings.append(
                    f"{row.get('event_id', '')}/{row.get('shot_id', '')}: {warning}"
                )

    return {
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "num_event_steps": num_steps,
        "num_event_caption_generated": num_event_caption,
        "num_trake_text_generated": num_trake_text,
        "num_trake_text_empty": num_trake_empty,
        "num_fallback_used": num_fallback,
        "event_mode_counts": event_mode_counts,
        "event_model_counts": event_model_counts,
        "prompt_version_counts": prompt_version_counts,
        "action_state_counts": action_state_counts,
        "temporal_role_counts": temporal_role_counts,
        "duplicate_document_ids": duplicate_document_ids,
        "duplicate_event_ids": duplicate_event_ids,
        "duplicate_shot_ids": duplicate_shot_ids,
        "num_event_warnings": len(event_warnings),
        "event_warnings": event_warnings[:50],
        "summary_warnings": summary_warnings[:50],
    }


def render_event_step_report_markdown(report: dict[str, Any]) -> str:
    """Render the Phase 4 event-step report as Markdown."""
    lines: list[str] = []
    lines.append(f"# Event-Step Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Event steps", "num_event_steps"),
        ("Event captions generated", "num_event_caption_generated"),
        ("TRAKE text generated", "num_trake_text_generated"),
        ("TRAKE text empty", "num_trake_text_empty"),
        ("Fallbacks used", "num_fallback_used"),
        ("Duplicate document IDs", "duplicate_document_ids"),
        ("Duplicate event IDs", "duplicate_event_ids"),
        ("Duplicate shot IDs", "duplicate_shot_ids"),
        ("Events with warnings", "num_event_warnings"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    lines.append("")

    _append_counts_table(lines, "Event Mode Counts", report.get("event_mode_counts") or {})
    _append_counts_table(lines, "Event Model Counts", report.get("event_model_counts") or {})
    _append_counts_table(lines, "Prompt Version Counts", report.get("prompt_version_counts") or {})
    _append_counts_table(lines, "Action State Counts", report.get("action_state_counts") or {})
    _append_counts_table(lines, "Temporal Role Counts", report.get("temporal_role_counts") or {})

    warnings = report.get("summary_warnings") or []
    if warnings:
        lines.append("## Warnings (first 50)")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    else:
        lines.append("## Status")
        lines.append("")
        lines.append("OK: No warnings. All event steps generated successfully.")
        lines.append("")

    lines.append("## Acceptance Checks")
    lines.append("")
    checks = [
        (
            report.get("num_event_caption_generated", 0) == report.get("num_event_steps", 0),
            f"All events captioned: {report.get('num_event_caption_generated')}/{report.get('num_event_steps')}",
        ),
        (
            report.get("num_trake_text_empty", 0) == 0,
            f"Empty TRAKE text: {report.get('num_trake_text_empty')}",
        ),
        (
            report.get("duplicate_document_ids", 0) == 0,
            f"Duplicate document_ids: {report.get('duplicate_document_ids')}",
        ),
        (
            report.get("duplicate_event_ids", 0) == 0,
            f"Duplicate event_ids: {report.get('duplicate_event_ids')}",
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


def _count_duplicates(rows: list[dict], key: str) -> int:
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        value = row.get(key, "")
        if value in seen:
            duplicates += 1
        seen.add(value)
    return duplicates
