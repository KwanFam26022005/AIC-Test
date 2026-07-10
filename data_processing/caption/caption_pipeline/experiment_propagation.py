"""Propagate frame-caption experiment outputs through the caption pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .caption_index import build_frame_index
from .caption_report import build_caption_report, render_caption_report_markdown
from .config import (
    FrameCaptionConfig,
    PipelineConfig,
    ShotCaptionConfig,
    TrakeEventConfig,
)
from .event_step_index import build_event_step_index, rebuild_compact_with_trake_text
from .event_step_report import (
    build_event_step_report,
    render_event_step_report_markdown,
)
from .io_utils import read_jsonl, utc_now_iso, write_json, write_jsonl, write_text
from .shot_caption_report import (
    build_shot_caption_report,
    render_shot_caption_report_markdown,
)
from .shot_captioner import fuse_shot_captions
from .shot_index import build_shot_index, rebuild_compact_with_frame_and_shot_captions
from .text_utils import normalize_whitespace
from .trake_event_builder import build_trake_event_steps


PROPAGATION_REPORT_SCHEMA = "caption_experiment_propagation_report_v1"
VISUAL_QUERY_SCHEMA = "caption_visual_query_set_v1"


def propagate_caption_experiment(
    video_id: str,
    baseline_dir: str | Path,
    output_root: str | Path,
    experiment_name: str,
    frame_caption_overrides: list[dict],
    shot_caption_mode: str = "template",
    trake_event_mode: str = "template",
    copy_reports: bool = False,
    write_visual_queries: bool = True,
) -> dict[str, Any]:
    """Rebuild shot/event/index outputs after applying frame caption overrides."""
    baseline_video_dir = _resolve_video_dir(Path(baseline_dir), video_id)
    output_video_dir = Path(output_root) / experiment_name / video_id

    frame_evidence = read_jsonl(baseline_video_dir / "evidence" / "frame_evidence.jsonl")
    shot_evidence = read_jsonl(baseline_video_dir / "evidence" / "shot_evidence.jsonl")
    compact_docs = read_jsonl(baseline_video_dir / "indexes" / "compact_search_index.jsonl")

    baseline_frame_index = read_jsonl(
        baseline_video_dir / "captions" / "frame_index.jsonl"
    )
    override_index, override_qa = normalize_frame_caption_overrides(
        frame_caption_overrides,
    )

    baseline_caption_records = _caption_records_from_frame_index(baseline_frame_index)
    caption_records, frame_update_stats = _apply_frame_override_records(
        baseline_caption_records,
        override_index,
    )
    frame_index = build_frame_index(frame_evidence, caption_records)

    cfg = _build_propagation_config(
        video_id=video_id,
        output_dir=Path(output_root) / experiment_name,
        shot_caption_mode=shot_caption_mode,
        trake_event_mode=trake_event_mode,
    )

    shot_caption_records = fuse_shot_captions(shot_evidence, frame_index, cfg)
    shot_index = build_shot_index(shot_evidence, frame_index, shot_caption_records)
    event_records = build_trake_event_steps(shot_index, shot_evidence, cfg)
    event_step_index = build_event_step_index(
        shot_index,
        shot_evidence,
        event_records,
    )

    compact_docs = rebuild_compact_with_frame_and_shot_captions(
        compact_docs,
        caption_records,
        shot_caption_records,
    )
    compact_docs = rebuild_compact_with_trake_text(compact_docs, event_records)

    _copy_required_context(baseline_video_dir, output_video_dir, copy_reports)
    write_jsonl(output_video_dir / "captions" / "frame_index.jsonl", frame_index)
    write_jsonl(output_video_dir / "captions" / "shot_index.jsonl", shot_index)
    write_jsonl(
        output_video_dir / "captions" / "event_step_index.jsonl",
        event_step_index,
    )
    write_jsonl(
        output_video_dir / "indexes" / "compact_search_index.jsonl",
        compact_docs,
    )

    frame_report = build_caption_report(video_id, frame_index, caption_records)
    shot_report = build_shot_caption_report(
        video_id,
        shot_index,
        shot_caption_records,
    )
    event_report = build_event_step_report(video_id, event_step_index, event_records)
    write_json(output_video_dir / "reports" / "frame_caption_report.json", frame_report)
    write_text(
        output_video_dir / "reports" / "frame_caption_report.md",
        render_caption_report_markdown(frame_report),
    )
    write_json(output_video_dir / "reports" / "shot_caption_report.json", shot_report)
    write_text(
        output_video_dir / "reports" / "shot_caption_report.md",
        render_shot_caption_report_markdown(shot_report),
    )
    write_json(output_video_dir / "reports" / "event_step_report.json", event_report)
    write_text(
        output_video_dir / "reports" / "event_step_report.md",
        render_event_step_report_markdown(event_report),
    )

    if write_visual_queries:
        write_jsonl(
            output_video_dir / "eval" / "visual_retrieval_queries.jsonl",
            build_visual_queries(video_id),
        )

    report = build_propagation_report(
        video_id=video_id,
        experiment_name=experiment_name,
        baseline_video_dir=baseline_video_dir,
        output_video_dir=output_video_dir,
        frame_evidence=frame_evidence,
        shot_evidence=shot_evidence,
        frame_index=frame_index,
        shot_index=shot_index,
        event_step_index=event_step_index,
        compact_docs=compact_docs,
        override_qa=override_qa,
        frame_update_stats=frame_update_stats,
        shot_caption_mode=shot_caption_mode,
        trake_event_mode=trake_event_mode,
    )
    write_json(output_video_dir / "reports" / "experiment_propagation_report.json", report)
    write_text(
        output_video_dir / "reports" / "experiment_propagation_report.md",
        render_propagation_report_markdown(report),
    )
    return report


def normalize_frame_caption_overrides(
    rows: list[dict],
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Index usable frame caption overrides and collect QA counters."""
    indexed: dict[str, dict] = {}
    duplicate_frame_ids: list[str] = []
    empty_caption_ids: list[str] = []
    missing_frame_id_count = 0
    too_short_ids: list[str] = []
    too_long_ids: list[str] = []

    for row in rows:
        frame_id = (
            row.get("canonical_frame_id")
            or row.get("frame_id")
            or row.get("unit_id")
            or ""
        )
        caption_text = normalize_whitespace(
            row.get("caption_text", "")
            or row.get("vlm_caption", "")
            or row.get("caption", "")
            or ""
        )
        if not frame_id:
            missing_frame_id_count += 1
            continue
        if not caption_text:
            empty_caption_ids.append(frame_id)
            continue
        if frame_id in indexed:
            duplicate_frame_ids.append(frame_id)
        if len(caption_text) < 12:
            too_short_ids.append(frame_id)
        if len(caption_text) > 700:
            too_long_ids.append(frame_id)

        item = dict(row)
        item["caption_text"] = caption_text
        indexed[frame_id] = item

    qa = {
        "num_input_rows": len(rows),
        "num_valid_overrides": len(indexed),
        "num_duplicate_frame_ids": len(duplicate_frame_ids),
        "duplicate_frame_ids": duplicate_frame_ids[:50],
        "num_empty_caption_rows": len(empty_caption_ids),
        "empty_caption_frame_ids": empty_caption_ids[:50],
        "num_missing_frame_id_rows": missing_frame_id_count,
        "num_too_short_captions": len(too_short_ids),
        "too_short_frame_ids": too_short_ids[:50],
        "num_too_long_captions": len(too_long_ids),
        "too_long_frame_ids": too_long_ids[:50],
    }
    return indexed, qa


def build_visual_queries(video_id: str) -> list[dict]:
    """Build visual-heavy queries for checking frame-caption impact."""
    templates = [
        (
            "q_visual_001",
            "boats on water storm warning",
            ["boat", "water", "storm"],
            "VLM should help with water/boat scenes.",
        ),
        (
            "q_visual_002",
            "people wearing traditional dress ao dai",
            ["people", "person", "dress", "ao", "dai"],
            "Visual apparel/person query.",
        ),
        (
            "q_visual_003",
            "classroom laptop sewing machine",
            ["classroom", "laptop", "sewing", "machine"],
            "Indoor object-rich scene query.",
        ),
        (
            "q_visual_004",
            "news studio presenter screen",
            ["presenter", "screen", "studio", "person"],
            "Presenter/screen visual query.",
        ),
        (
            "q_visual_005",
            "cartoon preview colorful characters",
            ["cartoon", "preview", "characters", "wolfoo"],
            "Program preview/cartoon visual query.",
        ),
        (
            "q_visual_006",
            "city street building road vehicles",
            ["city", "street", "building", "road", "vehicle"],
            "Outdoor scene query.",
        ),
        (
            "q_visual_007",
            "parade flowers people stage",
            ["parade", "flowers", "people", "stage"],
            "Event/parade visual query.",
        ),
        (
            "q_visual_008",
            "computer screen website text screenshot",
            ["computer", "screen", "website", "screenshot"],
            "Screen/screenshot visual query.",
        ),
    ]
    rows: list[dict] = []
    for query_id, query, expected, notes in templates:
        rows.append({
            "schema_version": VISUAL_QUERY_SCHEMA,
            "query_id": query_id,
            "video_id": video_id,
            "query": query,
            "route": "object_visual",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": expected,
            "notes": notes,
        })
    return rows


def build_propagation_report(
    video_id: str,
    experiment_name: str,
    baseline_video_dir: Path,
    output_video_dir: Path,
    frame_evidence: list[dict],
    shot_evidence: list[dict],
    frame_index: list[dict],
    shot_index: list[dict],
    event_step_index: list[dict],
    compact_docs: list[dict],
    override_qa: dict[str, Any],
    frame_update_stats: dict[str, Any],
    shot_caption_mode: str,
    trake_event_mode: str,
) -> dict[str, Any]:
    warnings = _build_warnings(
        frame_index=frame_index,
        shot_index=shot_index,
        event_step_index=event_step_index,
        override_qa=override_qa,
        frame_update_stats=frame_update_stats,
    )
    return {
        "schema_version": PROPAGATION_REPORT_SCHEMA,
        "video_id": video_id,
        "experiment_name": experiment_name,
        "created_at": utc_now_iso(),
        "baseline_video_dir": str(baseline_video_dir),
        "output_video_dir": str(output_video_dir),
        "shot_caption_mode": shot_caption_mode,
        "trake_event_mode": trake_event_mode,
        "num_frame_evidence": len(frame_evidence),
        "num_shot_evidence": len(shot_evidence),
        "num_frame_index": len(frame_index),
        "num_shot_index": len(shot_index),
        "num_event_step_index": len(event_step_index),
        "num_compact_docs": len(compact_docs),
        "frame_override_qa": override_qa,
        "frame_update_stats": frame_update_stats,
        "frame_caption_mode_counts": _quality_counts(frame_index, "caption_mode"),
        "shot_caption_mode_counts": _quality_counts(shot_index, "caption_mode"),
        "event_mode_counts": _quality_counts(event_step_index, "event_mode"),
        "num_frame_captions": sum(1 for row in frame_index if row.get("caption_text")),
        "num_shot_captions": sum(1 for row in shot_index if row.get("caption_text")),
        "num_event_steps_with_trake": sum(
            1 for row in event_step_index if row.get("trake_text")
        ),
        "warnings": warnings,
    }


def render_propagation_report_markdown(report: dict[str, Any]) -> str:
    """Render the Phase 8 propagation report as Markdown."""
    frame_stats = report.get("frame_update_stats") or {}
    override_qa = report.get("frame_override_qa") or {}

    lines: list[str] = []
    lines.append(f"# Caption Experiment Propagation Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Experiment: {report.get('experiment_name', '')}")
    lines.append(f"Created: {report.get('created_at', '')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Frame evidence rows", "num_frame_evidence"),
        ("Shot evidence rows", "num_shot_evidence"),
        ("Frame index rows", "num_frame_index"),
        ("Shot index rows", "num_shot_index"),
        ("Event-step rows", "num_event_step_index"),
        ("Compact docs", "num_compact_docs"),
        ("Frame captions", "num_frame_captions"),
        ("Shot captions", "num_shot_captions"),
        ("Event steps with TRAKE", "num_event_steps_with_trake"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    lines.append("")

    lines.append("## Frame Override QA")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    override_metrics = [
        ("Input override rows", "num_input_rows"),
        ("Valid overrides", "num_valid_overrides"),
        ("Duplicate frame IDs", "num_duplicate_frame_ids"),
        ("Empty caption rows", "num_empty_caption_rows"),
        ("Missing frame-id rows", "num_missing_frame_id_rows"),
        ("Too-short captions", "num_too_short_captions"),
        ("Too-long captions", "num_too_long_captions"),
    ]
    for label, key in override_metrics:
        lines.append(f"| {label} | {override_qa.get(key, 0)} |")
    lines.append("")

    lines.append("## Propagation")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    for label, key in [
        ("Frames updated from override", "num_frames_updated_with_override"),
        ("Frames kept from baseline", "num_frames_kept_baseline"),
        ("Unused frame overrides", "num_unused_frame_overrides"),
    ]:
        lines.append(f"| {label} | {frame_stats.get(key, 0)} |")
    lines.append("")

    _append_counts_table(
        lines,
        "Frame Caption Mode Counts",
        report.get("frame_caption_mode_counts") or {},
    )
    _append_counts_table(
        lines,
        "Shot Caption Mode Counts",
        report.get("shot_caption_mode_counts") or {},
    )
    _append_counts_table(lines, "Event Mode Counts", report.get("event_mode_counts") or {})

    lines.append("## Status")
    lines.append("")
    warnings = report.get("warnings") or []
    if warnings:
        for warning in warnings:
            lines.append(f"- [WARN] {warning}")
    else:
        lines.append("OK: Propagation completed without warnings.")
    lines.append("")
    return "\n".join(lines)


def _apply_frame_override_records(
    baseline_caption_records: list[dict],
    override_index: dict[str, dict],
) -> tuple[list[dict], dict[str, Any]]:
    caption_records: list[dict] = []
    updated_ids: set[str] = set()

    for rec in baseline_caption_records:
        frame_id = rec.get("canonical_frame_id", "")
        override = override_index.get(frame_id)
        if not override:
            caption_records.append(dict(rec))
            continue

        item = dict(rec)
        item["caption_text"] = override.get("caption_text", "")
        item["caption_mode"] = override.get("caption_mode", "vlm")
        item["caption_model"] = (
            override.get("caption_model")
            or override.get("model")
            or override.get("model_name")
            or ""
        )
        item["prompt_version"] = override.get("prompt_version", "")
        item["fallback_used"] = False
        item["warnings"] = []
        item["experiment_override"] = True
        if override.get("provider"):
            item["provider"] = override["provider"]
        caption_records.append(item)
        updated_ids.add(frame_id)

    unused = sorted(set(override_index) - {
        row.get("canonical_frame_id", "") for row in baseline_caption_records
    })
    return caption_records, {
        "num_frames_updated_with_override": len(updated_ids),
        "num_frames_kept_baseline": len(baseline_caption_records) - len(updated_ids),
        "num_unused_frame_overrides": len(unused),
        "unused_frame_override_ids": unused[:50],
    }


def _caption_records_from_frame_index(frame_index: list[dict]) -> list[dict]:
    records: list[dict] = []
    for row in frame_index:
        quality = row.get("quality") or {}
        records.append({
            "canonical_frame_id": row.get("canonical_frame_id", ""),
            "caption_text": row.get("caption_text", "") or "",
            "caption_mode": quality.get("caption_mode", "baseline"),
            "caption_model": quality.get("caption_model", ""),
            "prompt_version": quality.get("prompt_version", ""),
            "used_ocr": quality.get("used_ocr", False),
            "used_audio": quality.get("used_audio", False),
            "used_objects": quality.get("used_objects", False),
            "used_scene": quality.get("used_scene", False),
            "fallback_used": quality.get("fallback_used", False),
            "warnings": quality.get("warnings") or [],
        })
    return records


def _build_propagation_config(
    video_id: str,
    output_dir: str | Path,
    shot_caption_mode: str,
    trake_event_mode: str,
) -> PipelineConfig:
    cfg = PipelineConfig(
        video_id=video_id,
        output_dir=str(output_dir),
        enable_frame_captions=True,
        enable_shot_captions=True,
        enable_trake_events=True,
        frame_caption=FrameCaptionConfig(rebuild_compact_with_captions=True),
        shot_caption=ShotCaptionConfig(caption_mode=shot_caption_mode),
        trake_event=TrakeEventConfig(event_mode=trake_event_mode),
    )
    cfg.resolve_paths()
    return cfg


def _copy_required_context(
    baseline_video_dir: Path,
    output_video_dir: Path,
    copy_reports: bool,
) -> None:
    src_evidence = baseline_video_dir / "evidence"
    dst_evidence = output_video_dir / "evidence"
    if dst_evidence.exists():
        shutil.rmtree(dst_evidence)
    if src_evidence.exists():
        shutil.copytree(src_evidence, dst_evidence)

    for subdir in ["captions", "indexes", "reports", "eval"]:
        (output_video_dir / subdir).mkdir(parents=True, exist_ok=True)

    if copy_reports:
        src_reports = baseline_video_dir / "reports"
        dst_reports = output_video_dir / "reports"
        if src_reports.exists():
            for path in src_reports.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst_reports / path.name)


def _build_warnings(
    frame_index: list[dict],
    shot_index: list[dict],
    event_step_index: list[dict],
    override_qa: dict[str, Any],
    frame_update_stats: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if override_qa.get("num_valid_overrides", 0) == 0:
        warnings.append("no valid frame caption overrides")
    if override_qa.get("num_duplicate_frame_ids", 0):
        warnings.append(
            f"{override_qa['num_duplicate_frame_ids']} duplicate frame override ids"
        )
    if override_qa.get("num_empty_caption_rows", 0):
        warnings.append(f"{override_qa['num_empty_caption_rows']} empty override captions")
    if override_qa.get("num_missing_frame_id_rows", 0):
        warnings.append(
            f"{override_qa['num_missing_frame_id_rows']} override rows missing frame id"
        )
    if frame_update_stats.get("num_unused_frame_overrides", 0):
        warnings.append(
            f"{frame_update_stats['num_unused_frame_overrides']} unused frame overrides"
        )
    if any(not row.get("caption_text") for row in frame_index):
        warnings.append("one or more frame captions are empty")
    if any(not row.get("caption_text") for row in shot_index):
        warnings.append("one or more shot captions are empty")
    if any(not row.get("trake_text") for row in event_step_index):
        warnings.append("one or more event steps have empty trake_text")
    if len(shot_index) != len(event_step_index):
        warnings.append(
            f"shot count ({len(shot_index)}) differs from event-step count "
            f"({len(event_step_index)})"
        )
    return warnings


def _quality_counts(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        quality = row.get("quality") or {}
        value = quality.get(key, "unknown") or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


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


def _resolve_video_dir(root: Path, video_id: str) -> Path:
    if (root / "captions").exists() and (root / "indexes").exists():
        return root
    return root / video_id

