"""Materialize caption experiment directories from baseline outputs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, utc_now_iso, write_json, write_jsonl, write_text
from .retrieval_export import normalized_text, tokenize
from .text_utils import join_texts


def materialize_caption_experiment(
    video_id: str,
    baseline_dir: str | Path,
    output_root: str | Path,
    experiment_name: str,
    frame_caption_overrides: list[dict] | None = None,
    shot_caption_overrides: list[dict] | None = None,
    event_step_overrides: list[dict] | None = None,
    copy_reports: bool = False,
) -> dict[str, Any]:
    """Create an experiment caption directory with optional caption overrides."""
    baseline_video_dir = _resolve_video_dir(Path(baseline_dir), video_id)
    output_video_dir = Path(output_root) / experiment_name / video_id

    frame_index = read_jsonl(baseline_video_dir / "captions" / "frame_index.jsonl")
    shot_index = read_jsonl(baseline_video_dir / "captions" / "shot_index.jsonl")
    event_step_index = read_jsonl(baseline_video_dir / "captions" / "event_step_index.jsonl")
    compact_docs = read_jsonl(baseline_video_dir / "indexes" / "compact_search_index.jsonl")

    frame_overrides = _index_by_frame_id(frame_caption_overrides or [])
    shot_overrides = _index_by_shot_id(shot_caption_overrides or [])
    event_overrides = _index_by_event_or_shot(event_step_overrides or [])

    updated_frames = _apply_frame_overrides(frame_index, compact_docs, frame_overrides)
    updated_shots = _apply_shot_overrides(shot_index, compact_docs, shot_overrides)
    updated_events = _apply_event_overrides(event_step_index, compact_docs, event_overrides)

    _copy_required_context(baseline_video_dir, output_video_dir, copy_reports=copy_reports)
    write_jsonl(output_video_dir / "captions" / "frame_index.jsonl", frame_index)
    write_jsonl(output_video_dir / "captions" / "shot_index.jsonl", shot_index)
    write_jsonl(output_video_dir / "captions" / "event_step_index.jsonl", event_step_index)
    write_jsonl(output_video_dir / "indexes" / "compact_search_index.jsonl", compact_docs)

    report = {
        "schema_version": "caption_experiment_materialize_report_v1",
        "video_id": video_id,
        "experiment_name": experiment_name,
        "created_at": utc_now_iso(),
        "baseline_video_dir": str(baseline_video_dir),
        "output_video_dir": str(output_video_dir),
        "num_frame_index": len(frame_index),
        "num_shot_index": len(shot_index),
        "num_event_step_index": len(event_step_index),
        "num_compact_docs": len(compact_docs),
        "num_frame_overrides": len(frame_overrides),
        "num_shot_overrides": len(shot_overrides),
        "num_event_step_overrides": len(event_overrides),
        "num_frames_updated": updated_frames,
        "num_shots_updated": updated_shots,
        "num_event_steps_updated": updated_events,
        "warnings": _build_warnings(
            frame_overrides,
            shot_overrides,
            event_overrides,
            frame_index,
            shot_index,
            event_step_index,
        ),
    }
    write_json(output_video_dir / "reports" / "experiment_materialize_report.json", report)
    write_text(
        output_video_dir / "reports" / "experiment_materialize_report.md",
        render_materialize_report_markdown(report),
    )
    return report


def render_materialize_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Caption Experiment Materialize Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Experiment: {report.get('experiment_name', '')}")
    lines.append(f"Created: {report.get('created_at', '')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Frame index rows", "num_frame_index"),
        ("Shot index rows", "num_shot_index"),
        ("Event-step rows", "num_event_step_index"),
        ("Compact docs", "num_compact_docs"),
        ("Frame overrides", "num_frame_overrides"),
        ("Shot overrides", "num_shot_overrides"),
        ("Event-step overrides", "num_event_step_overrides"),
        ("Frames updated", "num_frames_updated"),
        ("Shots updated", "num_shots_updated"),
        ("Event steps updated", "num_event_steps_updated"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    lines.append("")
    warnings = report.get("warnings") or []
    lines.append("## Status")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- [WARN] {warning}")
    else:
        lines.append("OK: Experiment materialized without warnings.")
    lines.append("")
    return "\n".join(lines)


def _apply_frame_overrides(
    frame_index: list[dict],
    compact_docs: list[dict],
    overrides: dict[str, dict],
) -> int:
    updated = 0
    for doc in frame_index:
        frame_id = doc.get("canonical_frame_id", "") or doc.get("frame_id", "")
        override = overrides.get(frame_id)
        if not override:
            continue
        caption_text = override.get("caption_text", "") or override.get("vlm_caption", "")
        if not caption_text:
            continue
        doc["caption_text"] = caption_text
        doc["caption_text_search"] = normalized_text(caption_text)
        doc["caption_terms"] = tokenize(caption_text)
        quality = dict(doc.get("quality") or {})
        quality["has_caption"] = True
        quality["caption_mode"] = override.get("caption_mode", "experiment")
        quality["caption_model"] = override.get("caption_model", override.get("model", ""))
        quality["prompt_version"] = override.get("prompt_version", "")
        quality["experiment_override"] = True
        doc["quality"] = quality
        search_fields = dict(doc.get("search_fields") or {})
        evidence = doc.get("evidence_text") or {}
        search_fields["caption_boost_text"] = normalized_text(caption_text)
        search_fields["all_text"] = join_texts(
            evidence.get("ocr_text", ""),
            evidence.get("audio_text", ""),
            evidence.get("object_text", ""),
            evidence.get("scene_text", ""),
            caption_text,
        )
        doc["search_fields"] = search_fields
        updated += 1

    frame_caption_map = {
        doc.get("canonical_frame_id"): doc.get("caption_text", "")
        for doc in frame_index
        if doc.get("canonical_frame_id")
    }
    for doc in compact_docs:
        if doc.get("unit_type") != "frame":
            continue
        caption_text = frame_caption_map.get(doc.get("unit_id", ""), "")
        if caption_text:
            doc["caption_text"] = caption_text
            _refresh_compact_all_text(doc)
    return updated


def _apply_shot_overrides(
    shot_index: list[dict],
    compact_docs: list[dict],
    overrides: dict[str, dict],
) -> int:
    updated = 0
    for doc in shot_index:
        shot_id = doc.get("shot_id", "")
        override = overrides.get(shot_id)
        if not override:
            continue
        caption_text = override.get("caption_text", "") or override.get("shot_caption", "")
        temporal_caption = override.get("temporal_caption", "")
        if not caption_text and not temporal_caption:
            continue
        if caption_text:
            doc["caption_text"] = caption_text
            doc["caption_text_search"] = normalized_text(caption_text)
        if temporal_caption:
            doc["temporal_caption"] = temporal_caption
            doc["temporal_caption_search"] = normalized_text(temporal_caption)
        doc["caption_terms"] = tokenize(join_texts(
            doc.get("caption_text", ""),
            doc.get("temporal_caption", ""),
        ))
        quality = dict(doc.get("quality") or {})
        quality["has_caption"] = bool(doc.get("caption_text"))
        quality["caption_mode"] = override.get("caption_mode", "experiment")
        quality["caption_model"] = override.get("caption_model", override.get("model", ""))
        quality["prompt_version"] = override.get("prompt_version", "")
        quality["experiment_override"] = True
        doc["quality"] = quality
        search_fields = dict(doc.get("search_fields") or {})
        evidence = doc.get("evidence_text") or {}
        search_fields["caption_boost_text"] = normalized_text(join_texts(
            doc.get("caption_text", ""),
            doc.get("temporal_caption", ""),
        ))
        search_fields["all_text"] = join_texts(
            evidence.get("merged_ocr_text", ""),
            evidence.get("merged_audio_text", ""),
            evidence.get("object_text", ""),
            evidence.get("scene_text", ""),
            evidence.get("representative_frame_caption_text", ""),
            doc.get("caption_text", ""),
            doc.get("temporal_caption", ""),
        )
        search_fields["temporal_text"] = join_texts(
            (doc.get("memory") or {}).get("memory_before", ""),
            doc.get("temporal_caption", ""),
            (doc.get("memory") or {}).get("memory_after", ""),
        )
        doc["search_fields"] = search_fields
        updated += 1

    shot_caption_map = {
        doc.get("shot_id"): doc.get("caption_text", "")
        for doc in shot_index
        if doc.get("shot_id")
    }
    for doc in compact_docs:
        if doc.get("unit_type") != "shot":
            continue
        caption_text = shot_caption_map.get(doc.get("unit_id", ""), "")
        if caption_text:
            doc["caption_text"] = caption_text
            _refresh_compact_all_text(doc)
    return updated


def _apply_event_overrides(
    event_step_index: list[dict],
    compact_docs: list[dict],
    overrides: dict[str, dict],
) -> int:
    updated = 0
    for doc in event_step_index:
        key = doc.get("event_id", "") or doc.get("shot_id", "")
        override = overrides.get(key) or overrides.get(doc.get("shot_id", ""))
        if not override:
            continue
        for field in ["event_caption", "current_observation", "before_context", "after_context"]:
            if override.get(field):
                doc[field] = override[field]
        trake_text = override.get("trake_text", "")
        if trake_text:
            doc["trake_text"] = trake_text
            doc["trake_text_search"] = normalized_text(trake_text)
            doc["trake_terms"] = tokenize(trake_text)
        for field in ["action_state", "temporal_role", "scene"]:
            if override.get(field):
                doc[field] = override[field]
        quality = dict(doc.get("quality") or {})
        quality["has_event_caption"] = bool(doc.get("event_caption"))
        quality["has_trake_text"] = bool(doc.get("trake_text"))
        quality["event_mode"] = override.get("event_mode", "experiment")
        quality["event_model"] = override.get("event_model", override.get("model", ""))
        quality["prompt_version"] = override.get("prompt_version", "")
        quality["experiment_override"] = True
        doc["quality"] = quality
        updated += 1

    trake_map = {
        doc.get("shot_id"): doc.get("trake_text", "")
        for doc in event_step_index
        if doc.get("shot_id")
    }
    for doc in compact_docs:
        if doc.get("unit_type") != "shot":
            continue
        trake_text = trake_map.get(doc.get("unit_id", ""), "")
        if trake_text:
            doc["trake_text"] = trake_text
            _refresh_compact_all_text(doc)
    return updated


def _copy_required_context(
    baseline_video_dir: Path,
    output_video_dir: Path,
    copy_reports: bool,
) -> None:
    for subdir in ["evidence"]:
        src = baseline_video_dir / subdir
        dst = output_video_dir / subdir
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)
    for subdir in ["captions", "indexes", "reports"]:
        (output_video_dir / subdir).mkdir(parents=True, exist_ok=True)
    if copy_reports:
        src = baseline_video_dir / "reports"
        dst = output_video_dir / "reports"
        if src.exists():
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)


def _build_warnings(
    frame_overrides: dict[str, dict],
    shot_overrides: dict[str, dict],
    event_overrides: dict[str, dict],
    frame_index: list[dict],
    shot_index: list[dict],
    event_step_index: list[dict],
) -> list[str]:
    warnings: list[str] = []
    frame_ids = {row.get("canonical_frame_id") for row in frame_index}
    shot_ids = {row.get("shot_id") for row in shot_index}
    event_ids = {row.get("event_id") for row in event_step_index}
    event_shot_ids = {row.get("shot_id") for row in event_step_index}
    for key in sorted(set(frame_overrides) - frame_ids):
        warnings.append(f"unused frame override: {key}")
    for key in sorted(set(shot_overrides) - shot_ids):
        warnings.append(f"unused shot override: {key}")
    valid_event_keys = event_ids | event_shot_ids
    for key in sorted(set(event_overrides) - valid_event_keys):
        warnings.append(f"unused event override: {key}")
    return warnings[:100]


def _index_by_frame_id(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        key = (
            row.get("canonical_frame_id")
            or row.get("frame_id")
            or row.get("unit_id")
            or ""
        )
        if key:
            result[key] = row
    return result


def _index_by_shot_id(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        key = row.get("shot_id") or row.get("unit_id") or ""
        if key:
            result[key] = row
    return result


def _index_by_event_or_shot(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        key = row.get("event_id") or row.get("shot_id") or row.get("unit_id") or ""
        if key:
            result[key] = row
    return result


def _refresh_compact_all_text(doc: dict) -> None:
    doc["all_text"] = join_texts(
        doc.get("ocr_text", ""),
        doc.get("audio_text", ""),
        doc.get("object_text", ""),
        doc.get("scene_text", ""),
        doc.get("caption_text", ""),
        doc.get("trake_text", ""),
    )


def _resolve_video_dir(root: Path, video_id: str) -> Path:
    if (root / "captions").exists() and (root / "indexes").exists():
        return root
    return root / video_id
