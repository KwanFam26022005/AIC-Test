"""Acceptance checks for caption pipeline v2 outputs."""

from __future__ import annotations

from typing import Any

from .schemas import (
    FRAME_CAPTION_SCHEMA,
    SHOT_AUDIO_CONTEXT_SCHEMA,
    SHOT_CAPTION_SCHEMA,
    VALID_ACTION_STATES,
    VALID_CAPTION_MODES,
    VALID_SELECTION_ROLES,
    VALID_TEMPORAL_ROLES,
)


def validate_outputs(
    frames: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    frame_captions: list[dict[str, Any]],
    audio_contexts: list[dict[str, Any]],
    shot_captions: list[dict[str, Any]],
    max_fallback_rate: float = 0.15,
) -> tuple[dict[str, Any], list[str]]:
    """Return report stats and warnings/errors."""
    issues: list[str] = []

    frame_ids = [row["canonical_frame_id"] for row in frames]
    shot_ids = [row["shot_id"] for row in shots]

    _check_unique("input frame", frame_ids, issues)
    _check_unique("shot", shot_ids, issues)

    selection_by_frame = {row.get("canonical_frame_id"): row for row in selections}
    caption_by_frame = {row.get("canonical_frame_id"): row for row in frame_captions}
    audio_by_shot = {row.get("shot_id"): row for row in audio_contexts}
    shot_caption_by_shot = {row.get("shot_id"): row for row in shot_captions}

    if len(caption_by_frame) != len(frame_ids):
        issues.append(f"Expected {len(frame_ids)} frame captions, found {len(caption_by_frame)}")
    if len(audio_by_shot) != len(shot_ids):
        issues.append(f"Expected {len(shot_ids)} audio contexts, found {len(audio_by_shot)}")
    if len(shot_caption_by_shot) != len(shot_ids):
        issues.append(f"Expected {len(shot_ids)} shot captions, found {len(shot_caption_by_shot)}")

    missing_frame_captions = [fid for fid in frame_ids if fid not in caption_by_frame]
    if missing_frame_captions:
        issues.append(f"Missing frame captions: {missing_frame_captions[:10]}")

    missing_shot_captions = [sid for sid in shot_ids if sid not in shot_caption_by_shot]
    if missing_shot_captions:
        issues.append(f"Missing shot captions: {missing_shot_captions[:10]}")

    fallback_count = 0
    selected_count = 0
    for row in frame_captions:
        if row.get("schema_version") != FRAME_CAPTION_SCHEMA:
            issues.append(f"Frame caption has invalid schema: {row.get('canonical_frame_id')}")
        caption = row.get("caption") or {}
        mode = str(caption.get("mode", ""))
        if mode not in VALID_CAPTION_MODES:
            issues.append(f"Frame {row.get('canonical_frame_id')} has invalid caption mode: {mode}")
        if not str(caption.get("text", "")).strip():
            issues.append(f"Frame {row.get('canonical_frame_id')} has empty caption")
        if mode == "fallback":
            fallback_count += 1
        selection = row.get("selection") or {}
        role = str(selection.get("role", ""))
        if role not in VALID_SELECTION_ROLES:
            issues.append(f"Frame {row.get('canonical_frame_id')} has invalid selection role: {role}")
        if selection_by_frame.get(row.get("canonical_frame_id"), {}).get("selected_for_vlm"):
            selected_count += 1

    fallback_rate = fallback_count / len(frame_captions) if frame_captions else 0.0
    if fallback_rate > max_fallback_rate:
        issues.append(f"Frame fallback rate {fallback_rate:.3f} exceeds {max_fallback_rate:.3f}")

    audio_with_text = 0
    for row in audio_contexts:
        if row.get("schema_version") != SHOT_AUDIO_CONTEXT_SCHEMA:
            issues.append(f"Audio context has invalid schema: {row.get('shot_id')}")
        if row.get("has_usable_audio"):
            audio_with_text += 1

    for row in shot_captions:
        if row.get("schema_version") != SHOT_CAPTION_SCHEMA:
            issues.append(f"Shot caption has invalid schema: {row.get('shot_id')}")
        caption = row.get("caption") or {}
        for field in ("shot_caption", "event_caption", "trake_text"):
            if not str(caption.get(field, "")).strip():
                issues.append(f"Shot {row.get('shot_id')} has empty {field}")
        structure = row.get("structure") or {}
        action_state = str(structure.get("action_state", ""))
        temporal_role = str(structure.get("temporal_role", ""))
        if action_state not in VALID_ACTION_STATES:
            issues.append(f"Shot {row.get('shot_id')} has invalid action_state: {action_state}")
        if temporal_role not in VALID_TEMPORAL_ROLES:
            issues.append(f"Shot {row.get('shot_id')} has invalid temporal_role: {temporal_role}")
        raw = str(row)
        if "ocr_evidence" in raw or "object_detection" in raw or "ram_tags" in raw:
            issues.append(f"Shot {row.get('shot_id')} appears to contain old OCR/object evidence")

    stats = {
        "num_frames": len(frame_ids),
        "num_shots": len(shot_ids),
        "num_selected_for_vlm": selected_count,
        "num_frame_captions": len(frame_captions),
        "num_shot_audio_contexts": len(audio_contexts),
        "num_shot_captions": len(shot_captions),
        "num_frame_fallback": fallback_count,
        "frame_fallback_rate": round(fallback_rate, 4),
        "num_shots_with_audio": audio_with_text,
        "num_issues": len(issues),
    }
    return stats, issues


def _check_unique(label: str, values: list[str], issues: list[str]) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for value in values:
        if value in seen:
            dupes.append(value)
        seen.add(value)
    if dupes:
        issues.append(f"Duplicate {label} IDs: {dupes[:10]}")
