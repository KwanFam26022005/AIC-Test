"""TRAKE event-step builder for Phase 4 temporal retrieval."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import PipelineConfig
from .runtime import JsonlCheckpoint, TextLLMRuntime, stable_input_signature
from .runtime.json_output import (
    enum_value,
    optional_text,
    require_text,
    string_list,
)
from .text_utils import join_texts, normalize_whitespace, truncate

logger = logging.getLogger(__name__)

_ACTOR_LABELS = {
    "person", "people", "man", "woman", "child", "customer", "student",
    "player", "boy", "girl", "speaker", "presenter",
}

_ACTION_KEYWORDS = {
    "speaking": ["speak", "speaking", "spoken", "noi", "phat bieu", "trinh bay"],
    "presenting": ["present", "presenting", "presentation", "trinh bay"],
    "showing": ["show", "shows", "display", "hien thi"],
    "walking": ["walk", "walking", "di bo"],
    "driving": ["drive", "driving", "car", "truck", "oto", "xe"],
    "sitting": ["sit", "sitting", "ngoi"],
    "standing": ["stand", "standing", "dung"],
}

_RESULT_KEYWORDS = [
    "result", "finally", "completed", "ending", "conclusion",
    "ket qua", "cuoi cung", "hoan thanh", "ket thuc",
]


def build_trake_event_steps(
    shot_index: list[dict],
    shot_evidence: list[dict],
    cfg: PipelineConfig,
    llm_runtime: TextLLMRuntime | None = None,
    checkpoint: JsonlCheckpoint | None = None,
) -> list[dict]:
    """Build deterministic TRAKE event-step records from shot captions."""
    evidence_map = {
        row.get("shot_id"): row
        for row in shot_evidence
        if row.get("shot_id")
    }
    shots = sorted(
        shot_index,
        key=lambda row: (row.get("start_sec", 0.0), row.get("shot_id", "")),
    )
    mode = cfg.trake_event.event_mode
    results: list[dict] = []

    for idx, shot in enumerate(shots):
        shot_id = shot["shot_id"]
        prev_shot = shots[idx - 1] if idx > 0 else None
        next_shot = shots[idx + 1] if idx < len(shots) - 1 else None
        evidence = evidence_map.get(shot.get("shot_id", ""), {})
        input_signature = stable_input_signature({
            "shot": shot,
            "evidence": evidence,
            "previous_shot": prev_shot,
            "next_shot": next_shot,
        })
        if checkpoint:
            existing = checkpoint.get(shot_id, input_signature=input_signature)
            if existing:
                results.append(existing)
                continue

        rec = _template_event_step(
            shot=shot,
            evidence=evidence,
            prev_shot=prev_shot,
            next_shot=next_shot,
            cfg=cfg,
            idx=idx,
            num_steps=len(shots),
        )
        if mode != "template" and llm_runtime is not None:
            try:
                rec = _llm_event_step(rec, shot, evidence, cfg, llm_runtime)
            except Exception as exc:  # noqa: BLE001 - preserve deterministic event fallback.
                logger.exception("TRAKE LLM failed for %s", shot_id)
                rec.update({
                    "event_mode": "fallback",
                    "event_model": llm_runtime.model_name,
                    "provider": llm_runtime.provider,
                    "prompt_version": llm_runtime.prompt_version,
                    "fallback_used": True,
                })
                rec["warnings"].append(
                    f"llm_generation_failed:{type(exc).__name__}:{truncate(str(exc), 180)}"
                )
        elif mode != "template":
            rec["event_mode"] = "fallback"
            rec["fallback_used"] = True
            rec["warnings"].append("llm_runtime_unavailable_using_template")
        results.append(rec)
        if checkpoint:
            rec["input_signature"] = input_signature
            checkpoint.put(rec)

    if checkpoint:
        checkpoint.flush()
        logger.info("TRAKE checkpoint reused %d rows", checkpoint.num_reused)

    num_text = sum(1 for row in results if row.get("trake_text"))
    num_fallback = sum(1 for row in results if row.get("fallback_used"))
    logger.info(
        "TRAKE event builder: %d event steps (%d with trake_text, %d fallback)",
        len(results), num_text, num_fallback,
    )
    return results


def _llm_event_step(
    base: dict[str, Any],
    shot: dict,
    evidence: dict,
    cfg: PipelineConfig,
    runtime: TextLLMRuntime,
) -> dict[str, Any]:
    source = shot.get("evidence_text") or {}
    values = {
        "shot_id": shot["shot_id"],
        "caption_text": shot.get("caption_text", "") or "",
        "temporal_caption": shot.get("temporal_caption", "") or "",
        "before_context": base.get("before_context", "") or "",
        "after_context": base.get("after_context", "") or "",
        "object_text": source.get("object_text", "") or "",
        "scene_text": source.get("scene_text", "") or " ".join(evidence.get("merged_scene_tags") or []),
        "ocr_text": source.get("merged_ocr_text", "") or evidence.get("merged_ocr_text", "") or "",
        "audio_text": source.get("merged_audio_text", "") or evidence.get("merged_audio_text", "") or "",
        "deterministic_actors": json.dumps(base.get("actors") or [], ensure_ascii=False),
        "deterministic_actions": json.dumps(base.get("actions") or [], ensure_ascii=False),
        "max_trake_text_chars": cfg.trake_event.max_trake_text_chars,
    }
    allowed_action_states = {"start", "middle", "end", "transition", "result", "unknown"}
    allowed_temporal_roles = {"beginning", "continuation", "change", "completion", "unknown"}

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_caption": require_text(
                payload,
                "event_caption",
                cfg.trake_event.max_current_observation_chars,
            ),
            "action_state": enum_value(
                payload,
                "action_state",
                allowed_action_states,
            ),
            "temporal_role": enum_value(
                payload,
                "temporal_role",
                allowed_temporal_roles,
            ),
            "actors": string_list(payload, "actors", 8),
            "actions": string_list(payload, "actions", 8),
            "objects_involved": string_list(payload, "objects_involved", 12),
            "scene": optional_text(payload, "scene", 240),
            "before_context": optional_text(
                payload,
                "before_context",
                cfg.trake_event.max_before_context_chars,
            ),
            "current_observation": require_text(
                payload,
                "current_observation",
                cfg.trake_event.max_current_observation_chars,
            ),
            "after_context": optional_text(
                payload,
                "after_context",
                cfg.trake_event.max_after_context_chars,
            ),
            "trake_text": require_text(
                payload,
                "trake_text",
                cfg.trake_event.max_trake_text_chars,
            ),
        }

    required = (
        "event_caption",
        "action_state",
        "temporal_role",
        "actors",
        "actions",
        "objects_involved",
        "scene",
        "before_context",
        "current_observation",
        "after_context",
        "trake_text",
    )
    outcome = runtime.generate_task(
        "trake",
        values,
        required_fields=required,
        max_new_tokens=cfg.llm.trake_max_new_tokens,
        validator=validate,
    )
    result = dict(base)
    result.update(outcome.data)
    result.update({
        "event_mode": "llm",
        "event_model": runtime.model_name,
        "provider": runtime.provider,
        "prompt_version": runtime.prompt_version,
        "prompt_hash": outcome.prompt_hash,
        "prompt_path": outcome.prompt_path,
        "generation_attempts": outcome.attempts,
        "fallback_used": False,
        "warnings": [],
    })
    return result


def _template_event_step(
    shot: dict,
    evidence: dict,
    prev_shot: dict | None,
    next_shot: dict | None,
    cfg: PipelineConfig,
    idx: int,
    num_steps: int,
) -> dict[str, Any]:
    shot_id = shot["shot_id"]
    step_order = idx + 1
    video_id = shot.get("video_id", cfg.video_id)
    event_id = f"{video_id}_event_{step_order:04d}"

    source = shot.get("evidence_text") or {}
    caption_text = shot.get("caption_text", "") or ""
    temporal_caption = shot.get("temporal_caption", "") or ""
    object_counts = evidence.get("merged_object_counts") or {}
    scene_tags = evidence.get("merged_scene_tags") or []
    object_text = source.get("object_text", "") or _object_counts_to_text(object_counts, 12)
    scene_text = source.get("scene_text", "") or " ".join(scene_tags)
    merged_ocr = source.get("merged_ocr_text", "") or evidence.get("merged_ocr_text", "") or ""
    merged_audio = source.get("merged_audio_text", "") or evidence.get("merged_audio_text", "") or ""

    before_context = _context_from_shot(prev_shot, cfg.trake_event.max_before_context_chars)
    current_observation = _finalize_text(
        caption_text or temporal_caption,
        cfg.trake_event.max_current_observation_chars,
    )
    after_context = _context_from_shot(next_shot, cfg.trake_event.max_after_context_chars)
    event_caption = current_observation

    actors = _extract_actors(object_counts)
    objects_involved = _extract_objects(object_counts, actors, max_labels=10)
    actions = _extract_actions(join_texts(caption_text, temporal_caption, merged_audio))

    changed = _is_major_change(evidence, prev_shot)
    action_state = _action_state(
        idx=idx,
        num_steps=num_steps,
        text=join_texts(caption_text, temporal_caption, merged_audio),
        changed=changed,
    )
    temporal_role = _temporal_role(idx, num_steps, changed)

    trake_text = _build_trake_text(
        action_state=action_state,
        temporal_role=temporal_role,
        event_caption=event_caption,
        current_observation=current_observation,
        actors=actors,
        actions=actions,
        objects_involved=objects_involved,
        scene=scene_text,
        ocr_text=merged_ocr,
        audio_text=merged_audio,
        cfg=cfg,
    )

    warnings: list[str] = []
    if not trake_text:
        warnings.append("empty_trake_text")

    return {
        "shot_id": shot_id,
        "event_id": event_id,
        "step_order": step_order,
        "event_caption": event_caption,
        "current_observation": current_observation,
        "before_context": before_context,
        "after_context": after_context,
        "action_state": action_state,
        "temporal_role": temporal_role,
        "actors": actors,
        "actions": actions,
        "objects_involved": objects_involved,
        "scene": scene_text,
        "trake_text": trake_text,
        "event_mode": "template",
        "event_model": "",
        "prompt_version": "",
        "fallback_used": bool(warnings),
        "warnings": warnings,
    }


def _context_from_shot(shot: dict | None, max_chars: int) -> str:
    if not shot:
        return ""
    text = shot.get("temporal_caption", "") or shot.get("caption_text", "") or ""
    return _finalize_text(text, max_chars)


def _object_counts_to_text(counts: dict[str, int], max_labels: int) -> str:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_labels]
    return ", ".join(f"{cnt} {label}" for label, cnt in items if cnt > 0)


def _extract_actors(counts: dict[str, int]) -> list[str]:
    actors: list[str] = []
    for label, _cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        low = label.lower()
        if low in _ACTOR_LABELS and label not in actors:
            actors.append(label)
    return actors[:6]


def _extract_objects(counts: dict[str, int], actors: list[str], max_labels: int) -> list[str]:
    actor_set = {label.lower() for label in actors}
    objects: list[str] = []
    for label, _cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if label.lower() not in actor_set:
            objects.append(label)
        if len(objects) >= max_labels:
            break
    return objects


def _extract_actions(text: str) -> list[str]:
    normalized = text.lower()
    found: list[str] = []
    for action, keywords in _ACTION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            found.append(action)
    return found


def _action_state(idx: int, num_steps: int, text: str, changed: bool) -> str:
    low = text.lower()
    if idx == 0:
        return "start"
    if idx == num_steps - 1:
        return "end"
    if changed:
        return "transition"
    if any(keyword in low for keyword in _RESULT_KEYWORDS):
        return "result"
    return "middle"


def _temporal_role(idx: int, num_steps: int, changed: bool) -> str:
    if idx == 0:
        return "beginning"
    if idx == num_steps - 1:
        return "completion"
    if changed:
        return "change"
    return "continuation"


def _is_major_change(evidence: dict, prev_shot: dict | None) -> bool:
    if not prev_shot:
        return False

    prev_source = prev_shot.get("evidence_text") or {}
    current_scenes = set(evidence.get("merged_scene_tags") or [])
    prev_scenes = set((prev_source.get("scene_text", "") or "").split())

    current_objects = set((evidence.get("merged_object_counts") or {}).keys())
    prev_object_text = prev_source.get("object_text", "") or ""
    prev_objects = {
        token.strip()
        for chunk in prev_object_text.split(",")
        for token in [" ".join(chunk.strip().split()[1:])]
        if token.strip()
    }

    scene_overlap = _overlap_ratio(current_scenes, prev_scenes)
    object_overlap = _overlap_ratio(current_objects, prev_objects)
    if current_scenes and prev_scenes and current_objects and prev_objects:
        return scene_overlap < 0.25 and object_overlap < 0.25
    if current_scenes and prev_scenes:
        return scene_overlap < 0.15
    if current_objects and prev_objects:
        return object_overlap < 0.15
    return False


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _build_trake_text(
    action_state: str,
    temporal_role: str,
    event_caption: str,
    current_observation: str,
    actors: list[str],
    actions: list[str],
    objects_involved: list[str],
    scene: str,
    ocr_text: str,
    audio_text: str,
    cfg: PipelineConfig,
) -> str:
    return _finalize_text(
        join_texts(
            action_state,
            temporal_role,
            event_caption,
            current_observation,
            " ".join(actors),
            " ".join(actions),
            " ".join(objects_involved),
            scene,
            truncate(ocr_text, 180),
            truncate(audio_text, 240),
        ),
        cfg.trake_event.max_trake_text_chars,
    )


def _finalize_text(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = truncate(text, max_chars).rstrip(" ,;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text
