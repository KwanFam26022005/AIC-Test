"""Shot captioner - generate shot-level temporal captions from evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import PipelineConfig
from .progress import progress_iter
from .runtime import JsonlCheckpoint, TextLLMRuntime, stable_input_signature
from .runtime.json_output import optional_text, require_text
from .text_utils import normalize_whitespace, truncate

logger = logging.getLogger(__name__)


def fuse_shot_captions(
    shot_evidence: list[dict],
    frame_index: list[dict],
    cfg: PipelineConfig,
    llm_runtime: TextLLMRuntime | None = None,
    checkpoint: JsonlCheckpoint | None = None,
) -> list[dict]:
    """Generate shot caption records from shot evidence and frame captions."""
    frame_map = {
        row.get("canonical_frame_id"): row
        for row in frame_index
        if row.get("canonical_frame_id")
    }
    mode = cfg.shot_caption.caption_mode
    results: list[dict] = []
    memory_before = ""

    for idx, shot in progress_iter(
        enumerate(shot_evidence),
        total=len(shot_evidence),
        desc="Phase 3 shot captions",
        logger=logger,
        item_label=lambda item: item[1].get("shot_id", ""),
    ):
        shot_id = shot["shot_id"]
        rep_input = {
            frame_id: (frame_map.get(frame_id) or {}).get("caption_text", "") or ""
            for frame_id in shot.get("representative_frame_ids") or []
        }
        input_signature = stable_input_signature({
            "shot_evidence": shot,
            "representative_captions": rep_input,
            "memory_before": memory_before,
        })
        if checkpoint:
            existing = checkpoint.get(shot_id, input_signature=input_signature)
            if existing:
                memory_before = existing.get("memory_after", "") or ""
                results.append(existing)
                continue
        if mode == "template":
            rec = _template_shot_caption(
                shot, frame_map, cfg, idx, len(shot_evidence), memory_before,
            )
        elif llm_runtime is not None:
            try:
                rec = _llm_shot_caption(
                    shot,
                    frame_map,
                    cfg,
                    idx,
                    len(shot_evidence),
                    memory_before,
                    llm_runtime,
                )
            except Exception as exc:  # noqa: BLE001 - use deterministic record fallback.
                logger.exception("Shot caption LLM failed for %s", shot_id)
                rec = _template_shot_caption(
                    shot, frame_map, cfg, idx, len(shot_evidence), memory_before,
                )
                rec.update({
                    "caption_mode": "fallback",
                    "caption_model": llm_runtime.model_name,
                    "provider": llm_runtime.provider,
                    "prompt_version": llm_runtime.prompt_version,
                    "fallback_used": True,
                })
                rec["warnings"].append(
                    f"llm_generation_failed:{type(exc).__name__}:{truncate(str(exc), 180)}"
                )
        else:
            rec = _template_shot_caption(
                shot, frame_map, cfg, idx, len(shot_evidence), memory_before,
            )
            rec["caption_mode"] = "fallback"
            rec["fallback_used"] = True
            rec["warnings"].append("llm_runtime_unavailable_using_template")

        memory_before = rec.get("memory_after", "") or ""
        results.append(rec)
        if checkpoint:
            rec["input_signature"] = input_signature
            checkpoint.put(rec)

    if checkpoint:
        checkpoint.flush()
        logger.info("Shot caption checkpoint reused %d rows", checkpoint.num_reused)

    num_generated = sum(1 for r in results if r.get("caption_text"))
    num_temporal = sum(1 for r in results if r.get("temporal_caption"))
    num_empty = len(results) - num_generated
    num_fallback = sum(1 for r in results if r.get("fallback_used"))

    logger.info(
        "Shot captioner: %d shots (%d captions, %d temporal, %d empty, %d fallback)",
        len(results), num_generated, num_temporal, num_empty, num_fallback,
    )
    return results


def _llm_shot_caption(
    shot: dict,
    frame_map: dict[str, dict],
    cfg: PipelineConfig,
    shot_index: int,
    num_shots: int,
    memory_before: str,
    runtime: TextLLMRuntime,
) -> dict[str, Any]:
    sc = cfg.shot_caption
    rep_ids = shot.get("representative_frame_ids") or []
    rep_captions = [
        (frame_map.get(frame_id) or {}).get("caption_text", "") or ""
        for frame_id in rep_ids
    ]
    rep_captions = [caption for caption in rep_captions if caption]
    object_text = _object_counts_to_text(shot.get("merged_object_counts") or {}, max_labels=12)
    scene_text = " ".join(shot.get("merged_scene_tags") or [])
    ocr_text = truncate((shot.get("merged_ocr_text") or "").strip(), sc.max_ocr_chars)
    audio_text = truncate((shot.get("merged_audio_text") or "").strip(), sc.max_audio_chars)
    values = {
        "shot_id": shot["shot_id"],
        "start_sec": shot.get("start_sec", 0.0),
        "end_sec": shot.get("end_sec", shot.get("start_sec", 0.0)),
        "temporal_position": _temporal_position(shot_index, num_shots),
        "representative_frame_captions": json.dumps(rep_captions, ensure_ascii=False),
        "object_text": object_text,
        "scene_text": scene_text,
        "ocr_text": ocr_text,
        "audio_text": audio_text,
        "memory_before": memory_before,
        "max_caption_chars": sc.max_caption_chars,
        "max_temporal_caption_chars": sc.max_temporal_caption_chars,
        "max_memory_chars": sc.max_memory_chars,
    }

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "caption_text": require_text(payload, "caption_text", sc.max_caption_chars),
            "temporal_caption": require_text(
                payload,
                "temporal_caption",
                sc.max_temporal_caption_chars,
            ),
            "memory_after": optional_text(payload, "memory_after", sc.max_memory_chars),
        }

    outcome = runtime.generate_task(
        "shot",
        values,
        required_fields=("caption_text", "temporal_caption", "memory_after"),
        max_new_tokens=cfg.llm.shot_max_new_tokens,
        validator=validate,
    )
    data = outcome.data
    warnings: list[str] = []
    if not data["memory_after"]:
        data["memory_after"] = _build_memory_after(
            memory_before=memory_before,
            scene_text=scene_text,
            object_text=object_text,
            ocr_text=ocr_text,
            audio_text=audio_text,
            cfg=cfg,
        )
        warnings.append("llm_empty_memory_after_used_deterministic_memory")
    return {
        "shot_id": shot["shot_id"],
        "caption_text": _finalize_caption(data["caption_text"], sc.max_caption_chars),
        "temporal_caption": _finalize_caption(
            data["temporal_caption"],
            sc.max_temporal_caption_chars,
        ),
        "memory_before": memory_before,
        "memory_after": data["memory_after"],
        "caption_mode": "llm",
        "caption_model": runtime.model_name,
        "provider": runtime.provider,
        "prompt_version": runtime.prompt_version,
        "prompt_hash": outcome.prompt_hash,
        "prompt_path": outcome.prompt_path,
        "generation_attempts": outcome.attempts,
        "used_ocr": bool(ocr_text),
        "used_audio": bool(audio_text),
        "used_objects": bool(object_text),
        "used_scene": bool(scene_text),
        "used_frame_captions": bool(rep_captions),
        "fallback_used": False,
        "warnings": warnings,
    }


def _template_shot_caption(
    shot: dict,
    frame_map: dict[str, dict],
    cfg: PipelineConfig,
    shot_index: int,
    num_shots: int,
    memory_before: str,
) -> dict[str, Any]:
    sc = cfg.shot_caption
    warnings: list[str] = []

    rep_ids = shot.get("representative_frame_ids") or []
    rep_captions: list[str] = []
    for frame_id in rep_ids:
        frame_doc = frame_map.get(frame_id)
        caption = (frame_doc or {}).get("caption_text", "") or ""
        if caption:
            rep_captions.append(caption)
        else:
            warnings.append(f"missing_representative_frame_caption:{frame_id}")

    frame_caption_text = _join_limited(rep_captions, sc.max_frame_caption_chars)
    object_text = _object_counts_to_text(shot.get("merged_object_counts") or {}, max_labels=12)
    scene_text = " ".join(shot.get("merged_scene_tags") or [])
    ocr_text = truncate((shot.get("merged_ocr_text") or "").strip(), sc.max_ocr_chars)
    audio_text = truncate((shot.get("merged_audio_text") or "").strip(), sc.max_audio_chars)

    used_frame_captions = bool(rep_captions)
    used_objects = bool(object_text)
    used_scene = bool(scene_text)
    used_ocr = bool(ocr_text)
    used_audio = bool(audio_text)

    parts: list[str] = []
    start_sec = shot.get("start_sec", 0.0)
    end_sec = shot.get("end_sec", start_sec)
    parts.append(f"Shot covers {start_sec}-{end_sec} seconds.")

    if scene_text:
        parts.append(f"Scene shows {_format_list_phrase(scene_text.split())}.")
    if object_text:
        parts.append(f"Visible objects include {object_text}.")
    if frame_caption_text:
        parts.append(f"Representative frames show {_strip_terminal_punctuation(frame_caption_text)}.")
    if ocr_text:
        parts.append(f'On-screen text mentions "{_strip_terminal_punctuation(ocr_text)}".')
    if audio_text:
        parts.append(f"Spoken content mentions {_strip_terminal_punctuation(audio_text)}.")

    caption_text = _finalize_caption(" ".join(parts), sc.max_caption_chars)
    if not caption_text:
        fallback = (shot.get("shot_search_text") or "").strip()
        if fallback:
            caption_text = _finalize_caption(fallback, sc.max_caption_chars)
            warnings.append("no_structured_evidence_fallback_search_text")
        else:
            warnings.append("empty_caption_no_evidence")

    position = _temporal_position(shot_index, num_shots)
    temporal_parts = [position, caption_text]
    if memory_before:
        temporal_parts.append(f"Previous context: {_strip_terminal_punctuation(memory_before)}.")
    temporal_caption = _finalize_caption(
        " ".join(p for p in temporal_parts if p), sc.max_temporal_caption_chars,
    )

    memory_after = _build_memory_after(
        memory_before=memory_before,
        scene_text=scene_text,
        object_text=object_text,
        ocr_text=ocr_text,
        audio_text=audio_text,
        cfg=cfg,
    )

    return {
        "shot_id": shot["shot_id"],
        "caption_text": caption_text,
        "temporal_caption": temporal_caption,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "caption_mode": "template",
        "caption_model": "",
        "prompt_version": "",
        "used_ocr": used_ocr,
        "used_audio": used_audio,
        "used_objects": used_objects,
        "used_scene": used_scene,
        "used_frame_captions": used_frame_captions,
        "fallback_used": bool(warnings),
        "warnings": warnings,
    }


def _object_counts_to_text(counts: dict[str, int], max_labels: int) -> str:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_labels]
    return ", ".join(f"{cnt} {label}" for label, cnt in items if cnt > 0)


def _join_limited(parts: list[str], max_chars: int) -> str:
    text = ""
    for part in parts:
        candidate = normalize_whitespace(f"{text} {part}" if text else part)
        if len(candidate) > max_chars and text:
            break
        text = candidate
        if len(text) >= max_chars:
            break
    return truncate(text, max_chars)


def _format_list_phrase(items: list[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + " and " + items[-1]


def _temporal_position(idx: int, num_shots: int) -> str:
    if idx == 0:
        return "At the beginning of the video,"
    if idx == num_shots - 1:
        return "Near the end,"
    return "Then,"


def _build_memory_after(
    memory_before: str,
    scene_text: str,
    object_text: str,
    ocr_text: str,
    audio_text: str,
    cfg: PipelineConfig,
) -> str:
    pieces: list[str] = []
    if memory_before:
        pieces.append(memory_before)
    if scene_text:
        pieces.append(f"scene: {_strip_terminal_punctuation(scene_text)}")
    if object_text:
        pieces.append(f"objects: {_strip_terminal_punctuation(object_text)}")
    if ocr_text:
        pieces.append(f"ocr: {_strip_terminal_punctuation(truncate(ocr_text, 120))}")
    if audio_text:
        pieces.append(f"audio: {_strip_terminal_punctuation(truncate(audio_text, 180))}")
    return _finalize_caption("; ".join(pieces), cfg.shot_caption.max_memory_chars)


def _strip_terminal_punctuation(text: str) -> str:
    return normalize_whitespace(text).rstrip(".,;:!?")


def _finalize_caption(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = truncate(text, max_chars).rstrip(" ,;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text
