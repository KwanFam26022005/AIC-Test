"""Frame propagator — produce canonical frame-caption rows for every keyframe.

Phase 3 (§9): every valid keyframe produces exactly one canonical
frame-caption row, regardless of whether it was directly captioned or
propagated from a selected frame.

Modes:
  vlm_generated  — generated from this exact image
  propagated     — copied from a selected frame in the same shot
  fallback       — deterministic emergency output after failure
"""

from __future__ import annotations

import logging
from typing import Any

from .schemas import FRAME_CAPTION_SCHEMA, PIPELINE_VERSION

logger = logging.getLogger(__name__)


def propagate_captions(
    frames: list[dict],
    selections: list[dict],
    vlm_results: dict[str, dict],
    run_id: str,
    video_id: str,
    vlm_config: dict[str, Any],
    prompt_version: str = "",
    prompt_hash: str = "",
) -> list[dict[str, Any]]:
    """Produce canonical frame-caption rows for every keyframe.

    Args:
        frames: All keyframe dicts (sorted by timestamp).
        selections: Frame selection records from frame_selector.
        vlm_results: Dict mapping canonical_frame_id -> VLM result dict.
        run_id: Current run identifier.
        video_id: Video identifier.
        vlm_config: VLM config dict for generation metadata.
        prompt_version: Prompt version string.
        prompt_hash: Prompt hash string.

    Returns:
        List of canonical frame-caption dicts (caption_frame_v2 schema).
    """
    # Index selections by canonical_frame_id
    sel_by_id: dict[str, dict] = {}
    for sel in selections:
        sel_by_id[sel["canonical_frame_id"]] = sel

    results: list[dict[str, Any]] = []
    stats = {"vlm_generated": 0, "propagated": 0, "fallback": 0}

    for frame in frames:
        cfid = frame["canonical_frame_id"]
        sel = sel_by_id.get(cfid, {})
        is_selected = sel.get("selected_for_vlm", False)
        role = sel.get("selection_role", "propagated")
        source_frame_id = sel.get("source_frame_id")
        similarity_method = sel.get("similarity_method", "phash")
        similarity_score = sel.get("similarity_score")
        novelty_score = sel.get("novelty_score")

        if is_selected and cfid in vlm_results:
            # VLM-generated caption
            vlm = vlm_results[cfid]
            caption_text = vlm.get("text", "").strip()
            validation_warnings = vlm.get("validation_warnings", [])

            if caption_text and not any(
                w.startswith("empty") for w in validation_warnings
            ):
                mode = "vlm_generated"
                stats["vlm_generated"] += 1
                generation = _build_generation(
                    vlm, vlm_config, prompt_version, prompt_hash,
                )
            else:
                # Fallback for failed VLM
                mode = "fallback"
                caption_text = _fallback_caption(cfid)
                validation_warnings.append("vlm_empty_fallback")
                stats["fallback"] += 1
                generation = None

        elif source_frame_id and source_frame_id in vlm_results:
            # Propagated from a selected frame
            source_vlm = vlm_results[source_frame_id]
            caption_text = source_vlm.get("text", "").strip()

            if caption_text:
                mode = "propagated"
                stats["propagated"] += 1
                generation = None
                validation_warnings = []
            else:
                mode = "fallback"
                caption_text = _fallback_caption(cfid)
                validation_warnings = ["source_empty_fallback"]
                stats["fallback"] += 1
                generation = None

        else:
            # No source available — fallback
            mode = "fallback"
            caption_text = _fallback_caption(cfid)
            validation_warnings = ["no_source_fallback"]
            stats["fallback"] += 1
            generation = None

        # Word / char counts
        words = caption_text.split() if caption_text else []
        word_count = len(words)
        char_count = len(caption_text)

        # Quality assessment
        quality_level = "high"
        if mode == "fallback":
            quality_level = "fallback"
        elif validation_warnings:
            quality_level = "warning"

        row: dict[str, Any] = {
            "schema_version": FRAME_CAPTION_SCHEMA,
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "video_id": video_id,
            "canonical_frame_id": cfid,
            "keyframe_idx": frame.get("keyframe_idx", 0),
            "source_frame_idx": frame.get("source_frame_idx"),
            "timestamp_sec": frame["timestamp_sec"],
            "shot_id": sel.get("shot_id", ""),
            "image_relpath": frame.get("image_relpath", ""),
            "selection": {
                "role": role,
                "source_frame_id": source_frame_id,
                "similarity_method": similarity_method,
                "similarity_score": similarity_score,
                "novelty_score": novelty_score,
            },
            "caption": {
                "text": caption_text,
                "language": "en",
                "visual_only": True,
                "mode": mode,
            },
            "generation": generation,
            "quality": {
                "valid": mode != "fallback",
                "quality_level": quality_level,
                "word_count": word_count,
                "char_count": char_count,
                "warnings": validation_warnings,
            },
            "provenance": {
                "image_sha256": "",  # Populated by runner if needed
                "input_signature": "",  # Populated by runner
            },
            "created_at": "",  # Populated by runner
        }
        results.append(row)

    logger.info(
        "Frame propagator: %d total (%d vlm_generated, %d propagated, %d fallback)",
        len(results), stats["vlm_generated"], stats["propagated"], stats["fallback"],
    )
    return results


def _build_generation(
    vlm_result: dict,
    vlm_config: dict[str, Any],
    prompt_version: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """Build the generation metadata block."""
    return {
        "provider": vlm_config.get("provider", "transformers"),
        "model_id": vlm_config.get("model_id", ""),
        "model_revision": vlm_config.get("model_revision", ""),
        "prompt_version": prompt_version,
        "prompt_hash": f"sha256:{prompt_hash}" if prompt_hash else "",
        "attempts": vlm_result.get("attempts", 1),
        "max_visual_tokens": vlm_config.get("max_visual_tokens", 1280),
        "max_new_tokens": vlm_config.get("max_new_tokens", 64),
        "input_tokens": vlm_result.get("input_tokens"),
        "output_tokens": vlm_result.get("output_tokens"),
        "elapsed_ms": vlm_result.get("elapsed_ms", 0),
        "batch_id": vlm_result.get("batch_id", ""),
    }


def _fallback_caption(canonical_frame_id: str) -> str:
    """Generate a deterministic fallback caption."""
    return f"[Visual content of frame {canonical_frame_id}]"
