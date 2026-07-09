"""Frame caption fuser - generate frame-level captions from evidence.

Implements sections 4.1 and 5 of the Phase 2 plan.

Two modes:
- ``template``: deterministic, search-friendly captions from evidence text.
- ``llm``: (placeholder) will send prompts to an LLM and parse JSON output.
  Falls back to template mode on failure.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import PipelineConfig
from .text_utils import normalize_whitespace, truncate

logger = logging.getLogger(__name__)

# Template caption truncation limits
_OCR_MAX = 160
_AUDIO_MAX = 220
_OBJECT_MAX_LABELS = 12
_RAM_MAX_TAGS = 10


def fuse_frame_captions(
    frame_evidence: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """Generate caption records for each frame evidence.

    Returns a list of dicts with:
    - ``canonical_frame_id``
    - ``caption_text``
    - ``caption_mode``
    - ``used_ocr``, ``used_audio``, ``used_objects``, ``used_scene``
    - ``fallback_used``
    - ``warnings``
    """
    mode = cfg.frame_caption.caption_mode
    results: list[dict] = []

    for fe in frame_evidence:
        if mode == "template":
            caption_rec = _template_caption(fe, cfg)
        else:
            # Future: LLM mode with fallback
            caption_rec = _template_caption(fe, cfg)
            caption_rec["caption_mode"] = "fallback"
            caption_rec["fallback_used"] = True
            caption_rec["warnings"].append("llm_mode_not_implemented_using_template")

        results.append(caption_rec)

    num_generated = sum(1 for r in results if r.get("caption_text"))
    num_empty = sum(1 for r in results if not r.get("caption_text"))
    num_fallback = sum(1 for r in results if r.get("fallback_used"))

    logger.info(
        "Frame fuser: %d captions (%d generated, %d empty, %d fallback)",
        len(results), num_generated, num_empty, num_fallback,
    )
    return results


# ---------------------------------------------------------------------------
# Template caption builder
# ---------------------------------------------------------------------------

def _template_caption(
    fe: dict,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Build a deterministic template caption from frame evidence."""
    obj_ev = fe.get("object_evidence") or {}
    ocr_ev = fe.get("ocr_evidence") or {}
    aud_ev = fe.get("audio_evidence") or {}

    parts: list[str] = []
    used_ocr = False
    used_audio = False
    used_objects = False
    used_scene = False
    warnings: list[str] = []

    # --- Scene phrase ---
    scene_tags: list[str] = obj_ev.get("scene_tags") or []
    ram_tags: list[str] = obj_ev.get("ram_tags") or []

    if scene_tags:
        scene_phrase = _format_list_phrase(scene_tags)
        parts.append(f"Scene shows {scene_phrase}.")
        used_scene = True
    elif ram_tags:
        short_rams = [t for t in ram_tags if len(t.split()) <= 2][:_RAM_MAX_TAGS]
        if short_rams:
            scene_phrase = _format_list_phrase(short_rams)
            parts.append(f"Scene shows {scene_phrase}.")
            used_scene = True

    # --- Object phrase ---
    important: list[str] = obj_ev.get("important_objects") or []
    if important:
        obj_labels = important[:_OBJECT_MAX_LABELS]
        obj_phrase = ", ".join(obj_labels)
        parts.append(f"Visible objects: {obj_phrase}.")
        used_objects = True

    # --- OCR phrase ---
    ocr_text = (ocr_ev.get("ocr_text") or "").strip()
    if ocr_text:
        ocr_truncated = _strip_terminal_punctuation(truncate(ocr_text, _OCR_MAX))
        parts.append(f'On-screen text: "{ocr_truncated}".')
        used_ocr = True

    # --- Audio phrase ---
    audio_text = (aud_ev.get("audio_text") or "").strip()
    if audio_text:
        audio_truncated = _strip_terminal_punctuation(truncate(audio_text, _AUDIO_MAX))
        parts.append(f"Spoken content: {audio_truncated}.")
        used_audio = True

    # --- Assemble ---
    caption_text = normalize_whitespace(" ".join(parts))
    caption_text = _finalize_caption(caption_text)

    if not caption_text:
        # Fallback: use frame_search_text if available
        fst = (fe.get("frame_search_text") or "").strip()
        if fst:
            caption_text = _finalize_caption(truncate(fst, 400))
            warnings.append("no_structured_evidence_fallback_search_text")
        else:
            warnings.append("empty_caption_no_evidence")

    return {
        "canonical_frame_id": fe["canonical_frame_id"],
        "caption_text": caption_text,
        "caption_mode": "template",
        "caption_model": "",
        "prompt_version": "",
        "used_ocr": used_ocr,
        "used_audio": used_audio,
        "used_objects": used_objects,
        "used_scene": used_scene,
        "fallback_used": bool(warnings),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_list_phrase(items: list[str]) -> str:
    """Format a list into 'a, b and c' style."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + " and " + items[-1]


def _strip_terminal_punctuation(text: str) -> str:
    """Remove trailing punctuation before adding a template period."""
    return text.strip().rstrip(".,;:!?")


def _finalize_caption(text: str, max_chars: int = 420) -> str:
    """Keep generated captions compact and cleanly punctuated."""
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = truncate(text, max_chars).rstrip(" ,;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text
