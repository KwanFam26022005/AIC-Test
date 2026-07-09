"""Evidence builder — convert aligned frame dicts into FrameEvidence JSONL rows.

Implements §5.1 (FrameEvidence schema) and §8 (evidence builder rules)
from the code plan.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import PipelineConfig
from .io_utils import round_sec
from .schemas import FRAME_EVIDENCE_SCHEMA
from .text_utils import join_texts, truncate

logger = logging.getLogger(__name__)


def build_frame_evidence(
    aligned_frames: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """Build FrameEvidence records from aligned frame dicts.

    Each returned dict follows the ``caption_frame_evidence_v1`` schema.
    """
    eb = cfg.evidence_builder
    results: list[dict] = []

    for af in aligned_frames:
        obj: dict | None = af.get("object")
        ocr: dict | None = af.get("ocr")
        audio_segs: list[dict] = af.get("audio_segments") or []
        audio_window: list[float] = af.get("audio_window") or [0.0, 0.0]

        # ----- Object evidence -----
        if obj is not None:
            obj_counts = obj.get("object_counts") or {}
            # Cap objects
            capped_counts = _cap_counts(obj_counts, eb.max_objects)
            obj_tags = (obj.get("object_tags") or [])[:eb.max_object_tags]
            scene_tags = (obj.get("scene_tags") or [])[:eb.max_scene_tags]
            ram_tags = obj.get("ram_tags") or []
            important = obj.get("important_objects") or []
            obj_text = obj.get("object_text", "") or ""
            scene_text = obj.get("scene_text", "") or ""
            ram_tag_text = obj.get("ram_tag_text", "") or ""
            all_obj_text = obj.get("all_object_text", "") or ""

            object_evidence: dict[str, Any] = {
                "object_counts": capped_counts,
                "object_tags": obj_tags,
                "scene_tags": scene_tags,
                "ram_tags": ram_tags,
                "important_objects": important,
                "object_text": obj_text,
                "scene_text": scene_text,
                "all_object_text": all_obj_text,
            }
            has_object = bool(capped_counts)
            has_scene = bool(scene_tags)
        else:
            object_evidence = {
                "object_counts": {},
                "object_tags": [],
                "scene_tags": [],
                "ram_tags": [],
                "important_objects": [],
                "object_text": "",
                "scene_text": "",
                "all_object_text": "",
            }
            obj_text = ""
            scene_text = ""
            ram_tag_text = ""
            has_object = False
            has_scene = False

        # ----- OCR evidence -----
        if ocr is not None:
            ocr_text = truncate(ocr.get("ocr_text", "") or "", eb.max_ocr_chars)
            ocr_terms = ocr.get("ocr_terms") or []
            has_clean = ocr.get("has_clean_text", bool(ocr_text))
            review_text = ocr.get("ocr_text_review", "") or ""
            review_used = False  # §6.1: review never used in primary
        else:
            ocr_text = ""
            ocr_terms = []
            has_clean = False
            review_text = ""
            review_used = False

        ocr_evidence: dict[str, Any] = {
            "ocr_text": ocr_text,
            "ocr_terms": ocr_terms,
            "has_clean_text": has_clean,
            "review_text": review_text if eb.include_ocr_review else "",
            "review_used": review_used,
        }
        has_ocr = bool(ocr_text)

        # ----- Audio evidence -----
        audio_texts: list[str] = []
        for seg in audio_segs:
            t = (seg.get("caption_text") or "").strip()
            if t:
                audio_texts.append(t)
        audio_text = truncate(
            " ".join(audio_texts), eb.max_audio_chars_frame,
        )
        has_audio = bool(audio_text)

        audio_evidence: dict[str, Any] = {
            "window": audio_window,
            "segments": [
                {
                    "feature_id": s.get("feature_id", ""),
                    "start_sec": s.get("start_sec", 0.0),
                    "end_sec": s.get("end_sec", 0.0),
                    "caption_text": s.get("caption_text", ""),
                }
                for s in audio_segs
            ],
            "audio_text": audio_text,
            "num_segments": len(audio_segs),
        }

        # ----- Quality block -----
        warnings: list[str] = []
        if af.get("timestamp_mismatch"):
            warnings.append("timestamp_mismatch_ocr_object")
        if obj is None:
            warnings.append("missing_object_data")

        quality: dict[str, Any] = {
            "has_object": has_object,
            "has_scene_tags": has_scene,
            "has_ocr": has_ocr,
            "has_audio": has_audio,
            "alignment_warnings": warnings,
        }

        # ----- Frame search text  (§8) -----
        frame_search_text = join_texts(
            ocr_text,
            audio_text,
            obj_text,
            scene_text,
            ram_tag_text,
        )

        # ----- Assemble FrameEvidence -----
        fe: dict[str, Any] = {
            "schema_version": FRAME_EVIDENCE_SCHEMA,
            "video_id": af.get("video_id", cfg.video_id),
            "frame_id": af.get("frame_id", af["canonical_frame_id"]),
            "canonical_frame_id": af["canonical_frame_id"],
            "frame_name": af.get("frame_name", ""),
            "keyframe_idx": af.get("keyframe_idx"),
            "source_frame_idx": af.get("source_frame_idx"),
            "timestamp_sec": round_sec(af.get("timestamp_sec")),
            "image_path": af.get("image_path", ""),
            "image_relpath": af.get("image_relpath", ""),
            "object_evidence": object_evidence,
            "ocr_evidence": ocr_evidence,
            "audio_evidence": audio_evidence,
            "quality": quality,
            "frame_search_text": frame_search_text,
        }
        results.append(fe)

    logger.info("Evidence builder: %d frame evidence records built", len(results))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap_counts(counts: dict[str, int], max_objects: int) -> dict[str, int]:
    """Keep at most *max_objects* object labels, sorted by count desc."""
    if len(counts) <= max_objects:
        return dict(counts)
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return dict(sorted_items[:max_objects])
