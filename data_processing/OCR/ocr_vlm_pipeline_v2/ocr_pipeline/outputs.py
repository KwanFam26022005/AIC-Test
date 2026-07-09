"""
Output generation for OCR pipeline results.

Handles:
- Building group clean/review text aggregation
- Collecting final text outputs
- Exporting CSV (line-level, group-level)
- Exporting ES document JSON
- Exporting clean/review TXT
- Visualization with bounding boxes
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .structural_filter import classify_noise_text

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "ocr_vlm_pipeline_v2_es_2"


# ── Text aggregation ─────────────────────────────────────────────────


def build_group_text_clean_review(
    groups: list[dict],
    line_items: list[dict],
    cfg: dict,
) -> None:
    """Build group_text_clean and group_text_review from line results.

    CRITICAL RULE: group_text = group_text_clean, NEVER fallback to review.

    Modifies groups in place.
    """
    for group in groups:
        clean_lines = []
        review_lines = []
        num_keep = 0
        num_filtered = 0
        num_vlm = 0
        num_vintern = 0
        det_scores = []
        composite_scores = []
        rec_confs = []
        priorities = []

        for idx in group.get("line_indices", []):
            if idx >= len(line_items):
                continue
            line = line_items[idx]

            det_scores.append(line.get("det_score", 0.0))
            composite_scores.append(line.get("composite_score", 0.0))
            if line.get("rec_conf") is not None:
                rec_confs.append(float(line["rec_conf"]))
            priorities.append(line.get("priority", 0.0))

            if line.get("is_filtered", False):
                num_filtered += 1
                continue

            if line.get("send_to_vintern", False):
                num_vlm += 1
            if line.get("vintern_text") is not None:
                num_vintern += 1

            final_text = (line.get("final_text") or "").strip()
            noise_reason = classify_noise_text(final_text, cfg)
            if noise_reason:
                line["post_filter_status"] = "text_noise_filter"
                line["noise_reason"] = noise_reason
                line["is_filtered"] = True
                line["filter_status"] = "text_noise_filter"
                line["send_to_vintern"] = False
                line["keep_for_index"] = False
                line["need_review"] = False
                line["final_source"] = "text_noise_filter"
                num_filtered += 1
                continue

            if line.get("keep_for_index", False) and not line.get("need_review", False) and final_text:
                clean_lines.append(final_text)
                num_keep += 1
            elif final_text:
                review_lines.append(final_text)

        # IMPORTANT: group_text = group_text_clean, NOT review fallback
        group["group_text_clean"] = "\n".join(clean_lines)
        group["group_text_review"] = "\n".join(review_lines)
        group["group_text"] = group["group_text_clean"]
        group["group_text_lines"] = clean_lines
        group["group_review_lines"] = review_lines

        group["num_keep_lines"] = num_keep
        group["num_filtered_lines"] = num_filtered
        group["num_vlm_candidates"] = num_vlm
        group["num_vintern_lines"] = num_vintern

        group["mean_det_score"] = float(np.mean(det_scores)) if det_scores else 0.0
        group["mean_composite_score"] = float(np.mean(composite_scores)) if composite_scores else 0.0
        group["mean_rec_conf"] = float(np.mean(rec_confs)) if rec_confs else 0.0

        # Group VLM priority
        mean_priority = float(np.mean(priorities)) if priorities else 0.0
        bonus = 0.0
        if group.get("is_multiline_group", False):
            bonus += cfg.get("group_priority_bonus_multiline", 0.25)
        if num_vlm > 0:
            bonus += cfg.get("group_priority_bonus_vlm_candidate", 0.20)
        if any(line_items[idx].get("need_review", False)
               for idx in group.get("line_indices", [])
               if idx < len(line_items)):
            bonus += cfg.get("group_priority_bonus_need_review", 0.15)
        group["group_vlm_priority"] = mean_priority + bonus


def collect_final_texts(groups: list[dict]) -> tuple[str, str]:
    """Collect final clean and review text from all groups.

    Returns:
        (final_clean_text, final_review_text)
    """
    clean_parts = []
    review_parts = []

    for group in groups:
        clean = (group.get("group_text_clean") or "").strip()
        review = (group.get("group_text_review") or "").strip()
        if clean:
            clean_parts.append(clean)
        if review:
            review_parts.append(review)

    final_clean = "\n\n".join(clean_parts)
    final_review = "\n\n".join(review_parts)
    return final_clean, final_review


# ── CSV export ────────────────────────────────────────────────────────


def export_csv_lines(
    line_items: list[dict],
    output_path: str | Path,
) -> str:
    """Export line-level results to CSV."""
    columns = [
        "line_id", "group_id", "vietocr_text", "rec_conf", "rec_conf_source",
        "rec_conf_flat_run", "det_score", "composite_score", "quality_score",
        "lex_ratio", "diacritic_susp", "diacritic_suspicious_tokens",
        "oov_tokens", "charset_penalty", "repetition_penalty",
        "priority", "filter_status", "send_to_vintern",
        "vintern_text", "vintern_composite_score", "agreement_similarity",
        "final_text", "final_source", "keep_for_index", "need_review",
        "post_filter_status", "noise_reason", "vintern_noise_reason",
        "persp_crop_path", "group_crop_path", "bbox_xyxy",
    ]

    rows = []
    for line in line_items:
        row = {}
        for col in columns:
            val = line.get(col)
            if isinstance(val, (list, dict)):
                row[col] = str(val)
            else:
                row[col] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Exported line CSV: {path} ({len(rows)} rows)")
    return str(path)


def export_csv_groups(
    groups: list[dict],
    output_path: str | Path,
) -> str:
    """Export group-level results to CSV."""
    columns = [
        "reading_order", "group_id", "num_lines",
        "group_text", "group_text_clean", "group_text_review",
        "group_vintern_text", "group_vintern_composite_score",
        "group_vintern_agreement_similarity",
        "group_final_source", "group_keep_for_index", "need_review",
        "group_vintern_noise_reason",
        "group_crop_path", "bbox_xyxy",
        "mean_det_score", "mean_composite_score",
        "num_keep_lines", "num_vlm_candidates",
    ]

    rows = []
    for group in groups:
        row = {}
        for col in columns:
            val = group.get(col)
            if isinstance(val, (list, dict)):
                row[col] = str(val)
            else:
                row[col] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Exported group CSV: {path} ({len(rows)} rows)")
    return str(path)


# ── ES document ──────────────────────────────────────────────────────


def export_es_document(
    line_items: list[dict],
    groups: list[dict],
    clean_text: str,
    review_text: str,
    cfg: dict,
    image_path: str,
    output_path: str | Path,
    timing: dict | None = None,
) -> str:
    """Export ES-compatible JSON document."""
    doc = build_es_document(line_items, groups, clean_text, review_text, cfg, image_path, timing)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info(f"Exported ES document: {path}")
    return str(path)


def build_es_document(
    line_items: list[dict],
    groups: list[dict],
    clean_text: str,
    review_text: str,
    cfg: dict,
    image_path: str,
    timing: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build an ES-friendly frame OCR document.

    The top-level search fields are intentionally denormalized:
    - ocr_text_search: clean/indexable text only by default
    - ocr_text_unaccent: accent-stripped search helper for Vietnamese fuzzy search
    - ocr_terms: normalized unique tokens for lazy autocomplete/filtering

    Line/group arrays keep evidence for UI inspection and debugging.
    """
    image = Path(image_path)
    metadata = metadata or {}
    frame_name = str(metadata.get("frame_name") or image.stem)
    frame_number = _parse_frame_number(frame_name)
    keyframe_idx = metadata.get("keyframe_idx")
    if keyframe_idx is None:
        keyframe_idx = frame_number
    video_id = str(metadata.get("video_id") or image.parent.name)
    canonical_frame_id = str(metadata.get("canonical_frame_id") or f"{video_id}_{frame_name}")
    legacy_frame_id = frame_name

    clean_text = (clean_text or "").strip()
    review_text = (review_text or "").strip()
    line_texts = [
        (line.get("final_text") or line.get("vietocr_text") or "").strip()
        for line in line_items
        if line.get("keep_for_index")
        and not line.get("need_review")
        and (line.get("final_text") or line.get("vietocr_text") or "").strip()
        and not classify_noise_text((line.get("final_text") or line.get("vietocr_text") or ""), cfg)
    ]
    group_texts = [
        (group.get("group_text_clean") or "").strip()
        for group in groups
        if isinstance(group.get("group_text_clean"), str)
        and group.get("group_text_clean", "").strip()
        and not classify_noise_text(group.get("group_text_clean", ""), cfg)
    ]
    raw_evidence_parts = [clean_text, *group_texts, *line_texts]
    evidence_parts = _dedupe_text_parts(raw_evidence_parts)
    search_parts = [clean_text]
    if cfg.get("es_include_review_in_search", False):
        search_parts.append(review_text)
    search_parts_clean = [part for part in search_parts if _normalize_space(part)]
    deduped_search_parts = _dedupe_text_parts(search_parts_clean)
    search_text = _normalize_space("\n".join(deduped_search_parts))
    search_unaccent = _strip_accents(search_text)
    terms = _extract_terms(search_unaccent)
    review_included = bool(cfg.get("es_include_review_in_search", False))
    search_duplicate_count = max(0, len(search_parts_clean) - len(deduped_search_parts))
    evidence_duplicate_count = max(0, len([p for p in raw_evidence_parts if _normalize_space(p)]) - len(evidence_parts))

    quality = {
        "lines_detected": len(line_items),
        "groups_formed": len(groups),
        "clean_chars": len(clean_text),
        "review_chars": len(review_text),
        "has_clean_text": bool(clean_text),
        "has_review_text": bool(review_text),
        "num_keep_lines": sum(1 for line in line_items if line.get("keep_for_index")),
        "num_primary_keep_lines": sum(
            1 for line in line_items
            if line.get("keep_for_index") and not line.get("need_review")
        ),
        "num_review_lines": sum(1 for line in line_items if line.get("need_review")),
        "num_search_lines": len([text for text in line_texts if text]),
        "num_search_duplicates": search_duplicate_count,
        "num_evidence_duplicates": evidence_duplicate_count,
        "review_included_in_search": review_included,
        "num_need_review_lines_in_primary_search": 0 if not review_included else sum(
            1 for line in line_items
            if line.get("need_review") and line.get("keep_for_index")
        ),
        "num_noise_filtered_lines": sum(1 for line in line_items if line.get("post_filter_status") == "text_noise_filter"),
        "num_vintern_lines": sum(1 for line in line_items if line.get("vintern_text")),
        "num_vlm_candidates": sum(1 for line in line_items if line.get("send_to_vintern")),
        "gating_stats": _count_values(line.get("filter_status", "unknown") for line in line_items),
        "final_source_stats": _count_values(line.get("final_source", "unknown") for line in line_items),
    }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "document_id": f"ocr:{canonical_frame_id}",
        "video_id": video_id,
        "frame_id": canonical_frame_id,
        "canonical_frame_id": canonical_frame_id,
        "legacy_frame_id": legacy_frame_id,
        "frame_name": frame_name,
        "frame_number": frame_number,
        "keyframe_idx": keyframe_idx,
        "source_frame_idx": metadata.get("source_frame_idx"),
        "timestamp_sec": metadata.get("timestamp_sec"),
        "timestamp_source": metadata.get("timestamp_source", "none"),
        "image_path": str(image_path),
        "image_relpath": metadata.get("image_relpath") or f"{video_id}/{image.name}",
        "media": {
            "video_id": video_id,
            "frame_id": canonical_frame_id,
            "canonical_frame_id": canonical_frame_id,
            "legacy_frame_id": legacy_frame_id,
            "frame_name": frame_name,
            "frame_number": frame_number,
            "keyframe_idx": keyframe_idx,
            "source_frame_idx": metadata.get("source_frame_idx"),
            "timestamp_sec": metadata.get("timestamp_sec"),
            "image_path": str(image_path),
            "image_relpath": metadata.get("image_relpath") or f"{video_id}/{image.name}",
        },
        "ocr_pipeline": {
            "detector": cfg.get("det_model_name", "PP-OCRv6_medium_det"),
            "detector_device": os.environ.get("OCR_V2_PADDLE_DEVICE") or cfg.get("paddle_device"),
            "detector_engine": os.environ.get("OCR_V2_PADDLE_ENGINE") or cfg.get("paddle_engine", "paddle_static"),
            "line_recognizer": f"VietOCR/{cfg.get('vietocr_config', 'vgg_transformer')}",
            "fallback_vlm_line": cfg.get("vintern_model_id", "5CD-AI/Vintern-1B-v3_5"),
            "fallback_vlm_group": cfg.get("vintern_model_id", "5CD-AI/Vintern-1B-v3_5"),
            "wordlist_enabled": True,
            "group_text_rule": "group_text_clean only; group_text_review is not indexed",
            "search_text_rule": "ocr_text_search uses clean_text only unless es_include_review_in_search=true",
        },
        "timing": timing or {},
        "quality": quality,
        "ocr_text_clean": clean_text,
        "ocr_text_review": review_text,
        "ocr_text_search": search_text,
        "ocr_text_unaccent": search_unaccent,
        "ocr_terms": terms,
        "ocr_text_evidence": _normalize_space("\n".join(evidence_parts)),
        "all_ocr_text": _normalize_space("\n".join(_dedupe_text_parts([clean_text, review_text]))),
        "ocr_group_texts_clean": [g.get("group_text_clean", "") for g in groups],
        "ocr_group_texts_review": [g.get("group_text_review", "") for g in groups],
        "ocr_lines": _serialize_lines(line_items),
        "ocr_groups": _serialize_groups(groups),
    }
    return _safe_json_value(doc)


def export_es_jsonl(docs: list[dict], output_path: str | Path) -> str:
    """Export one JSON document per line for ES bulk preparation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(_safe_json_value(doc), ensure_ascii=False))
            f.write("\n")
    logger.info(f"Exported ES JSONL: {path} ({len(docs)} docs)")
    return str(path)


def _serialize_lines(line_items: list[dict]) -> list[dict]:
    """Serialize line items for JSON export."""
    safe_keys = [
        "line_id", "group_id", "vietocr_text", "rec_conf", "rec_conf_source",
        "det_score", "composite_score", "quality_score", "lex_ratio",
        "diacritic_susp", "diacritic_suspicious_tokens", "oov_tokens",
        "charset_penalty", "repetition_penalty", "priority",
        "filter_status", "send_to_vintern", "vintern_text",
        "vintern_composite_score", "agreement_similarity",
        "final_text", "final_source", "keep_for_index", "need_review",
        "post_filter_status", "noise_reason", "vintern_noise_reason",
        "bbox_xyxy", "persp_crop_path", "group_crop_path",
    ]
    result = []
    for line in line_items:
        entry = {}
        for key in safe_keys:
            val = line.get(key)
            entry[key] = _safe_json_value(val)
        result.append(entry)
    return result


def _serialize_groups(groups: list[dict]) -> list[dict]:
    """Serialize groups for JSON export."""
    safe_keys = [
        "group_id", "reading_order", "num_lines", "line_indices",
        "group_text_clean", "group_text_review", "group_text",
        "group_vintern_text", "group_vintern_composite_score",
        "group_vintern_agreement_similarity", "group_final_source",
        "group_keep_for_index", "need_review", "bbox_xyxy",
        "group_vintern_noise_reason",
        "group_crop_path", "mean_det_score", "mean_composite_score",
        "mean_rec_conf", "num_keep_lines", "num_filtered_lines",
        "num_vlm_candidates", "num_vintern_lines", "group_vlm_priority",
    ]
    result = []
    for group in groups:
        entry = {}
        for key in safe_keys:
            val = group.get(key)
            entry[key] = _safe_json_value(val)
        result.append(entry)
    return result


def _parse_frame_number(frame_id: str) -> int | None:
    try:
        return int(frame_id)
    except ValueError:
        match = re.search(r"(\d+)$", frame_id)
        return int(match.group(1)) if match else None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _dedupe_text_parts(parts: list[str]) -> list[str]:
    out = []
    seen = set()
    for part in parts:
        clean = _normalize_space(part)
        if not clean:
            continue
        key = _strip_accents(clean)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return _normalize_space(without_marks.casefold())


def _extract_terms(text: str, max_terms: int = 512) -> list[str]:
    terms = []
    seen = set()
    for term in re.findall(r"[\w]+", text or "", flags=re.UNICODE):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    return value


# ── Text export ──────────────────────────────────────────────────────


def export_clean_text(text: str, output_path: str | Path) -> str:
    """Export clean text to TXT file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info(f"Exported clean text: {path}")
    return str(path)


def export_review_text(text: str, output_path: str | Path) -> str:
    """Export review text to TXT file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info(f"Exported review text: {path}")
    return str(path)


# ── Visualization ────────────────────────────────────────────────────


# Color scheme for different statuses
_STATUS_COLORS = {
    "auto_accept": (0, 200, 0),       # Green
    "vlm_candidate": (255, 165, 0),    # Orange
    "structural_filter": (128, 128, 128),  # Gray
    "review_only": (255, 255, 0),      # Yellow
    "text_noise_filter": (90, 90, 90),  # Dark gray
}

_SOURCE_COLORS = {
    "vintern_line_agree": (0, 200, 255),     # Cyan
    "vintern_line_override": (255, 0, 200),  # Magenta
    "vintern_group_fallback": (200, 0, 255), # Purple
}


def export_visualization(
    img_rgb: np.ndarray,
    line_items: list[dict],
    groups: list[dict],
    output_path: str | Path,
) -> str:
    """Export visualization with bounding boxes and annotations.

    Color coding:
    - Green: auto_accept
    - Orange: vlm_candidate
    - Gray: structural_filter
    - Yellow: review_only
    - Cyan/Magenta: Vintern results
    """
    vis = img_rgb.copy()
    H, W = vis.shape[:2]

    for line in line_items:
        bbox = line.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # Determine color
        status = line.get("filter_status", "")
        source = line.get("final_source", "")

        if source in _SOURCE_COLORS:
            color = _SOURCE_COLORS[source]
        elif status in _STATUS_COLORS:
            color = _STATUS_COLORS[status]
        else:
            color = (200, 200, 200)

        # Draw bbox
        thickness = 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        text = line.get("final_text", "")
        score = line.get("composite_score", 0.0)
        label = f"{text[:30]}  [{score:.2f}]" if text else f"[{score:.2f}]"

        font_scale = 0.4
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Background for text
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Draw group bboxes
    for group in groups:
        bbox = group.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        color = (100, 100, 255)  # Light blue for groups
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        group_id = group.get("group_id", -1)
        cv2.putText(vis, f"G{group_id}", (x1, y2 + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    logger.info(f"Exported visualization: {path}")
    return str(path)
