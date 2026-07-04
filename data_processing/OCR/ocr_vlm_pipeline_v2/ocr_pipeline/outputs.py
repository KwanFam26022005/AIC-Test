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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

            if line.get("keep_for_index", False) and final_text:
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
    from ..scoring.wordlist import load_wordlist

    frame_id = Path(image_path).stem

    doc = {
        "frame_id": frame_id,
        "image_path": str(image_path),
        "ocr_pipeline": {
            "detector": cfg.get("det_model_name", "PP-OCRv6_medium_det"),
            "line_recognizer": f"VietOCR/{cfg.get('vietocr_config', 'vgg_transformer')}",
            "fallback_vlm_line": cfg.get("vintern_model_id", "5CD-AI/Vintern-1B-v3_5"),
            "fallback_vlm_group": cfg.get("vintern_model_id", "5CD-AI/Vintern-1B-v3_5"),
            "wordlist_enabled": True,
            "group_text_rule": "group_text_clean only; group_text_review is not indexed",
        },
        "timing": timing or {},
        "ocr_text_clean": clean_text,
        "ocr_text_review": review_text,
        "ocr_group_texts_clean": [g.get("group_text_clean", "") for g in groups],
        "ocr_group_texts_review": [g.get("group_text_review", "") for g in groups],
        "ocr_lines": _serialize_lines(line_items),
        "ocr_groups": _serialize_groups(groups),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    logger.info(f"Exported ES document: {path}")
    return str(path)


def _serialize_lines(line_items: list[dict]) -> list[dict]:
    """Serialize line items for JSON export."""
    safe_keys = [
        "line_id", "group_id", "vietocr_text", "rec_conf", "rec_conf_source",
        "det_score", "composite_score", "quality_score", "lex_ratio",
        "diacritic_susp", "charset_penalty", "repetition_penalty",
        "filter_status", "send_to_vintern", "vintern_text",
        "vintern_composite_score", "agreement_similarity",
        "final_text", "final_source", "keep_for_index", "need_review",
        "bbox_xyxy",
    ]
    result = []
    for line in line_items:
        entry = {}
        for key in safe_keys:
            val = line.get(key)
            if isinstance(val, np.generic):
                val = val.item()
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            entry[key] = val
        result.append(entry)
    return result


def _serialize_groups(groups: list[dict]) -> list[dict]:
    """Serialize groups for JSON export."""
    safe_keys = [
        "group_id", "reading_order", "num_lines", "line_indices",
        "group_text_clean", "group_text_review", "group_text",
        "group_vintern_text", "group_vintern_composite_score",
        "group_final_source", "group_keep_for_index", "need_review",
        "bbox_xyxy", "mean_det_score", "mean_composite_score",
        "num_keep_lines", "num_vlm_candidates",
    ]
    result = []
    for group in groups:
        entry = {}
        for key in safe_keys:
            val = group.get(key)
            if isinstance(val, np.generic):
                val = val.item()
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            entry[key] = val
        result.append(entry)
    return result


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
