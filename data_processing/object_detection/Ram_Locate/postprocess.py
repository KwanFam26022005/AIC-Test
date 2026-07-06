# -*- coding: utf-8 -*-
"""Postprocessing for LocateAnything detections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


LABEL_NORMALIZE_MAP = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "people": "person",
    "human": "person",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "bike": "bicycle",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "baseball cap": "hat",
    "cap": "hat",
    "couch": "sofa",
    "stool": "chair",
}


@dataclass(frozen=True)
class PostprocessConfig:
    min_box_width_px: float = 4.0
    min_box_height_px: float = 4.0
    min_box_area_px: float = 16.0
    scene_area_threshold: float = 0.65
    nms_iou_threshold: float = 0.75
    repeated_box_iou_threshold: float = 0.98


def normalize_label(label: str) -> str:
    clean = str(label).lower().replace("_", " ").strip()
    clean = " ".join(clean.split())
    clean = LABEL_NORMALIZE_MAP.get(clean, clean)
    return clean.upper().replace(" ", "_")


def clamp_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def area_ratio(box: list[float], img_w: int, img_h: int) -> float:
    img_area = float(img_w * img_h)
    return box_area(box) / img_area if img_area > 0 else 0.0


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _is_valid_box(box: list[float], cfg: PostprocessConfig) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width < cfg.min_box_width_px or height < cfg.min_box_height_px:
        return False
    return box_area(box) >= cfg.min_box_area_px


def _dedupe_repeated_boxes(
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for det in detections:
        duplicate = False
        for existing in kept:
            if det["label"] == existing["label"] and compute_iou(det["box"], existing["box"]) >= iou_threshold:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(det)
    return kept, removed


def class_agnostic_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float,
    img_w: int,
    img_h: int,
) -> tuple[list[dict[str, Any]], int]:
    if not detections:
        return [], 0

    # LocateAnything has no calibrated score. Prefer smaller, more specific boxes.
    sorted_dets = sorted(
        detections,
        key=lambda d: (area_ratio(d["box"], img_w, img_h), d.get("raw_index", 0)),
    )

    kept: list[dict[str, Any]] = []
    removed = 0
    for det in sorted_dets:
        duplicate = any(compute_iou(det["box"], existing["box"]) >= iou_threshold for existing in kept)
        if duplicate:
            removed += 1
        else:
            kept.append(det)

    kept.sort(key=lambda d: d.get("raw_index", 0))
    return kept, removed


def postprocess_detections(
    raw_detections: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    cfg: PostprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    cfg = cfg or PostprocessConfig()
    stats = {
        "num_raw_boxes": len(raw_detections),
        "num_removed_malformed": 0,
        "num_removed_degenerate": 0,
        "num_removed_scene": 0,
        "num_removed_repeated": 0,
        "num_removed_nms": 0,
        "num_final_boxes": 0,
    }

    normalized: list[dict[str, Any]] = []
    scene_labels: list[str] = []

    for idx, det in enumerate(raw_detections):
        box = det.get("box")
        label = det.get("label")
        if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
            stats["num_removed_malformed"] += 1
            continue

        try:
            clean_box = clamp_box([float(v) for v in box], img_w, img_h)
        except (TypeError, ValueError):
            stats["num_removed_malformed"] += 1
            continue

        if not _is_valid_box(clean_box, cfg):
            stats["num_removed_degenerate"] += 1
            continue

        clean_label = normalize_label(str(label))
        ratio = area_ratio(clean_box, img_w, img_h)
        if ratio > cfg.scene_area_threshold:
            stats["num_removed_scene"] += 1
            scene_labels.append(clean_label.lower())
            continue

        normalized.append({
            "label": clean_label,
            "box": clean_box,
            "source": det.get("source", "locateanything"),
            "raw_label": str(label),
            "raw_index": int(det.get("raw_index", idx)),
            "area_ratio": round(ratio, 6),
        })

    deduped, removed_repeated = _dedupe_repeated_boxes(
        normalized,
        cfg.repeated_box_iou_threshold,
    )
    stats["num_removed_repeated"] = removed_repeated

    final_objects, removed_nms = class_agnostic_nms(
        deduped,
        cfg.nms_iou_threshold,
        img_w,
        img_h,
    )
    stats["num_removed_nms"] = removed_nms
    stats["num_final_boxes"] = len(final_objects)

    for obj in final_objects:
        obj.pop("raw_index", None)

    return final_objects, sorted(set(scene_labels)), stats


def summarize_objects(objects: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    counts = Counter(obj["label"] for obj in objects)
    return sorted(counts.keys()), dict(counts)

