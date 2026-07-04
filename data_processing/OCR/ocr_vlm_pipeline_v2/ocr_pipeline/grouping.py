"""
Spatial grouping of detected text lines using Union-Find.

Groups nearby/stacked lines that likely belong to the same text block,
enabling context-aware Vintern group fallback.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Geometry helpers ──────────────────────────────────────────────────


def x_overlap_ratio(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Compute horizontal overlap ratio between two bboxes.

    Returns overlap / min(width_a, width_b).
    """
    x1_a, _, x2_a, _ = bbox_a
    x1_b, _, x2_b, _ = bbox_b

    overlap_start = max(x1_a, x1_b)
    overlap_end = min(x2_a, x2_b)
    overlap = max(0, overlap_end - overlap_start)

    width_a = max(1, x2_a - x1_a)
    width_b = max(1, x2_b - x1_b)

    return overlap / min(width_a, width_b)


def vertical_gap(bbox_a: list[int], bbox_b: list[int]) -> int:
    """Compute vertical gap between two bboxes (0 if overlapping)."""
    _, y1_a, _, y2_a = bbox_a
    _, y1_b, _, y2_b = bbox_b
    return max(0, max(y1_a, y1_b) - min(y2_a, y2_b))


def bbox_height(bbox: list[int]) -> int:
    """Get height of bbox [x1, y1, x2, y2]."""
    return max(1, bbox[3] - bbox[1])


def bbox_union(bboxes: list[list[int]]) -> list[int]:
    """Compute the union bounding box of multiple bboxes."""
    if not bboxes:
        return [0, 0, 0, 0]
    x1s = [b[0] for b in bboxes]
    y1s = [b[1] for b in bboxes]
    x2s = [b[2] for b in bboxes]
    y2s = [b[3] for b in bboxes]
    return [min(x1s), min(y1s), max(x2s), max(y2s)]


# ── Union-Find ────────────────────────────────────────────────────────


class UnionFind:
    """Disjoint set / Union-Find data structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ── Grouping logic ───────────────────────────────────────────────────


def should_group_lines(a: dict, b: dict, cfg: dict) -> bool:
    """Check if two line items should be grouped.

    Conditions (from plan section 8.1):
    - x_overlap_ratio >= group_min_x_overlap_ratio
    - vertical_gap <= avg_line_height * group_max_vertical_gap_ratio
    """
    bbox_a = a["bbox_xyxy"]
    bbox_b = b["bbox_xyxy"]

    x_ov = x_overlap_ratio(bbox_a, bbox_b)
    if x_ov < cfg.get("group_min_x_overlap_ratio", 0.12):
        return False

    y_gap = vertical_gap(bbox_a, bbox_b)
    avg_h = (bbox_height(bbox_a) + bbox_height(bbox_b)) / 2.0
    max_gap = avg_h * cfg.get("group_max_vertical_gap_ratio", 1.80)

    return y_gap <= max_gap


def group_line_items(line_items: list[dict], cfg: dict) -> list[dict]:
    """Group detected lines using Union-Find.

    Args:
        line_items: List of line_item dicts with bbox_xyxy
        cfg: Configuration dict

    Returns:
        List of group dicts, each with:
        - group_id, reading_order, line_indices, num_lines, is_multiline_group
        - bbox_xyxy (union of all line bboxes)
    """
    n = len(line_items)
    if n == 0:
        return []

    uf = UnionFind(n)

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            if should_group_lines(line_items[i], line_items[j], cfg):
                uf.union(i, j)

    # Collect components
    components: dict[int, list[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        components.setdefault(root, []).append(idx)

    # Build groups
    groups = []
    for component_indices in components.values():
        # Sort lines within group: top-to-bottom, then left-to-right
        component_indices.sort(key=lambda idx: (
            line_items[idx]["bbox_xyxy"][1],  # y1
            line_items[idx]["bbox_xyxy"][0],  # x1
        ))

        bboxes = [line_items[idx]["bbox_xyxy"] for idx in component_indices]
        group_bbox = bbox_union(bboxes)

        group = {
            "group_id": -1,  # assigned later
            "reading_order": -1,
            "line_indices": component_indices,
            "num_lines": len(component_indices),
            "is_multiline_group": len(component_indices) >= cfg.get("group_min_lines", 2),
            "bbox_xyxy": group_bbox,
            "group_crop_path": None,
            # Text fields filled later
            "group_text_clean": "",
            "group_text_review": "",
            "group_text": "",
            "group_text_lines": [],
            "group_review_lines": [],
            # Stats filled later
            "num_keep_lines": 0,
            "num_filtered_lines": 0,
            "num_vlm_candidates": 0,
            "num_vintern_lines": 0,
            "mean_det_score": 0.0,
            "mean_composite_score": 0.0,
            "mean_rec_conf": 0.0,
            "group_vlm_priority": 0.0,
            # Vintern group fields
            "group_vintern_text": None,
            "group_vintern_composite_score": None,
            "group_vintern_agreement_similarity": None,
            "group_final_source": None,
            "group_keep_for_index": None,
            "need_review": False,
        }
        groups.append(group)

    # Assign reading order: sort groups by (y_center, x_center)
    groups.sort(key=lambda g: (
        (g["bbox_xyxy"][1] + g["bbox_xyxy"][3]) / 2,  # y center
        (g["bbox_xyxy"][0] + g["bbox_xyxy"][2]) / 2,  # x center
    ))
    for order, group in enumerate(groups):
        group["group_id"] = order
        group["reading_order"] = order

    # Assign group_id back to line_items
    for group in groups:
        for idx in group["line_indices"]:
            line_items[idx]["group_id"] = group["group_id"]

    logger.info(f"Grouped {n} lines into {len(groups)} groups "
                f"({sum(1 for g in groups if g['is_multiline_group'])} multi-line)")
    return groups
