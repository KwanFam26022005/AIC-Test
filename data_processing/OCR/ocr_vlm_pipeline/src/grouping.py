from __future__ import annotations

import re

from .shape_utils import normalize_bbox, to_plain_list


TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
CHANNEL_RE = re.compile(r"\b(?:HTV|VTV|THVL|VTC|ANTV|QPVN|VOV|TV|HD|LIVE)\w*\b", re.IGNORECASE)


def bbox_union(bboxes: list[list[int]]) -> list[int]:
    xs1, ys1, xs2, ys2 = zip(*bboxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def bbox_iou(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def axis_overlap_ratio(a: list[int], b: list[int], axis: str) -> float:
    if axis == "x":
        left = max(a[0], b[0])
        right = min(a[2], b[2])
        denom = min(max(1, a[2] - a[0]), max(1, b[2] - b[0]))
    else:
        left = max(a[1], b[1])
        right = min(a[3], b[3])
        denom = min(max(1, a[3] - a[1]), max(1, b[3] - b[1]))
    return max(0, right - left) / denom


def axis_gap(a: list[int], b: list[int], axis: str) -> int:
    if axis == "x":
        return max(0, max(a[0], b[0]) - min(a[2], b[2]))
    return max(0, max(a[1], b[1]) - min(a[3], b[3]))


def classify_region(group: dict, image_width: int | None = None, image_height: int | None = None) -> str:
    text = (group.get("raw_group_text") or "").strip()
    compact_text = re.sub(r"\s+", "", text)
    bbox = normalize_bbox(group.get("merged_bbox"))
    line_count = len(to_plain_list(group.get("line_indices"), []))
    box_width = max(1, bbox[2] - bbox[0])
    box_height = max(1, bbox[3] - bbox[1])
    width_ratio = box_width / max(1, image_width or box_width)
    height_ratio = box_height / max(1, image_height or box_height)
    y1_ratio = bbox[1] / max(1, image_height or bbox[3] or 1)
    y2_ratio = bbox[3] / max(1, image_height or bbox[3] or 1)
    in_top_band = bool(image_height and y2_ratio <= 0.22)
    in_lower_band = bool(image_height and y1_ratio >= 0.62)

    if TIME_RE.search(text) and (in_top_band or len(compact_text) <= 24):
        return "timestamp"
    if in_top_band and CHANNEL_RE.search(text):
        return "channel_logo"
    if len(compact_text) <= 2 or (in_top_band and len(compact_text) <= 4 and width_ratio <= 0.25 and height_ratio <= 0.22):
        return "logo"
    if line_count >= 5:
        return "document_block"
    if in_lower_band and (line_count >= 2 or width_ratio >= 0.35):
        return "lower_third"
    if image_height and y1_ratio >= 0.72:
        return "subtitle"
    return "scene_text"


def group_ocr_lines(df, cfg):
    import pandas as pd

    rows = []
    for frame_id, frame_df in df.groupby("frame_id", sort=False):
        frame_rows = []
        for row in frame_df.to_dict("records"):
            row["bbox"] = normalize_bbox(row.get("bbox"))
            frame_rows.append(row)
        image_width, image_height = _frame_size(frame_rows)
        for group_id, component in enumerate(_spatial_components(frame_rows, cfg, image_width, image_height)):
            rows.append(_emit_group(frame_id, group_id, component, cfg, image_width, image_height))
    return pd.DataFrame(rows)


def _frame_size(rows: list[dict]) -> tuple[int | None, int | None]:
    if not rows:
        return None, None
    first = rows[0]
    width = _positive_int(first.get("width"))
    height = _positive_int(first.get("height"))
    if width and height:
        return width, height

    frame_path = first.get("frame_path")
    if not frame_path:
        return None, None
    try:
        from PIL import Image

        with Image.open(frame_path) as img:
            return img.size
    except Exception:
        return None, None


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _spatial_components(rows: list[dict], cfg, image_width: int | None = None, image_height: int | None = None) -> list[list[dict]]:
    if not rows:
        return []
    parent = list(range(len(rows)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _should_link_boxes(rows[i]["bbox"], rows[j]["bbox"], cfg, image_width, image_height):
                union(i, j)

    components: dict[int, list[dict]] = {}
    for idx, row in enumerate(rows):
        components.setdefault(find(idx), []).append(row)

    grouped = []
    for component in components.values():
        grouped.append(sorted(component, key=_line_sort_key))
    return sorted(grouped, key=_component_sort_key)


def _should_link_boxes(a: list[int], b: list[int], cfg, image_width: int | None = None, image_height: int | None = None) -> bool:
    vertical_gap = int(cfg.grouping.get("vertical_gap", 15))
    horizontal_gap = int(cfg.grouping.get("horizontal_gap", 15))
    iou_threshold = float(cfg.grouping.get("iou_threshold", 0.05))
    min_horizontal_overlap = float(cfg.grouping.get("min_horizontal_overlap", 0.25))
    min_vertical_overlap = float(cfg.grouping.get("min_vertical_overlap", 0.25))

    if bbox_iou(a, b) >= iou_threshold:
        return True

    stacked_vertically = axis_gap(a, b, "y") <= vertical_gap and axis_overlap_ratio(a, b, "x") >= min_horizontal_overlap
    if stacked_vertically:
        return True

    same_text_row = axis_gap(a, b, "x") <= horizontal_gap and axis_overlap_ratio(a, b, "y") >= min_vertical_overlap
    if same_text_row:
        return True

    return _should_link_lower_third_boxes(a, b, cfg, image_width, image_height)


def _should_link_lower_third_boxes(a: list[int], b: list[int], cfg, image_width: int | None, image_height: int | None) -> bool:
    if not image_width or not image_height:
        return False

    y_start_ratio = float(cfg.grouping.get("lower_third_y_start_ratio", 0.55))
    if a[1] / image_height < y_start_ratio or b[1] / image_height < y_start_ratio:
        return False

    max_horizontal_gap = int(cfg.grouping.get("lower_third_horizontal_gap", max(180, image_width * 0.15)))
    max_vertical_gap = int(cfg.grouping.get("lower_third_vertical_gap", max(45, image_height * 0.07)))
    min_vertical_overlap = float(cfg.grouping.get("lower_third_min_vertical_overlap", 0.12))

    horizontally_close = axis_gap(a, b, "x") <= max_horizontal_gap
    vertically_related = axis_gap(a, b, "y") <= max_vertical_gap or axis_overlap_ratio(a, b, "y") >= min_vertical_overlap
    return horizontally_close and vertically_related


def _line_sort_key(row: dict) -> tuple[int, int, int]:
    bbox = normalize_bbox(row.get("bbox"))
    return (bbox[1], bbox[0], int(row.get("line_idx", 0)))


def _component_sort_key(rows: list[dict]) -> tuple[int, int, int]:
    bbox = bbox_union([normalize_bbox(row.get("bbox")) for row in rows])
    first_idx = min(int(row.get("line_idx", 0)) for row in rows)
    return (bbox[1], bbox[0], first_idx)


def _emit_group(frame_id: str, group_id: int, lines: list[dict], cfg, image_width: int | None = None, image_height: int | None = None) -> dict:
    bboxes = [normalize_bbox(line.get("bbox")) for line in lines]
    raw_text = "\n".join((line.get("ocr_text") or "").strip() for line in lines)
    group = {
        "video_id": lines[0].get("video_id"),
        "frame_id": frame_id,
        "frame_number": lines[0].get("frame_number"),
        "frame_path": lines[0].get("frame_path"),
        "group_id": group_id,
        "line_indices": [int(line.get("line_idx", idx)) for idx, line in enumerate(lines)],
        "merged_bbox": bbox_union(bboxes),
        "crop_path": None,
        "crop_phash": None,
        "raw_group_text": raw_text,
        "min_confidence": min(float(line.get("confidence") or 0.0) for line in lines),
        "avg_confidence": sum(float(line.get("confidence") or 0.0) for line in lines) / max(1, len(lines)),
        "max_risk_score": max(float(line.get("risk_score") or 0.0) for line in lines),
    }
    group["region_type"] = classify_region(group, image_width=image_width, image_height=image_height)
    vlm_region_types = {"signboard", "complex_signboard", "document_block", "table", "lower_third", "subtitle"}
    excluded_region_types = {"timestamp", "logo", "channel_logo"}
    group["need_vlm_group"] = (
        group["max_risk_score"] >= float(cfg.risk.vlm_risk_threshold)
        or group["region_type"] in vlm_region_types
    ) and group["region_type"] not in excluded_region_types
    return group


def line_to_group_map(groups) -> dict[tuple[str, int], int]:
    mapping = {}
    for row in groups.to_dict("records"):
        for line_idx in to_plain_list(row.get("line_indices"), []):
            mapping[(row["frame_id"], int(line_idx))] = int(row["group_id"])
    return mapping


def line_to_region_map(groups) -> dict[tuple[str, int], str]:
    mapping = {}
    for row in groups.to_dict("records"):
        for line_idx in to_plain_list(row.get("line_indices"), []):
            mapping[(row["frame_id"], int(line_idx))] = row.get("region_type", "unknown")
    return mapping
