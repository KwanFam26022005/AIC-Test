from __future__ import annotations

from .shape_utils import normalize_bbox, to_plain_list


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


def classify_region(group: dict, image_height: int | None = None) -> str:
    text = (group.get("raw_group_text") or "").strip()
    bbox = normalize_bbox(group.get("merged_bbox"))
    if len(text) <= 2:
        return "logo"
    if ":" in text and any(ch.isdigit() for ch in text) and len(text) <= 12:
        return "timestamp"
    if image_height and bbox[1] > image_height * 0.70:
        return "subtitle"
    if len(to_plain_list(group.get("line_indices"), [])) >= 5:
        return "document_block"
    return "scene_text"


def group_ocr_lines(df, cfg):
    import pandas as pd

    rows = []
    for frame_id, frame_df in df.groupby("frame_id", sort=False):
        frame_rows = []
        for row in frame_df.to_dict("records"):
            row["bbox"] = normalize_bbox(row.get("bbox"))
            frame_rows.append(row)
        for group_id, component in enumerate(_spatial_components(frame_rows, cfg)):
            rows.append(_emit_group(frame_id, group_id, component, cfg))
    return pd.DataFrame(rows)


def _spatial_components(rows: list[dict], cfg) -> list[list[dict]]:
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
            if _should_link_boxes(rows[i]["bbox"], rows[j]["bbox"], cfg):
                union(i, j)

    components: dict[int, list[dict]] = {}
    for idx, row in enumerate(rows):
        components.setdefault(find(idx), []).append(row)

    grouped = []
    for component in components.values():
        grouped.append(sorted(component, key=_line_sort_key))
    return sorted(grouped, key=_component_sort_key)


def _should_link_boxes(a: list[int], b: list[int], cfg) -> bool:
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
    return same_text_row


def _line_sort_key(row: dict) -> tuple[int, int, int]:
    bbox = normalize_bbox(row.get("bbox"))
    return (bbox[1], bbox[0], int(row.get("line_idx", 0)))


def _component_sort_key(rows: list[dict]) -> tuple[int, int, int]:
    bbox = bbox_union([normalize_bbox(row.get("bbox")) for row in rows])
    first_idx = min(int(row.get("line_idx", 0)) for row in rows)
    return (bbox[1], bbox[0], first_idx)


def _emit_group(frame_id: str, group_id: int, lines: list[dict], cfg) -> dict:
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
    group["region_type"] = classify_region(group)
    group["need_vlm_group"] = (
        group["max_risk_score"] >= float(cfg.risk.vlm_risk_threshold)
        or group["region_type"] in ["signboard", "document_block", "table"]
    ) and group["region_type"] not in ["timestamp", "logo"]
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
