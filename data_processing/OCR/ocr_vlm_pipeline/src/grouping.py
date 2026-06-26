from __future__ import annotations

from .shape_utils import normalize_bbox, to_plain_list


def bbox_union(bboxes: list[list[int]]) -> list[int]:
    xs1, ys1, xs2, ys2 = zip(*bboxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


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
        frame_df = frame_df.sort_values(["frame_number", "line_idx"])
        group_id = 0
        current = []
        last_bbox = None
        for row in frame_df.to_dict("records"):
            bbox = normalize_bbox(row.get("bbox"))
            row["bbox"] = bbox
            should_split = False
            if last_bbox is not None:
                vertical_gap = bbox[1] - last_bbox[3]
                horizontal_gap = abs(bbox[0] - last_bbox[0])
                should_split = (
                    vertical_gap > int(cfg.grouping.vertical_gap)
                    or horizontal_gap > max(int(cfg.grouping.horizontal_gap), (last_bbox[2] - last_bbox[0]) * 2)
                )
            if should_split and current:
                rows.append(_emit_group(frame_id, group_id, current, cfg))
                group_id += 1
                current = []
            current.append(row)
            last_bbox = bbox
        if current:
            rows.append(_emit_group(frame_id, group_id, current, cfg))
    return pd.DataFrame(rows)


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
