# -*- coding: utf-8 -*-
"""
stage4_iou_merger.py — Stage 4: IoU Merge + Containment Assignment

Môi trường: Bất kỳ (CPU only, không cần GPU)

Input:  JSON files từ Stage 1 (tags), Stage 2 (GDINO dets), Stage 3 (Co-DETR dets)
Output: JSONL file chứa merged metadata cho từng frame (sẵn sàng index vào Elasticsearch)

Logic merge:
    1. Co-DETR = nguồn CHÍNH cho bbox (label cụ thể, bbox tight)
    2. GDINO = nguồn BỔ SUNG (detect objects mà Co-DETR miss, VD: person)
    3. IoU matching: nếu GDINO bbox overlap cao (IoU > threshold) với Co-DETR bbox
       → bỏ GDINO (giữ Co-DETR vì label cụ thể hơn, VD: COW thay vì animal)
    4. Loại bỏ GDINO bbox scene-level (chiếm > 50% diện tích ảnh)
    5. Chuyển scene-level labels vào trường tags
    6. Gán parent_id cho containment (HAT ⊂ PERSON) bằng IoMin

Sử dụng:
    python stage4_iou_merger.py
    python stage4_iou_merger.py --stage1 /path/to/s1.json --stage2 /path/to/s2.json \\
                                --stage3 /path/to/s3.json --output /path/to/output.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    STAGE1_OUTPUT, STAGE2_OUTPUT, STAGE3_OUTPUT, STAGE4_OUTPUT,
    MERGE_IOU_THRESHOLD, CONTAINMENT_IOMIN_THRESHOLD,
    GDINO_MAX_AREA_RATIO,
    CONTAINER_CLASSES, PART_CLASSES,
    VIDEO_ID,
)
from config import ensure_output_dirs, get_quadrant
from utils import (
    compute_iou, compute_iomin, compute_area_ratio,
    normalize_tags, filter_large_boxes,
    load_intermediate, save_final_jsonl,
    Timer,
)


def merge_detections_for_frame(gdino_dets, codetr_dets, img_size,
                                iou_threshold=MERGE_IOU_THRESHOLD,
                                max_area_ratio=GDINO_MAX_AREA_RATIO):
    """Merge detections từ GDINO và Co-DETR cho 1 frame.

    Logic:
        1. Giữ TOÀN BỘ Co-DETR detections (nguồn chính)
        2. Với mỗi GDINO detection:
           a. Nếu bbox chiếm > max_area_ratio diện tích ảnh → chuyển vào scene_labels
           b. Nếu IoU > threshold với bất kỳ Co-DETR det nào → BỎ (trùng)
           c. Nếu không trùng → THÊM (object mới mà Co-DETR không detect)

    Args:
        gdino_dets: list[dict] — detections từ GroundingDINO
        codetr_dets: list[dict] — detections từ Co-DETR
        img_size: [width, height]
        iou_threshold: ngưỡng IoU merge
        max_area_ratio: ngưỡng loại bbox scene-level

    Returns:
        tuple: (merged_objects, scene_labels)
    """
    img_w, img_h = img_size
    merged = []
    scene_labels = []

    # Step 1: Giữ toàn bộ Co-DETR detections
    for det in codetr_dets:
        merged.append({
            "label": det["label"],
            "source": "codetr",
            "score": det["score"],
            "box": det["box"],
        })

    # Step 2: Xử lý từng GDINO detection
    for gdet in gdino_dets:
        # 2a. Kiểm tra scene-level
        area_ratio = compute_area_ratio(gdet["box"], img_w, img_h)
        if area_ratio > max_area_ratio:
            scene_labels.append(gdet["label"])
            continue

        # 2b. Kiểm tra trùng lặp với Co-DETR
        is_duplicate = False
        for cdet in codetr_dets:
            iou = compute_iou(gdet["box"], cdet["box"])
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            # 2c. Object mới — thêm vào merged (normalize label thành UPPER)
            merged.append({
                "label": gdet["label"].upper().strip(),
                "source": "gdino",
                "score": gdet["score"],
                "box": gdet["box"],
            })

    return merged, scene_labels


def assign_parent_ids(objects, img_size,
                      iomin_threshold=CONTAINMENT_IOMIN_THRESHOLD):
    """Gán parent_id cho quan hệ containment (parent-child).

    Logic:
        - Chỉ xét child nếu label ∈ PART_CLASSES (HAT, SHIRT, SHOE, ...)
        - Chỉ xét parent nếu label ∈ CONTAINER_CLASSES (PERSON, CAR, ...)
        - Dùng IoMin (Intersection-over-Min) thay vì IoU vì child bbox rất nhỏ
        - Chọn parent có IoMin cao nhất nếu nhiều candidates

    Args:
        objects: list[dict] — danh sách objects đã merge
        img_size: [width, height]
        iomin_threshold: ngưỡng IoMin tối thiểu

    Returns:
        list[dict]: objects đã gán obj_id, parent_id, quadrant, area_ratio
    """
    img_w, img_h = img_size

    # Gán obj_id
    for i, obj in enumerate(objects):
        obj["obj_id"] = i

    # Tính area_ratio và quadrant cho từng object
    for obj in objects:
        obj["area_ratio"] = round(compute_area_ratio(obj["box"], img_w, img_h), 6)
        obj["quadrant"] = get_quadrant(obj["box"], img_w, img_h)

    # Gán parent_id
    for child in objects:
        child["parent_id"] = -1  # Default: top-level

        if child["label"] not in PART_CLASSES:
            continue

        best_parent_id = -1
        best_iomin = iomin_threshold

        for parent in objects:
            if parent["obj_id"] == child["obj_id"]:
                continue
            if parent["label"] not in CONTAINER_CLASSES:
                continue

            iomin = compute_iomin(child["box"], parent["box"])
            if iomin > best_iomin:
                best_iomin = iomin
                best_parent_id = parent["obj_id"]

        child["parent_id"] = best_parent_id

    return objects


def build_document(frame_name, video_id, ram_tags, merged_objects,
                   scene_labels, img_size):
    """Xây dựng document Elasticsearch cho 1 frame.

    Args:
        frame_name: str — tên file (VD: "089.jpg")
        video_id: str — ID video (VD: "L21_V001")
        ram_tags: list[str] — tags gốc từ RAM++
        merged_objects: list[dict] — objects đã merge + gán parent
        scene_labels: list[str] — labels scene-level từ GDINO bị loại
        img_size: [width, height]

    Returns:
        dict: document sẵn sàng index vào Elasticsearch
    """
    # Chuẩn hóa tags và thêm scene_labels bị loại từ bbox
    filtered_tags = normalize_tags(ram_tags)
    # Thêm scene labels (đã bị loại khỏi bbox) vào tags nếu chưa có
    for label in scene_labels:
        label_lower = label.lower()
        if label_lower not in filtered_tags:
            filtered_tags.append(label_lower)

    # Object summary: danh sách unique class names
    object_summary = sorted(set(obj["label"] for obj in merged_objects))

    # Class counts
    class_counts = defaultdict(int)
    for obj in merged_objects:
        class_counts[obj["label"]] += 1

    # Frame ID
    frame_idx = frame_name.split(".")[0]  # "089.jpg" → "089"
    frame_id = f"{video_id}_frame_{frame_idx}"

    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_idx": int(frame_idx) if frame_idx.isdigit() else frame_idx,
        "image_size": img_size,

        # Tags (scene context — cho text search)
        "tags": filtered_tags,
        "tag_count": len(filtered_tags),

        # Objects (bbox detections — cho spatial search)
        "object_summary": object_summary,
        "object_count": len(merged_objects),
        "objects": merged_objects,

        # Class counts (cho count queries)
        "class_counts": dict(class_counts),
    }


def run_stage4(stage1_path=None, stage2_path=None, stage3_path=None,
               output_path=None, video_id=None):
    """Chạy Stage 4: IoU Merge + Containment → Final JSONL.

    Args:
        stage1_path: Path — file JSON Stage 1 (RAM++ tags)
        stage2_path: Path — file JSON Stage 2 (GDINO dets)
        stage3_path: Path — file JSON Stage 3 (Co-DETR dets)
        output_path: Path — file JSONL output
        video_id: str — ID video

    Returns:
        list[dict]: danh sách documents
    """
    stage1_path = Path(stage1_path) if stage1_path else STAGE1_OUTPUT
    stage2_path = Path(stage2_path) if stage2_path else STAGE2_OUTPUT
    stage3_path = Path(stage3_path) if stage3_path else STAGE3_OUTPUT
    output_path = Path(output_path) if output_path else STAGE4_OUTPUT
    video_id = video_id or VIDEO_ID

    ensure_output_dirs()

    # Đọc kết quả các stage trước
    stage1_data = load_intermediate(stage1_path)
    stage2_data = load_intermediate(stage2_path)
    stage3_data = load_intermediate(stage3_path)

    tags_per_frame = stage1_data["results"]
    gdino_per_frame = stage2_data["results"]
    codetr_per_frame = stage3_data["results"]

    # Lấy danh sách tất cả frames (union từ cả 3 stages)
    all_frames = sorted(set(tags_per_frame.keys()) |
                        set(gdino_per_frame.keys()) |
                        set(codetr_per_frame.keys()))
    total = len(all_frames)

    print(f"\n[Merge] Bắt đầu merge {total} frames ...")
    print(f"  Stage 1 (RAM++):     {len(tags_per_frame)} frames")
    print(f"  Stage 2 (GDINO):     {len(gdino_per_frame)} frames")
    print(f"  Stage 3 (Co-DETR):   {len(codetr_per_frame)} frames")
    print(f"  IoU threshold:       {MERGE_IOU_THRESHOLD}")
    print(f"  IoMin threshold:     {CONTAINMENT_IOMIN_THRESHOLD}")
    print(f"  Max area ratio:      {GDINO_MAX_AREA_RATIO}")

    # Thống kê merge
    stats = {
        "total_codetr_kept": 0,
        "total_gdino_kept": 0,
        "total_gdino_duplicate": 0,
        "total_gdino_scene": 0,
        "total_containments": 0,
    }

    documents = []
    start_time = time.time()

    for idx, frame_name in enumerate(all_frames):
        # Lấy dữ liệu từ mỗi stage (có thể thiếu)
        ram_tags = tags_per_frame.get(frame_name, [])

        gdino_data = gdino_per_frame.get(frame_name, {})
        gdino_dets = gdino_data.get("detections", []) if isinstance(gdino_data, dict) else []

        codetr_data = codetr_per_frame.get(frame_name, {})
        codetr_dets = codetr_data.get("detections", []) if isinstance(codetr_data, dict) else []

        # Xác định image_size (ưu tiên từ Stage 2 vì dùng PIL, chính xác hơn)
        img_size = (gdino_data.get("image_size") or
                    codetr_data.get("image_size") or
                    [1280, 720])  # Fallback

        # Step 1: Merge detections
        merged_objects, scene_labels = merge_detections_for_frame(
            gdino_dets, codetr_dets, img_size
        )

        # Đếm thống kê
        n_codetr = sum(1 for o in merged_objects if o.get("source") == "codetr")
        n_gdino = sum(1 for o in merged_objects if o.get("source") == "gdino")
        n_gdino_dup = len(gdino_dets) - n_gdino - len(scene_labels)
        stats["total_codetr_kept"] += n_codetr
        stats["total_gdino_kept"] += n_gdino
        stats["total_gdino_duplicate"] += max(0, n_gdino_dup)
        stats["total_gdino_scene"] += len(scene_labels)

        # Step 2: Gán parent_id (containment)
        merged_objects = assign_parent_ids(merged_objects, img_size)
        n_containments = sum(1 for o in merged_objects if o["parent_id"] >= 0)
        stats["total_containments"] += n_containments

        # Step 3: Build document
        doc = build_document(frame_name, video_id, ram_tags,
                             merged_objects, scene_labels, img_size)
        documents.append(doc)

        # In tiến độ
        if (idx + 1) % 100 == 0 or idx == total - 1:
            print(f"  [{idx + 1}/{total}] {frame_name}: "
                  f"{n_codetr} codetr + {n_gdino} gdino = "
                  f"{len(merged_objects)} objects, "
                  f"{n_containments} containments")

    elapsed = time.time() - start_time

    # Thống kê tổng hợp
    print(f"\n{'='*60}")
    print(f"📊 STAGE 4 — IoU MERGE REPORT")
    print(f"{'='*60}")
    print(f"  Tổng frames:              {total}")
    print(f"  Thời gian merge:          {elapsed:.2f}s")
    print(f"  ─────────────────────────────────────")
    print(f"  Co-DETR objects giữ:       {stats['total_codetr_kept']}")
    print(f"  GDINO objects giữ (bổ sung): {stats['total_gdino_kept']}")
    print(f"  GDINO objects bỏ (trùng):  {stats['total_gdino_duplicate']}")
    print(f"  GDINO labels → tags:       {stats['total_gdino_scene']}")
    print(f"  Containments (parent-child): {stats['total_containments']}")
    print(f"  ─────────────────────────────────────")
    total_objs = stats['total_codetr_kept'] + stats['total_gdino_kept']
    print(f"  Tổng objects final:        {total_objs}")
    print(f"  Trung bình objects/frame:  {total_objs / total:.1f}")
    print(f"{'='*60}")

    # Lưu output JSONL
    save_final_jsonl(documents, output_path)

    # Lưu thêm bản report JSON
    report = {
        "stage": "stage4_iou_merger",
        "total_frames": total,
        "merge_time_s": round(elapsed, 2),
        "iou_threshold": MERGE_IOU_THRESHOLD,
        "iomin_threshold": CONTAINMENT_IOMIN_THRESHOLD,
        "max_area_ratio": GDINO_MAX_AREA_RATIO,
        "stats": stats,
    }
    report_path = output_path.parent / "merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[I/O] Report → {report_path}")

    # In 1 document mẫu
    if documents:
        print(f"\n📄 DOCUMENT MẪU (frame đầu tiên):")
        print(json.dumps(documents[0], ensure_ascii=False, indent=2)[:2000])

    return documents


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: IoU Merge + Containment")
    parser.add_argument("--stage1", type=str, default=None,
                        help="File JSON Stage 1 — RAM++ tags (default: config)")
    parser.add_argument("--stage2", type=str, default=None,
                        help="File JSON Stage 2 — GDINO dets (default: config)")
    parser.add_argument("--stage3", type=str, default=None,
                        help="File JSON Stage 3 — Co-DETR dets (default: config)")
    parser.add_argument("--output", type=str, default=None,
                        help="File JSONL output (default: config)")
    parser.add_argument("--video-id", type=str, default=None,
                        help="Video ID (default: config)")
    args = parser.parse_args()

    with Timer("Stage 4 — IoU Merge + Containment"):
        run_stage4(
            stage1_path=args.stage1,
            stage2_path=args.stage2,
            stage3_path=args.stage3,
            output_path=args.output,
            video_id=args.video_id,
        )
