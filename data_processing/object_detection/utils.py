# -*- coding: utf-8 -*-
"""
utils.py — Tiện ích dùng chung cho Object Detection Pipeline

Bao gồm:
    - Tính toán IoU (Intersection over Union)
    - Tính toán IoMin (Intersection over Min) cho containment detection
    - Lọc tag RAM++ (loại verb/adj)
    - Lọc bbox scene-level (loại bbox quá lớn)
    - Tiện ích I/O (đọc/ghi JSON intermediate files)
"""

import json
import time
from pathlib import Path
from datetime import datetime


# ===========================================================================
# 1. TÍNH TOÁN HÌNH HỌC BBOX
# ===========================================================================

def compute_iou(box_a, box_b):
    """Tính IoU (Intersection over Union) giữa 2 bbox.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float: IoU trong khoảng [0, 1]
    """
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_iomin(box_small, box_large):
    """Tính IoMin (Intersection over Min area) cho containment detection.

    IoMin = Intersection / min(area_small, area_large)
    Nếu box_small nằm gọn hoàn toàn trong box_large → IoMin = 1.0

    Đây là metric chính xác hơn IoU khi kiểm tra quan hệ chứa đựng
    (containment), vì IoU bị thấp khi 2 box chênh lệch kích thước lớn.

    Ví dụ thực tế:
        HAT [843,400,860,412] vs PERSON [828,397,865,467]
        IoU = 0.079 (rất thấp, dễ bị reject)
        IoMin = 1.0 (chính xác, HAT nằm gọn trong PERSON)

    Args:
        box_small: [x1, y1, x2, y2] — bbox nhỏ hơn (candidate child)
        box_large: [x1, y1, x2, y2] — bbox lớn hơn (candidate parent)

    Returns:
        float: IoMin trong khoảng [0, 1]
    """
    x0 = max(box_small[0], box_large[0])
    y0 = max(box_small[1], box_large[1])
    x1 = min(box_small[2], box_large[2])
    y1 = min(box_small[3], box_large[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_small = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    area_large = (box_large[2] - box_large[0]) * (box_large[3] - box_large[1])
    min_area = min(area_small, area_large)

    return intersection / min_area if min_area > 0 else 0.0


def compute_area_ratio(bbox, img_width, img_height):
    """Tính tỷ lệ diện tích bbox so với diện tích ảnh.

    Args:
        bbox: [x1, y1, x2, y2]
        img_width: chiều rộng ảnh (pixels)
        img_height: chiều cao ảnh (pixels)

    Returns:
        float: tỷ lệ trong khoảng [0, 1]
    """
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    img_area = img_width * img_height
    return box_area / img_area if img_area > 0 else 0.0


# ===========================================================================
# 2. CHUẨN HÓA TAG RAM++
# ===========================================================================

def normalize_tags(tags):
    """Chuẩn hóa danh sách tags từ RAM++ (lowercase + strip).

    Args:
        tags: list[str] — danh sách tags gốc từ RAM++

    Returns:
        list[str]: danh sách tags đã chuẩn hóa
    """
    return [t.lower().strip() for t in tags]


# ===========================================================================
# 3. LỌC BBOX SCENE-LEVEL
# ===========================================================================

def filter_large_boxes(detections, img_width, img_height, max_area_ratio=0.5):
    """Loại bỏ bbox chiếm quá nhiều diện tích ảnh (scene-level).

    Ví dụ thực tế từ benchmark:
        field    → bbox chiếm ~80% ảnh → LOẠI
        farmland → bbox chiếm ~95% ảnh → LOẠI
        animal   → bbox chiếm ~3% ảnh  → GIỮ

    Args:
        detections: list[dict] — mỗi dict có key "box": [x1, y1, x2, y2]
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh
        max_area_ratio: ngưỡng tối đa (default 0.5 = 50%)

    Returns:
        tuple: (kept_detections, removed_labels)
            - kept_detections: list[dict] — bbox hợp lệ
            - removed_labels: list[str] — tên labels bị loại (dùng để chuyển vào tags)
    """
    kept = []
    removed_labels = []

    for det in detections:
        ratio = compute_area_ratio(det["box"], img_width, img_height)
        if ratio <= max_area_ratio:
            kept.append(det)
        else:
            removed_labels.append(det["label"])

    return kept, removed_labels


# ===========================================================================
# 4. TIỆN ÍCH I/O — ĐỌC/GHI FILE TRUNG GIAN
# ===========================================================================

def save_intermediate(data, output_path):
    """Lưu kết quả trung gian ra file JSON.

    Args:
        data: dict hoặc list — dữ liệu cần lưu
        output_path: str hoặc Path — đường dẫn file output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[I/O] Đã lưu kết quả → {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def load_intermediate(input_path):
    """Đọc kết quả trung gian từ file JSON.

    Args:
        input_path: str hoặc Path — đường dẫn file input

    Returns:
        dict hoặc list: dữ liệu đã đọc
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file intermediate: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[I/O] Đã đọc ← {input_path} ({input_path.stat().st_size / 1024:.1f} KB)")
    return data


def save_final_jsonl(documents, output_path):
    """Lưu kết quả cuối cùng ra file JSONL (mỗi dòng = 1 document).

    Args:
        documents: list[dict] — danh sách documents
        output_path: str hoặc Path — đường dẫn file output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[I/O] Đã lưu {len(documents)} documents → {output_path} "
          f"({output_path.stat().st_size / 1024:.1f} KB)")


# ===========================================================================
# 5. TIỆN ÍCH ĐO THỜI GIAN
# ===========================================================================

class Timer:
    """Context manager đo thời gian thực thi.

    Usage:
        with Timer("Stage 1 — RAM++"):
            run_stage1()
    """

    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        print(f"\n{'='*60}")
        print(f"⏱️  BẮT ĐẦU: {self.name}")
        print(f"{'='*60}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        minutes = int(self.elapsed // 60)
        seconds = self.elapsed % 60
        print(f"\n{'='*60}")
        print(f"✅ HOÀN THÀNH: {self.name}")
        print(f"⏱️  Thời gian: {minutes}m {seconds:.2f}s ({self.elapsed:.2f}s)")
        print(f"{'='*60}\n")


# ===========================================================================
# 6. TIỆN ÍCH LIỆT KÊ FRAMES
# ===========================================================================

def list_keyframe_paths(input_dir, extensions=(".jpg", ".jpeg", ".png")):
    """Liệt kê tất cả file ảnh keyframe trong thư mục, sắp xếp theo tên.

    Args:
        input_dir: str hoặc Path — thư mục chứa keyframes
        extensions: tuple — các đuôi file ảnh được chấp nhận

    Returns:
        list[Path]: danh sách đường dẫn file, đã sắp xếp
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Thư mục keyframe không tồn tại: {input_dir}")

    paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])

    print(f"[I/O] Tìm thấy {len(paths)} keyframes trong {input_dir}")
    return paths
