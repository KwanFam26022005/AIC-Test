# -*- coding: utf-8 -*-
"""
config.py — Cấu hình trung tâm cho Object Detection Pipeline (4 Stages)

Pipeline:
    Stage 1 (RAM++)  →  Stage 2 (GroundingDINO)  →  Stage 3 (Co-DETR)  →  Stage 4 (IoU Merge)

Lưu ý về môi trường:
    - Stage 1 + 2 chạy chung env: transformers==4.46.3 + torch mới
    - Stage 3 chạy env riêng: conda "codetr", Python 3.9, torch==1.12.1+cu113
    - Stage 4 chạy CPU only, bất kỳ env nào cũng được

Cấu hình BENCHMARK (FP32, không tối ưu) — để đo baseline performance.
"""

import os
from pathlib import Path

# ===========================================================================
# 1. ĐƯỜNG DẪN CHUNG
# ===========================================================================

# Thư mục gốc dự án (tự detect dựa trên vị trí file config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # AIC2026/

# Thư mục chứa keyframes cần xử lý (thay đổi theo nhu cầu)
# Mặc định: thư mục test nhỏ L21_V001 (307 frames)
KEYFRAME_INPUT_DIR = PROJECT_ROOT / "keyframe_test" / "L21_V001"

# Thư mục output chứa kết quả trung gian và cuối cùng
OUTPUT_DIR = PROJECT_ROOT / "output" / "object_detection"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final"
VISUALIZE_DIR = OUTPUT_DIR / "visualized"

# Tên video (dùng làm prefix cho frame_id)
VIDEO_ID = "L21_V001"

# ===========================================================================
# 2. CẤU HÌNH STAGE 1 — RAM++ (Tag Detection)
# ===========================================================================

RAM_IMAGE_SIZE = 384
RAM_CHECKPOINT = "pretrained/ram_plus_swin_large_14m.pth"
RAM_CHECKPOINT_URL = (
    "https://huggingface.co/xinyu1205/recognize-anything-plus-model"
    "/resolve/main/ram_plus_swin_large_14m.pth"
)
RAM_VIT_TYPE = "swin_l"

# Output file cho Stage 1
STAGE1_OUTPUT = INTERMEDIATE_DIR / "stage1_ram_tags.json"

# ===========================================================================
# 3. CẤU HÌNH STAGE 2 — GroundingDINO (Open-vocab Object Detection)
# ===========================================================================

# Checkpoint từ HuggingFace transformers (sử dụng bản base cho benchmark)
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Ngưỡng detection
GDINO_BOX_THRESHOLD = 0.35
GDINO_TEXT_THRESHOLD = 0.25

# Ngưỡng loại bỏ bbox scene-level (bbox chiếm > 50% diện tích ảnh → loại)
GDINO_MAX_AREA_RATIO = 0.5

# Output file cho Stage 2
STAGE2_OUTPUT = INTERMEDIATE_DIR / "stage2_gdino_detections.json"

# ===========================================================================
# 4. CẤU HÌNH STAGE 3 — Co-DETR ViT-L (LVIS 1203-class Detection)
# ===========================================================================

# Config file MMDetection (relative path trong repo Co-DETR đã clone)
CODETR_CONFIG = "projects/configs/co_dino_vit/co_dino_5scale_lsj_vit_large_lvis.py"

# Checkpoint (tải từ HuggingFace)
CODETR_CHECKPOINT = "checkpoints/pytorch_model_lvis.pth"
CODETR_HF_REPO = "zongzhuofan/co-detr-vit-large-lvis"
CODETR_HF_FILENAME = "pytorch_model.pth"

# Ngưỡng confidence cho Co-DETR
CODETR_SCORE_THRESHOLD = 0.3

# Output file cho Stage 3
STAGE3_OUTPUT = INTERMEDIATE_DIR / "stage3_codetr_detections.json"

# ===========================================================================
# 5. CẤU HÌNH STAGE 4 — IoU Merge
# ===========================================================================

# Ngưỡng IoU để merge bbox trùng lặp giữa GDINO và Co-DETR
MERGE_IOU_THRESHOLD = 0.5

# Ngưỡng IoMin (Intersection-over-Min) để xác định containment (parent-child)
CONTAINMENT_IOMIN_THRESHOLD = 0.7

# Các lớp thường là "container" (parent candidates)
CONTAINER_CLASSES = frozenset({
    "PERSON", "CAR", "BUS", "TRUCK", "HORSE", "COW", "BULL",
    "BICYCLE", "MOTORCYCLE", "BOAT", "AIRPLANE", "ELEPHANT",
    "DOG", "CAT", "BEAR", "GIRAFFE", "ZEBRA", "SHEEP",
})

# Các lớp thường là "part/accessory" (child candidates)
PART_CLASSES = frozenset({
    "HAT", "SHIRT", "SHOE", "GLOVE", "TIE", "BELT", "BACKPACK",
    "HANDBAG", "HELMET", "MASK", "GLASSES", "WATCH", "SCARF",
    "JACKET", "COAT", "SOCK", "DRESS", "SKIRT", "SHORTS",
    "WHEEL", "HEADLIGHT", "LICENSE_PLATE", "WINDOW_(OF_A_VEHICLE)",
    "MIRROR", "BUMPER",
})

# Output file cuối cùng
STAGE4_OUTPUT = FINAL_OUTPUT_DIR / "merged_metadata.jsonl"

# ===========================================================================
# 6. CẤU HÌNH QUADRANT (phân vùng không gian cho spatial search)
# ===========================================================================

def get_quadrant(bbox, img_width, img_height):
    """Xác định quadrant (vùng không gian) của bbox trong ảnh.

    Chia ảnh thành 9 vùng (3×3 grid):
        top_left    | top_center    | top_right
        center_left | center        | center_right
        bottom_left | bottom_center | bottom_right

    Args:
        bbox: [x1, y1, x2, y2]
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh

    Returns:
        str: tên quadrant
    """
    cx = (bbox[0] + bbox[2]) / 2  # tâm x
    cy = (bbox[1] + bbox[3]) / 2  # tâm y

    # Xác định cột (left / center / right)
    if cx < img_width / 3:
        col = "left"
    elif cx < img_width * 2 / 3:
        col = "center"
    else:
        col = "right"

    # Xác định hàng (top / center / bottom)
    if cy < img_height / 3:
        row = "top"
    elif cy < img_height * 2 / 3:
        row = "center"
    else:
        row = "bottom"

    # Ghép tên
    if row == "center" and col == "center":
        return "center"
    elif row == "center":
        return f"center_{col}"
    elif col == "center":
        return f"{row}_center"
    else:
        return f"{row}_{col}"


# ===========================================================================
# 8. TIỆN ÍCH TẠO THƯ MỤC
# ===========================================================================

def ensure_output_dirs():
    """Tạo tất cả thư mục output nếu chưa tồn tại."""
    for d in [OUTPUT_DIR, INTERMEDIATE_DIR, FINAL_OUTPUT_DIR, VISUALIZE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
