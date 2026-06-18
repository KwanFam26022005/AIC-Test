# -*- coding: utf-8 -*-
"""
run_pipeline.py — Orchestrator chạy pipeline 4 stages

Pipeline đầy đủ:
    Stage 1 (RAM++)  →  Stage 2 (GroundingDINO)  →  Stage 3 (Co-DETR)  →  Stage 4 (IoU Merge)

⚠️ LƯU Ý VỀ MÔI TRƯỜNG:
    - Stage 1 + 2 + 4 chạy trong env CHÍNH (transformers==4.46.3)
    - Stage 3 PHẢI chạy riêng trong conda env "codetr" (torch==1.12.1)

    → File này chỉ chạy được Stage 1 + 2 + 4 trong env chính.
    → Stage 3 cần chạy thủ công (hoặc qua subprocess) trong env codetr.

Sử dụng:
    # Chạy Stage 1 + 2 (env transformers):
    python run_pipeline.py --stages 1,2

    # Chạy Stage 3 riêng (env codetr — xem hướng dẫn Colab):
    # (chạy stage3_codetr_detector.py trực tiếp)

    # Chạy Stage 4 sau khi có đủ output từ S1, S2, S3:
    python run_pipeline.py --stages 4

    # Hoặc chạy S1 + S2 + S4 liên tục (bỏ qua S3 nếu đã chạy trước):
    python run_pipeline.py --stages 1,2,4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    KEYFRAME_INPUT_DIR, STAGE1_OUTPUT, STAGE2_OUTPUT,
    STAGE3_OUTPUT, STAGE4_OUTPUT, VIDEO_ID,
)
from config import ensure_output_dirs
from utils import Timer


def run_stages(stages, input_dir=None, video_id=None):
    """Chạy các stages được chỉ định.

    Args:
        stages: list[int] — danh sách stages cần chạy (VD: [1, 2, 4])
        input_dir: str — thư mục chứa keyframes (None → dùng config)
        video_id: str — ID video (None → dùng config)
    """
    input_dir = input_dir or str(KEYFRAME_INPUT_DIR)
    video_id = video_id or VIDEO_ID

    ensure_output_dirs()

    print(f"\n{'#'*60}")
    print(f"#  OBJECT DETECTION PIPELINE — BENCHMARK MODE (FP32)")
    print(f"#  Stages: {stages}")
    print(f"#  Input:  {input_dir}")
    print(f"#  Video:  {video_id}")
    print(f"{'#'*60}\n")

    # ── STAGE 1: RAM++ ──
    if 1 in stages:
        with Timer("Stage 1 — RAM++ Tag Detection"):
            from stage1_ram_tagger import run_stage1
            run_stage1(input_dir=input_dir)

    # ── STAGE 2: GroundingDINO ──
    if 2 in stages:
        # Kiểm tra output Stage 1 tồn tại
        if not STAGE1_OUTPUT.exists():
            print(f"\n❌ LỖI: Cần chạy Stage 1 trước! "
                  f"File không tồn tại: {STAGE1_OUTPUT}")
            sys.exit(1)

        with Timer("Stage 2 — GroundingDINO Detection"):
            from stage2_gdino_detector import run_stage2
            run_stage2(input_dir=input_dir)

    # ── STAGE 3: Co-DETR ──
    if 3 in stages:
        print(f"\n{'!'*60}")
        print(f"⚠️  STAGE 3 (Co-DETR) KHÔNG THỂ CHẠY TRONG ENV NÀY!")
        print(f"")
        print(f"  Co-DETR yêu cầu env riêng: conda 'codetr'")
        print(f"  (Python 3.9, torch==1.12.1, mmcv-full==1.6.0, mmdet==2.25.3)")
        print(f"")
        print(f"  Chạy lệnh sau trong env codetr:")
        print(f"")
        print(f"  $ENV_PYTHON stage3_codetr_detector.py \\")
        print(f"      --input {input_dir} \\")
        print(f"      --output {STAGE3_OUTPUT} \\")
        print(f"      --codetr-dir /content/Co-DETR")
        print(f"{'!'*60}\n")

    # ── STAGE 4: IoU Merge ──
    if 4 in stages:
        # Kiểm tra outputs từ các stage trước
        missing = []
        if not STAGE1_OUTPUT.exists():
            missing.append(f"Stage 1: {STAGE1_OUTPUT}")
        if not STAGE2_OUTPUT.exists():
            missing.append(f"Stage 2: {STAGE2_OUTPUT}")
        if not STAGE3_OUTPUT.exists():
            missing.append(f"Stage 3: {STAGE3_OUTPUT}")

        if missing:
            print(f"\n❌ LỖI: Cần chạy các stages sau trước khi merge:")
            for m in missing:
                print(f"  - {m}")
            sys.exit(1)

        with Timer("Stage 4 — IoU Merge + Containment"):
            from stage4_iou_merger import run_stage4
            run_stage4(video_id=video_id)

    # ── Tổng kết ──
    print(f"\n{'#'*60}")
    print(f"#  PIPELINE HOÀN THÀNH — Stages: {stages}")
    print(f"#")
    if STAGE1_OUTPUT.exists():
        print(f"#  Stage 1 output: {STAGE1_OUTPUT}")
    if STAGE2_OUTPUT.exists():
        print(f"#  Stage 2 output: {STAGE2_OUTPUT}")
    if STAGE3_OUTPUT.exists():
        print(f"#  Stage 3 output: {STAGE3_OUTPUT}")
    if STAGE4_OUTPUT.exists():
        print(f"#  Stage 4 output: {STAGE4_OUTPUT}")
    print(f"{'#'*60}\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object Detection Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Chạy Stage 1 + 2 (env transformers==4.46.3):
  python run_pipeline.py --stages 1,2

  # Chạy Stage 3 riêng (env conda codetr):
  # → Xem hướng dẫn trong TEST_COLAB_GUIDE.md

  # Chạy Stage 4 sau khi có đủ output:
  python run_pipeline.py --stages 4

  # Chạy liên tục S1 + S2 + S4 (bỏ S3 nếu đã chạy):
  python run_pipeline.py --stages 1,2,4

  # Chỉ định thư mục keyframes và video ID:
  python run_pipeline.py --stages 1,2 --input /path/to/keyframes --video-id L21_V001
        """
    )
    parser.add_argument("--stages", type=str, default="1,2,4",
                        help="Danh sách stages cần chạy, cách nhau bằng dấu phẩy "
                             "(VD: 1,2,4). Lưu ý: stage 3 cần env riêng.")
    parser.add_argument("--input", type=str, default=None,
                        help="Thư mục chứa keyframes (default: config)")
    parser.add_argument("--video-id", type=str, default=None,
                        help="Video ID (default: config)")
    args = parser.parse_args()

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(",")]

    with Timer("FULL PIPELINE"):
        run_stages(stages, input_dir=args.input, video_id=args.video_id)
