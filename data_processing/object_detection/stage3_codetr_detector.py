# -*- coding: utf-8 -*-
"""
stage3_codetr_detector.py — Stage 3: Co-DETR ViT-L (LVIS) Object Detection

⚠️ MÔI TRƯỜNG RIÊNG BIỆT — KHÔNG chạy chung env với Stage 1+2!

Môi trường: conda "codetr" — Python 3.9, torch==1.12.1+cu113,
            mmcv-full==1.6.0, mmdet==2.25.3

Input:  Thư mục chứa keyframes (*.jpg)
Output: JSON file chứa LVIS detections cho từng frame

Cấu hình FP32 chuẩn (benchmark mode — không tối ưu).

=== CHẠY TRÊN COLAB (conda env codetr) ===
    $ENV_PYTHON stage3_codetr_detector.py --input /path/to/keyframes \\
                                          --output /path/to/output.json \\
                                          --codetr-dir /content/Co-DETR

=== CHẠY TRÊN SERVER (conda env codetr) ===
    conda activate codetr
    python stage3_codetr_detector.py --input /path/to/keyframes \\
                                     --output /path/to/output.json \\
                                     --codetr-dir /path/to/Co-DETR
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

# LƯU Ý: KHÔNG import config.py ở đây vì config.py có thể import
# các module không tương thích với env codetr (torch cũ).
# Thay vào đó, nhận tất cả config qua CLI arguments.


def load_codetr_model(config_path, checkpoint_path, device="cuda:0"):
    """Nạp mô hình Co-DETR ViT-L LVIS.

    Args:
        config_path: str — đường dẫn tới file config MMDetection
        checkpoint_path: str — đường dẫn tới file checkpoint .pth
        device: str — thiết bị (default: "cuda:0")

    Returns:
        model: MMDetection model đã load weights
    """
    from mmdet.apis import init_detector
    from mmdet.datasets.lvis import LVISV1Dataset

    print(f"[Co-DETR] Đang nạp mô hình Co-DETR ViT-Large (LVIS) ...")
    print(f"[Co-DETR] Config:     {config_path}")
    print(f"[Co-DETR] Checkpoint: {checkpoint_path}")
    print(f"[Co-DETR] Device:     {device}")

    model = init_detector(config_path, checkpoint_path, device=device)

    # Ép mô hình nạp đúng 1203 tên lớp từ LVISV1Dataset
    model.CLASSES = LVISV1Dataset.CLASSES
    print(f"[Co-DETR] Đã đồng bộ {len(model.CLASSES)} lớp LVIS.")

    # Đo VRAM
    import torch
    if "cuda" in device:
        dev = torch.device(device)
        vram_mb = torch.cuda.memory_allocated(dev) / 1024 / 1024
        print(f"[Co-DETR] VRAM sử dụng (model loaded): {vram_mb:.0f} MB")

    print("[Co-DETR] Mô hình đã sẵn sàng.")
    return model


def detect_objects_codetr(model, img_path, score_threshold=0.3):
    """Chạy Co-DETR inference trên 1 ảnh.

    Args:
        model: MMDetection model
        img_path: str — đường dẫn tới ảnh
        score_threshold: float — ngưỡng confidence

    Returns:
        list[dict]: mỗi dict = {"label": str, "score": float, "box": [x1,y1,x2,y2]}
    """
    from mmdet.apis import inference_detector
    import cv2

    # Chạy inference
    result = inference_detector(model, img_path)

    # Đọc kích thước ảnh
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    detections = []
    for class_idx, bboxes in enumerate(result):
        class_name = model.CLASSES[class_idx]
        for bbox in bboxes:
            score = float(bbox[4])
            if score >= score_threshold:
                box_coords = [round(float(coord), 2) for coord in bbox[:4]]
                detections.append({
                    "label": class_name.upper(),
                    "score": round(score, 4),
                    "box": box_coords,  # [x1, y1, x2, y2]
                })

    return detections, [img_w, img_h]


def run_stage3(input_dir, output_path, codetr_dir,
               score_threshold=0.3, device="cuda:0"):
    """Chạy Stage 3: Co-DETR ViT-L LVIS detection trên toàn bộ keyframes.

    Args:
        input_dir: str — thư mục chứa keyframes
        output_path: str — file output JSON
        codetr_dir: str — thư mục gốc của repo Co-DETR đã clone
        score_threshold: float — ngưỡng confidence
        device: str — thiết bị GPU
    """
    import torch

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    codetr_dir = Path(codetr_dir)

    # Thêm Co-DETR vào sys.path
    sys.path.insert(0, str(codetr_dir))
    print(f"[Co-DETR] Đã thêm {codetr_dir} vào sys.path")

    # Xác định đường dẫn config và checkpoint
    config_path = str(codetr_dir / "projects" / "configs" / "co_dino_vit" /
                      "co_dino_5scale_lsj_vit_large_lvis.py")
    checkpoint_path = str(codetr_dir / "checkpoints" / "pytorch_model_lvis.pth")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config không tồn tại: {config_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

    # Liệt kê keyframes
    extensions = (".jpg", ".jpeg", ".png")
    frame_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])
    total = len(frame_paths)
    print(f"[Co-DETR] Tìm thấy {total} keyframes trong {input_dir}")

    if total == 0:
        print("[Co-DETR] Không tìm thấy keyframe nào!")
        return

    # Load model
    model = load_codetr_model(config_path, checkpoint_path, device=device)

    # Inference
    results = {}
    total_time = 0.0
    total_detections = 0

    print(f"\n[Co-DETR] Bắt đầu xử lý {total} keyframes (score >= {score_threshold}) ...")

    for idx, frame_path in enumerate(frame_paths):
        frame_name = frame_path.name

        start = time.time()
        detections, img_size = detect_objects_codetr(
            model, str(frame_path), score_threshold=score_threshold
        )
        elapsed = time.time() - start
        total_time += elapsed
        total_detections += len(detections)

        results[frame_name] = {
            "image_size": img_size,
            "detections": detections,
        }

        # In tiến độ
        if (idx + 1) % 50 == 0 or idx == total - 1:
            avg_fps = (idx + 1) / total_time if total_time > 0 else 0
            eta_s = (total - idx - 1) / avg_fps if avg_fps > 0 else 0
            print(f"  [{idx + 1}/{total}] {frame_name}: {len(detections)} dets | "
                  f"{elapsed:.3f}s/frame | avg {avg_fps:.2f} fps | "
                  f"ETA: {eta_s:.0f}s")

    # VRAM peak
    peak_vram = 0
    if "cuda" in device:
        dev = torch.device(device)
        peak_vram = torch.cuda.max_memory_allocated(dev) / 1024 / 1024
        print(f"\n[Co-DETR] VRAM peak: {peak_vram:.0f} MB")

    # Thống kê
    avg_time = total_time / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"📊 STAGE 3 — Co-DETR ViT-L LVIS BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"  Model:              Co-DETR ViT-L (LVIS 1203 classes)")
    print(f"  Tổng frames:        {total}")
    print(f"  Tổng thời gian:     {total_time:.2f}s ({total_time/60:.1f} phút)")
    print(f"  Trung bình/frame:   {avg_time:.4f}s ({1/avg_time:.2f} fps)")
    print(f"  Tổng detections:    {total_detections}")
    print(f"  Trung bình dets:    {total_detections/total:.1f} dets/frame")
    print(f"  Score threshold:    {score_threshold}")
    if "cuda" in device:
        print(f"  VRAM peak:          {peak_vram:.0f} MB")
    print(f"{'='*60}")

    # Đóng gói output
    output_data = {
        "stage": "stage3_codetr_detector",
        "model": "Co-DETR ViT-L (co_dino_5scale_lsj_vit_large_lvis)",
        "device": device,
        "precision": "FP32",
        "score_threshold": score_threshold,
        "total_frames": total,
        "total_time_s": round(total_time, 2),
        "avg_time_per_frame_s": round(avg_time, 4),
        "avg_fps": round(1 / avg_time, 2) if avg_time > 0 else 0,
        "total_detections": total_detections,
        "vram_peak_mb": round(peak_vram, 0) if "cuda" in device else None,
        "results": results,
    }

    # Lưu output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"[I/O] Đã lưu kết quả → {output_path} "
          f"({output_path.stat().st_size / 1024:.1f} KB)")

    # Giải phóng VRAM
    del model
    torch.cuda.empty_cache()
    print("[Co-DETR] Đã giải phóng VRAM.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Co-DETR ViT-L LVIS Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Trên Colab (dùng conda env codetr):
  $ENV_PYTHON stage3_codetr_detector.py \\
      --input /content/drive/MyDrive/keyframe_test/L21_V001 \\
      --output /content/output/intermediate/stage3_codetr_detections.json \\
      --codetr-dir /content/Co-DETR \\
      --score-threshold 0.3

  # Trên server:
  conda activate codetr
  python stage3_codetr_detector.py \\
      --input /data/keyframes/L21_V001 \\
      --output /output/intermediate/stage3_codetr_detections.json \\
      --codetr-dir /opt/Co-DETR
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Thư mục chứa keyframes")
    parser.add_argument("--output", type=str, required=True,
                        help="File output JSON")
    parser.add_argument("--codetr-dir", type=str, required=True,
                        help="Thư mục gốc repo Co-DETR đã clone")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Ngưỡng confidence (default: 0.3)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Thiết bị GPU (default: cuda:0)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"⏱️  BẮT ĐẦU: Stage 3 — Co-DETR ViT-L LVIS Detection")
    print(f"{'='*60}")

    start_total = time.time()
    run_stage3(
        input_dir=args.input,
        output_path=args.output,
        codetr_dir=args.codetr_dir,
        score_threshold=args.score_threshold,
        device=args.device,
    )
    elapsed_total = time.time() - start_total

    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH: Stage 3 — Co-DETR ViT-L LVIS Detection")
    print(f"⏱️  Thời gian tổng: {elapsed_total:.2f}s ({elapsed_total/60:.1f} phút)")
    print(f"{'='*60}\n")
