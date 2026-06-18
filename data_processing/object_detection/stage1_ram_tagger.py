# -*- coding: utf-8 -*-
"""
stage1_ram_tagger.py — Stage 1: RAM++ Tag Detection

Môi trường: transformers==4.46.3 + torch (mới) + recognize-anything
(Chung env với Stage 2 — GroundingDINO)

Input:  Thư mục chứa keyframes (*.jpg)
Output: JSON file chứa tags cho từng frame

Cấu hình FP32 chuẩn (benchmark mode — không tối ưu).

Sử dụng:
    python stage1_ram_tagger.py
    python stage1_ram_tagger.py --input /path/to/keyframes --output /path/to/output.json
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Thêm thư mục cha vào sys.path để import config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    KEYFRAME_INPUT_DIR, RAM_IMAGE_SIZE, RAM_CHECKPOINT,
    RAM_CHECKPOINT_URL, RAM_VIT_TYPE, STAGE1_OUTPUT,
)
from config import ensure_output_dirs
from utils import Timer, list_keyframe_paths, save_intermediate


def download_ram_checkpoint(checkpoint_path, url):
    """Tải checkpoint RAM++ nếu chưa có."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"[RAM++] Checkpoint đã tồn tại: {checkpoint_path}")
        return

    print(f"[RAM++] Đang tải checkpoint từ {url} ...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(url, str(checkpoint_path))
    print(f"[RAM++] Đã tải xong: {checkpoint_path}")


def load_ram_model(device, checkpoint_path=RAM_CHECKPOINT,
                   image_size=RAM_IMAGE_SIZE, vit_type=RAM_VIT_TYPE):
    """Nạp mô hình RAM++ Swin-Large.

    Args:
        device: torch.device
        checkpoint_path: đường dẫn tới file .pth
        image_size: kích thước ảnh đầu vào (384)
        vit_type: loại backbone ('swin_l')

    Returns:
        tuple: (model, transform)
    """
    from ram.models import ram_plus
    from ram import get_transform

    print(f"[RAM++] Đang nạp mô hình RAM++ ({vit_type}, img={image_size}) ...")
    print(f"[RAM++] Checkpoint: {checkpoint_path}")
    print(f"[RAM++] Device: {device}")

    model = ram_plus(pretrained=checkpoint_path, image_size=image_size, vit=vit_type)
    model.eval()
    model = model.to(device)

    transform = get_transform(image_size=image_size)

    # Đo VRAM sau khi load model
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"[RAM++] VRAM sử dụng (model loaded): {vram_mb:.0f} MB")

    print("[RAM++] Mô hình đã sẵn sàng.")
    return model, transform


def get_ram_tags(image, model, transform, device):
    """Chạy RAM++ trên 1 ảnh PIL, trả về danh sách tag.

    Args:
        image: PIL.Image — ảnh đầu vào
        model: RAM++ model
        transform: RAM++ transform
        device: torch.device

    Returns:
        list[str]: danh sách tags (VD: ["person", "field", "cow"])
    """
    from ram import inference_ram as inference

    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        res = inference(image_tensor, model)

    raw_tags = res[0]  # VD: "dog | grass | playing"
    tag_list = [t.strip() for t in raw_tags.split("|") if t.strip()]
    return tag_list


def run_stage1(input_dir=None, output_path=None):
    """Chạy Stage 1: RAM++ tag extraction trên toàn bộ keyframes.

    Args:
        input_dir: Path — thư mục chứa keyframes (None → dùng config default)
        output_path: Path — file output JSON (None → dùng config default)

    Returns:
        dict: kết quả {frame_name: [tags]}
    """
    input_dir = Path(input_dir) if input_dir else KEYFRAME_INPUT_DIR
    output_path = Path(output_path) if output_path else STAGE1_OUTPUT

    ensure_output_dirs()

    # Liệt kê keyframes
    frame_paths = list_keyframe_paths(input_dir)
    if not frame_paths:
        print("[RAM++] Không tìm thấy keyframe nào!")
        return {}

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RAM++] Device: {device}")

    # Download checkpoint nếu cần
    download_ram_checkpoint(RAM_CHECKPOINT, RAM_CHECKPOINT_URL)

    # Load model
    model, transform = load_ram_model(device)

    # Inference
    results = {}
    total = len(frame_paths)
    total_time = 0.0

    print(f"\n[RAM++] Bắt đầu xử lý {total} keyframes ...")

    for idx, frame_path in enumerate(frame_paths):
        frame_name = frame_path.name  # VD: "001.jpg"

        start = time.time()
        image = Image.open(frame_path).convert("RGB")
        tags = get_ram_tags(image, model, transform, device)
        elapsed = time.time() - start
        total_time += elapsed

        results[frame_name] = tags

        # In tiến độ mỗi 50 frames hoặc frame cuối
        if (idx + 1) % 50 == 0 or idx == total - 1:
            avg_fps = (idx + 1) / total_time
            eta_s = (total - idx - 1) / avg_fps if avg_fps > 0 else 0
            print(f"  [{idx + 1}/{total}] {frame_name}: {len(tags)} tags | "
                  f"{elapsed:.3f}s/frame | avg {avg_fps:.1f} fps | "
                  f"ETA: {eta_s:.0f}s")

    # Đo VRAM peak
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"\n[RAM++] VRAM peak: {peak_vram:.0f} MB")

    # Thống kê
    total_tags = sum(len(t) for t in results.values())
    avg_tags = total_tags / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"📊 STAGE 1 — RAM++ BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"  Tổng frames:        {total}")
    print(f"  Tổng thời gian:     {total_time:.2f}s ({total_time/60:.1f} phút)")
    print(f"  Trung bình/frame:   {avg_time:.4f}s ({1/avg_time:.1f} fps)")
    print(f"  Tổng tags:          {total_tags}")
    print(f"  Trung bình tags:    {avg_tags:.1f} tags/frame")
    if device.type == "cuda":
        print(f"  VRAM peak:          {peak_vram:.0f} MB")
    print(f"{'='*60}")

    # Đóng gói output
    output_data = {
        "stage": "stage1_ram_tagger",
        "model": f"RAM++ {RAM_VIT_TYPE} (img={RAM_IMAGE_SIZE})",
        "device": str(device),
        "precision": "FP32",
        "total_frames": total,
        "total_time_s": round(total_time, 2),
        "avg_time_per_frame_s": round(avg_time, 4),
        "avg_fps": round(1 / avg_time, 2) if avg_time > 0 else 0,
        "vram_peak_mb": round(peak_vram, 0) if device.type == "cuda" else None,
        "results": results,
    }

    save_intermediate(output_data, output_path)

    # Giải phóng VRAM
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[RAM++] Đã giải phóng VRAM.")

    return results


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: RAM++ Tag Detection")
    parser.add_argument("--input", type=str, default=None,
                        help="Thư mục chứa keyframes (default: config)")
    parser.add_argument("--output", type=str, default=None,
                        help="File output JSON (default: config)")
    args = parser.parse_args()

    with Timer("Stage 1 — RAM++ Tag Detection"):
        run_stage1(input_dir=args.input, output_path=args.output)
