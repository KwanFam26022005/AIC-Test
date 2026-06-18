# -*- coding: utf-8 -*-
"""
stage2_gdino_detector.py — Stage 2: GroundingDINO Object Detection

Môi trường: transformers==4.46.3 + torch (mới)
(Chung env với Stage 1 — RAM++)

Input:  JSON file từ Stage 1 (chứa tags cho từng frame) + thư mục keyframes
Output: JSON file chứa bounding box detections cho từng frame

Cấu hình FP32 chuẩn (benchmark mode — không tối ưu).

Sử dụng:
    python stage2_gdino_detector.py
    python stage2_gdino_detector.py --input /path/to/stage1_output.json --output /path/to/output.json
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    KEYFRAME_INPUT_DIR, GDINO_MODEL_ID,
    GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD,
    STAGE1_OUTPUT, STAGE2_OUTPUT,
)
from config import ensure_output_dirs
from utils import Timer, load_intermediate, save_intermediate


def load_gdino_model(device, model_id=GDINO_MODEL_ID):
    """Nạp mô hình GroundingDINO từ HuggingFace transformers.

    Args:
        device: torch.device
        model_id: HuggingFace model ID

    Returns:
        tuple: (model, processor)
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"[GDINO] Đang nạp mô hình: {model_id} ...")
    print(f"[GDINO] Device: {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # Đo VRAM sau khi load model
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"[GDINO] VRAM sử dụng (model loaded): {vram_mb:.0f} MB")

    print("[GDINO] Mô hình đã sẵn sàng.")
    return model, processor


def build_dino_prompt(tag_list):
    """Chuyển danh sách tag từ RAM++ thành text-prompt cho GroundingDINO.

    Quy ước chuẩn: 1 chuỗi duy nhất, chữ thường, các phrase cách nhau bằng ". "
    và KẾT THÚC bằng dấu chấm.

    Ví dụ: ["person", "kitchen", "apron"] → "person. kitchen. apron."

    Args:
        tag_list: list[str] — danh sách tags

    Returns:
        str: text prompt
    """
    phrases = [tag.lower().strip() for tag in tag_list if tag.strip()]
    return ". ".join(phrases) + "."


def detect_objects_gdino(image, tag_list, model, processor, device,
                         box_threshold=GDINO_BOX_THRESHOLD,
                         text_threshold=GDINO_TEXT_THRESHOLD):
    """Chạy GroundingDINO zero-shot detection với prompt = tags từ RAM++.

    Args:
        image: PIL.Image — ảnh đầu vào
        tag_list: list[str] — danh sách tags làm text prompt
        model: GroundingDINO model
        processor: GroundingDINO processor
        device: torch.device
        box_threshold: ngưỡng confidence cho bounding box
        text_threshold: ngưỡng confidence cho text matching

    Returns:
        list[dict]: mỗi dict = {"label": str, "score": float, "box": [x1,y1,x2,y2]}
    """
    # Ép ảnh về RGB 3 kênh
    image = image.convert("RGB")

    text_prompt = build_dino_prompt(tag_list)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )
    result = results[0]  # batch size = 1

    detections = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        detections.append({
            "label": label.strip(),
            "score": round(score.item(), 4),
            "box": [round(coord, 2) for coord in box.tolist()],  # [x1, y1, x2, y2]
        })

    return detections


def run_stage2(stage1_output=None, output_path=None, input_dir=None):
    """Chạy Stage 2: GroundingDINO detection trên toàn bộ keyframes.

    Args:
        stage1_output: Path — file JSON từ Stage 1 (None → dùng config)
        output_path: Path — file output JSON (None → dùng config)
        input_dir: Path — thư mục chứa keyframes (None → dùng config)

    Returns:
        dict: kết quả {frame_name: [detections]}
    """
    stage1_output = Path(stage1_output) if stage1_output else STAGE1_OUTPUT
    output_path = Path(output_path) if output_path else STAGE2_OUTPUT
    input_dir = Path(input_dir) if input_dir else KEYFRAME_INPUT_DIR

    ensure_output_dirs()

    # Đọc kết quả Stage 1
    stage1_data = load_intermediate(stage1_output)
    tags_per_frame = stage1_data["results"]
    total = len(tags_per_frame)

    if total == 0:
        print("[GDINO] Không có dữ liệu từ Stage 1!")
        return {}

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GDINO] Device: {device}")

    # Load model
    model, processor = load_gdino_model(device)

    # Inference
    results = {}
    total_time = 0.0
    total_detections = 0

    print(f"\n[GDINO] Bắt đầu xử lý {total} keyframes ...")

    for idx, (frame_name, tags) in enumerate(tags_per_frame.items()):
        frame_path = input_dir / frame_name

        if not frame_path.exists():
            print(f"  [SKIP] {frame_name} — file không tồn tại")
            results[frame_name] = []
            continue

        start = time.time()
        image = Image.open(frame_path).convert("RGB")
        img_w, img_h = image.size

        detections = detect_objects_gdino(image, tags, model, processor, device)
        elapsed = time.time() - start
        total_time += elapsed
        total_detections += len(detections)

        # Lưu kèm kích thước ảnh (cần cho Stage 4)
        results[frame_name] = {
            "image_size": [img_w, img_h],
            "tags_used": tags,
            "detections": detections,
        }

        # In tiến độ
        if (idx + 1) % 50 == 0 or idx == total - 1:
            avg_fps = (idx + 1) / total_time if total_time > 0 else 0
            eta_s = (total - idx - 1) / avg_fps if avg_fps > 0 else 0
            print(f"  [{idx + 1}/{total}] {frame_name}: {len(detections)} dets | "
                  f"{elapsed:.3f}s/frame | avg {avg_fps:.1f} fps | "
                  f"ETA: {eta_s:.0f}s")

    # Đo VRAM peak
    peak_vram = 0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"\n[GDINO] VRAM peak: {peak_vram:.0f} MB")

    # Thống kê
    avg_time = total_time / total if total > 0 else 0
    avg_dets = total_detections / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"📊 STAGE 2 — GROUNDING DINO BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"  Model:              {GDINO_MODEL_ID}")
    print(f"  Tổng frames:        {total}")
    print(f"  Tổng thời gian:     {total_time:.2f}s ({total_time/60:.1f} phút)")
    print(f"  Trung bình/frame:   {avg_time:.4f}s ({1/avg_time:.1f} fps)")
    print(f"  Tổng detections:    {total_detections}")
    print(f"  Trung bình dets:    {avg_dets:.1f} dets/frame")
    print(f"  Box threshold:      {GDINO_BOX_THRESHOLD}")
    print(f"  Text threshold:     {GDINO_TEXT_THRESHOLD}")
    if device.type == "cuda":
        print(f"  VRAM peak:          {peak_vram:.0f} MB")
    print(f"{'='*60}")

    # Đóng gói output
    output_data = {
        "stage": "stage2_gdino_detector",
        "model": GDINO_MODEL_ID,
        "device": str(device),
        "precision": "FP32",
        "box_threshold": GDINO_BOX_THRESHOLD,
        "text_threshold": GDINO_TEXT_THRESHOLD,
        "total_frames": total,
        "total_time_s": round(total_time, 2),
        "avg_time_per_frame_s": round(avg_time, 4),
        "avg_fps": round(1 / avg_time, 2) if avg_time > 0 else 0,
        "total_detections": total_detections,
        "vram_peak_mb": round(peak_vram, 0) if device.type == "cuda" else None,
        "results": results,
    }

    save_intermediate(output_data, output_path)

    # Giải phóng VRAM
    del model, processor
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[GDINO] Đã giải phóng VRAM.")

    return results


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: GroundingDINO Detection")
    parser.add_argument("--stage1-output", type=str, default=None,
                        help="File JSON từ Stage 1 (default: config)")
    parser.add_argument("--output", type=str, default=None,
                        help="File output JSON (default: config)")
    parser.add_argument("--input", type=str, default=None,
                        help="Thư mục chứa keyframes (default: config)")
    args = parser.parse_args()

    with Timer("Stage 2 — GroundingDINO Detection"):
        run_stage2(
            stage1_output=args.stage1_output,
            output_path=args.output,
            input_dir=args.input,
        )
