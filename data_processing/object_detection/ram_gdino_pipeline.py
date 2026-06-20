# -*- coding: utf-8 -*-
"""
ram_gdino_pipeline.py — RAM++ → GroundingDINO Object Detection Pipeline

Pipeline chính cho object detection trong hệ thống MMRS.
Sử dụng RAM++ để sinh tags, GroundingDINO để detect bounding boxes,
sau đó apply tag canonicalization (dedup/NMS/normalize) và output JSONL.

Môi trường yêu cầu:
    - transformers==4.46.3 (PHẢI PIN đúng version — xem groundingdino_ram++.py)
    - recognize-anything (RAM++)
    - torch + torchvision (GPU recommended)
    - tqdm, Pillow

Chạy được trên cả:
    - Server (toàn bộ frames, batch processing)
    - Google Colab (test thử với keyframe_test/L21_V001)

Output schema (mỗi dòng trong JSONL):
    {
        "frame_id": "L21_V001_001",
        "video_id": "L21_V001",
        "frame_idx": 1,
        "image_size": [1280, 720],
        "tags": ["fish", "person", "market"],
        "objects": [
            {"label": "PERSON", "score": 0.82, "box": [97.5, 182.0, 519.5, 718.4]}
        ],
        "object_summary": ["PERSON", "FISH"],
        "object_counts": {"PERSON": 2, "FISH": 1}
    }

Usage:
    # Single video (Colab hoặc server)
    python ram_gdino_pipeline.py \\
        --input /path/to/keyframes/L21_V001 \\
        --output /path/to/output/L21_V001.jsonl \\
        --video-id L21_V001

    # Batch — tất cả video dirs trong 1 thư mục cha
    python ram_gdino_pipeline.py \\
        --input /path/to/all_keyframes \\
        --output /path/to/output \\
        --batch
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── Local imports ────────────────────────────────────────────────────────────
# tag_canonicalization.py phải nằm cùng thư mục
from tag_canonicalization import (
    filter_tags_for_dino,
    build_dino_prompt,
    canonicalize_detections,
)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# RAM++ config
RAM_IMAGE_SIZE = 384
RAM_CHECKPOINT_FILENAME = "ram_plus_swin_large_14m.pth"
RAM_CHECKPOINT_URL = (
    "https://huggingface.co/xinyu1205/recognize-anything-plus-model"
    "/resolve/main/ram_plus_swin_large_14m.pth"
)

# GroundingDINO config
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Detection thresholds (có thể override qua CLI)
DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_NMS_IOU_THRESHOLD = 0.7


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device():
    """Auto-detect GPU/CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — running on CPU (will be slow)")
    return device


def find_ram_checkpoint():
    """Tìm checkpoint RAM++ theo thứ tự ưu tiên.

    Thứ tự tìm:
        1. ./pretrained/ram_plus_swin_large_14m.pth  (Colab)
        2. ../pretrained/ram_plus_swin_large_14m.pth
        3. Biến môi trường RAM_CHECKPOINT_PATH
    """
    candidates = [
        os.path.join("pretrained", RAM_CHECKPOINT_FILENAME),
        os.path.join("..", "pretrained", RAM_CHECKPOINT_FILENAME),
        os.path.join(os.path.dirname(__file__), "pretrained", RAM_CHECKPOINT_FILENAME),
    ]

    # Check env var
    env_path = os.environ.get("RAM_CHECKPOINT_PATH")
    if env_path:
        candidates.insert(0, env_path)

    for path in candidates:
        if os.path.isfile(path):
            log.info(f"RAM++ checkpoint found: {path}")
            return path

    raise FileNotFoundError(
        f"RAM++ checkpoint '{RAM_CHECKPOINT_FILENAME}' not found.\n"
        f"Searched: {candidates}\n"
        f"Download: wget -O pretrained/{RAM_CHECKPOINT_FILENAME} {RAM_CHECKPOINT_URL}"
    )


def load_ram_model(device):
    """Load RAM++ model (Swin-Large, 14M tags)."""
    from ram.models import ram_plus
    from ram import get_transform

    checkpoint = find_ram_checkpoint()
    log.info("Loading RAM++ model (Swin-Large)...")

    model = ram_plus(pretrained=checkpoint, image_size=RAM_IMAGE_SIZE, vit="swin_l")
    model.eval()
    model = model.to(device)

    transform = get_transform(image_size=RAM_IMAGE_SIZE)
    log.info("RAM++ ready.")
    return model, transform


def load_gdino_model(device):
    """Load GroundingDINO model (HuggingFace transformers)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    log.info(f"Loading GroundingDINO: {GDINO_MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)
    model = model.to(device)
    model.eval()

    log.info("GroundingDINO ready.")
    return model, processor


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def get_ram_tags(image, ram_model, ram_transform, device):
    """Chạy RAM++ trên 1 ảnh PIL → list[str] tags.

    Args:
        image: PIL.Image — ảnh đã convert RGB
        ram_model: loaded RAM++ model
        ram_transform: RAM++ image transform
        device: torch.device

    Returns:
        list[str]: raw tags từ RAM++ (chưa filter)
    """
    from ram import inference_ram as inference

    tensor = ram_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        result = inference(tensor, ram_model)

    raw_tags_str = result[0]  # e.g. "dog | grass | playing"
    tags = [t.strip() for t in raw_tags_str.split("|") if t.strip()]
    return tags


def detect_objects_gdino(image, tag_list, gdino_model, gdino_processor, device,
                         box_threshold=DEFAULT_BOX_THRESHOLD,
                         text_threshold=DEFAULT_TEXT_THRESHOLD):
    """Chạy GroundingDINO zero-shot detection.

    Args:
        image: PIL.Image — ảnh RGB
        tag_list: list[str] — filtered tags (from tag_canonicalization)
        gdino_model: loaded GroundingDINO model
        gdino_processor: GroundingDINO processor
        device: torch.device
        box_threshold: confidence threshold cho bbox
        text_threshold: confidence threshold cho text matching

    Returns:
        list[dict]: [{"label": str, "score": float, "box": [x0,y0,x1,y1]}]
    """
    if not tag_list:
        return []

    text_prompt = build_dino_prompt(tag_list)
    inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )
    result = results[0]  # batch_size = 1

    detections = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        detections.append({
            "label": label.strip(),
            "score": round(score.item(), 4),
            "box": [round(c, 2) for c in box.tolist()],
        })

    return detections


# =============================================================================
# FRAME PROCESSING
# =============================================================================

def process_single_frame(image_path, video_id, frame_idx,
                         ram_model, ram_transform,
                         gdino_model, gdino_processor,
                         device, box_threshold, text_threshold,
                         nms_iou_threshold):
    """Xử lý 1 frame: RAM++ → filter → GDINO → NMS → normalize → output dict.

    Args:
        image_path: str — đường dẫn tới ảnh
        video_id: str — ID video (e.g. "L21_V001")
        frame_idx: int — index của frame (từ filename)
        ram_model, ram_transform: RAM++ model + transform
        gdino_model, gdino_processor: GroundingDINO model + processor
        device: torch.device
        box_threshold, text_threshold: detection thresholds
        nms_iou_threshold: IoU threshold cho class-agnostic NMS

    Returns:
        dict: frame metadata theo output schema
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # Build frame_id from video_id + frame filename (without extension)
    frame_name = Path(image_path).stem
    frame_id = f"{video_id}_{frame_name}"

    # Step 1: RAM++ → raw tags
    raw_tags = get_ram_tags(image, ram_model, ram_transform, device)

    # Step 2: Filter tags (Layer 1 — tag canonicalization)
    filtered_tags = filter_tags_for_dino(raw_tags)

    # Step 3: GroundingDINO → raw detections
    raw_detections = detect_objects_gdino(
        image, filtered_tags,
        gdino_model, gdino_processor, device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Step 4: Canonicalize — NMS (Layer 2) + normalize/scene-filter (Layer 3)
    final_objects, scene_labels = canonicalize_detections(
        raw_detections, img_w, img_h,
        nms_iou_threshold=nms_iou_threshold,
    )

    # Step 5: Build output
    # Merge scene labels back into tags (scene-level detections become tags)
    all_tags = list(dict.fromkeys(
        [t.lower() for t in raw_tags] + scene_labels
    ))

    # Object summary and counts
    label_counts = Counter(obj["label"] for obj in final_objects)
    object_summary = sorted(label_counts.keys())

    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "image_size": [img_w, img_h],
        "tags": all_tags,
        "objects": final_objects,
        "object_summary": object_summary,
        "object_counts": dict(label_counts),
    }


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def discover_frames(input_dir):
    """Tìm tất cả ảnh trong thư mục, sorted theo tên.

    Args:
        input_dir: str — đường dẫn thư mục chứa keyframes

    Returns:
        list[str]: sorted list đường dẫn tuyệt đối tới ảnh
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    frames = sorted([
        str(f) for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not frames:
        raise FileNotFoundError(f"No image files found in: {input_dir}")

    log.info(f"Found {len(frames)} frames in {input_dir}")
    return frames


def discover_video_dirs(input_dir):
    """Tìm tất cả video directories (batch mode).

    Quy ước: mỗi thư mục con chứa keyframes = 1 video.
    Tên thư mục = video_id.

    Args:
        input_dir: str — thư mục cha chứa nhiều video dirs

    Returns:
        list[tuple]: [(video_id, video_dir_path), ...]
    """
    input_path = Path(input_dir)
    video_dirs = []

    for d in sorted(input_path.iterdir()):
        if d.is_dir():
            # Check if dir contains at least 1 image
            has_images = any(
                f.suffix.lower() in IMAGE_EXTENSIONS
                for f in d.iterdir() if f.is_file()
            )
            if has_images:
                video_dirs.append((d.name, str(d)))

    if not video_dirs:
        raise FileNotFoundError(
            f"No video directories (with images) found in: {input_dir}"
        )

    log.info(f"Found {len(video_dirs)} video directories in {input_dir}")
    return video_dirs


def load_existing_frame_ids(output_path):
    """Load frame_ids đã xử lý từ JSONL file (resume support).

    Args:
        output_path: str — đường dẫn file JSONL

    Returns:
        set[str]: tập frame_ids đã có trong output
    """
    existing = set()
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        doc = json.loads(line)
                        existing.add(doc.get("frame_id", ""))
                    except json.JSONDecodeError:
                        continue
        if existing:
            log.info(f"Resume: found {len(existing)} existing frames in {output_path}")
    return existing


def extract_frame_idx(filename):
    """Trích xuất frame index từ tên file.

    Hỗ trợ: "001.jpg" → 1, "frame_089.jpg" → 89, "L21_V001_001.jpg" → 1

    Args:
        filename: str — tên file (có hoặc không có extension)

    Returns:
        int: frame index
    """
    stem = Path(filename).stem
    # Lấy phần số cuối cùng trong tên file
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        else:
            if digits:
                break
    return int(digits) if digits else 0


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_single_video(input_dir, output_path, video_id,
                     ram_model, ram_transform,
                     gdino_model, gdino_processor,
                     device, args):
    """Xử lý tất cả frames trong 1 video directory.

    Args:
        input_dir: str — thư mục chứa keyframes
        output_path: str — đường dẫn file JSONL output
        video_id: str — ID video
        ram_model, ram_transform: RAM++ model
        gdino_model, gdino_processor: GroundingDINO model
        device: torch.device
        args: parsed CLI arguments

    Returns:
        int: số frames đã xử lý thành công
    """
    frames = discover_frames(input_dir)

    # Resume support
    existing_ids = set()
    if args.resume:
        existing_ids = load_existing_frame_ids(output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    # Open in append mode for resume, write mode otherwise
    write_mode = "a" if args.resume and existing_ids else "w"

    with open(output_path, write_mode, encoding="utf-8") as f_out:
        pbar = tqdm(frames, desc=f"[{video_id}]", unit="frame")
        for image_path in pbar:
            frame_idx = extract_frame_idx(image_path)
            frame_name = Path(image_path).stem
            frame_id = f"{video_id}_{frame_name}"

            # Skip if already processed (resume)
            if frame_id in existing_ids:
                skipped += 1
                continue

            try:
                result = process_single_frame(
                    image_path, video_id, frame_idx,
                    ram_model, ram_transform,
                    gdino_model, gdino_processor,
                    device,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                )

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()  # Flush after each frame for safety
                processed += 1

                # Update progress bar
                n_obj = len(result["objects"])
                pbar.set_postfix(
                    objects=n_obj,
                    tags=len(result["tags"]),
                    ok=processed,
                    err=errors,
                )

            except Exception as e:
                errors += 1
                log.error(f"Error processing {image_path}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()

    elapsed = time.time() - start_time
    fps = processed / elapsed if elapsed > 0 else 0

    log.info(
        f"[{video_id}] Done: {processed} processed, {skipped} skipped, "
        f"{errors} errors | {elapsed:.1f}s ({fps:.1f} fps)"
    )
    log.info(f"Output: {output_path}")

    return processed


def run_batch(input_dir, output_dir, ram_model, ram_transform,
              gdino_model, gdino_processor, device, args):
    """Batch mode: xử lý tất cả video dirs trong thư mục cha.

    Args:
        input_dir: str — thư mục cha chứa nhiều video directories
        output_dir: str — thư mục output (1 JSONL per video)
        ram_model, ram_transform: RAM++ model
        gdino_model, gdino_processor: GroundingDINO model
        device: torch.device
        args: parsed CLI arguments

    Returns:
        int: tổng số frames đã xử lý
    """
    video_dirs = discover_video_dirs(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    total_videos = len(video_dirs)

    for idx, (video_id, video_path) in enumerate(video_dirs, 1):
        log.info(f"=== Video {idx}/{total_videos}: {video_id} ===")

        output_path = os.path.join(output_dir, f"{video_id}.jsonl")
        n = run_single_video(
            video_path, output_path, video_id,
            ram_model, ram_transform,
            gdino_model, gdino_processor,
            device, args,
        )
        total_processed += n

    log.info(f"=== Batch complete: {total_processed} frames across {total_videos} videos ===")
    return total_processed


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="RAM++ → GroundingDINO Object Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video (Colab test)
  python ram_gdino_pipeline.py \\
      --input /content/keyframes_test/L21_V001 \\
      --output /content/output/L21_V001.jsonl \\
      --video-id L21_V001

  # Batch — all videos
  python ram_gdino_pipeline.py \\
      --input /data/keyframes \\
      --output /data/output \\
      --batch
        """,
    )

    # Required
    parser.add_argument(
        "--input", required=True,
        help="Input directory (keyframes for single video, or parent dir for batch)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path (JSONL file for single, directory for batch)",
    )

    # Mode
    parser.add_argument(
        "--video-id", default=None,
        help="Video ID (single video mode). If not set, inferred from input dir name.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: process all video subdirectories in --input",
    )

    # Thresholds
    parser.add_argument(
        "--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD,
        help=f"GroundingDINO box confidence threshold (default: {DEFAULT_BOX_THRESHOLD})",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD,
        help=f"GroundingDINO text confidence threshold (default: {DEFAULT_TEXT_THRESHOLD})",
    )
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=DEFAULT_NMS_IOU_THRESHOLD,
        help=f"Class-agnostic NMS IoU threshold (default: {DEFAULT_NMS_IOU_THRESHOLD})",
    )

    # Options
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume processing: skip frames already in output JSONL",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print full traceback on errors",
    )

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    # Validate transformers version
    try:
        import transformers
        ver = transformers.__version__
        if not ver.startswith("4.46"):
            log.warning(
                f"transformers=={ver} detected. Recommended: 4.46.3 "
                f"(required for RAM++ compatibility). "
                f"Run: pip install transformers==4.46.3"
            )
    except ImportError:
        log.error("transformers not installed. Run: pip install transformers==4.46.3")
        sys.exit(1)

    # Setup
    device = get_device()

    log.info("Loading models...")
    ram_model, ram_transform = load_ram_model(device)
    gdino_model, gdino_processor = load_gdino_model(device)

    log.info(
        f"Config: box_threshold={args.box_threshold}, "
        f"text_threshold={args.text_threshold}, "
        f"nms_iou_threshold={args.nms_iou_threshold}"
    )

    # Run
    if args.batch:
        run_batch(
            args.input, args.output,
            ram_model, ram_transform,
            gdino_model, gdino_processor,
            device, args,
        )
    else:
        # Single video mode
        video_id = args.video_id or Path(args.input).name
        run_single_video(
            args.input, args.output, video_id,
            ram_model, ram_transform,
            gdino_model, gdino_processor,
            device, args,
        )


if __name__ == "__main__":
    main()
