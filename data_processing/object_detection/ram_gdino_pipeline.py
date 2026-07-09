# -*- coding: utf-8 -*-
"""RAM++ -> GroundingDINO object detection runner for A5000 servers.

The pipeline is independent from OCR. It reads keyframes, uses RAM++ to
generate image tags, filters those tags into GroundingDINO prompts, runs
zero-shot detection, postprocesses boxes, and appends one JSON object per
frame to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from tqdm import tqdm

from tag_canonicalization import (
    build_dino_prompt,
    canonicalize_detections,
    filter_tags_for_dino,
)


LOG = logging.getLogger("object_detection")

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCHEMA_VERSION = "ram_gdino_object_detection_v1"

RAM_CHECKPOINT_FILENAME = "ram_plus_swin_large_14m.pth"
RAM_CHECKPOINT_URL = (
    "https://huggingface.co/xinyu1205/recognize-anything-plus-model"
    "/resolve/main/ram_plus_swin_large_14m.pth"
)

DEFAULTS: dict[str, Any] = {
    "input": None,
    "output": None,
    "video_id": None,
    "batch": False,
    "pattern": "*.jpg",
    "limit": None,
    "resume": False,
    "summary_output": None,
    "device": "auto",
    "hf_cache_dir": None,
    "ram_checkpoint": None,
    "ram_image_size": 384,
    "ram_batch_size": 16,
    "max_prompt_tags": 25,
    "gdino_model_id": "IDEA-Research/grounding-dino-base",
    "image_max_side": 1280,
    "box_threshold": 0.30,
    "text_threshold": 0.25,
    "nms_iou_threshold": 0.70,
    "scene_area_threshold": 0.60,
    "amp": True,
    "allow_tf32": True,
    "flush_every": 1,
    "quiet": False,
    "debug": False,
}

CONFIG_KEY_MAP = {
    "io.input": "input",
    "io.output": "output",
    "io.video_id": "video_id",
    "io.batch": "batch",
    "io.pattern": "pattern",
    "io.limit": "limit",
    "io.resume": "resume",
    "io.summary_output": "summary_output",
    "io.flush_every": "flush_every",
    "runtime.device": "device",
    "runtime.hf_cache_dir": "hf_cache_dir",
    "runtime.amp": "amp",
    "runtime.allow_tf32": "allow_tf32",
    "ram.checkpoint": "ram_checkpoint",
    "ram.image_size": "ram_image_size",
    "ram.batch_size": "ram_batch_size",
    "ram.max_prompt_tags": "max_prompt_tags",
    "grounding_dino.model_id": "gdino_model_id",
    "grounding_dino.image_max_side": "image_max_side",
    "grounding_dino.box_threshold": "box_threshold",
    "grounding_dino.text_threshold": "text_threshold",
    "postprocess.nms_iou_threshold": "nms_iou_threshold",
    "postprocess.scene_area_threshold": "scene_area_threshold",
}


def setup_logging(quiet: bool = False, debug: bool = False) -> None:
    if debug:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def flatten_dict(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(flatten_dict(item, name))
        else:
            flat[name] = item
    return flat


def load_config_defaults(config_path: str | None) -> dict[str, Any]:
    defaults = dict(DEFAULTS)
    if not config_path:
        return defaults

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using --config.") from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    flat = flatten_dict(payload)
    for key, value in flat.items():
        if key in defaults:
            defaults[key] = value
        elif key in CONFIG_KEY_MAP:
            defaults[CONFIG_KEY_MAP[key]] = value

    return defaults


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAM++ -> GroundingDINO object detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional YAML config path.")

    parser.add_argument("--input", default=defaults["input"], help="Frame directory, or parent directory with --batch.")
    parser.add_argument("--output", default=defaults["output"], help="Output JSONL path, or output directory with --batch.")
    parser.add_argument("--video-id", default=defaults["video_id"], help="Video ID for single-directory mode.")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--batch", dest="batch", action="store_true", default=defaults["batch"], help="Process child video directories under --input.")
    mode_group.add_argument("--single", dest="batch", action="store_false", help="Force single video mode.")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=defaults["resume"], help="Skip frame_ids already present in output JSONL.")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Overwrite outputs instead of resuming.")

    parser.add_argument("--pattern", default=defaults["pattern"], help="Glob pattern for frames inside each video directory.")
    parser.add_argument("--limit", type=int, default=defaults["limit"], help="Limit frames per video.")
    parser.add_argument("--summary-output", default=defaults["summary_output"], help="Optional summary JSON path.")
    parser.add_argument("--flush-every", type=int, default=defaults["flush_every"], help="Flush output JSONL every N records.")

    parser.add_argument("--device", default=defaults["device"], help="Torch device: auto, cuda, cuda:0, cpu.")
    parser.add_argument("--hf-cache-dir", default=defaults["hf_cache_dir"], help="Optional HuggingFace cache dir.")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=defaults["amp"], help="Use CUDA autocast fp16 during inference.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA autocast.")

    tf32_group = parser.add_mutually_exclusive_group()
    tf32_group.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=defaults["allow_tf32"], help="Enable TF32 matmul/cudnn on CUDA.")
    tf32_group.add_argument("--no-tf32", dest="allow_tf32", action="store_false", help="Disable TF32.")

    parser.add_argument("--ram-checkpoint", default=defaults["ram_checkpoint"], help="Path to ram_plus_swin_large_14m.pth.")
    parser.add_argument("--ram-image-size", type=int, default=defaults["ram_image_size"], help="RAM++ input image size.")
    parser.add_argument("--ram-batch-size", type=int, default=defaults["ram_batch_size"], help="RAM++ batch size.")
    parser.add_argument("--max-prompt-tags", type=int, default=defaults["max_prompt_tags"], help="Max tags to send to GroundingDINO per frame.")

    parser.add_argument("--gdino-model-id", default=defaults["gdino_model_id"], help="GroundingDINO HF model id or local path.")
    parser.add_argument("--image-max-side", type=int, default=defaults["image_max_side"], help="Resize longest side before GroundingDINO; <=0 disables.")
    parser.add_argument("--box-threshold", type=float, default=defaults["box_threshold"], help="GroundingDINO box confidence threshold.")
    parser.add_argument("--text-threshold", type=float, default=defaults["text_threshold"], help="GroundingDINO text threshold.")
    parser.add_argument("--nms-iou-threshold", type=float, default=defaults["nms_iou_threshold"], help="Class-agnostic NMS IoU threshold.")
    parser.add_argument("--scene-area-threshold", type=float, default=defaults["scene_area_threshold"], help="Move boxes above this area ratio to tags instead of object counts.")

    parser.add_argument("--quiet", action="store_true", default=defaults["quiet"], help="Reduce logs.")
    parser.add_argument("--debug", action="store_true", default=defaults["debug"], help="Raise/print detailed errors.")
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = load_config_defaults(pre_args.config)
    parser = build_parser(defaults)
    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required, either via CLI or config.")
    if not args.output:
        parser.error("--output is required, either via CLI or config.")

    args.ram_batch_size = max(1, int(args.ram_batch_size))
    args.flush_every = max(1, int(args.flush_every))
    if args.limit is not None:
        args.limit = max(0, int(args.limit))
    return args


def validate_transformers_version() -> None:
    try:
        import transformers
    except ImportError:
        LOG.error("transformers is not installed. Install requirements.txt first.")
        sys.exit(1)

    version = transformers.__version__
    if not version.startswith("4.46"):
        LOG.warning(
            "transformers==%s detected. Recommended for this workflow: 4.46.3.",
            version,
        )


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.hf_cache_dir:
        os.environ.setdefault("HF_HOME", str(args.hf_cache_dir))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = True


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_device(device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        LOG.info("GPU detected: %s (%.1f GB)", props.name, props.total_memory / (1024 ** 3))
    else:
        LOG.warning("Using CPU. This will be slow for full-scale processing.")


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return tuple(key)


def discover_frames(input_dir: str | Path, pattern: str, limit: int | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")

    frames = [
        path for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=natural_sort_key)

    if limit is not None:
        frames = frames[:limit]

    if not frames:
        raise FileNotFoundError(f"No image files found in: {root} with pattern={pattern}")

    LOG.info("Found %d frames in %s", len(frames), root)
    return frames


def discover_video_dirs(input_dir: str | Path, pattern: str) -> list[tuple[str, Path]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")

    video_dirs: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir(), key=natural_sort_key):
        if not child.is_dir():
            continue
        has_images = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in child.glob(pattern)
        )
        if has_images:
            video_dirs.append((child.name, child))

    if not video_dirs:
        raise FileNotFoundError(f"No video directories with images found in: {root}")

    LOG.info("Found %d video directories in %s", len(video_dirs), root)
    return video_dirs


def extract_frame_idx(filename: str | Path) -> int:
    stem = Path(filename).stem
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else 0


def make_frame_id(video_id: str, image_path: str | Path) -> str:
    return f"{video_id}_{Path(image_path).stem}"


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                LOG.warning("Skipping corrupt JSONL line %s:%d: %s", path, line_no, exc)
                continue
            if isinstance(doc, dict):
                yield doc


def load_existing_frame_ids(output_path: str | Path) -> set[str]:
    path = Path(output_path)
    if not path.is_file():
        return set()

    existing: set[str] = set()
    for doc in iter_jsonl(path):
        frame_id = doc.get("frame_id")
        if frame_id:
            existing.add(str(frame_id))

    if existing:
        LOG.info("Resume: found %d existing frames in %s", len(existing), path)
    return existing


def write_jsonl_record(handle, doc: dict[str, Any], flush: bool) -> None:
    handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    if flush:
        handle.flush()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def find_ram_checkpoint(cli_path: str | None = None) -> str:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    env_path = os.environ.get("RAM_CHECKPOINT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            Path("pretrained") / RAM_CHECKPOINT_FILENAME,
            here / "pretrained" / RAM_CHECKPOINT_FILENAME,
            here.parent / "pretrained" / RAM_CHECKPOINT_FILENAME,
        ]
    )

    for path in candidates:
        if path.is_file():
            LOG.info("RAM++ checkpoint found: %s", path)
            return str(path)

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"RAM++ checkpoint not found: {RAM_CHECKPOINT_FILENAME}\n"
        f"Searched:\n{searched}\n"
        f"Download: {RAM_CHECKPOINT_URL}"
    )


def load_ram_model(device, checkpoint_path: str | None, image_size: int):
    from ram import get_transform
    from ram.models import ram_plus

    checkpoint = find_ram_checkpoint(checkpoint_path)
    LOG.info("Loading RAM++ Swin-Large checkpoint: %s", checkpoint)
    model = ram_plus(pretrained=checkpoint, image_size=image_size, vit="swin_l")
    model.eval().to(device)
    transform = get_transform(image_size=image_size)
    LOG.info("RAM++ ready.")
    return model, transform


def load_gdino_model(device, model_id: str, cache_dir: str | None = None):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    LOG.info("Loading GroundingDINO: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval().to(device)
    LOG.info("GroundingDINO ready.")
    return model, processor


def _split_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in str(raw).split("|") if tag.strip()]


def _extract_english_outputs(result, expected: int) -> list[str] | None:
    english = result[0] if isinstance(result, (tuple, list)) and result else result
    if isinstance(english, str):
        return [english] if expected == 1 else None
    if isinstance(english, (list, tuple)):
        outputs = [str(item) for item in english]
        return outputs if len(outputs) == expected else None
    return None


def autocast_context(device, enabled: bool):
    if enabled and getattr(device, "type", None) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def infer_ram_tags_batch(
    images: list[Image.Image],
    model,
    transform,
    device,
    amp_enabled: bool,
) -> list[list[str]]:
    from ram import inference_ram as inference

    if not images:
        return []

    tensors = torch.stack([transform(image) for image in images], dim=0).to(device)
    try:
        with torch.inference_mode(), autocast_context(device, amp_enabled):
            result = inference(tensors, model)
        outputs = _extract_english_outputs(result, len(images))
        if outputs is not None:
            return [_split_tags(raw) for raw in outputs]
    except Exception as exc:
        if len(images) == 1:
            raise
        LOG.warning("RAM++ batch inference failed; falling back to per-image: %s", exc)
    finally:
        del tensors

    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()

    tags: list[list[str]] = []
    for image in images:
        tensor = transform(image).unsqueeze(0).to(device)
        try:
            with torch.inference_mode(), autocast_context(device, amp_enabled):
                result = inference(tensor, model)
        finally:
            del tensor
        outputs = _extract_english_outputs(result, 1)
        tags.append(_split_tags(outputs[0] if outputs else ""))
    return tags


def resize_for_model(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0

    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image, 1.0

    scale = max_side / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC), scale


def detect_objects_gdino(
    image: Image.Image,
    tag_list: list[str],
    gdino_model,
    gdino_processor,
    device,
    box_threshold: float,
    text_threshold: float,
    image_max_side: int,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    if not tag_list:
        return []

    text_prompt = build_dino_prompt(tag_list)
    if not text_prompt:
        return []

    model_image, scale = resize_for_model(image, image_max_side)
    inputs = gdino_processor(images=model_image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode(), autocast_context(device, amp_enabled):
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[model_image.size[::-1]],
    )
    result = results[0]

    detections: list[dict[str, Any]] = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        coords = box.tolist()
        if scale != 1.0:
            coords = [coord / scale for coord in coords]
        detections.append(
            {
                "label": str(label).strip(),
                "score": round(float(score.item()), 4),
                "box": [round(float(coord), 2) for coord in coords],
                "source": "groundingdino",
            }
        )

    return detections


def load_images(frame_paths: list[Path]) -> tuple[list[Path], list[Image.Image], int]:
    loaded_paths: list[Path] = []
    images: list[Image.Image] = []
    errors = 0

    for frame_path in frame_paths:
        try:
            images.append(Image.open(frame_path).convert("RGB"))
            loaded_paths.append(frame_path)
        except Exception as exc:
            errors += 1
            LOG.error("Failed to load image %s: %s", frame_path, exc)

    return loaded_paths, images, errors


def build_frame_doc(
    image_path: Path,
    image: Image.Image,
    video_id: str,
    raw_tags: list[str],
    prompt_tags: list[str],
    raw_detections: list[dict[str, Any]],
    final_objects: list[dict[str, Any]],
    scene_labels: list[str],
    timing: dict[str, float],
) -> dict[str, Any]:
    img_w, img_h = image.size
    label_counts = Counter(obj["label"] for obj in final_objects)
    all_tags = unique_preserve_order([tag.lower() for tag in raw_tags] + scene_labels)

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": make_frame_id(video_id, image_path),
        "video_id": video_id,
        "frame_idx": extract_frame_idx(image_path),
        "frame_name": image_path.stem,
        "image_path": str(image_path.resolve()),
        "image_size": [img_w, img_h],
        "tags": all_tags,
        "raw_tags": raw_tags,
        "object_prompt_tags": prompt_tags,
        "objects": final_objects,
        "object_summary": sorted(label_counts.keys()),
        "object_counts": dict(label_counts),
        "quality": {
            "num_raw_tags": len(raw_tags),
            "num_prompt_tags": len(prompt_tags),
            "num_raw_boxes": len(raw_detections),
            "num_final_boxes": len(final_objects),
            "num_scene_labels": len(scene_labels),
        },
        "timing": timing,
    }


def process_frame_batch(
    frame_paths: list[Path],
    video_id: str,
    ram_model,
    ram_transform,
    gdino_model,
    gdino_processor,
    device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    loaded_paths, images, load_errors = load_images(frame_paths)
    if not images:
        return [], load_errors

    ram_start = time.time()
    ram_tag_groups = infer_ram_tags_batch(
        images,
        ram_model,
        ram_transform,
        device,
        amp_enabled=args.amp,
    )
    ram_elapsed = time.time() - ram_start
    ram_sec = ram_elapsed / max(1, len(images))

    if len(ram_tag_groups) != len(images):
        raise RuntimeError(
            f"RAM++ returned {len(ram_tag_groups)} outputs for {len(images)} images."
        )

    docs: list[dict[str, Any]] = []
    for image_path, image, raw_tags in zip(loaded_paths, images, ram_tag_groups):
        prompt_tags = filter_tags_for_dino(raw_tags, max_tags=args.max_prompt_tags)

        detect_start = time.time()
        raw_detections = detect_objects_gdino(
            image=image,
            tag_list=prompt_tags,
            gdino_model=gdino_model,
            gdino_processor=gdino_processor,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            image_max_side=args.image_max_side,
            amp_enabled=args.amp,
        )
        gdino_sec = time.time() - detect_start

        post_start = time.time()
        final_objects, scene_labels = canonicalize_detections(
            raw_detections,
            image.size[0],
            image.size[1],
            nms_iou_threshold=args.nms_iou_threshold,
            scene_area_threshold=args.scene_area_threshold,
        )
        post_sec = time.time() - post_start

        docs.append(
            build_frame_doc(
                image_path=image_path,
                image=image,
                video_id=video_id,
                raw_tags=raw_tags,
                prompt_tags=prompt_tags,
                raw_detections=raw_detections,
                final_objects=final_objects,
                scene_labels=scene_labels,
                timing={
                    "ram_sec": round(ram_sec, 4),
                    "gdino_sec": round(gdino_sec, 4),
                    "postprocess_sec": round(post_sec, 4),
                },
            )
        )

    return docs, load_errors


def run_single_video(
    input_dir: str | Path,
    output_path: str | Path,
    video_id: str,
    ram_model,
    ram_transform,
    gdino_model,
    gdino_processor,
    device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames = discover_frames(input_dir, pattern=args.pattern, limit=args.limit)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_frame_ids(output_path) if args.resume else set()
    write_mode = "a" if args.resume and existing_ids else "w"

    processed = 0
    skipped = 0
    errors = 0
    total_objects = 0
    record_counter = 0
    start_time = time.time()
    pending: list[Path] = []

    def flush_pending(handle) -> None:
        nonlocal pending, processed, errors, total_objects, record_counter
        if not pending:
            return
        try:
            docs, batch_errors = process_frame_batch(
                pending,
                video_id,
                ram_model,
                ram_transform,
                gdino_model,
                gdino_processor,
                device,
                args,
            )
            errors += batch_errors
        except Exception as exc:
            errors += len(pending)
            LOG.error("[%s] Failed batch of %d frames: %s", video_id, len(pending), exc)
            if args.debug:
                raise
            pending = []
            return

        for doc in docs:
            record_counter += 1
            flush_now = args.flush_every == 1 or record_counter % args.flush_every == 0
            write_jsonl_record(handle, doc, flush=flush_now)
            processed += 1
            total_objects += len(doc["objects"])
        pending = []

    with open(output_path, write_mode, encoding="utf-8") as out_handle:
        pbar = tqdm(frames, desc=f"RAM+GDINO {video_id}", unit="frame")
        for frame_path in pbar:
            frame_id = make_frame_id(video_id, frame_path)
            if frame_id in existing_ids:
                skipped += 1
                pbar.set_postfix(ok=processed, skip=skipped, err=errors)
                continue

            pending.append(frame_path)
            if len(pending) >= args.ram_batch_size:
                flush_pending(out_handle)
                elapsed = max(time.time() - start_time, 1e-6)
                pbar.set_postfix(
                    ok=processed,
                    skip=skipped,
                    err=errors,
                    fps=f"{processed / elapsed:.2f}",
                )

        flush_pending(out_handle)
        if args.flush_every > 1:
            out_handle.flush()

    elapsed = time.time() - start_time
    summary = {
        "video_id": video_id,
        "input": str(Path(input_dir)),
        "output": str(output_path),
        "num_frames": len(frames),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "total_objects": total_objects,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(processed / elapsed, 4) if elapsed > 0 else None,
    }
    LOG.info(
        "[%s] Done: %d processed, %d skipped, %d errors, %.1fs",
        video_id,
        processed,
        skipped,
        errors,
        elapsed,
    )
    LOG.info("[%s] Output: %s", video_id, output_path)
    return summary


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    ram_model,
    ram_transform,
    gdino_model,
    gdino_processor,
    device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    video_dirs = discover_video_dirs(input_dir, pattern=args.pattern)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for index, (video_id, video_path) in enumerate(video_dirs, 1):
        LOG.info("=== Video %d/%d: %s ===", index, len(video_dirs), video_id)
        output_path = output_dir / f"{video_id}_objects.jsonl"
        summaries.append(
            run_single_video(
                video_path,
                output_path,
                video_id,
                ram_model,
                ram_transform,
                gdino_model,
                gdino_processor,
                device,
                args,
            )
        )

    return summaries


def write_summary(summary_path: str | Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ram_image_size": args.ram_image_size,
            "ram_batch_size": args.ram_batch_size,
            "max_prompt_tags": args.max_prompt_tags,
            "gdino_model_id": args.gdino_model_id,
            "image_max_side": args.image_max_side,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "scene_area_threshold": args.scene_area_threshold,
            "amp": args.amp,
            "allow_tf32": args.allow_tf32,
        },
        "runs": summaries,
        "total_processed": sum(int(item.get("processed") or 0) for item in summaries),
        "total_errors": sum(int(item.get("errors") or 0) for item in summaries),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOG.info("Summary written: %s", path)


def main() -> None:
    args = parse_args()
    setup_logging(quiet=args.quiet, debug=args.debug)
    validate_transformers_version()
    configure_runtime(args)

    device = get_device(args.device)
    log_device(device)

    LOG.info(
        "Config: ram_batch_size=%s, max_prompt_tags=%s, image_max_side=%s, "
        "box_threshold=%.2f, text_threshold=%.2f, nms_iou=%.2f",
        args.ram_batch_size,
        args.max_prompt_tags,
        args.image_max_side,
        args.box_threshold,
        args.text_threshold,
        args.nms_iou_threshold,
    )

    ram_model, ram_transform = load_ram_model(device, args.ram_checkpoint, args.ram_image_size)
    gdino_model, gdino_processor = load_gdino_model(device, args.gdino_model_id, args.hf_cache_dir)

    if args.batch:
        summaries = run_batch(
            args.input,
            args.output,
            ram_model,
            ram_transform,
            gdino_model,
            gdino_processor,
            device,
            args,
        )
    else:
        video_id = args.video_id or Path(args.input).name
        summaries = [
            run_single_video(
                args.input,
                args.output,
                video_id,
                ram_model,
                ram_transform,
                gdino_model,
                gdino_processor,
                device,
                args,
            )
        ]

    if args.summary_output:
        write_summary(args.summary_output, summaries, args)


if __name__ == "__main__":
    main()
