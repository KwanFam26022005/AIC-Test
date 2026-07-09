# -*- coding: utf-8 -*-
"""RAM++ -> GroundingDINO object detection runner for A5000 servers.

The pipeline is independent from OCR. It reads keyframes, uses RAM++ to
generate image tags, filters those tags into GroundingDINO prompts, runs
zero-shot detection, postprocesses boxes, and appends one JSON object per
frame to a JSONL file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    build_search_fields,
    canonicalize_detections,
    filter_tags_for_dino,
)


LOG = logging.getLogger("object_detection")

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCHEMA_VERSION = "ram_gdino_object_detection_v1_1"

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
    "resume_legacy": True,
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
    # Metadata — Phase 1
    "video_manifest": None,
    "keyframe_map": None,
    "frames_root": None,
    "timestamp_strategy": "map_or_uniform",
}

CONFIG_KEY_MAP = {
    "io.input": "input",
    "io.output": "output",
    "io.video_id": "video_id",
    "io.batch": "batch",
    "io.pattern": "pattern",
    "io.limit": "limit",
    "io.resume": "resume",
    "io.resume_legacy": "resume_legacy",
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
    "metadata.video_manifest": "video_manifest",
    "metadata.keyframe_map": "keyframe_map",
    "metadata.frames_root": "frames_root",
    "metadata.timestamp_strategy": "timestamp_strategy",
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
    parser.add_argument("--resume-legacy", dest="resume_legacy", action="store_true", default=defaults["resume_legacy"], help="When resuming, skip frames without fingerprint match (legacy behavior).")

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

    # Metadata — Phase 1
    parser.add_argument("--video-manifest", default=defaults["video_manifest"], help="Path to video_manifest.jsonl for timestamp fallback.")
    parser.add_argument("--keyframe-map", default=defaults["keyframe_map"], help="Path to keyframe map CSV (or directory of CSVs for batch mode).")
    parser.add_argument("--frames-root", default=defaults["frames_root"], help="Root directory of keyframes for image_relpath resolution.")
    parser.add_argument("--timestamp-strategy", default=defaults["timestamp_strategy"],
                        choices=["map_or_uniform", "map_only", "uniform", "none"],
                        help="How to assign timestamps: map_or_uniform|map_only|uniform|none.")

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


def load_existing_records(output_path: str | Path) -> dict[str, str | None]:
    """Load existing frame_id -> frame_fingerprint mapping from output JSONL.

    Returns:
        dict: frame_id -> frame_fingerprint (or None if legacy record)
    """
    path = Path(output_path)
    if not path.is_file():
        return {}

    existing: dict[str, str | None] = {}
    for doc in iter_jsonl(path):
        frame_id = doc.get("frame_id")
        if frame_id:
            existing[str(frame_id)] = doc.get("frame_fingerprint")

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


# ============================================================================
# KEYFRAME MAP & TIMESTAMP RESOLVER
# ============================================================================

def load_keyframe_map_csv(csv_path: str | Path) -> dict[int, dict[str, Any]]:
    """Load keyframe map CSV vào dict keyed by `n` (keyframe ordinal).

    CSV format expected:
        n,pts_time,fps,frame_idx
        1,0.0,30.0,0
        2,3.0,30.0,90

    Args:
        csv_path: Path to CSV file

    Returns:
        dict: {keyframe_idx: {"pts_time": float, "fps": float, "frame_idx": int}}
    """
    result: dict[int, dict[str, Any]] = {}
    path = Path(csv_path)
    if not path.is_file():
        LOG.warning("Keyframe map CSV not found: %s", csv_path)
        return result

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["n"])
                result[n] = {
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row.get("fps", 0)),
                    "frame_idx": int(row.get("frame_idx", 0)),
                }
            except (KeyError, ValueError) as exc:
                LOG.warning("Skipping malformed keyframe map row: %s (%s)", row, exc)

    LOG.info("Loaded keyframe map: %d entries from %s", len(result), csv_path)
    return result


def find_keyframe_map_for_video(keyframe_map_path: str | Path | None, video_id: str) -> dict[int, dict[str, Any]]:
    """Find and load keyframe map CSV for a given video_id.

    Supports:
        - Direct CSV file path
        - Directory containing <video_id>/<video_id>.csv
        - Directory containing <video_id>.csv

    Args:
        keyframe_map_path: CLI/config path (file or directory)
        video_id: video identifier

    Returns:
        dict: keyframe map or empty dict
    """
    if not keyframe_map_path:
        return {}

    path = Path(keyframe_map_path)

    if path.is_file():
        return load_keyframe_map_csv(path)

    if path.is_dir():
        # Try <dir>/<video_id>/<video_id>.csv
        candidate = path / video_id / f"{video_id}.csv"
        if candidate.is_file():
            return load_keyframe_map_csv(candidate)
        # Try <dir>/<video_id>.csv
        candidate = path / f"{video_id}.csv"
        if candidate.is_file():
            return load_keyframe_map_csv(candidate)

    LOG.debug("No keyframe map found for video_id=%s at %s", video_id, keyframe_map_path)
    return {}


def load_video_manifest(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load video_manifest.jsonl for fallback metadata.

    Returns:
        dict: {video_id: manifest_doc}
    """
    if not manifest_path:
        return {}

    path = Path(manifest_path)
    if not path.is_file():
        LOG.warning("Video manifest not found: %s", manifest_path)
        return {}

    result: dict[str, dict[str, Any]] = {}
    for doc in iter_jsonl(path):
        vid = doc.get("video_id")
        if vid:
            result[str(vid)] = doc

    LOG.info("Loaded video manifest: %d entries from %s", len(result), manifest_path)
    return result


def resolve_timestamp(
    keyframe_idx: int,
    keyframe_map: dict[int, dict[str, Any]],
    num_keyframes: int,
    video_manifest_entry: dict[str, Any] | None,
    strategy: str,
) -> tuple[float | None, str]:
    """Resolve timestamp for a keyframe.

    Args:
        keyframe_idx: 1-based keyframe ordinal (from frame filename)
        keyframe_map: loaded CSV map {n: {pts_time, fps, frame_idx}}
        num_keyframes: total number of keyframes in this video
        video_manifest_entry: optional manifest with duration_sec, fps
        strategy: "map_or_uniform"|"map_only"|"uniform"|"none"

    Returns:
        tuple: (timestamp_sec, timestamp_source)
    """
    if strategy == "none":
        return None, "none"

    # Try map CSV first
    if strategy in ("map_or_uniform", "map_only"):
        if keyframe_idx in keyframe_map:
            row = keyframe_map[keyframe_idx]
            return row["pts_time"], "keyframe_map_csv"

    if strategy == "map_only":
        return None, "map_missing"

    # Uniform interpolation fallback
    if strategy in ("map_or_uniform", "uniform"):
        duration = None
        if video_manifest_entry:
            duration = video_manifest_entry.get("duration_sec")
        if duration and duration > 0 and num_keyframes > 1:
            ts = (keyframe_idx - 1) * duration / max(1, num_keyframes - 1)
            return round(ts, 6), "uniform_interpolation"

    return None, "unavailable"


# ============================================================================
# RESUME FINGERPRINT
# ============================================================================

def compute_run_config_hash(args: argparse.Namespace) -> str:
    """Compute a hash of the run configuration for resume fingerprinting."""
    config_parts = [
        f"schema={SCHEMA_VERSION}",
        f"ram_image_size={args.ram_image_size}",
        f"gdino_model_id={args.gdino_model_id}",
        f"image_max_side={args.image_max_side}",
        f"box_threshold={args.box_threshold}",
        f"text_threshold={args.text_threshold}",
        f"nms_iou_threshold={args.nms_iou_threshold}",
        f"scene_area_threshold={args.scene_area_threshold}",
        f"max_prompt_tags={args.max_prompt_tags}",
    ]
    config_str = "|".join(config_parts)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def compute_frame_fingerprint(run_config_hash: str, image_path: Path) -> str:
    """Compute fingerprint for a specific frame + run config combination."""
    fp_str = f"{run_config_hash}|{image_path.name}"
    return hashlib.md5(fp_str.encode()).hexdigest()[:16]


def should_skip_frame(
    frame_id: str,
    frame_fingerprint: str,
    existing_records: dict[str, str | None],
    resume_legacy: bool,
) -> bool:
    """Determine whether a frame should be skipped during resume.

    Args:
        frame_id: frame identifier
        frame_fingerprint: current run's fingerprint
        existing_records: {frame_id: existing_fingerprint or None}
        resume_legacy: if True, skip even when existing has no fingerprint

    Returns:
        bool: True if frame should be skipped
    """
    if frame_id not in existing_records:
        return False

    existing_fp = existing_records[frame_id]
    if existing_fp is None:
        # Legacy record — no fingerprint
        if resume_legacy:
            LOG.debug("Resume (legacy): skipping %s (no fingerprint in output)", frame_id)
            return True
        LOG.debug("Resume: re-running %s (legacy output, no fingerprint)", frame_id)
        return False

    if existing_fp == frame_fingerprint:
        return True

    LOG.debug("Resume: re-running %s (fingerprint mismatch: %s vs %s)",
              frame_id, existing_fp, frame_fingerprint)
    return False


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
    quality_stats: dict[str, Any],
    timing: dict[str, float],
    *,
    keyframe_idx: int | None = None,
    source_frame_idx: int | None = None,
    timestamp_sec: float | None = None,
    timestamp_source: str = "none",
    frame_fingerprint: str | None = None,
    run_config_hash: str | None = None,
) -> dict[str, Any]:
    img_w, img_h = image.size
    frame_id = make_frame_id(video_id, image_path)
    all_tags = unique_preserve_order([tag.lower() for tag in raw_tags] + scene_labels)

    # Build search/caption fields
    search_fields = build_search_fields(final_objects, scene_labels, raw_tags)

    # image_relpath: video_id/frame_name.ext
    image_relpath = f"{video_id}/{image_path.name}"

    doc = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "frame_id": frame_id,
        "canonical_frame_id": frame_id,
        "frame_name": image_path.stem,
        "frame_idx": extract_frame_idx(image_path),
        "keyframe_idx": keyframe_idx,
        "source_frame_idx": source_frame_idx,
        "timestamp_sec": timestamp_sec,
        "timestamp_source": timestamp_source,
        "image_path": str(image_path.resolve()),
        "image_relpath": image_relpath,
        "image_size": [img_w, img_h],
        "tags": all_tags,
        "raw_tags": raw_tags,
        "ram_tags": search_fields["ram_tags"],
        "object_prompt_tags": prompt_tags,
        "scene_tags": search_fields["scene_tags"],
        "object_tags": search_fields["object_tags"],
        "objects": final_objects,
        "object_summary": sorted(search_fields["object_counts"].keys()),
        "object_counts": search_fields["object_counts"],
        "object_counts_normalized": search_fields["object_counts_normalized"],
        "object_count_items": search_fields["object_count_items"],
        "important_objects": search_fields["important_objects"],
        "object_text": search_fields["object_text"],
        "scene_text": search_fields["scene_text"],
        "ram_tag_text": search_fields["ram_tag_text"],
        "all_object_text": search_fields["all_object_text"],
        "quality": {
            "num_raw_tags": len(raw_tags),
            "num_prompt_tags": len(prompt_tags),
            "num_raw_boxes": len(raw_detections),
            "num_final_boxes": len(final_objects),
            **quality_stats,
        },
        "timing": timing,
    }

    # Resume fingerprint (Phase 7)
    if frame_fingerprint:
        doc["frame_fingerprint"] = frame_fingerprint
    if run_config_hash:
        doc["run_config_hash"] = run_config_hash

    return doc


def process_frame_batch(
    frame_paths: list[Path],
    video_id: str,
    ram_model,
    ram_transform,
    gdino_model,
    gdino_processor,
    device,
    args: argparse.Namespace,
    *,
    keyframe_map: dict[int, dict[str, Any]] | None = None,
    num_keyframes: int = 0,
    video_manifest_entry: dict[str, Any] | None = None,
    run_config_hash: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    loaded_paths, images, load_errors = load_images(frame_paths)
    if not images:
        return [], load_errors

    if keyframe_map is None:
        keyframe_map = {}

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
        final_objects, scene_labels, quality_stats = canonicalize_detections(
            raw_detections,
            image.size[0],
            image.size[1],
            nms_iou_threshold=args.nms_iou_threshold,
            scene_area_threshold=args.scene_area_threshold,
        )
        post_sec = time.time() - post_start

        # Timestamp resolution (Phase 1)
        kf_idx = extract_frame_idx(image_path)
        ts_sec, ts_source = resolve_timestamp(
            keyframe_idx=kf_idx,
            keyframe_map=keyframe_map,
            num_keyframes=num_keyframes,
            video_manifest_entry=video_manifest_entry,
            strategy=getattr(args, "timestamp_strategy", "map_or_uniform"),
        )

        # Source frame idx from keyframe map
        src_frame_idx = None
        if kf_idx in keyframe_map:
            src_frame_idx = keyframe_map[kf_idx].get("frame_idx")

        # Frame fingerprint (Phase 7)
        fp = compute_frame_fingerprint(run_config_hash, image_path) if run_config_hash else None

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
                quality_stats=quality_stats,
                timing={
                    "ram_sec": round(ram_sec, 4),
                    "gdino_sec": round(gdino_sec, 4),
                    "postprocess_sec": round(post_sec, 4),
                },
                keyframe_idx=kf_idx,
                source_frame_idx=src_frame_idx,
                timestamp_sec=ts_sec,
                timestamp_source=ts_source,
                frame_fingerprint=fp,
                run_config_hash=run_config_hash,
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
    *,
    keyframe_map: dict[int, dict[str, Any]] | None = None,
    video_manifest_entry: dict[str, Any] | None = None,
    run_config_hash: str | None = None,
) -> dict[str, Any]:
    frames = discover_frames(input_dir, pattern=args.pattern, limit=args.limit)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume with fingerprint support (Phase 7)
    existing_records = load_existing_records(output_path) if args.resume else {}
    write_mode = "a" if args.resume and existing_records else "w"

    # Precompute run_config_hash if not provided
    if run_config_hash is None:
        run_config_hash = compute_run_config_hash(args)

    # Load keyframe map if not provided
    if keyframe_map is None:
        keyframe_map = find_keyframe_map_for_video(
            getattr(args, "keyframe_map", None), video_id
        )

    num_keyframes = len(frames)

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
                keyframe_map=keyframe_map,
                num_keyframes=num_keyframes,
                video_manifest_entry=video_manifest_entry,
                run_config_hash=run_config_hash,
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

            # Fingerprint-aware resume (Phase 7)
            if existing_records:
                fp = compute_frame_fingerprint(run_config_hash, frame_path)
                if should_skip_frame(frame_id, fp, existing_records,
                                     getattr(args, "resume_legacy", True)):
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
        "run_config_hash": run_config_hash,
        "schema_version": SCHEMA_VERSION,
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

    # Precompute config hash once
    run_config_hash = compute_run_config_hash(args)

    # Load video manifest once for all videos
    video_manifest = load_video_manifest(getattr(args, "video_manifest", None))

    summaries: list[dict[str, Any]] = []
    for index, (video_id, video_path) in enumerate(video_dirs, 1):
        LOG.info("=== Video %d/%d: %s ===", index, len(video_dirs), video_id)
        output_path = output_dir / f"{video_id}_objects.jsonl"

        # Per-video keyframe map
        kf_map = find_keyframe_map_for_video(
            getattr(args, "keyframe_map", None), video_id
        )
        vm_entry = video_manifest.get(video_id)

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
                keyframe_map=kf_map,
                video_manifest_entry=vm_entry,
                run_config_hash=run_config_hash,
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

    run_config_hash = compute_run_config_hash(args)
    LOG.info(
        "Config: ram_batch_size=%s, max_prompt_tags=%s, image_max_side=%s, "
        "box_threshold=%.2f, text_threshold=%.2f, nms_iou=%.2f, "
        "schema=%s, config_hash=%s",
        args.ram_batch_size,
        args.max_prompt_tags,
        args.image_max_side,
        args.box_threshold,
        args.text_threshold,
        args.nms_iou_threshold,
        SCHEMA_VERSION,
        run_config_hash,
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
        # Load metadata for single video mode
        kf_map = find_keyframe_map_for_video(
            getattr(args, "keyframe_map", None), video_id
        )
        vm = load_video_manifest(getattr(args, "video_manifest", None))
        vm_entry = vm.get(video_id)

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
                keyframe_map=kf_map,
                video_manifest_entry=vm_entry,
                run_config_hash=run_config_hash,
            )
        ]

    if args.summary_output:
        write_summary(args.summary_output, summaries, args)


if __name__ == "__main__":
    main()
