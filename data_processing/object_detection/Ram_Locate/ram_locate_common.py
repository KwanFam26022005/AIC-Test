# -*- coding: utf-8 -*-
"""Shared utilities for the RAM++ + LocateAnything pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


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


def discover_frames(input_dir: str | Path, pattern: str = "*", limit: int | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")

    frames = [
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=natural_sort_key)

    if limit is not None:
        frames = frames[:limit]

    if not frames:
        raise FileNotFoundError(f"No image files found in: {root}")

    return frames


def discover_video_dirs(input_dir: str | Path, pattern: str = "*") -> list[tuple[str, Path]]:
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
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping corrupt JSONL line %s:%d: %s", path, line_no, exc)
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
    return existing


def write_jsonl_record(handle, doc: dict[str, Any], flush: bool = True) -> None:
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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def get_torch_device(device_arg: str = "auto"):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_device(device) -> None:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        memory_gb = props.total_memory / (1024 ** 3)
        logging.info("GPU detected: %s (%.1f GB)", props.name, memory_gb)
    else:
        logging.warning("Using CPU. This will be slow for full-scale processing.")


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parent


def ensure_local_imports(file_path: str | Path) -> None:
    root = str(project_root_from_file(file_path))
    if root not in sys.path:
        sys.path.insert(0, root)


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

