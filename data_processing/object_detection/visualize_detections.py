# -*- coding: utf-8 -*-
"""Visualize RAM++ + GroundingDINO JSONL detections on keyframes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


LOG = logging.getLogger("visualize_detections")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (0, 119, 182),
    (106, 153, 78),
    (214, 40, 40),
    (80, 81, 79),
]


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


def resolve_image_path(
    doc: dict[str, Any],
    frames_dir: str | Path | None = None,
    frames_root: str | Path | None = None,
) -> Path | None:
    raw_path = doc.get("image_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.is_file():
            return path

    frame_name = str(doc.get("frame_name") or Path(str(raw_path or "")).stem)
    video_id = str(doc.get("video_id") or "")
    candidates: list[Path] = []

    if frames_dir and frame_name:
        base = Path(frames_dir)
        candidates.extend(base / f"{frame_name}{ext}" for ext in IMAGE_EXTENSIONS)

    if frames_root and video_id and frame_name:
        base = Path(frames_root) / video_id
        candidates.extend(base / f"{frame_name}{ext}" for ext in IMAGE_EXTENSIONS)

    for path in candidates:
        if path.is_file():
            return path
    return None


def label_color(label: str) -> tuple[int, int, int]:
    return PALETTE[sum(ord(ch) for ch in label) % len(PALETTE)]


def clamp_box(box: list[Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_font(size: int = 16):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    y0 = max(0, y - text_h - 6)
    draw.rectangle([x, y0, x + text_w + 8, y0 + text_h + 6], fill=color)
    draw.text((x + 4, y0 + 3), text, fill=(255, 255, 255), font=font)


def draw_detections(doc: dict[str, Any], image_path: Path, output_path: Path) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = get_font()
    width, height = image.size

    objects = doc.get("objects") or []
    drawn = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        box = clamp_box(obj.get("box"), width, height)
        if box is None:
            continue

        label = str(obj.get("label") or "OBJECT")
        score = obj.get("score")
        label_text = f"{label} {float(score):.2f}" if isinstance(score, (int, float)) else label
        color = label_color(label)
        x1, y1, x2, y2 = box

        line_width = max(2, round(min(width, height) / 400))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        draw_label(draw, (x1, y1), label_text, color, font)
        drawn += 1

    summary = ", ".join(f"{key}:{value}" for key, value in sorted((doc.get("object_counts") or {}).items()))
    header = f"{doc.get('frame_id', image_path.stem)} | objects={drawn}"
    if summary:
        header = f"{header} | {summary}"
    draw_label(draw, (8, 8), header, (20, 20, 20), font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return drawn


def build_output_path(doc: dict[str, Any], image_path: Path, output_dir: Path) -> Path:
    video_id = str(doc.get("video_id") or "unknown_video")
    frame_id = str(doc.get("frame_id") or image_path.stem)
    return output_dir / video_id / f"{frame_id}_det.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw object detection JSONL boxes onto source keyframes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", required=True, help="Detection JSONL from ram_gdino_pipeline.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for annotated images.")
    parser.add_argument("--frames-dir", default=None, help="Fallback frame directory for single-video JSONL.")
    parser.add_argument("--frames-root", default=None, help="Fallback parent directory containing video_id subdirs.")
    parser.add_argument("--limit", type=int, default=50, help="Max records to visualize.")
    parser.add_argument("--only-with-objects", action="store_true", help="Skip frames with no final objects.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    seen = 0
    written = 0
    missing = 0

    for doc in iter_jsonl(args.jsonl):
        if args.only_with_objects and not doc.get("objects"):
            continue
        if args.limit is not None and written >= args.limit:
            break

        image_path = resolve_image_path(doc, frames_dir=args.frames_dir, frames_root=args.frames_root)
        if image_path is None:
            missing += 1
            LOG.warning("Cannot resolve image for frame_id=%s", doc.get("frame_id"))
            continue

        out_path = build_output_path(doc, image_path, output_dir)
        num_boxes = draw_detections(doc, image_path, out_path)
        LOG.info("Wrote %s (%d boxes)", out_path, num_boxes)
        seen += 1
        written += 1

    print("Visualization complete")
    print(f"Read/written: {seen}/{written}")
    print(f"Missing images: {missing}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
