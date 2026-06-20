# -*- coding: utf-8 -*-
"""
visualize.py — Vẽ bounding boxes lên ảnh từ output JSONL

Đọc output JSONL từ ram_gdino_pipeline.py + ảnh gốc → vẽ bbox + label
lên ảnh. Chỉ cần CPU (Pillow + numpy, không cần GPU).

Output format: "LABEL (0.82)" với bbox màu theo label category.

Usage:
    python visualize.py \\
        --images /path/to/keyframes/L21_V001 \\
        --metadata /path/to/output/L21_V001.jsonl \\
        --output /path/to/visualized \\
        --limit 20

    # Visualize specific frames
    python visualize.py \\
        --images /path/to/keyframes/L21_V001 \\
        --metadata /path/to/output/L21_V001.jsonl \\
        --output /path/to/visualized \\
        --frames 001 010 089
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Color palette — visually distinct, modern look ───────────────────────────
# Grouped by semantic category for consistent coloring
CATEGORY_COLORS = {
    # People
    "PERSON": "#FF3B30",
    "MAN": "#FF3B30",
    "WOMAN": "#FF3B30",
    "CHILD": "#FF6961",
    "WORKER": "#FF6961",
    # Vehicles
    "CAR": "#007AFF",
    "BUS": "#5856D6",
    "TRUCK": "#5AC8FA",
    "MOTORCYCLE": "#34AADC",
    "BICYCLE": "#5AC8FA",
    "BOAT": "#5856D6",
    "TRAIN": "#5856D6",
    "AIRPLANE": "#5AC8FA",
    "SHIP": "#5856D6",
    # Animals
    "COW": "#34C759",
    "HORSE": "#30D158",
    "DOG": "#32D74B",
    "CAT": "#30D158",
    "FISH": "#00C7BE",
    "BIRD": "#32D74B",
    "SHEEP": "#34C759",
    "ELEPHANT": "#30D158",
    # Objects
    "CHAIR": "#FF9500",
    "TABLE": "#FF9500",
    "BED": "#FFCC00",
    "SOFA": "#FF9500",
    # Clothing
    "HAT": "#AF52DE",
    "SHOE": "#AF52DE",
    "GLOVE": "#BF5AF2",
    # Nature
    "TREE": "#34C759",
    "FLOWER": "#FF2D55",
}

# Fallback palette for labels not in CATEGORY_COLORS
FALLBACK_PALETTE = [
    "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE",
    "#00C7BE", "#FFD60A", "#FF2D55", "#5856D6", "#FF6B6B",
    "#64D2FF", "#30D158", "#BF5AF2", "#5AC8FA", "#FFCC00",
]


def get_color(label, idx=0):
    """Lấy màu cho label — ưu tiên category color, fallback theo index.

    Args:
        label: str — label name (uppercase)
        idx: int — fallback index nếu label không có trong category map

    Returns:
        str: hex color code
    """
    label_upper = label.upper().replace("_", " ").strip()
    # Thử match exact
    if label_upper in CATEGORY_COLORS:
        return CATEGORY_COLORS[label_upper]
    # Thử match partial (e.g. "PERSON" in "YOUNG_PERSON")
    for key, color in CATEGORY_COLORS.items():
        if key in label_upper:
            return color
    # Fallback
    return FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]


def load_font(size=16):
    """Load font, fallback to default if not found.

    Args:
        size: int — font size in pixels

    Returns:
        ImageFont: loaded font
    """
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
    ]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple.

    Args:
        hex_color: str — e.g. "#FF3B30"

    Returns:
        tuple: (R, G, B)
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def wrap_text(text, font, max_width, draw):
    """Chia văn bản thành các dòng phù hợp với chiều rộng tối đa (pixels)."""
    words = text.split(" ")
    lines = []
    current_line = []

    for word in words:
        if not word:
            continue
        test_line = " ".join(current_line + [word]) if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))
    return lines


def draw_frame(image, objects, tags=None, font=None):
    """Vẽ bounding boxes + labels + tags danh sách lên 1 ảnh.

    Args:
        image: PIL.Image — ảnh gốc
        objects: list[dict] — [{"label", "score", "box"}]
        tags: list[str] — danh sách tags của frame từ RAM++
        font: ImageFont — optional, sẽ auto-load nếu None

    Returns:
        PIL.Image: ảnh đã annotate
    """
    if font is None:
        font = load_font(size=16)

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    # 1. Vẽ bounding boxes của các objects
    for idx, obj in enumerate(objects):
        label = obj["label"]
        score = obj["score"]
        box = obj["box"]  # [x1, y1, x2, y2]

        x0, y0, x1, y1 = box
        color = get_color(label, idx)
        rgb = hex_to_rgb(color)

        # Draw bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Draw label background + text
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Position: above bbox, or below if no room above
        label_y = y0 - text_h - 6
        if label_y < 0:
            label_y = y1 + 2

        # Background rectangle
        draw.rectangle(
            [x0, label_y, x0 + text_w + 8, label_y + text_h + 6],
            fill=color,
        )
        # Text (white on colored background)
        draw.text((x0 + 4, label_y + 2), text, fill="white", font=font)

    # 2. Vẽ bảng thông tin (Summary & Tags) ở góc trên bên trái
    summary_text = f"{len(objects)} objects detected"
    raw_tags_text = f"Tags: {', '.join(tags)}" if tags else "Tags: none"

    # Tính toán độ rộng tối đa để xuống dòng cho tags (tránh tràn viền ảnh)
    max_text_width = min(800, annotated.width - 40)
    wrapped_tag_lines = wrap_text(raw_tags_text, font, max_text_width, draw)

    # Tính toán kích thước cho background box
    line_spacing = 6
    padding = 12

    bbox_sum = draw.textbbox((0, 0), summary_text, font=font)
    h_sum = bbox_sum[3] - bbox_sum[1]
    w_sum = bbox_sum[2] - bbox_sum[0]

    h_tags = []
    w_tags = []
    for line in wrapped_tag_lines:
        bbox_l = draw.textbbox((0, 0), line, font=font)
        h_tags.append(bbox_l[3] - bbox_l[1])
        w_tags.append(bbox_l[2] - bbox_l[0])

    total_text_height = h_sum + sum(h_tags) + line_spacing * (len(wrapped_tag_lines) + 1)
    max_w = max([w_sum] + w_tags)

    box_w = max_w + padding * 2
    box_h = total_text_height + padding * 2

    # Vẽ background box màu xám đậm, bo viền nhẹ
    draw.rectangle(
        [10, 10, 10 + box_w, 10 + box_h],
        fill=(25, 25, 25),
        outline=(80, 80, 80),
        width=1
    )

    # Vẽ nội dung text
    current_y = 10 + padding
    draw.text((10 + padding, current_y), summary_text, fill=(255, 255, 255), font=font)
    current_y += h_sum + line_spacing + 2  # Separator space

    # Vẽ một đường gạch ngang phân cách nhẹ
    draw.line(
        [10 + padding, current_y, 10 + padding + max_w, current_y],
        fill=(80, 80, 80),
        width=1
    )
    current_y += line_spacing

    for line, h_l in zip(wrapped_tag_lines, h_tags):
        draw.text((10 + padding, current_y), line, fill=(200, 200, 200), font=font)
        current_y += h_l + line_spacing

    return annotated


def visualize_from_jsonl(images_dir, metadata_path, output_dir,
                         limit=None, frame_names=None):
    """Đọc JSONL + ảnh → vẽ bbox → lưu.

    Args:
        images_dir: str — thư mục chứa ảnh gốc
        metadata_path: str — đường dẫn file JSONL
        output_dir: str — thư mục output
        limit: int or None — giới hạn số frames
        frame_names: list[str] or None — chỉ vẽ các frame cụ thể (stem names)
    """
    # Load metadata
    metadata = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            # Map by frame filename stem (e.g. "001", "089")
            # frame_id = "L21_V001_001" → extract last part after video_id
            fid = doc["frame_id"]
            vid = doc.get("video_id", "")
            if vid and fid.startswith(vid + "_"):
                stem = fid[len(vid) + 1:]
            else:
                stem = fid
            metadata[stem] = doc

    print(f"Loaded metadata for {len(metadata)} frames from {metadata_path}")

    # Find images to process
    images_path = Path(images_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted([
        f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if frame_names:
        # Filter to specific frames
        frame_set = set(frame_names)
        all_images = [f for f in all_images if f.stem in frame_set]

    if limit:
        all_images = all_images[:limit]

    if not all_images:
        print("No images to visualize.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load font once
    font = load_font(size=16)

    processed = 0
    for img_path in all_images:
        stem = img_path.stem
        if stem not in metadata:
            print(f"  SKIP {img_path.name}: no metadata")
            continue

        doc = metadata[stem]
        objects = doc.get("objects", [])
        tags = doc.get("tags", [])

        image = Image.open(str(img_path)).convert("RGB")
        annotated = draw_frame(image, objects, tags=tags, font=font)

        out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
        annotated.save(out_path, quality=90)
        processed += 1

        print(f"  [{processed}] {img_path.name}: {len(objects)} objects → {out_path}")

    print(f"\nDone: {processed} frames visualized → {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize object detection results (draw bbox on images)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--images", required=True,
        help="Directory containing original keyframe images",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to JSONL metadata file (from ram_gdino_pipeline.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for annotated images",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of frames to visualize",
    )
    parser.add_argument(
        "--frames", nargs="+", default=None,
        help="Specific frame names to visualize (e.g. 001 010 089)",
    )

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    visualize_from_jsonl(
        images_dir=args.images,
        metadata_path=args.metadata,
        output_dir=args.output,
        limit=args.limit,
        frame_names=args.frames,
    )


if __name__ == "__main__":
    main()
