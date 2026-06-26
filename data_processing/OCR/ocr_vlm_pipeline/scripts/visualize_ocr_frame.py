from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.io_utils import ensure_dir, read_table
from src.shape_utils import normalize_bbox


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser(description="Visualize one frame as PP-OCR detect -> crop -> final."))
    parser.add_argument("--frame-id", default=None, help="Frame id to visualize, e.g. 017.")
    parser.add_argument("--frame-number", type=int, default=None, help="Frame number to visualize when frame-id is unknown.")
    parser.add_argument("--output-image", default=None, help="Where to save the three-panel visualization.")
    parser.add_argument("--panel-width", type=int, default=720, help="Width of each visualization panel.")
    parser.add_argument("--max-crops", type=int, default=8, help="Maximum VLM crop thumbnails to show.")
    args = parser.parse_args()

    if args.frame_id is None and args.frame_number is None:
        raise SystemExit("Pass either --frame-id or --frame-number.")

    cfg = load_config(args.config)
    out_dir = Path(cfg.project.output_dir)
    raw = _read_required(out_dir / "ppocr_raw.parquet")
    merged = _read_required(out_dir / "ocr_merged_lines.parquet")
    groups = read_table(out_dir / "ocr_groups.parquet") if (out_dir / "ocr_groups.parquet").exists() else None
    jobs = read_table(out_dir / "vlm_jobs.parquet") if (out_dir / "vlm_jobs.parquet").exists() else None

    raw_frame = select_frame_rows(raw, args.frame_id, args.frame_number)
    merged_frame = select_frame_rows(merged, args.frame_id, args.frame_number)
    if raw_frame.empty and merged_frame.empty:
        raise SystemExit("No OCR rows found for the requested frame.")

    frame_rows = merged_frame if not merged_frame.empty else raw_frame
    frame_id = str(frame_rows.iloc[0]["frame_id"])
    frame_path = Path(str(frame_rows.iloc[0]["frame_path"]))
    if not frame_path.exists():
        raise SystemExit(f"Frame image does not exist: {frame_path}")

    base = _load_image(frame_path)
    raw_panel = draw_line_overlay(
        base,
        raw_frame,
        title="1. PP-OCRv6 Detect",
        text_col="ocr_text",
        mode="raw",
        panel_width=args.panel_width,
    )
    crop_panel = draw_crop_panel(
        base,
        select_frame_rows(jobs if jobs is not None else groups, frame_id, None) if (jobs is not None or groups is not None) else None,
        title="2. VLM Crops",
        panel_width=args.panel_width,
        max_crops=args.max_crops,
    )
    final_panel = draw_line_overlay(
        base,
        merged_frame,
        title="3. Final Corrected",
        text_col="corrected_text",
        mode="final",
        panel_width=args.panel_width,
    )
    canvas = combine_panels([raw_panel, crop_panel, final_panel])

    output_image = Path(args.output_image) if args.output_image else out_dir / "qa_frames" / f"{frame_id}_detect_crop_final.jpg"
    ensure_dir(output_image.parent)
    canvas.save(output_image, quality=95)
    print_frame_table(merged_frame if not merged_frame.empty else raw_frame)
    print(f"\nSaved visualization: {output_image}")


def _read_required(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}. Run the previous pipeline stages first.")
    return read_table(path)


def select_frame_rows(df, frame_id: str | None, frame_number: int | None):
    if df is None:
        return None
    if frame_id is not None:
        return df[df["frame_id"].astype(str) == str(frame_id)].copy()
    return df[df["frame_number"].astype(int) == int(frame_number)].copy()


def draw_line_overlay(image, rows, title: str, text_col: str, mode: str, panel_width: int):
    from PIL import ImageDraw, ImageFont

    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    if rows is not None and not rows.empty:
        for row in rows.sort_values("line_idx").to_dict("records"):
            bbox = normalize_bbox(row.get("bbox"))
            changed = _truthy(row.get("text_changed"))
            vlm_corrected = _truthy(row.get("vlm_corrected"))
            color = _line_color(mode, changed, vlm_corrected)
            draw.rectangle(bbox, outline=color, width=3)
            text = str(row.get(text_col) or row.get("ocr_text") or "")
            label = f"{int(row['line_idx'])}: {text}"[:90]
            _draw_label(draw, (bbox[0], max(0, bbox[1] - 16)), label, font)

    return add_title_and_resize(panel, title, panel_width)


def draw_crop_panel(image, rows, title: str, panel_width: int, max_crops: int):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    crops = []
    if rows is not None and not rows.empty:
        for row in rows.sort_values("group_id").to_dict("records")[:max_crops]:
            crop_path = row.get("crop_path")
            if not isinstance(crop_path, str) or not crop_path:
                continue
            path = Path(crop_path)
            if not path.exists():
                continue
            crop = _load_image(path)
            label = f"g{int(row.get('group_id', 0))}: {str(row.get('raw_group_text') or '')[:70]}"
            crops.append((crop, label))

    panel_height = max(420, int(image.height * panel_width / max(1, image.width)))
    panel = Image.new("RGB", (panel_width, panel_height), (245, 247, 250))
    draw = ImageDraw.Draw(panel)
    if not crops:
        message = "No VLM crops for this frame"
        bbox = draw.textbbox((0, 0), message, font=font)
        draw.text(((panel_width - (bbox[2] - bbox[0])) // 2, panel_height // 2), message, fill=(40, 40, 40), font=font)
        return add_title(panel, title)

    cols = 2
    gap = 12
    cell_w = (panel_width - gap * (cols + 1)) // cols
    cell_h = max(120, (panel_height - gap * 5) // 4)
    for idx, (crop, label) in enumerate(crops):
        row = idx // cols
        col = idx % cols
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)
        if y + cell_h > panel_height - gap:
            break
        thumb = fit_image(crop, cell_w, cell_h - 22)
        panel.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb.width, y + thumb.height), outline=(0, 130, 255), width=2)
        _draw_label(draw, (x, y + thumb.height + 2), label[:80], font)
    return add_title(panel, title)


def combine_panels(panels):
    from PIL import Image, ImageDraw, ImageFont

    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, height + 34), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    legend = "Raw detect: orange | VLM corrected: blue | Text changed in final: green"
    _draw_label(draw, (10, height + 8), legend, font)
    return canvas


def add_title_and_resize(image, title: str, panel_width: int):
    return add_title(fit_image(image, panel_width, int(image.height * panel_width / max(1, image.width))), title)


def add_title(image, title: str):
    from PIL import Image, ImageDraw, ImageFont

    title_h = 28
    canvas = Image.new("RGB", (image.width, image.height + title_h), (22, 24, 28))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def fit_image(image, max_width: int, max_height: int):
    resampling = _resampling()
    ratio = min(max_width / max(1, image.width), max_height / max(1, image.height))
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(size, resampling)


def print_frame_table(rows) -> None:
    cols = [
        col
        for col in ["frame_id", "frame_number", "line_idx", "ocr_text", "corrected_text", "confidence", "vlm_corrected", "text_changed", "bbox"]
        if col in rows.columns
    ]
    print("\n=== Visualized Frame Lines ===")
    print(rows.sort_values("line_idx")[cols].to_string(index=False))


def _draw_label(draw, xy, text: str, font) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def _line_color(mode: str, changed: bool, vlm_corrected: bool):
    if mode == "raw":
        return (255, 170, 0)
    if changed:
        return (0, 200, 80)
    if vlm_corrected:
        return (0, 130, 255)
    return (255, 170, 0)


def _truthy(value) -> bool:
    return str(value).lower() in ["true", "1", "yes"]


def _load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _resampling():
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


if __name__ == "__main__":
    main()
