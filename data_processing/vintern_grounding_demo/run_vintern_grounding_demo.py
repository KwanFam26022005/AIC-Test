#!/usr/bin/env python3
"""Compare Vintern coordinate grounding modes on one frame.

The script intentionally reuses the OCR pipeline's Vintern loader and image
preprocess utilities so that the demo runs under the same environment as the
existing OCR fallback path.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OCR_V2_ROOT = REPO_ROOT / "data_processing" / "OCR" / "ocr_vlm_pipeline_v2"
sys.path.insert(0, str(OCR_V2_ROOT))

from ocr_pipeline.config import make_config  # noqa: E402
from ocr_pipeline.recognizers.vintern_recognizer import (  # noqa: E402
    _build_generation_config,
    _preprocess_for_vintern,
    load_vintern_model,
    vintern_ocr_from_pixel_values,
)


LOGGER = logging.getLogger("vintern_grounding_demo")


def parse_bbox(raw: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must have exactly 4 numbers: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(round(float(p))) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bbox values must be numbers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--bbox must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    bw = x2 - x1
    bh = y2 - y1
    pad = max(min_padding, int(round(max(bw, bh) * padding_ratio)))
    return clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def draw_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (255, 32, 32),
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    line_width = max(3, int(round(max(out.size) / 320)))
    for offset in range(line_width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(16, line_width * 5))
    except Exception:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    label_h = text_bbox[3] - text_bbox[1] + 8
    label_y = max(0, y1 - label_h)
    draw.rectangle((x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 12, label_y + label_h), fill=color)
    draw.text((x1 + 6, label_y + 4), label, fill=(255, 255, 255), font=font)
    return out


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def current_vram() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }


def run_case(
    *,
    model: Any,
    tokenizer: Any,
    image_path: Path,
    prompt: str,
    cfg: dict[str, Any],
    max_tiles: int,
) -> dict[str, Any]:
    case_cfg = dict(cfg)
    case_cfg["vintern_max_tiles"] = max_tiles
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pixel_values = _preprocess_for_vintern(
        image_path,
        input_size=case_cfg.get("vintern_input_size", 448),
        max_tiles=max_tiles,
    )
    preprocess_s = time.perf_counter() - started
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=torch.float16)
    gen_cfg = _build_generation_config(tokenizer, case_cfg)
    infer_started = time.perf_counter()
    response = vintern_ocr_from_pixel_values(model, tokenizer, pixel_values, prompt, gen_cfg)
    infer_s = time.perf_counter() - infer_started
    return {
        "prompt": prompt,
        "response": response,
        "num_tiles": int(pixel_values.shape[0]),
        "preprocess_seconds": round(preprocess_s, 3),
        "inference_seconds": round(infer_s, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "vram": current_vram(),
    }


def build_prompts(
    bbox: tuple[int, int, int, int],
    ocr_text: str,
    language: str,
) -> dict[str, str]:
    x1, y1, x2, y2 = bbox
    wrong_text_hint = f"\nOCR text nghi ngờ hiện tại: {ocr_text!r}" if ocr_text else ""
    lang_hint = "Vietnamese or English" if language == "auto" else language
    return {
        "full_coord": (
            "You are testing coordinate grounding for OCR.\n"
            f"Read only the text inside bbox [x1={x1}, y1={y1}, x2={x2}, y2={y2}] "
            "in the original image pixel coordinate system.\n"
            f"The text may be {lang_hint}.{wrong_text_hint}\n"
            "Return JSON only: {\"text\":\"...\", \"confidence\":\"high|medium|low\", \"note\":\"...\"}"
        ),
        "full_box": (
            "Read only the text inside the red rectangle labeled TARGET.\n"
            f"The text may be {lang_hint}.{wrong_text_hint}\n"
            "Return JSON only: {\"text\":\"...\", \"confidence\":\"high|medium|low\", \"note\":\"...\"}"
        ),
        "crop_expanded": (
            "Read all visible text in this cropped image region as accurately as possible.\n"
            f"The text may be {lang_hint}.{wrong_text_hint}\n"
            "Return JSON only: {\"text\":\"...\", \"confidence\":\"high|medium|low\", \"note\":\"...\"}"
        ),
    }


def write_html(result: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for name, case in result["cases"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{case['num_tiles']}</td>"
            f"<td>{case['inference_seconds']:.3f}s</td>"
            f"<td><pre>{html.escape(case['response'])}</pre></td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vintern Grounding Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #102033; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #ccd6e2; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #d6e0ea; padding: 10px; vertical-align: top; }}
    th {{ background: #eef4fb; text-align: left; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
    .meta {{ color: #52657a; }}
  </style>
</head>
<body>
  <h1>Vintern Grounding Demo</h1>
  <p class="meta">Image: {html.escape(result['image'])}</p>
  <p class="meta">BBox: {html.escape(str(result['bbox']))}</p>
  <div class="grid">
    <section>
      <h2>Full Frame With Box</h2>
      <img src="full_with_box.jpg" alt="full frame with target box">
    </section>
    <section>
      <h2>Expanded Crop</h2>
      <img src="crop_expanded.jpg" alt="expanded crop">
    </section>
  </div>
  <table>
    <thead><tr><th>Case</th><th>Tiles</th><th>Inference</th><th>Response</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "summary.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Vintern coordinate grounding on one frame.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the source frame image.")
    parser.add_argument("--bbox", required=True, type=parse_bbox, help="Target bbox in original pixels: x1,y1,x2,y2.")
    parser.add_argument("--ocr-text", default="", help="Optional existing/wrong OCR text hint.")
    parser.add_argument("--model-id", default="5CD-AI/Vintern-3B-beta", help="HF model id.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for demo artifacts.")
    parser.add_argument("--device", default=None, help="Torch device, default auto cuda:0/cpu.")
    parser.add_argument("--quantization", default="4bit_nf4", choices=["4bit_nf4", "none"], help="Model quantization.")
    parser.add_argument("--attn", default=None, help="Attention implementation: flash_attention_2, sdpa, or eager.")
    parser.add_argument("--language", default="auto", help="Text language hint, e.g. Vietnamese, English, auto.")
    parser.add_argument("--full-max-tiles", type=int, default=6, help="Max Vintern tiles for full-frame cases.")
    parser.add_argument("--crop-max-tiles", type=int, default=2, help="Max Vintern tiles for crop case.")
    parser.add_argument("--input-size", type=int, default=448, help="Vintern tile size.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation max_new_tokens.")
    parser.add_argument("--padding-ratio", type=float, default=0.12, help="Expanded crop padding ratio.")
    parser.add_argument("--min-padding", type=int, default=18, help="Expanded crop minimum padding in pixels.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    bbox = clamp_bbox(args.bbox, width, height)
    expanded = expand_bbox(bbox, width, height, args.padding_ratio, args.min_padding)

    full_with_box = draw_bbox(image, bbox, "TARGET")
    crop = image.crop(expanded)
    full_path = args.output_dir / "source.jpg"
    boxed_path = args.output_dir / "full_with_box.jpg"
    crop_path = args.output_dir / "crop_expanded.jpg"
    save_image(image, full_path)
    save_image(full_with_box, boxed_path)
    save_image(crop, crop_path)

    cfg = make_config(
        vintern_model_id=args.model_id,
        vintern_device=args.device,
        vintern_quantization=None if args.quantization == "none" else args.quantization,
        vintern_attn_implementation=args.attn,
        vintern_input_size=args.input_size,
        vintern_max_new_tokens=args.max_new_tokens,
        vintern_do_sample=False,
    )

    LOGGER.info("Loading model: %s", args.model_id)
    model, tokenizer = load_vintern_model(cfg)
    prompts = build_prompts(bbox, args.ocr_text, args.language)

    cases = {
        "full_coord": run_case(
            model=model,
            tokenizer=tokenizer,
            image_path=full_path,
            prompt=prompts["full_coord"],
            cfg=cfg,
            max_tiles=args.full_max_tiles,
        ),
        "full_box": run_case(
            model=model,
            tokenizer=tokenizer,
            image_path=boxed_path,
            prompt=prompts["full_box"],
            cfg=cfg,
            max_tiles=args.full_max_tiles,
        ),
        "crop_expanded": run_case(
            model=model,
            tokenizer=tokenizer,
            image_path=crop_path,
            prompt=prompts["crop_expanded"],
            cfg=cfg,
            max_tiles=args.crop_max_tiles,
        ),
    }

    result = {
        "image": str(args.image),
        "image_size": [width, height],
        "bbox": list(bbox),
        "expanded_bbox": list(expanded),
        "model_id": args.model_id,
        "device": str(next(model.parameters()).device),
        "cases": cases,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(result, args.output_dir)

    LOGGER.info("Wrote result JSON -> %s", result_path)
    LOGGER.info("Wrote HTML summary -> %s", args.output_dir / "summary.html")
    for name, case in cases.items():
        LOGGER.info("%s: %s", name, case["response"].replace("\n", " ")[:240])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
