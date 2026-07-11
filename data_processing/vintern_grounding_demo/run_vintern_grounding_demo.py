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


def infer_video_id(frame_id: str) -> str | None:
    parts = frame_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return docs


def resolve_ocr_jsonl(repo_root: Path, video_id: str, ocr_root: Path | None, ocr_jsonl: Path | None) -> Path:
    if ocr_jsonl is not None:
        return ocr_jsonl
    root = ocr_root or (repo_root / "outputs" / "ocr_vlm_pipeline_v2")
    candidates = [
        root / video_id / f"{video_id}_ocr_es_docs.jsonl",
        root / f"{video_id}_ocr_es_docs.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find OCR JSONL. Tried: "
        + ", ".join(str(path) for path in candidates)
        + ". Pass --ocr-jsonl explicitly if output is elsewhere."
    )


def find_ocr_doc(docs: list[dict[str, Any]], frame_id: str) -> dict[str, Any]:
    normalized = frame_id.removesuffix(".jpg").removesuffix(".png")
    for doc in docs:
        values = {
            str(doc.get("frame_id") or ""),
            str(doc.get("canonical_frame_id") or ""),
            str(doc.get("legacy_frame_id") or ""),
            str(doc.get("frame_name") or ""),
            Path(str(doc.get("image_path") or "")).stem,
        }
        media = doc.get("media") if isinstance(doc.get("media"), dict) else {}
        values.update({
            str(media.get("frame_id") or ""),
            str(media.get("canonical_frame_id") or ""),
            str(media.get("legacy_frame_id") or ""),
            str(media.get("frame_name") or ""),
        })
        values = {value.removesuffix(".jpg").removesuffix(".png") for value in values if value}
        if normalized in values or frame_id in values:
            return doc
    raise KeyError(f"Frame id not found in OCR JSONL: {frame_id}")


def resolve_image_from_doc(doc: dict[str, Any], frames_root: Path | None) -> Path:
    media = doc.get("media") if isinstance(doc.get("media"), dict) else {}
    for raw in (doc.get("image_path"), media.get("image_path")):
        if raw:
            path = Path(str(raw))
            if path.exists():
                return path
    rel = doc.get("image_relpath") or media.get("image_relpath")
    if rel and frames_root is not None:
        path = frames_root / str(rel)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot resolve frame image from OCR doc. Pass --image explicitly, "
        "or pass --frames-root pointing to keyframe_test."
    )


def line_candidate_score(line: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = line.get("bbox_xyxy") or [0, 0, 0, 0]
    try:
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except Exception:
        area = 0.0
    text = (line.get("final_text") or line.get("vietocr_text") or line.get("vintern_text") or "").strip()
    composite = line.get("composite_score")
    try:
        composite_value = float(composite)
    except Exception:
        composite_value = 1.0
    suspicious = 1.0 if line.get("send_to_vintern") or line.get("need_review") else 0.0
    return suspicious, 1.0 - composite_value, float(len(text)), area


def ranked_ocr_lines(doc: dict[str, Any]) -> list[dict[str, Any]]:
    lines = [line for line in doc.get("ocr_lines", []) if line.get("bbox_xyxy")]
    if not lines:
        raise ValueError("OCR doc has no ocr_lines with bbox_xyxy.")
    return sorted(lines, key=line_candidate_score, reverse=True)


def pick_ocr_line(doc: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    ranked = ranked_ocr_lines(doc)
    if candidate_index < 0 or candidate_index >= len(ranked):
        raise IndexError(f"--candidate-index {candidate_index} out of range; frame has {len(ranked)} candidates.")
    return ranked[candidate_index]


def line_text_hint(line: dict[str, Any] | None) -> str:
    if not line:
        return ""
    return (
        line.get("final_text")
        or line.get("vietocr_text")
        or line.get("vintern_text")
        or ""
    )


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


def write_all_html(results: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for result in results:
        candidate_dir = html.escape(result["candidate_dir"])
        cases = result.get("cases", {})
        response_cells = []
        for case_name in ("full_coord", "full_box", "crop_expanded"):
            response = cases.get(case_name, {}).get("response", "")
            response_cells.append(f"<td><pre>{html.escape(response)}</pre></td>")
        rows.append(
            "<tr>"
            f"<td>{result['candidate_index']}</td>"
            f"<td>{html.escape(str(result.get('line_id')))}</td>"
            f"<td>{html.escape(str(result['bbox']))}</td>"
            f"<td>{html.escape(str(result.get('ocr_text_hint', '')))}</td>"
            f"<td><a href=\"{candidate_dir}/summary.html\">open</a></td>"
            + "".join(response_cells)
            + "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vintern Grounding Demo - All Candidates</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #102033; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
    th, td {{ border: 1px solid #d6e0ea; padding: 8px; vertical-align: top; }}
    th {{ background: #eef4fb; text-align: left; position: sticky; top: 0; }}
    pre {{ white-space: pre-wrap; margin: 0; max-width: 360px; }}
  </style>
</head>
<body>
  <h1>Vintern Grounding Demo - All Candidates</h1>
  <p>Total candidates: {len(results)}</p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Line</th><th>BBox</th><th>OCR Hint</th><th>Detail</th>
        <th>Full Coord</th><th>Full Box</th><th>Crop Expanded</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "summary_all.html").write_text(html_text, encoding="utf-8")


def prepare_candidate_artifacts(
    *,
    image: Image.Image,
    image_path: Path,
    bbox: tuple[int, int, int, int],
    ocr_text: str,
    output_dir: Path,
    candidate_index: int,
    line: dict[str, Any] | None,
    padding_ratio: float,
    min_padding: int,
) -> dict[str, Any]:
    width, height = image.size
    safe_bbox = clamp_bbox(bbox, width, height)
    expanded = expand_bbox(safe_bbox, width, height, padding_ratio, min_padding)
    label = f"TARGET {candidate_index}"
    full_with_box = draw_bbox(image, safe_bbox, label)
    crop = image.crop(expanded)

    full_path = output_dir / "source.jpg"
    boxed_path = output_dir / "full_with_box.jpg"
    crop_path = output_dir / "crop_expanded.jpg"
    save_image(image, full_path)
    save_image(full_with_box, boxed_path)
    save_image(crop, crop_path)

    return {
        "candidate_index": candidate_index,
        "line_id": line.get("line_id") if line else None,
        "image": str(image_path),
        "image_size": [width, height],
        "bbox": list(safe_bbox),
        "expanded_bbox": list(expanded),
        "ocr_text_hint": ocr_text,
        "candidate_dir": output_dir.name,
        "paths": {
            "source": str(full_path),
            "full_with_box": str(boxed_path),
            "crop_expanded": str(crop_path),
        },
        "line": {
            "line_id": line.get("line_id"),
            "group_id": line.get("group_id"),
            "final_text": line.get("final_text"),
            "vietocr_text": line.get("vietocr_text"),
            "vintern_text": line.get("vintern_text"),
            "composite_score": line.get("composite_score"),
            "send_to_vintern": line.get("send_to_vintern"),
            "need_review": line.get("need_review"),
        } if line else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Vintern coordinate grounding on one frame.")
    parser.add_argument("--image", type=Path, help="Path to the source frame image. Optional when --frame-id is used.")
    parser.add_argument("--bbox", type=parse_bbox, help="Target bbox in original pixels: x1,y1,x2,y2. Optional when --frame-id is used.")
    parser.add_argument("--frame-id", help="Frame id to look up from old OCR output, e.g. L22_V012_001.")
    parser.add_argument("--video-id", help="Video id, e.g. L22_V012. Inferred from --frame-id when omitted.")
    parser.add_argument("--ocr-jsonl", type=Path, help="Exact OCR ES JSONL path.")
    parser.add_argument("--ocr-root", type=Path, help="OCR output root. Default: <repo>/outputs/ocr_vlm_pipeline_v2.")
    parser.add_argument("--frames-root", type=Path, help="Frames root used to resolve image_relpath. Default: <repo>/keyframe_test.")
    parser.add_argument("--candidate-index", type=int, default=0, help="Ranked OCR line candidate index within the frame.")
    parser.add_argument("--max-candidates", type=int, default=1, help="Number of ranked OCR boxes to test from --candidate-index.")
    parser.add_argument("--all-candidates", action="store_true", help="Test all OCR boxes in the selected frame.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve OCR bbox and write crop/box artifacts without loading Vintern.")
    parser.add_argument("--ocr-text", default=None, help="Optional existing/wrong OCR text hint. Defaults to selected OCR line text.")
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

    if args.image is not None and not args.image.exists():
        raise FileNotFoundError(args.image)

    selected_lines: list[tuple[int, dict[str, Any]]] = []
    selected_doc: dict[str, Any] | None = None

    if args.frame_id and (args.image is None or args.bbox is None or args.all_candidates or args.max_candidates != 1):
        video_id = args.video_id or infer_video_id(args.frame_id)
        if not video_id:
            raise ValueError("--video-id is required when it cannot be inferred from --frame-id")
        ocr_jsonl = resolve_ocr_jsonl(REPO_ROOT, video_id, args.ocr_root, args.ocr_jsonl)
        LOGGER.info("Reading OCR output: %s", ocr_jsonl)
        selected_doc = find_ocr_doc(read_jsonl(ocr_jsonl), args.frame_id)
        ranked_lines = ranked_ocr_lines(selected_doc)
        if args.candidate_index < 0 or args.candidate_index >= len(ranked_lines):
            raise IndexError(
                f"--candidate-index {args.candidate_index} out of range; "
                f"frame has {len(ranked_lines)} candidates."
            )
        if args.all_candidates:
            count = len(ranked_lines) - args.candidate_index
        else:
            count = max(1, args.max_candidates)
        selected_lines = list(enumerate(
            ranked_lines[args.candidate_index:args.candidate_index + count],
            start=args.candidate_index,
        ))
        line = selected_lines[0][1]
        if args.image is None:
            args.image = resolve_image_from_doc(selected_doc, args.frames_root or (REPO_ROOT / "keyframe_test"))
        if args.bbox is None:
            args.bbox = parse_bbox(",".join(str(v) for v in line["bbox_xyxy"]))
        if args.ocr_text is None:
            args.ocr_text = line_text_hint(line)
        LOGGER.info(
            "Selected %d OCR candidate(s), first line_id=%s bbox=%s text=%r score=%s send_to_vintern=%s need_review=%s",
            len(selected_lines),
            line.get("line_id"),
            args.bbox,
            args.ocr_text,
            line.get("composite_score"),
            line.get("send_to_vintern"),
            line.get("need_review"),
        )

    if args.image is None or args.bbox is None:
        raise ValueError("Provide either --frame-id with old OCR output, or both --image and --bbox.")
    if args.ocr_text is None:
        args.ocr_text = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    if not selected_lines:
        selected_lines = [(
            args.candidate_index,
            {
                "line_id": "manual",
                "bbox_xyxy": list(args.bbox),
                "final_text": args.ocr_text,
            },
        )]

    candidate_specs = []
    for candidate_index, line in selected_lines:
        bbox = parse_bbox(",".join(str(v) for v in line["bbox_xyxy"]))
        ocr_text = args.ocr_text if len(selected_lines) == 1 and args.ocr_text else line_text_hint(line)
        candidate_dir = args.output_dir
        if len(selected_lines) > 1:
            candidate_dir = args.output_dir / f"candidate_{candidate_index:03d}_{line.get('line_id') or 'line'}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_specs.append({
            "candidate_index": candidate_index,
            "line": line,
            "bbox": bbox,
            "ocr_text": ocr_text,
            "output_dir": candidate_dir,
        })

    if args.dry_run:
        dry_results = []
        for spec in candidate_specs:
            dry_result = prepare_candidate_artifacts(
                image=image,
                image_path=args.image,
                bbox=spec["bbox"],
                ocr_text=spec["ocr_text"],
                output_dir=spec["output_dir"],
                candidate_index=spec["candidate_index"],
                line=spec["line"],
                padding_ratio=args.padding_ratio,
                min_padding=args.min_padding,
            )
            dry_result["note"] = "dry_run_only_no_vintern_loaded"
            dry_path = spec["output_dir"] / "selected_region.json"
            dry_path.write_text(json.dumps(dry_result, ensure_ascii=False, indent=2), encoding="utf-8")
            dry_results.append(dry_result)
        if len(dry_results) > 1:
            all_path = args.output_dir / "selected_regions_all.json"
            all_path.write_text(json.dumps(dry_results, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("Dry run wrote %d selected regions -> %s", len(dry_results), all_path)
        else:
            LOGGER.info("Dry run wrote selected region -> %s", candidate_specs[0]["output_dir"] / "selected_region.json")
        return 0

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
    results = []
    total = len(candidate_specs)
    for pos, spec in enumerate(candidate_specs, start=1):
        LOGGER.info(
            "[%d/%d] Running candidate=%s line_id=%s bbox=%s",
            pos,
            total,
            spec["candidate_index"],
            spec["line"].get("line_id"),
            spec["bbox"],
        )
        base = prepare_candidate_artifacts(
            image=image,
            image_path=args.image,
            bbox=spec["bbox"],
            ocr_text=spec["ocr_text"],
            output_dir=spec["output_dir"],
            candidate_index=spec["candidate_index"],
            line=spec["line"],
            padding_ratio=args.padding_ratio,
            min_padding=args.min_padding,
        )
        prompts = build_prompts(tuple(base["bbox"]), spec["ocr_text"], args.language)
        paths = base["paths"]
        cases = {
            "full_coord": run_case(
                model=model,
                tokenizer=tokenizer,
                image_path=Path(paths["source"]),
                prompt=prompts["full_coord"],
                cfg=cfg,
                max_tiles=args.full_max_tiles,
            ),
            "full_box": run_case(
                model=model,
                tokenizer=tokenizer,
                image_path=Path(paths["full_with_box"]),
                prompt=prompts["full_box"],
                cfg=cfg,
                max_tiles=args.full_max_tiles,
            ),
            "crop_expanded": run_case(
                model=model,
                tokenizer=tokenizer,
                image_path=Path(paths["crop_expanded"]),
                prompt=prompts["crop_expanded"],
                cfg=cfg,
                max_tiles=args.crop_max_tiles,
            ),
        }

        result = {
            **base,
            "model_id": args.model_id,
            "device": str(next(model.parameters()).device),
            "cases": cases,
        }
        result_path = spec["output_dir"] / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_html(result, spec["output_dir"])
        results.append(result)

        LOGGER.info("Wrote candidate result JSON -> %s", result_path)
        for name, case in cases.items():
            LOGGER.info("%s candidate=%s: %s", name, spec["candidate_index"], case["response"].replace("\n", " ")[:240])

    if len(results) > 1:
        all_path = args.output_dir / "result_all.json"
        all_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        write_all_html(results, args.output_dir)
        LOGGER.info("Wrote all-candidate result JSON -> %s", all_path)
        LOGGER.info("Wrote all-candidate HTML summary -> %s", args.output_dir / "summary_all.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
