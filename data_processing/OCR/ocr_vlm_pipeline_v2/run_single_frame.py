"""
OCR VLM Pipeline v2 — Single Frame Runner.

End-to-end pipeline:
  PP-OCRv6 DET → Perspective Crop → Grouping → VietOCR → Wordlist Scoring
  → Gating → Vintern Line/Group Fallback → Export

Usage:
    python run_single_frame.py --image /path/to/frame.jpg [--output_dir ./output]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_pipeline.config import make_config
from ocr_pipeline.detector import create_detector, detect_lines
from ocr_pipeline.cropping import (
    crop_perspective_line,
    prepare_crop_for_ocr,
    crop_axis_from_bbox,
    save_line_crop,
    save_group_crop,
)
from ocr_pipeline.grouping import group_line_items
from ocr_pipeline.recognizers.vietocr_recognizer import (
    create_vietocr_predictor,
    recognize_all_lines,
)
from ocr_pipeline.recognizers.vintern_recognizer import (
    load_vintern_model,
    run_vintern_line_fallback,
    run_vintern_group_fallback,
)
from ocr_pipeline.scoring.wordlist import load_wordlist
from ocr_pipeline.scoring.gating import score_and_gate_line, detect_rec_conf_flat
from ocr_pipeline.outputs import (
    build_group_text_clean_review,
    collect_final_texts,
    export_csv_lines,
    export_csv_groups,
    export_es_document,
    export_clean_text,
    export_review_text,
    export_visualization,
)

logger = logging.getLogger(__name__)


def run_ocr_pipeline(
    image_path: str,
    cfg: dict | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the full OCR pipeline on a single frame.

    Args:
        image_path: Path to input frame image
        cfg: Configuration dict (uses defaults if None)
        output_dir: Override output directory

    Returns:
        Dict with pipeline results and timing info
    """
    pipeline_start = time.time()

    if cfg is None:
        cfg = make_config()
    if output_dir:
        cfg["output_dir"] = output_dir

    frame_id = Path(image_path).stem
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = cfg.get("output_suffix", "wordlist_gating_v2")
    line_crops_dir = out_dir / cfg.get("line_crops_subdir", "ppocr_line_perspective_crops")
    group_crops_dir = out_dir / cfg.get("group_crops_subdir", "ppocr_stacked_group_crops")

    # ══════════════════════════════════════════════════════════════════
    # 1. Load image
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"Loading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    logger.info(f"Image size: {W}x{H}")

    # ══════════════════════════════════════════════════════════════════
    # 2. Load wordlist
    # ══════════════════════════════════════════════════════════════════
    logger.info("Loading wordlist...")
    wordset, base_to_variants, wordlist_enabled = load_wordlist(
        cfg.get("wordlist_paths", []),
        cfg.get("wordlist_min_entries_to_enable", 1000),
    )

    # ══════════════════════════════════════════════════════════════════
    # 3. Text detection
    # ══════════════════════════════════════════════════════════════════
    logger.info("Running PP-OCRv6 detection...")
    detector = create_detector(cfg)
    line_items, det_time = detect_lines(detector, img_rgb, cfg)
    logger.info(f"Detected {len(line_items)} valid lines in {det_time:.3f}s")

    # ══════════════════════════════════════════════════════════════════
    # 4. Perspective crop each line
    # ══════════════════════════════════════════════════════════════════
    logger.info("Cropping lines (perspective transform)...")
    for line in line_items:
        box = np.array(line["box"], dtype=np.float32)
        crop = crop_perspective_line(img_rgb, box, pad=cfg.get("perspective_padding", 12))
        crop = prepare_crop_for_ocr(
            crop,
            min_height=cfg.get("crop_min_height_for_ocr", 72),
            max_upscale_factor=cfg.get("crop_upscale_max_factor", 3.5),
            border=cfg.get("add_white_border", 10),
            contrast_factor=cfg.get("contrast_factor", 1.25),
        )
        crop_path = save_line_crop(crop, line_crops_dir, line["line_id"])
        line["persp_crop_path"] = crop_path

    # ══════════════════════════════════════════════════════════════════
    # 5. Group nearby lines
    # ══════════════════════════════════════════════════════════════════
    logger.info("Grouping lines...")
    groups = group_line_items(line_items, cfg)

    # Crop group context
    for group in groups:
        group_crop = crop_axis_from_bbox(
            img_rgb, group["bbox_xyxy"],
            pad=cfg.get("group_crop_padding", 28),
        )
        crop_path = save_group_crop(group_crop, group_crops_dir, group["group_id"])
        group["group_crop_path"] = crop_path

        # Also set group_crop_path on each line in this group
        for idx in group["line_indices"]:
            line_items[idx]["group_crop_path"] = crop_path

    # ══════════════════════════════════════════════════════════════════
    # 6. VietOCR recognition
    # ══════════════════════════════════════════════════════════════════
    logger.info("Running VietOCR recognition...")
    vietocr_predictor = create_vietocr_predictor(cfg)
    vietocr_time = recognize_all_lines(vietocr_predictor, line_items, cfg)

    # ══════════════════════════════════════════════════════════════════
    # 7. Detect rec_conf flat
    # ══════════════════════════════════════════════════════════════════
    rec_flat = detect_rec_conf_flat(line_items, cfg)
    if rec_flat:
        logger.warning("rec_conf is FLAT — switching to rec_missing gating rule")
        for line in line_items:
            line["rec_conf_flat_run"] = True

    # ══════════════════════════════════════════════════════════════════
    # 8. Score + Gate each line
    # ══════════════════════════════════════════════════════════════════
    logger.info("Scoring and gating lines...")
    for line in line_items:
        score_and_gate_line(
            line, cfg, wordset, base_to_variants, wordlist_enabled,
            W, H, use_rec_missing_rule=rec_flat,
        )

    # Log gating distribution
    status_counts = {}
    for line in line_items:
        status = line.get("filter_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    logger.info(f"Gating distribution: {status_counts}")

    # ══════════════════════════════════════════════════════════════════
    # 9. Vintern line fallback
    # ══════════════════════════════════════════════════════════════════
    vintern_line_time = 0.0
    vintern_group_time = 0.0

    if cfg.get("use_vintern_line_fallback", True) or cfg.get("use_vintern_group_fallback", True):
        logger.info("Loading Vintern model...")
        vintern_model, vintern_tokenizer = load_vintern_model(cfg)

        if cfg.get("use_vintern_line_fallback", True):
            vintern_line_time = run_vintern_line_fallback(
                vintern_model, vintern_tokenizer, line_items, cfg,
                wordset, base_to_variants, wordlist_enabled,
            )

        # ══════════════════════════════════════════════════════════════
        # 10. Build group text (before group Vintern)
        # ══════════════════════════════════════════════════════════════
        build_group_text_clean_review(groups, line_items, cfg)

        # ══════════════════════════════════════════════════════════════
        # 11. Vintern group fallback
        # ══════════════════════════════════════════════════════════════
        if cfg.get("use_vintern_group_fallback", True):
            vintern_group_time = run_vintern_group_fallback(
                vintern_model, vintern_tokenizer, groups, line_items, cfg,
                wordset, base_to_variants, wordlist_enabled,
            )
    else:
        build_group_text_clean_review(groups, line_items, cfg)

    # ══════════════════════════════════════════════════════════════════
    # 12. Collect final texts
    # ══════════════════════════════════════════════════════════════════
    final_clean, final_review = collect_final_texts(groups)
    logger.info(f"Clean text: {len(final_clean)} chars")
    logger.info(f"Review text: {len(final_review)} chars")

    # ══════════════════════════════════════════════════════════════════
    # 13. Export outputs
    # ══════════════════════════════════════════════════════════════════
    logger.info("Exporting outputs...")

    timing = {
        "det_time_sec": round(det_time, 3),
        "vietocr_total_time_sec": round(vietocr_time, 3),
        "vintern_line_total_time_sec": round(vintern_line_time, 3),
        "vintern_group_total_time_sec": round(vintern_group_time, 3),
        "total_pipeline_sec": round(time.time() - pipeline_start, 3),
    }

    export_csv_lines(
        line_items,
        out_dir / f"{frame_id}_ocr_lines_{suffix}.csv",
    )
    export_csv_groups(
        groups,
        out_dir / f"{frame_id}_ocr_groups_{suffix}.csv",
    )
    export_es_document(
        line_items, groups, final_clean, final_review, cfg,
        image_path,
        out_dir / f"{frame_id}_ocr_es_doc_{suffix}.json",
        timing,
    )
    export_clean_text(
        final_clean,
        out_dir / f"{frame_id}_ocr_clean_text_{suffix}.txt",
    )
    export_review_text(
        final_review,
        out_dir / f"{frame_id}_ocr_review_text_{suffix}.txt",
    )
    export_visualization(
        img_rgb, line_items, groups,
        out_dir / f"{frame_id}_ocr_vis_{suffix}.png",
    )

    pipeline_time = time.time() - pipeline_start
    timing["total_pipeline_sec"] = round(pipeline_time, 3)

    logger.info(f"Pipeline complete in {pipeline_time:.2f}s")
    logger.info(f"  Detection: {det_time:.2f}s")
    logger.info(f"  VietOCR: {vietocr_time:.2f}s")
    logger.info(f"  Vintern line: {vintern_line_time:.2f}s")
    logger.info(f"  Vintern group: {vintern_group_time:.2f}s")

    return {
        "frame_id": frame_id,
        "image_path": str(image_path),
        "line_items": line_items,
        "groups": groups,
        "final_clean_text": final_clean,
        "final_review_text": final_review,
        "timing": timing,
        "gating_stats": status_counts,
        "rec_conf_flat": rec_flat,
        "wordlist_enabled": wordlist_enabled,
        "wordlist_size": len(wordset),
    }


def main():
    parser = argparse.ArgumentParser(description="OCR VLM Pipeline v2 — Single Frame")
    parser.add_argument("--image", required=True, help="Path to input frame image")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--no_vintern", action="store_true", help="Disable Vintern fallback")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = make_config()
    if args.no_vintern:
        cfg["use_vintern_line_fallback"] = False
        cfg["use_vintern_group_fallback"] = False

    result = run_ocr_pipeline(args.image, cfg, args.output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Frame: {result['frame_id']}")
    print(f"Lines detected: {len(result['line_items'])}")
    print(f"Groups formed: {len(result['groups'])}")
    print(f"Gating: {result['gating_stats']}")
    print(f"rec_conf flat: {result['rec_conf_flat']}")
    print(f"Wordlist: enabled={result['wordlist_enabled']}, size={result['wordlist_size']}")
    print(f"\nClean text ({len(result['final_clean_text'])} chars):")
    print(result["final_clean_text"][:500] if result["final_clean_text"] else "(empty)")
    print(f"\nReview text ({len(result['final_review_text'])} chars):")
    print(result["final_review_text"][:500] if result["final_review_text"] else "(empty)")
    print(f"\nTiming: {result['timing']}")


if __name__ == "__main__":
    main()
