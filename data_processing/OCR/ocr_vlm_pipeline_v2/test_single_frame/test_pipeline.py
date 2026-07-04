"""
Test Pipeline — Visualize every stage of the OCR pipeline on a single frame.

This script runs the pipeline step-by-step and generates detailed visualizations
and reports for each stage, allowing inspection before production runs.

Usage:
    python test_pipeline.py --image /path/to/frame.jpg [--output_dir ./test_output]

Output structure:
    test_output/
    ├── 01_input_frame.png
    ├── 02_detection_boxes.png
    ├── 03_line_crops/
    │   ├── line_000_persp.png
    │   └── ...
    ├── 04_groups_visualization.png
    ├── 05_group_crops/
    │   ├── group_000.png
    │   └── ...
    ├── 06_vietocr_results.png
    ├── 07_scoring_heatmap.png
    ├── 08_gating_distribution.png
    ├── 09_vintern_line_results.png
    ├── 10_vintern_group_results.png
    ├── 11_final_visualization.png
    ├── pipeline_report.txt
    ├── line_details.csv
    └── group_details.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PIPELINE_DIR))

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
from ocr_pipeline.scoring.wordlist import load_wordlist, extract_tokens, get_eval_tokens
from ocr_pipeline.scoring.gating import score_and_gate_line, detect_rec_conf_flat
from ocr_pipeline.outputs import (
    build_group_text_clean_review,
    collect_final_texts,
    export_csv_lines,
    export_csv_groups,
    export_visualization,
)

logger = logging.getLogger(__name__)


# ── Visualization helpers ────────────────────────────────────────────


def _draw_text_bg(img, text, pos, font_scale=0.45, color=(255, 255, 255),
                  bg_color=(0, 0, 0), thickness=1):
    """Draw text with background rectangle."""
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def _save_vis(img_rgb, path, title=None):
    """Save visualization image."""
    vis = img_rgb.copy()
    if title:
        _draw_text_bg(vis, title, (10, 25), font_scale=0.7, color=(255, 255, 0),
                      bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_detection(img_rgb, line_items, output_path):
    """Stage 2: Visualize detected bounding boxes."""
    vis = img_rgb.copy()
    for line in line_items:
        bbox = line["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        det_score = line.get("det_score", 0.0)

        # Color by det_score: green=high, red=low
        r = int(255 * (1 - det_score))
        g = int(255 * det_score)
        color = (r, g, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"L{line['line_id']} det={det_score:.2f}"
        _draw_text_bg(vis, label, (x1, y1 - 2), font_scale=0.35, bg_color=color)

    _draw_text_bg(vis, f"Detection: {len(line_items)} lines", (10, 25),
                  font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_groups(img_rgb, line_items, groups, output_path):
    """Stage 4: Visualize grouping results."""
    vis = img_rgb.copy()

    # Generate distinct colors for each group
    n_groups = len(groups)
    colors = []
    for i in range(n_groups):
        hue = int(180 * i / max(1, n_groups))
        hsv = np.uint8([[[hue, 255, 200]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(int(c) for c in rgb))

    for group in groups:
        gid = group["group_id"]
        color = colors[gid % len(colors)] if colors else (200, 200, 200)

        # Draw group bbox
        gbbox = group["bbox_xyxy"]
        cv2.rectangle(vis, (gbbox[0], gbbox[1]), (gbbox[2], gbbox[3]), color, 3)

        # Draw line bboxes within group
        for idx in group["line_indices"]:
            bbox = line_items[idx]["bbox_xyxy"]
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

        # Group label
        label = f"G{gid} ({group['num_lines']}L)"
        _draw_text_bg(vis, label, (gbbox[0], gbbox[1] - 2), font_scale=0.45, bg_color=color)

    _draw_text_bg(vis, f"Groups: {len(groups)} ({sum(1 for g in groups if g['is_multiline_group'])} multi-line)",
                  (10, 25), font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_vietocr_results(img_rgb, line_items, output_path):
    """Stage 6: Visualize VietOCR recognition results."""
    vis = img_rgb.copy()
    for line in line_items:
        bbox = line["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        text = line.get("vietocr_text", "")
        conf = line.get("rec_conf")
        source = line.get("rec_conf_source", "")

        # Color by confidence
        if conf is not None:
            r = int(255 * (1 - conf))
            g = int(255 * conf)
            color = (r, g, 0)
        else:
            color = (200, 200, 0)  # Yellow for missing conf

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        conf_str = f"{conf:.2f}" if conf is not None else "N/A"
        label = f"{text[:25]} [{conf_str}]"
        _draw_text_bg(vis, label, (x1, y2 + 12), font_scale=0.35, bg_color=(0, 0, 0))

    _draw_text_bg(vis, "VietOCR Results", (10, 25),
                  font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_scoring(img_rgb, line_items, output_path):
    """Stage 7: Visualize composite scores as heatmap."""
    vis = img_rgb.copy()
    for line in line_items:
        bbox = line["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        score = line.get("composite_score", 0.0)
        lex = line.get("lex_ratio", 0.0)
        dia = line.get("diacritic_susp", 0.0)

        # Heatmap color: green=high score, red=low
        r = int(255 * (1 - score))
        g = int(255 * score)
        color = (r, g, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"CS={score:.2f} lex={lex:.2f} dia={dia:.2f}"
        _draw_text_bg(vis, label, (x1, y1 - 2), font_scale=0.3, bg_color=color)

    _draw_text_bg(vis, "Composite Scores", (10, 25),
                  font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_gating(img_rgb, line_items, output_path):
    """Stage 8: Visualize gating decisions."""
    vis = img_rgb.copy()

    status_colors = {
        "auto_accept": (0, 220, 0),
        "vlm_candidate": (255, 165, 0),
        "structural_filter": (128, 128, 128),
        "review_only": (255, 255, 0),
    }

    for line in line_items:
        bbox = line["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        status = line.get("filter_status", "unknown")
        color = status_colors.get(status, (200, 200, 200))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text = line.get("vietocr_text", "")
        label = f"[{status[:6]}] {text[:20]}"
        _draw_text_bg(vis, label, (x1, y1 - 2), font_scale=0.3, bg_color=color)

    # Legend
    y_legend = 30
    for status, color in status_colors.items():
        cv2.rectangle(vis, (10, y_legend - 12), (25, y_legend), color, -1)
        cv2.putText(vis, status, (30, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_legend += 20

    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_vintern_lines(img_rgb, line_items, output_path):
    """Stage 9: Visualize Vintern line fallback results."""
    vis = img_rgb.copy()
    for line in line_items:
        bbox = line["bbox_xyxy"]
        x1, y1, x2, y2 = bbox

        vintern_text = line.get("vintern_text")
        if vintern_text is not None:
            # This line was processed by Vintern
            similarity = line.get("agreement_similarity", 0.0)
            source = line.get("final_source", "")

            if "agree" in source:
                color = (0, 255, 200)  # Cyan - agreement
            elif "override" in source:
                color = (255, 0, 200)  # Magenta - override
            else:
                color = (255, 100, 100)  # Red-ish - insufficient

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            vietocr = line.get("vietocr_text", "")
            label1 = f"V: {vietocr[:20]}"
            label2 = f"Vn: {vintern_text[:20]} sim={similarity:.2f}"
            _draw_text_bg(vis, label1, (x1, y2 + 12), font_scale=0.3, bg_color=(0, 0, 0))
            _draw_text_bg(vis, label2, (x1, y2 + 26), font_scale=0.3, bg_color=color)
        else:
            # Not processed by Vintern
            cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)

    _draw_text_bg(vis, "Vintern Line Fallback", (10, 25),
                  font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def visualize_vintern_groups(img_rgb, groups, output_path):
    """Stage 10: Visualize Vintern group fallback results."""
    vis = img_rgb.copy()
    for group in groups:
        bbox = group["bbox_xyxy"]
        x1, y1, x2, y2 = bbox

        gv_text = group.get("group_vintern_text")
        if gv_text is not None:
            gv_score = group.get("group_vintern_composite_score", 0.0)
            accepted = group.get("group_keep_for_index", False)
            color = (0, 255, 0) if accepted else (255, 0, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            label = f"G{group['group_id']}: {gv_text[:30]} [cs={gv_score:.2f}]"
            _draw_text_bg(vis, label, (x1, y2 + 12), font_scale=0.35, bg_color=color)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)

    _draw_text_bg(vis, "Vintern Group Fallback", (10, 25),
                  font_scale=0.6, color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ── Report generation ────────────────────────────────────────────────


def generate_report(
    result: dict,
    line_items: list[dict],
    groups: list[dict],
    output_path: str | Path,
) -> None:
    """Generate a detailed text report of the pipeline results."""
    lines = []
    lines.append("=" * 70)
    lines.append("OCR VLM PIPELINE v2 — TEST REPORT")
    lines.append("=" * 70)
    lines.append("")

    # 1. Overview
    lines.append("─── OVERVIEW ───")
    lines.append(f"  Frame:          {result['frame_id']}")
    lines.append(f"  Image:          {result['image_path']}")
    lines.append(f"  Lines detected: {len(line_items)}")
    lines.append(f"  Groups formed:  {len(groups)}")
    lines.append(f"  Wordlist:       enabled={result['wordlist_enabled']}, size={result['wordlist_size']}")
    lines.append(f"  rec_conf flat:  {result['rec_conf_flat']}")
    lines.append("")

    # 2. Timing
    lines.append("─── TIMING ───")
    timing = result.get("timing", {})
    for key, val in timing.items():
        lines.append(f"  {key}: {val}")
    lines.append("")

    # 3. Gating stats
    lines.append("─── GATING DISTRIBUTION ───")
    stats = result.get("gating_stats", {})
    total_non_structural = sum(v for k, v in stats.items() if k != "structural_filter")
    for status, count in sorted(stats.items()):
        lines.append(f"  {status}: {count}")
    if total_non_structural > 0:
        auto = stats.get("auto_accept", 0)
        vlm = stats.get("vlm_candidate", 0)
        lines.append(f"  auto_accept_rate (non-structural): {auto / total_non_structural:.2%}")
        lines.append(f"  escalation_rate (non-structural): {vlm / total_non_structural:.2%}")
    lines.append("")

    # 4. rec_conf distribution
    lines.append("─── REC_CONF DISTRIBUTION ───")
    source_counts = {}
    confs = []
    for line in line_items:
        src = line.get("rec_conf_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        if line.get("rec_conf") is not None:
            confs.append(float(line["rec_conf"]))
    for src, cnt in sorted(source_counts.items()):
        lines.append(f"  {src}: {cnt}")
    if confs:
        lines.append(f"  rec_conf: min={min(confs):.4f}, max={max(confs):.4f}, "
                      f"mean={np.mean(confs):.4f}, std={np.std(confs):.4f}")
    lines.append("")

    # 5. Line details
    lines.append("─── LINE DETAILS ───")
    for line in line_items:
        lid = line.get("line_id", "?")
        gid = line.get("group_id", "?")
        text = line.get("vietocr_text", "")
        final = line.get("final_text", "")
        status = line.get("filter_status", "?")
        cs = line.get("composite_score", 0.0)
        lex = line.get("lex_ratio", 0.0)
        dia = line.get("diacritic_susp", 0.0)
        src = line.get("final_source", "?")
        keep = line.get("keep_for_index", False)
        review = line.get("need_review", False)
        vintern = line.get("vintern_text")
        sim = line.get("agreement_similarity")

        lines.append(f"  Line {lid} (G{gid}) [{status}]")
        lines.append(f"    VietOCR: {text!r}")
        if vintern is not None:
            lines.append(f"    Vintern: {vintern!r} (sim={sim:.2f})" if sim else f"    Vintern: {vintern!r}")
        lines.append(f"    Final:   {final!r} (source={src})")
        lines.append(f"    Scores:  CS={cs:.3f} lex={lex:.2f} dia={dia:.2f}")
        lines.append(f"    Index:   keep={keep} review={review}")
        lines.append("")

    # 6. Group details
    lines.append("─── GROUP DETAILS ───")
    for group in groups:
        gid = group["group_id"]
        nl = group["num_lines"]
        clean = group.get("group_text_clean", "")
        review = group.get("group_text_review", "")
        gv = group.get("group_vintern_text")
        gv_score = group.get("group_vintern_composite_score")
        gsrc = group.get("group_final_source")

        lines.append(f"  Group {gid} ({nl} lines, multiline={group['is_multiline_group']})")
        lines.append(f"    Clean text:  {clean!r}")
        lines.append(f"    Review text: {review!r}")
        if gv is not None:
            lines.append(f"    Vintern:     {gv!r} (score={gv_score:.2f})" if gv_score else f"    Vintern: {gv!r}")
            lines.append(f"    Source:      {gsrc}")
        lines.append("")

    # 7. Final text
    lines.append("─── FINAL CLEAN TEXT (for indexing) ───")
    lines.append(result.get("final_clean_text", "(empty)") or "(empty)")
    lines.append("")
    lines.append("─── FINAL REVIEW TEXT (not indexed) ───")
    lines.append(result.get("final_review_text", "(empty)") or "(empty)")

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    print(report)


# ── Main test function ───────────────────────────────────────────────


def test_pipeline(image_path: str, output_dir: str = None, no_vintern: bool = False):
    """Run the full pipeline with stage-by-stage visualization."""
    cfg = make_config()
    if no_vintern:
        cfg["use_vintern_line_fallback"] = False
        cfg["use_vintern_group_fallback"] = False

    frame_id = Path(image_path).stem
    out_dir = Path(output_dir or str(_SCRIPT_DIR / "test_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override config output dirs
    line_crops_dir = out_dir / "03_line_crops"
    group_crops_dir = out_dir / "05_group_crops"

    pipeline_start = time.time()

    # ── 1. Load image ────────────────────────────────────────────────
    print("\n[1/11] Loading image...")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print(f"  Size: {W}x{H}")

    _save_vis(img_rgb, out_dir / "01_input_frame.png", f"Input: {W}x{H}")

    # ── 2. Load wordlist ─────────────────────────────────────────────
    print("\n[2/11] Loading wordlist...")
    wordset, base_to_variants, wordlist_enabled = load_wordlist(
        cfg["wordlist_paths"], cfg["wordlist_min_entries_to_enable"]
    )
    print(f"  Wordlist: {len(wordset)} entries, enabled={wordlist_enabled}")

    # ── 3. Detection ─────────────────────────────────────────────────
    print("\n[3/11] Running detection...")
    detector = create_detector(cfg)
    line_items, det_time = detect_lines(detector, img_rgb, cfg)
    print(f"  Found {len(line_items)} lines in {det_time:.3f}s")

    visualize_detection(img_rgb, line_items, out_dir / "02_detection_boxes.png")

    # ── 4. Perspective crop ──────────────────────────────────────────
    print("\n[4/11] Cropping lines...")
    for line in line_items:
        box = np.array(line["box"], dtype=np.float32)
        crop = crop_perspective_line(img_rgb, box, pad=cfg["perspective_padding"])
        crop = prepare_crop_for_ocr(
            crop,
            min_height=cfg["crop_min_height_for_ocr"],
            max_upscale_factor=cfg["crop_upscale_max_factor"],
            border=cfg["add_white_border"],
            contrast_factor=cfg["contrast_factor"],
        )
        crop_path = save_line_crop(crop, line_crops_dir, line["line_id"])
        line["persp_crop_path"] = crop_path
    print(f"  Saved {len(line_items)} line crops")

    # ── 5. Grouping ──────────────────────────────────────────────────
    print("\n[5/11] Grouping lines...")
    groups = group_line_items(line_items, cfg)
    print(f"  {len(groups)} groups ({sum(1 for g in groups if g['is_multiline_group'])} multi-line)")

    for group in groups:
        group_crop = crop_axis_from_bbox(img_rgb, group["bbox_xyxy"], pad=cfg["group_crop_padding"])
        crop_path = save_group_crop(group_crop, group_crops_dir, group["group_id"])
        group["group_crop_path"] = crop_path
        for idx in group["line_indices"]:
            line_items[idx]["group_crop_path"] = crop_path

    visualize_groups(img_rgb, line_items, groups, out_dir / "04_groups_visualization.png")

    # ── 6. VietOCR ───────────────────────────────────────────────────
    print("\n[6/11] Running VietOCR...")
    vietocr_predictor = create_vietocr_predictor(cfg)
    vietocr_time = recognize_all_lines(vietocr_predictor, line_items, cfg)
    print(f"  VietOCR done in {vietocr_time:.2f}s")

    visualize_vietocr_results(img_rgb, line_items, out_dir / "06_vietocr_results.png")

    # ── 7. Detect rec_conf flat ──────────────────────────────────────
    print("\n[7/11] Checking rec_conf flat...")
    rec_flat = detect_rec_conf_flat(line_items, cfg)
    print(f"  rec_conf flat: {rec_flat}")
    if rec_flat:
        for line in line_items:
            line["rec_conf_flat_run"] = True

    # ── 8. Score + Gate ──────────────────────────────────────────────
    print("\n[8/11] Scoring and gating...")
    for line in line_items:
        score_and_gate_line(
            line, cfg, wordset, base_to_variants, wordlist_enabled,
            W, H, use_rec_missing_rule=rec_flat,
        )

    status_counts = {}
    for line in line_items:
        s = line.get("filter_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Gating: {status_counts}")

    visualize_scoring(img_rgb, line_items, out_dir / "07_scoring_heatmap.png")
    visualize_gating(img_rgb, line_items, out_dir / "08_gating_distribution.png")

    # ── 9. Vintern line fallback ─────────────────────────────────────
    vintern_line_time = 0.0
    vintern_group_time = 0.0

    if cfg.get("use_vintern_line_fallback") or cfg.get("use_vintern_group_fallback"):
        print("\n[9/11] Loading Vintern + line fallback...")
        vintern_model, vintern_tokenizer = load_vintern_model(cfg)

        if cfg.get("use_vintern_line_fallback"):
            vintern_line_time = run_vintern_line_fallback(
                vintern_model, vintern_tokenizer, line_items, cfg,
                wordset, base_to_variants, wordlist_enabled,
            )
            print(f"  Vintern line done in {vintern_line_time:.2f}s")
        else:
            print("  Vintern line fallback: DISABLED")

        visualize_vintern_lines(img_rgb, line_items, out_dir / "09_vintern_line_results.png")

        # ── 10. Build group text + Vintern group fallback ────────────
        print("\n[10/11] Building group text + Vintern group fallback...")
        build_group_text_clean_review(groups, line_items, cfg)

        if cfg.get("use_vintern_group_fallback"):
            vintern_group_time = run_vintern_group_fallback(
                vintern_model, vintern_tokenizer, groups, line_items, cfg,
                wordset, base_to_variants, wordlist_enabled,
            )
            print(f"  Vintern group done in {vintern_group_time:.2f}s")
        else:
            print("  Vintern group fallback: DISABLED")

        visualize_vintern_groups(img_rgb, groups, out_dir / "10_vintern_group_results.png")
    else:
        print("\n[9/11] Vintern: DISABLED")
        print("[10/11] Building group text...")
        build_group_text_clean_review(groups, line_items, cfg)

    # ── 11. Final outputs ────────────────────────────────────────────
    print("\n[11/11] Generating final outputs...")
    final_clean, final_review = collect_final_texts(groups)

    pipeline_time = time.time() - pipeline_start
    timing = {
        "det_time_sec": round(det_time, 3),
        "vietocr_total_time_sec": round(vietocr_time, 3),
        "vintern_line_total_time_sec": round(vintern_line_time, 3),
        "vintern_group_total_time_sec": round(vintern_group_time, 3),
        "total_pipeline_sec": round(pipeline_time, 3),
    }

    # Final visualization
    export_visualization(img_rgb, line_items, groups, out_dir / "11_final_visualization.png")

    # CSV exports
    export_csv_lines(line_items, out_dir / "line_details.csv")
    export_csv_groups(groups, out_dir / "group_details.csv")

    # Build result dict
    result = {
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

    # Report
    generate_report(result, line_items, groups, out_dir / "pipeline_report.txt")

    print("\n" + "=" * 60)
    print(f"Test complete! All outputs saved to: {out_dir}")
    print(f"Total pipeline time: {pipeline_time:.2f}s")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test OCR VLM Pipeline v2 — Full Visualization")
    parser.add_argument("--image", required=True, help="Path to test frame image")
    parser.add_argument("--output_dir", default=None, help="Output directory for test results")
    parser.add_argument("--no_vintern", action="store_true", help="Skip Vintern (faster test)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    test_pipeline(args.image, args.output_dir, args.no_vintern)


if __name__ == "__main__":
    main()
