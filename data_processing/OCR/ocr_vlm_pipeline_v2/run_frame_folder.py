"""
Batch runner for OCR VLM Pipeline v2 on an extracted frame folder.

Use this for one video worth of keyframes:
  PP-OCRv6 DET -> crops/groups -> VietOCR -> optional Vintern -> ES JSONL

The detector, VietOCR predictor, wordlist, and optional Vintern model are loaded
once, then reused for every frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean

# Add this directory to sys.path so the script works when called directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_pipeline.config import make_config
from ocr_pipeline.detector import create_detector
from ocr_pipeline.outputs import build_es_document
from ocr_pipeline.recognizers.vietocr_recognizer import create_vietocr_predictor
from ocr_pipeline.scoring.wordlist import load_wordlist
from run_single_frame import run_ocr_pipeline

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_frame_key(path: Path) -> tuple[int, str]:
    """Sort numeric frame names naturally, falling back to lexical order."""
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**18, path.name


def collect_frames(frames_dir: Path, pattern: str, limit: int | None = None) -> list[Path]:
    frames = [
        p for p in frames_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    frames.sort(key=natural_frame_key)
    if limit is not None:
        frames = frames[:limit]
    return frames


def build_context(cfg: dict, use_vintern: bool) -> tuple[dict, dict]:
    """Initialize reusable OCR/VLM components once for a frame batch."""
    context: dict = {}
    init_timing: dict[str, float] = {}

    start = time.time()
    context["wordlist"] = load_wordlist(
        cfg.get("wordlist_paths", []),
        cfg.get("wordlist_min_entries_to_enable", 1000),
    )
    init_timing["wordlist_init_time_sec"] = round(time.time() - start, 3)

    start = time.time()
    context["detector"] = create_detector(cfg)
    init_timing["detector_init_time_sec"] = round(time.time() - start, 3)

    start = time.time()
    context["vietocr_predictor"] = create_vietocr_predictor(cfg)
    init_timing["vietocr_init_time_sec"] = round(time.time() - start, 3)

    if use_vintern:
        from ocr_pipeline.recognizers.vintern_recognizer import load_vintern_model

        start = time.time()
        context["vintern"] = load_vintern_model(cfg)
        init_timing["vintern_init_time_sec"] = round(time.time() - start, 3)
    else:
        init_timing["vintern_init_time_sec"] = 0.0

    return context, init_timing


def append_jsonl(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False))
        f.write("\n")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timing_stats(rows: list[dict]) -> dict:
    timing_keys = sorted({
        key for row in rows for key in row.keys()
        if key.endswith("_sec") and isinstance(row.get(key), (int, float))
    })
    stats = {}
    for key in timing_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        stats[key] = {
            "sum": round(sum(values), 3),
            "mean": round(mean(values), 3),
            "max": round(max(values), 3),
        }
    return stats


def format_eta(done: int, total: int, elapsed: float) -> str:
    if done <= 0:
        return "?"
    remaining = max(total - done, 0)
    eta_sec = remaining * (elapsed / done)
    return f"{eta_sec / 60:.1f}m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR VLM Pipeline v2 on a folder of frames")
    parser.add_argument("--frames_dir", required=True, help="Folder containing extracted frames")
    parser.add_argument("--output_dir", default="./outputs_video", help="Batch output root")
    parser.add_argument("--video_id", default=None, help="Video id; defaults to frames folder name")
    parser.add_argument("--pattern", default="*", help="Frame glob pattern, e.g. '*.jpg'")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N frames")
    parser.add_argument("--no_vintern", action="store_true", help="Disable Vintern fallback")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    frames_dir = Path(args.frames_dir)
    video_id = args.video_id or frames_dir.name
    output_root = Path(args.output_dir) / video_id
    per_frame_root = output_root / "frames"
    es_jsonl_path = output_root / f"{video_id}_ocr_es_docs.jsonl"
    summary_csv_path = output_root / f"{video_id}_ocr_frame_summary.csv"
    summary_json_path = output_root / f"{video_id}_ocr_timing_summary.json"

    frames = collect_frames(frames_dir, args.pattern, args.limit)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir} with pattern={args.pattern!r}")

    output_root.mkdir(parents=True, exist_ok=True)
    es_jsonl_path.write_text("", encoding="utf-8")

    cfg = make_config(output_dir=str(output_root))
    if args.no_vintern:
        cfg["use_vintern_line_fallback"] = False
        cfg["use_vintern_group_fallback"] = False
    use_vintern = cfg.get("use_vintern_line_fallback", True) or cfg.get("use_vintern_group_fallback", True)

    logger.info("Video id: %s", video_id)
    logger.info("Frames: %d from %s", len(frames), frames_dir)
    logger.info("Output root: %s", output_root)
    logger.info("Vintern enabled: %s", use_vintern)

    batch_start = time.time()
    context, init_timing = build_context(cfg, use_vintern)
    logger.info("Initialization timing: %s", init_timing)

    rows: list[dict] = []
    for idx, frame_path in enumerate(frames, start=1):
        frame_start = time.time()
        frame_id = frame_path.stem
        frame_output_dir = per_frame_root / frame_id

        logger.info("[%d/%d] Processing frame=%s", idx, len(frames), frame_id)
        frame_cfg = dict(cfg)
        result = run_ocr_pipeline(
            str(frame_path),
            frame_cfg,
            output_dir=str(frame_output_dir),
            context=context,
        )

        doc = build_es_document(
            result["line_items"],
            result["groups"],
            result["final_clean_text"],
            result["final_review_text"],
            frame_cfg,
            str(frame_path),
            result["timing"],
        )
        doc["video_id"] = video_id
        doc["document_id"] = f"{video_id}:{result['frame_id']}"
        doc["batch_output_paths"] = result.get("output_paths", {})
        append_jsonl(es_jsonl_path, doc)

        frame_elapsed = time.time() - frame_start
        timing = result.get("timing", {})
        row = {
            "video_id": video_id,
            "frame_id": result["frame_id"],
            "frame_number": doc.get("frame_number"),
            "image_path": str(frame_path),
            "lines_detected": len(result["line_items"]),
            "groups_formed": len(result["groups"]),
            "clean_chars": len(result["final_clean_text"]),
            "review_chars": len(result["final_review_text"]),
            "gating_stats": json.dumps(result["gating_stats"], ensure_ascii=False),
            "rec_conf_flat": result["rec_conf_flat"],
            "frame_wall_time_sec": round(frame_elapsed, 3),
            **timing,
        }
        rows.append(row)

        elapsed = time.time() - batch_start
        logger.info(
            "[%d/%d] frame=%s wall=%.2fs det=%.2fs vietocr=%.2fs vintern=%.2fs clean=%d review=%d ETA=%s",
            idx,
            len(frames),
            frame_id,
            frame_elapsed,
            timing.get("det_time_sec", 0.0),
            timing.get("vietocr_total_time_sec", 0.0),
            timing.get("vintern_line_total_time_sec", 0.0)
            + timing.get("vintern_group_total_time_sec", 0.0),
            len(result["final_clean_text"]),
            len(result["final_review_text"]),
            format_eta(idx, len(frames), elapsed),
        )

    total_elapsed = time.time() - batch_start
    write_summary_csv(summary_csv_path, rows)
    summary = {
        "video_id": video_id,
        "frames_dir": str(frames_dir),
        "num_frames": len(frames),
        "output_root": str(output_root),
        "es_jsonl_path": str(es_jsonl_path),
        "summary_csv_path": str(summary_csv_path),
        "init_timing": init_timing,
        "total_wall_time_sec": round(total_elapsed, 3),
        "avg_wall_time_sec_per_frame": round(total_elapsed / len(frames), 3),
        "timing_stats": timing_stats(rows),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("BATCH OCR RESULTS")
    print("=" * 72)
    print(f"Video: {video_id}")
    print(f"Frames: {len(frames)}")
    print(f"Total wall time: {total_elapsed:.2f}s")
    print(f"Avg/frame: {total_elapsed / len(frames):.2f}s")
    print(f"ES JSONL: {es_jsonl_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Timing JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
