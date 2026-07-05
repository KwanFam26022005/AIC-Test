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
import os
import shutil
import sys
import time
import warnings
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


def configure_runtime_noise(quiet: bool) -> None:
    """Reduce repetitive third-party logs during long video runs."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("FLAGS_minloglevel", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")
    warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Using `TRANSFORMERS_CACHE` is deprecated.*")

    if quiet:
        logging.getLogger("ocr_pipeline").setLevel(logging.WARNING)
        logging.getLogger("run_single_frame").setLevel(logging.WARNING)


def apply_artifact_mode(cfg: dict, mode: str) -> None:
    """Control expensive per-frame artifacts for batch runs."""
    if mode == "full":
        return

    cfg["export_line_csv"] = False
    cfg["export_group_csv"] = False
    cfg["export_clean_text"] = False
    cfg["export_review_text"] = False
    cfg["export_visualization"] = False
    cfg["export_es_doc"] = False


def cleanup_frame_crops(frame_output_dir: Path, cfg: dict) -> None:
    """Remove generated crop folders after ES JSONL has been written."""
    for subdir in (
        cfg.get("line_crops_subdir", "ppocr_line_perspective_crops"),
        cfg.get("group_crops_subdir", "ppocr_stacked_group_crops"),
    ):
        target = frame_output_dir / subdir
        if target.exists() and target.is_dir():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR VLM Pipeline v2 on a folder of frames")
    parser.add_argument("--frames_dir", required=True, help="Folder containing extracted frames")
    parser.add_argument("--output_dir", default="./outputs_video", help="Batch output root")
    parser.add_argument("--video_id", default=None, help="Video id; defaults to frames folder name")
    parser.add_argument("--pattern", default="*", help="Frame glob pattern, e.g. '*.jpg'")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N frames")
    parser.add_argument("--no_vintern", action="store_true", help="Disable Vintern fallback")
    parser.add_argument("--vietocr_batch_size", type=int, default=None, help="Override VietOCR CNN batch size")
    parser.add_argument("--vietocr_batch", action="store_true", help="Enable experimental VietOCR batch decoder")
    parser.add_argument("--no_vietocr_batch", action="store_true", help="Disable VietOCR batch decoder")
    parser.add_argument("--vintern_max_candidates", type=int, default=None, help="Max line crops sent to Vintern per frame")
    parser.add_argument("--vintern_group_max_candidates", type=int, default=None, help="Max group crops sent to Vintern per frame")
    parser.add_argument(
        "--batch_artifacts",
        choices=("minimal", "full"),
        default="minimal",
        help="minimal writes ES JSONL + summaries only; full also writes per-frame CSV/TXT/PNG/JSON",
    )
    parser.add_argument("--cleanup_crops", action="store_true", help="Delete per-frame crop folders after each frame")
    parser.add_argument("--quiet", action="store_true", help="Hide per-stage logs; keep per-frame timing summaries")
    parser.add_argument("--show_vintern_progress", action="store_true", help="Show per-crop Vintern tqdm bars")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    quiet = args.quiet and not args.verbose
    configure_runtime_noise(quiet)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s" if quiet else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    configure_runtime_noise(quiet)

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
    apply_artifact_mode(cfg, args.batch_artifacts)
    cfg["vintern_progress"] = bool(args.show_vintern_progress)
    if args.no_vintern:
        cfg["use_vintern_line_fallback"] = False
        cfg["use_vintern_group_fallback"] = False
    if args.vietocr_batch_size is not None:
        cfg["vietocr_batch_size"] = args.vietocr_batch_size
    if args.vietocr_batch:
        cfg["vietocr_use_batch"] = True
    if args.no_vietocr_batch:
        cfg["vietocr_use_batch"] = False
    if args.vintern_max_candidates is not None:
        cfg["vintern_max_candidates"] = args.vintern_max_candidates
    if args.vintern_group_max_candidates is not None:
        cfg["vintern_group_max_candidates"] = args.vintern_group_max_candidates
    use_vintern = cfg.get("use_vintern_line_fallback", True) or cfg.get("use_vintern_group_fallback", True)

    logger.info("Video id: %s", video_id)
    logger.info("Frames: %d from %s", len(frames), frames_dir)
    logger.info("Output root: %s", output_root)
    logger.info("Vintern enabled: %s", use_vintern)
    logger.info(
        "Batch settings: artifacts=%s cleanup_crops=%s vietocr_use_batch=%s "
        "vietocr_batch_size=%s vintern_progress=%s vintern_max_candidates=%s "
        "vintern_group_max_candidates=%s",
        args.batch_artifacts,
        args.cleanup_crops,
        cfg.get("vietocr_use_batch"),
        cfg.get("vietocr_batch_size"),
        cfg.get("vintern_progress"),
        cfg.get("vintern_max_candidates"),
        cfg.get("vintern_group_max_candidates"),
    )

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

        if args.cleanup_crops:
            cleanup_frame_crops(frame_output_dir, frame_cfg)

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
