#!/usr/bin/env python3
"""Phase 8: propagate frame-caption experiments through shot/event captions."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.experiment_propagation import propagate_caption_experiment
from caption_pipeline.io_utils import read_jsonl

logger = logging.getLogger("caption_phase8_propagate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply frame-caption overrides, rebuild shot captions, rebuild "
            "TRAKE event steps, and refresh compact search index."
        ),
    )
    parser.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Baseline caption root or baseline caption/<video_id> dir",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Experiment root, e.g. /path/to/caption_experiments",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment folder name, e.g. vlm_frame_qwen25vl_7b_propagated",
    )
    parser.add_argument(
        "--frame-captions",
        required=True,
        help="JSONL with canonical_frame_id/frame_id and caption_text",
    )
    parser.add_argument(
        "--shot-caption-mode",
        default="template",
        choices=["template", "llm"],
        help="Shot caption propagation mode. llm currently falls back to template.",
    )
    parser.add_argument(
        "--trake-event-mode",
        default="template",
        choices=["template", "llm"],
        help="TRAKE event propagation mode. llm currently falls back to template.",
    )
    parser.add_argument(
        "--copy-reports",
        action="store_true",
        help="Copy baseline reports before writing new propagation reports",
    )
    parser.add_argument(
        "--no-visual-queries",
        action="store_true",
        help="Do not write eval/visual_retrieval_queries.jsonl",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if propagation warnings are present",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.monotonic()

    frame_caption_path = Path(args.frame_captions)
    if not frame_caption_path.exists():
        raise FileNotFoundError(f"Missing frame caption override JSONL: {frame_caption_path}")
    frame_rows = read_jsonl(frame_caption_path)

    logger.info("=== Caption Phase 8 Propagation - %s ===", args.video_id)
    logger.info("Baseline:    %s", args.baseline_dir)
    logger.info("OutputRoot:  %s", args.output_root)
    logger.info("Experiment:  %s", args.experiment_name)
    logger.info("Frame caps:  %s (%d rows)", frame_caption_path, len(frame_rows))
    logger.info("Shot mode:   %s", args.shot_caption_mode)
    logger.info("TRAKE mode:  %s", args.trake_event_mode)

    report = propagate_caption_experiment(
        video_id=args.video_id,
        baseline_dir=args.baseline_dir,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        frame_caption_overrides=frame_rows,
        shot_caption_mode=args.shot_caption_mode,
        trake_event_mode=args.trake_event_mode,
        copy_reports=args.copy_reports,
        write_visual_queries=not args.no_visual_queries,
    )

    elapsed = time.monotonic() - t0
    frame_stats = report.get("frame_update_stats") or {}
    override_qa = report.get("frame_override_qa") or {}

    logger.info("Output:      %s", report["output_video_dir"])
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Override rows:   %d", override_qa.get("num_input_rows", 0))
    logger.info("  Valid overrides: %d", override_qa.get("num_valid_overrides", 0))
    logger.info(
        "  Frames updated:  %d",
        frame_stats.get("num_frames_updated_with_override", 0),
    )
    logger.info("  Frames kept:     %d", frame_stats.get("num_frames_kept_baseline", 0))
    logger.info("  Frames:          %d", report["num_frame_index"])
    logger.info("  Shots:           %d", report["num_shot_index"])
    logger.info("  Event steps:     %d", report["num_event_step_index"])
    logger.info("  Index docs:      %d", report["num_compact_docs"])
    logger.info("  Warnings:        %d", len(report.get("warnings") or []))

    if args.strict and report.get("warnings"):
        logger.warning("Propagation warnings: %s", "; ".join(report["warnings"]))
        return 1
    logger.info("All caption propagation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
