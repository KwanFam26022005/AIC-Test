#!/usr/bin/env python3
"""Materialize a caption experiment directory from baseline outputs.

Use this before Phase 5/6 when you have alternative frame, shot, or TRAKE
captions from a VLM/FuseCap/LLM experiment.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.experiment_materializer import (
    materialize_caption_experiment,
)
from caption_pipeline.io_utils import read_jsonl

logger = logging.getLogger("caption_experiment_materialize")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create caption experiment outputs from baseline + overrides.",
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
        help="Experiment folder name, e.g. vlm_fuser",
    )
    parser.add_argument(
        "--frame-captions",
        default="",
        help="Optional JSONL with canonical_frame_id/frame_id and caption_text",
    )
    parser.add_argument(
        "--shot-captions",
        default="",
        help="Optional JSONL with shot_id and caption_text/temporal_caption",
    )
    parser.add_argument(
        "--event-steps",
        default="",
        help="Optional JSONL with event_id or shot_id and trake_text/event fields",
    )
    parser.add_argument(
        "--copy-reports",
        action="store_true",
        help="Copy baseline reports into experiment reports dir before writing materialize report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if materialization warnings are present",
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

    frame_rows = _read_optional_jsonl(args.frame_captions)
    shot_rows = _read_optional_jsonl(args.shot_captions)
    event_rows = _read_optional_jsonl(args.event_steps)

    logger.info("=== Caption Experiment Materialize - %s ===", args.video_id)
    logger.info("Baseline:   %s", args.baseline_dir)
    logger.info("OutputRoot: %s", args.output_root)
    logger.info("Experiment: %s", args.experiment_name)
    logger.info("Overrides:  frame=%d shot=%d event=%d",
                len(frame_rows), len(shot_rows), len(event_rows))

    report = materialize_caption_experiment(
        video_id=args.video_id,
        baseline_dir=args.baseline_dir,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        frame_caption_overrides=frame_rows,
        shot_caption_overrides=shot_rows,
        event_step_overrides=event_rows,
        copy_reports=args.copy_reports,
    )

    elapsed = time.monotonic() - t0
    logger.info("Output:    %s", report["output_video_dir"])
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Frames updated:      %d", report["num_frames_updated"])
    logger.info("  Shots updated:       %d", report["num_shots_updated"])
    logger.info("  Event steps updated: %d", report["num_event_steps_updated"])
    logger.info("  Warnings:            %d", len(report.get("warnings") or []))

    if args.strict and report.get("warnings"):
        logger.warning("Materialize warnings: %s", "; ".join(report["warnings"]))
        return 1
    logger.info("All caption experiment materialize checks passed.")
    return 0


def _read_optional_jsonl(path: str) -> list[dict]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Missing override JSONL: {target}")
    return read_jsonl(target)


if __name__ == "__main__":
    sys.exit(main())
