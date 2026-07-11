#!/usr/bin/env python3
"""Render an HTML visualization for caption pipeline outputs."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.visualization import write_caption_visualization

logger = logging.getLogger("caption_visualize")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frame and shot review HTML pages for one caption run.",
    )
    parser.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    parser.add_argument(
        "--caption-dir",
        required=True,
        help="Base caption output directory containing <video_id>/",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional index HTML path. Defaults to <caption-dir>/<video_id>/reports/visualization/index.html",
    )
    parser.add_argument(
        "--keyframes-root",
        default="",
        help="Optional keyframe root so representative frame thumbnails can be rendered.",
    )
    parser.add_argument(
        "--max-frames-per-shot",
        type=int,
        default=3,
        help="Maximum representative frame thumbnails per shot.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("embed", "copy", "link"),
        default="copy",
        help=(
            "How images are packaged: copy writes a reusable assets folder, embed "
            "creates portable but heavier HTML, and link references original files."
        ),
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
    try:
        output_path = write_caption_visualization(
            caption_dir=args.caption_dir,
            video_id=args.video_id,
            output_html=Path(args.output_html) if args.output_html else None,
            keyframes_root=Path(args.keyframes_root) if args.keyframes_root else None,
            max_frames_per_shot=args.max_frames_per_shot,
            image_mode=args.image_mode,
        )
    except Exception:
        logger.exception("Failed to render caption visualization")
        return 2

    logger.info("Wrote caption visualization -> %s", output_path)
    logger.info("=== Done in %.1fs ===", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
