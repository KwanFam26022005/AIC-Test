#!/usr/bin/env python3
"""Render a static HTML visualization for caption pipeline v2 outputs."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline_v2.visualization import write_caption_v2_visualization

logger = logging.getLogger("caption_v2_visualize")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one static HTML review page for caption_v2 outputs.",
    )
    parser.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    parser.add_argument(
        "--caption-dir",
        required=True,
        help="Base caption v2 output directory containing <video_id>/",
    )
    parser.add_argument(
        "--keyframes-root",
        default="",
        help="Keyframe root used to resolve image thumbnails.",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help="Output HTML path. Defaults to <caption-dir>/<video_id>/reports/visualization_v2/index.html",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "link", "embed"),
        default="copy",
        help="copy packages thumbnails in an assets folder, link references source files, embed makes a large standalone HTML.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
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
        output_path = write_caption_v2_visualization(
            caption_dir=args.caption_dir,
            video_id=args.video_id,
            output_html=Path(args.output_html) if args.output_html else None,
            keyframes_root=Path(args.keyframes_root) if args.keyframes_root else None,
            image_mode=args.image_mode,
        )
    except Exception:
        logger.exception("Failed to render caption v2 visualization")
        return 2

    logger.info("Wrote caption v2 visualization -> %s", output_path)
    logger.info("=== Done in %.1fs ===", time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
