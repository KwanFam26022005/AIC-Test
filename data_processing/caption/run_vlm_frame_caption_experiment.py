#!/usr/bin/env python3
"""Run a VLM frame-caption override experiment."""

from __future__ import annotations

import argparse
import logging
import sys

from caption_pipeline.vlm_frame_experiment import (
    load_env_file,
    load_vlm_experiment_config,
    run_vlm_frame_experiment,
)

logger = logging.getLogger("caption_vlm_frame_experiment")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLM frame-caption overrides for caption experiments.",
    )
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Optional .env file loaded before expanding YAML env vars. Can be repeated.",
    )
    parser.add_argument(
        "--env-override",
        action="store_true",
        help="Allow later --env-file values to override existing environment variables",
    )
    parser.add_argument("--video-id", default="", help="Override experiment.video_id")
    parser.add_argument("--experiment-name", default="", help="Override experiment.name")
    parser.add_argument("--provider", default="", help="Override model.provider")
    parser.add_argument("--model-name", default="", help="Override model.model_name")
    parser.add_argument("--shard-index", type=int, default=None, help="Override batch.shard_index")
    parser.add_argument("--num-shards", type=int, default=None, help="Override batch.num_shards")
    parser.add_argument("--frame-selection", default="", help="representative, sampled, or all")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed in this run")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output rows")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for env_file in args.env_file:
        load_env_file(env_file, override=args.env_override)
    cfg = load_vlm_experiment_config(args.config)
    _apply_overrides(cfg, args)
    report = run_vlm_frame_experiment(cfg)
    logger.info("=== VLM frame experiment done ===")
    logger.info("  Output rows: %d", report["num_output_rows"])
    logger.info("  Processed:   %d", report["num_processed_this_run"])
    logger.info("  Failures:    %d", report["num_failures"])
    return 1 if report["num_failures"] else 0


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    cfg.setdefault("experiment", {})
    cfg.setdefault("model", {})
    cfg.setdefault("batch", {})
    if args.video_id:
        cfg["experiment"]["video_id"] = args.video_id
    if args.experiment_name:
        cfg["experiment"]["name"] = args.experiment_name
    if args.provider:
        cfg["model"]["provider"] = args.provider
    if args.model_name:
        cfg["model"]["model_name"] = args.model_name
    if args.shard_index is not None:
        cfg["batch"]["shard_index"] = args.shard_index
    if args.num_shards is not None:
        cfg["batch"]["num_shards"] = args.num_shards
    if args.frame_selection:
        cfg["batch"]["frame_selection"] = args.frame_selection
    if args.max_frames is not None:
        cfg["batch"]["max_frames"] = args.max_frames
    if args.no_resume:
        cfg["batch"]["resume"] = False


if __name__ == "__main__":
    sys.exit(main())




