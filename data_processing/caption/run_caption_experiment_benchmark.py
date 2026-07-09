#!/usr/bin/env python3
"""Phase 6 caption experiment benchmark CLI.

The script compares Phase 5 retrieval outputs from a baseline run against
optional experiment runs. It does not regenerate captions and does not call
LLM/VLM models; it only reads existing eval/export artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.experiment_benchmark import (
    build_benchmark_report,
    build_manual_review_samples,
    load_experiment,
    render_benchmark_report_markdown,
)
from caption_pipeline.io_utils import write_json, write_jsonl, write_text

logger = logging.getLogger("caption_phase6_benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare caption retrieval experiments using Phase 5 outputs.",
    )
    parser.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Baseline caption dir. Accepts caption root or caption/<video_id>.",
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline",
        help="Label for the baseline run",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment in label=path form. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output dir. Default: <baseline video dir>/benchmarks/<benchmark-name>",
    )
    parser.add_argument(
        "--benchmark-name",
        default="phase6_retrieval_benchmark",
        help="Benchmark folder/report name",
    )
    parser.add_argument(
        "--top-n-samples",
        type=int,
        default=3,
        help="Top results per query to include in manual review samples",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if benchmark warnings are present",
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

    logger.info("=== Caption Phase 6 Benchmark - %s ===", args.video_id)
    baseline = load_experiment(args.baseline_label, args.baseline_dir, args.video_id)
    experiments = [
        load_experiment(label, path, args.video_id)
        for label, path in _parse_experiments(args.experiment)
    ]

    output_dir = _resolve_output_dir(
        args.output_dir,
        args.baseline_dir,
        args.video_id,
        args.benchmark_name,
    )
    logger.info("Baseline:    %s -> %s", args.baseline_label, baseline["video_dir"])
    for exp in experiments:
        logger.info("Experiment:  %s -> %s", exp["label"], exp["video_dir"])
    logger.info("Output:      %s", output_dir)

    report = build_benchmark_report(
        video_id=args.video_id,
        baseline=baseline,
        experiments=experiments,
        top_n_samples=args.top_n_samples,
    )
    samples = build_manual_review_samples(
        baseline=baseline,
        experiments=experiments,
        top_n=args.top_n_samples,
    )

    report_json_path = output_dir / f"{args.benchmark_name}.json"
    report_md_path = output_dir / f"{args.benchmark_name}.md"
    samples_path = output_dir / "manual_review_samples.jsonl"
    write_json(report_json_path, report)
    write_text(report_md_path, render_benchmark_report_markdown(report))
    write_jsonl(samples_path, samples)

    elapsed = time.monotonic() - t0
    logger.info("Wrote benchmark JSON -> %s", report_json_path)
    logger.info("Wrote benchmark MD -> %s", report_md_path)
    logger.info("Wrote manual review samples -> %s", samples_path)
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Candidates: %d", report["num_candidates"])
    logger.info("  Samples:    %d queries", len(samples))
    logger.info("  Warnings:   %d", len(report.get("warnings") or []))

    if args.strict and report.get("warnings"):
        logger.warning("Benchmark warnings: %s", "; ".join(report["warnings"]))
        return 1
    logger.info("All caption experiment benchmark checks passed.")
    return 0


def _parse_experiments(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --experiment '{item}'. Expected label=path."
            )
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(
                f"Invalid --experiment '{item}'. Label and path are required."
            )
        parsed.append((label, path))
    return parsed


def _resolve_output_dir(
    output_dir: str,
    baseline_dir: str,
    video_id: str,
    benchmark_name: str,
) -> Path:
    if output_dir:
        return Path(output_dir)
    root = Path(baseline_dir)
    video_dir = root if (root / "eval").exists() else root / video_id
    return video_dir / "benchmarks" / benchmark_name


if __name__ == "__main__":
    sys.exit(main())
