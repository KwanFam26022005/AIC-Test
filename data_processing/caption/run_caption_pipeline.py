#!/usr/bin/env python3
"""Caption evidence pipeline - CLI entry point.

Usage:
    python run_caption_pipeline.py \\
        --video-id L22_V012 \\
        --ocr-jsonl outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \\
        --object-jsonl outputs/object_detection/L22_V012_objects.jsonl \\
        --audio-features outputs/audio/L22_V012/features/audio_features.jsonl \\
        --output-dir outputs/caption

Outputs:
    outputs/caption/<video_id>/
      evidence/
        frame_evidence.jsonl
        shot_evidence.jsonl
      indexes/
        compact_search_index.jsonl
      reports/
        evidence_alignment_report.json
        evidence_alignment_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.compact_index import build_compact_index
from caption_pipeline.config import (
    AudioAlignmentConfig,
    EvidenceBuilderConfig,
    PipelineConfig,
    ShotGroupingConfig,
)
from caption_pipeline.evidence_alignment import align_frames
from caption_pipeline.evidence_builder import build_frame_evidence
from caption_pipeline.io_utils import write_json, write_jsonl, write_text
from caption_pipeline.load_inputs import load_audio_features, load_objects, load_ocr
from caption_pipeline.shot_grouper import group_shots
from caption_pipeline.validation import build_report, render_report_markdown

logger = logging.getLogger("caption_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caption evidence alignment & indexing pipeline (Phase 0/1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required inputs
    p.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    p.add_argument("--ocr-jsonl", required=True, help="Path to OCR ES docs JSONL")
    p.add_argument("--object-jsonl", required=True, help="Path to object detection JSONL")
    p.add_argument("--audio-features", required=True, help="Path to audio features JSONL")

    # Output
    p.add_argument("--output-dir", default="outputs/caption", help="Base output directory")

    # Optional: keyframe map (for future timestamp fallback)
    p.add_argument("--keyframe-map", default="", help="Keyframe map directory (unused in Phase 0)")

    # Audio alignment
    p.add_argument("--frame-window-before", type=float, default=5.0,
                   help="Audio window seconds before frame timestamp")
    p.add_argument("--frame-window-after", type=float, default=5.0,
                   help="Audio window seconds after frame timestamp")

    # Shot grouping
    p.add_argument("--shot-max-gap-sec", type=float, default=5.0,
                   help="Max gap between frames to stay in same shot")
    p.add_argument("--shot-max-duration-sec", type=float, default=20.0,
                   help="Max shot duration in seconds")

    # Flags
    p.add_argument("--include-ocr-only-frames", action="store_true",
                   help="Include frames with OCR but no object detection")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from CLI arguments."""
    cfg = PipelineConfig(
        video_id=args.video_id,
        ocr_jsonl=args.ocr_jsonl,
        object_jsonl=args.object_jsonl,
        audio_features_jsonl=args.audio_features,
        keyframe_map=args.keyframe_map,
        output_dir=args.output_dir,
        include_ocr_only_frames=args.include_ocr_only_frames,
        audio_alignment=AudioAlignmentConfig(
            frame_window_sec_before=args.frame_window_before,
            frame_window_sec_after=args.frame_window_after,
        ),
        evidence_builder=EvidenceBuilderConfig(),
        shot_grouping=ShotGroupingConfig(
            max_gap_sec=args.shot_max_gap_sec,
            max_shot_duration_sec=args.shot_max_duration_sec,
        ),
    )
    cfg.resolve_paths()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)
    t0 = time.monotonic()

    logger.info("=== Caption Pipeline - %s ===", cfg.video_id)
    logger.info("OCR:    %s", cfg.ocr_jsonl)
    logger.info("Object: %s", cfg.object_jsonl)
    logger.info("Audio:  %s", cfg.audio_features_jsonl)
    logger.info("Output: %s", cfg.output_dir)

    # ---- Phase 0: Load inputs ----
    logger.info("--- Phase 0: Loading inputs ---")
    ocr_frames = load_ocr(cfg.ocr_jsonl)
    object_frames, object_stats = load_objects(cfg.object_jsonl, return_stats=True)
    audio_segments, audio_stats = load_audio_features(cfg.audio_features_jsonl, return_stats=True)

    # ---- Phase 0: Align evidence ----
    logger.info("--- Phase 0: Aligning evidence ---")
    aligned, alignment_stats = align_frames(object_frames, ocr_frames, audio_segments, cfg)
    alignment_stats.update(object_stats)
    alignment_stats.update(audio_stats)

    # ---- Phase 0: Build frame evidence ----
    logger.info("--- Phase 0: Building frame evidence ---")
    frame_evidence = build_frame_evidence(aligned, cfg)

    # ---- Phase 1: Shot grouping ----
    logger.info("--- Phase 1: Grouping shots ---")
    shot_evidence = group_shots(frame_evidence, cfg)

    # ---- Phase 1: Compact search index ----
    logger.info("--- Phase 1: Building compact search index ---")
    compact_docs = build_compact_index(frame_evidence, shot_evidence)

    # ---- Write outputs ----
    logger.info("--- Writing outputs ---")

    # Evidence
    fe_path = Path(cfg.evidence_dir) / "frame_evidence.jsonl"
    write_jsonl(fe_path, frame_evidence)
    logger.info("Wrote %d frame evidence -> %s", len(frame_evidence), fe_path)

    se_path = Path(cfg.evidence_dir) / "shot_evidence.jsonl"
    write_jsonl(se_path, shot_evidence)
    logger.info("Wrote %d shot evidence -> %s", len(shot_evidence), se_path)

    # Search index
    idx_path = Path(cfg.indexes_dir) / "compact_search_index.jsonl"
    write_jsonl(idx_path, compact_docs)
    logger.info("Wrote %d search docs -> %s", len(compact_docs), idx_path)

    # Validation report
    report = build_report(cfg.video_id, frame_evidence, shot_evidence, alignment_stats, audio_segments)
    rpt_json_path = Path(cfg.reports_dir) / "evidence_alignment_report.json"
    write_json(rpt_json_path, report)
    logger.info("Wrote report JSON -> %s", rpt_json_path)

    rpt_md_path = Path(cfg.reports_dir) / "evidence_alignment_report.md"
    md_text = render_report_markdown(report)
    write_text(rpt_md_path, md_text)
    logger.info("Wrote report MD -> %s", rpt_md_path)

    elapsed = time.monotonic() - t0

    # ---- Summary ----
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Frames:  %d", len(frame_evidence))
    logger.info("  Shots:   %d", len(shot_evidence))
    logger.info("  Index:   %d docs", len(compact_docs))
    logger.info("  Warnings: %d", len(report.get("warnings", [])))

    # Print quick acceptance
    problems = []
    if report["num_timestamp_mismatch"] > 0:
        problems.append(f"{report['num_timestamp_mismatch']} timestamp mismatches")
    if report["num_frames_missing_ocr"] > 0:
        problems.append(f"{report['num_frames_missing_ocr']} frames missing OCR")
    if report["num_frame_evidence"] != report["num_joined_frames"]:
        problems.append("frame evidence count != joined frames")

    if problems:
        logger.warning("Acceptance issues: %s", "; ".join(problems))
        return 1
    else:
        logger.info("All acceptance checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
