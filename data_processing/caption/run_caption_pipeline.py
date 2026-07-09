#!/usr/bin/env python3
"""Caption evidence pipeline - CLI entry point.

Phase 0/1: Evidence alignment, shot grouping, compact search index.
Phase 2:   Frame-level caption generation (template or LLM mode).
Phase 3:   Shot-level caption generation (template or LLM mode).
Phase 4:   TRAKE event-step generation (template or LLM mode).

Usage:
    python run_caption_pipeline.py \\
        --video-id L22_V012 \\
        --ocr-jsonl outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \\
        --object-jsonl outputs/object_detection/L22_V012_objects.jsonl \\
        --audio-features outputs/audio/L22_V012/features/audio_features.jsonl \\
        --output-dir outputs/caption \\
        --enable-frame-captions \\
        --caption-mode template \\
        --rebuild-compact-with-captions

Outputs:
    outputs/caption/<video_id>/
      evidence/
        frame_evidence.jsonl
        shot_evidence.jsonl
      captions/
        frame_index.jsonl          (Phase 2)
        shot_index.jsonl           (Phase 3)
        event_step_index.jsonl     (Phase 4)
      indexes/
        compact_search_index.jsonl
      reports/
        evidence_alignment_report.json
        evidence_alignment_report.md
        frame_caption_report.json   (Phase 2)
        frame_caption_report.md     (Phase 2)
        shot_caption_report.json    (Phase 3)
        shot_caption_report.md      (Phase 3)
        event_step_report.json      (Phase 4)
        event_step_report.md        (Phase 4)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.caption_index import build_frame_index, rebuild_compact_with_captions
from caption_pipeline.caption_report import build_caption_report, render_caption_report_markdown
from caption_pipeline.compact_index import build_compact_index
from caption_pipeline.config import (
    AudioAlignmentConfig,
    EvidenceBuilderConfig,
    FrameCaptionConfig,
    PipelineConfig,
    ShotCaptionConfig,
    ShotGroupingConfig,
    TrakeEventConfig,
)
from caption_pipeline.evidence_alignment import align_frames
from caption_pipeline.evidence_builder import build_frame_evidence
from caption_pipeline.event_step_index import (
    build_event_step_index,
    rebuild_compact_with_trake_text,
)
from caption_pipeline.event_step_report import (
    build_event_step_report,
    render_event_step_report_markdown,
)
from caption_pipeline.frame_caption_fuser import fuse_frame_captions
from caption_pipeline.io_utils import write_json, write_jsonl, write_text
from caption_pipeline.load_inputs import load_audio_features, load_objects, load_ocr
from caption_pipeline.shot_caption_report import (
    build_shot_caption_report,
    render_shot_caption_report_markdown,
)
from caption_pipeline.shot_captioner import fuse_shot_captions
from caption_pipeline.shot_grouper import group_shots
from caption_pipeline.shot_index import (
    build_shot_index,
    rebuild_compact_with_frame_and_shot_captions,
)
from caption_pipeline.trake_event_builder import build_trake_event_steps
from caption_pipeline.validation import build_report, render_report_markdown

logger = logging.getLogger("caption_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caption evidence, caption, shot, and TRAKE event pipeline.",
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

    # Phase 2: Frame captions
    p.add_argument("--enable-frame-captions", action="store_true",
                   help="Enable Phase 2 frame caption generation")
    p.add_argument("--caption-mode", default="template",
                   choices=["template", "llm"],
                   help="Caption generation mode (default: template)")
    p.add_argument("--rebuild-compact-with-captions", action="store_true",
                   help="Rebuild compact search index with caption_text")

    # Phase 3: Shot captions
    p.add_argument("--enable-shot-captions", action="store_true",
                   help="Enable Phase 3 shot caption generation")
    p.add_argument("--shot-caption-mode", default="template",
                   choices=["template", "llm"],
                   help="Shot caption generation mode (default: template)")

    # Phase 4: TRAKE events
    p.add_argument("--enable-trake-events", action="store_true",
                   help="Enable Phase 4 TRAKE event-step generation")
    p.add_argument("--trake-event-mode", default="template",
                   choices=["template", "llm"],
                   help="TRAKE event generation mode (default: template)")

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
        enable_frame_captions=args.enable_frame_captions,
        enable_shot_captions=args.enable_shot_captions,
        enable_trake_events=args.enable_trake_events,
        audio_alignment=AudioAlignmentConfig(
            frame_window_sec_before=args.frame_window_before,
            frame_window_sec_after=args.frame_window_after,
        ),
        evidence_builder=EvidenceBuilderConfig(),
        shot_grouping=ShotGroupingConfig(
            max_gap_sec=args.shot_max_gap_sec,
            max_shot_duration_sec=args.shot_max_duration_sec,
        ),
        frame_caption=FrameCaptionConfig(
            caption_mode=args.caption_mode,
            rebuild_compact_with_captions=args.rebuild_compact_with_captions,
        ),
        shot_caption=ShotCaptionConfig(
            caption_mode=args.shot_caption_mode,
        ),
        trake_event=TrakeEventConfig(
            event_mode=args.trake_event_mode,
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
    if cfg.enable_shot_captions and not cfg.enable_frame_captions:
        print(
            "ERROR: Phase 3 requires --enable-frame-captions in this integrated pipeline.",
            file=sys.stderr,
        )
        return 2
    if cfg.enable_trake_events and not cfg.enable_shot_captions:
        print(
            "ERROR: Phase 4 requires --enable-shot-captions in this integrated pipeline.",
            file=sys.stderr,
        )
        return 2

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

    # ---- Write Phase 0/1 outputs ----
    logger.info("--- Writing Phase 0/1 outputs ---")

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

    # ---- Phase 2: Frame captions ----
    caption_records = []
    frame_index = []
    shot_caption_records = []
    shot_index = []
    event_records = []
    event_step_index = []
    if cfg.enable_frame_captions:
        logger.info("--- Phase 2: Generating frame captions ---")
        caption_records = fuse_frame_captions(frame_evidence, cfg)

        logger.info("--- Phase 2: Building frame index ---")
        frame_index = build_frame_index(frame_evidence, caption_records)

        # Write frame index
        fi_path = Path(cfg.captions_dir) / "frame_index.jsonl"
        write_jsonl(fi_path, frame_index)
        logger.info("Wrote %d frame index -> %s", len(frame_index), fi_path)

        # Rebuild compact index with captions if requested
        if cfg.frame_caption.rebuild_compact_with_captions:
            logger.info("--- Phase 2: Rebuilding compact index with captions ---")
            compact_docs = rebuild_compact_with_captions(compact_docs, caption_records)
            write_jsonl(idx_path, compact_docs)
            logger.info("Rebuilt %d search docs -> %s", len(compact_docs), idx_path)

        # Caption report
        caption_report = build_caption_report(cfg.video_id, frame_index, caption_records)
        crpt_json_path = Path(cfg.reports_dir) / "frame_caption_report.json"
        write_json(crpt_json_path, caption_report)
        logger.info("Wrote caption report JSON -> %s", crpt_json_path)

        crpt_md_path = Path(cfg.reports_dir) / "frame_caption_report.md"
        crpt_md = render_caption_report_markdown(caption_report)
        write_text(crpt_md_path, crpt_md)
        logger.info("Wrote caption report MD -> %s", crpt_md_path)

    # ---- Phase 3: Shot captions ----
    if cfg.enable_shot_captions:
        logger.info("--- Phase 3: Generating shot captions ---")
        shot_caption_records = fuse_shot_captions(shot_evidence, frame_index, cfg)

        logger.info("--- Phase 3: Building shot index ---")
        shot_index = build_shot_index(shot_evidence, frame_index, shot_caption_records)

        si_path = Path(cfg.captions_dir) / "shot_index.jsonl"
        write_jsonl(si_path, shot_index)
        logger.info("Wrote %d shot index -> %s", len(shot_index), si_path)

        if cfg.frame_caption.rebuild_compact_with_captions:
            logger.info("--- Phase 3: Rebuilding compact index with frame and shot captions ---")
            compact_docs = rebuild_compact_with_frame_and_shot_captions(
                compact_docs,
                caption_records,
                shot_caption_records,
            )
            write_jsonl(idx_path, compact_docs)
            logger.info("Rebuilt %d search docs -> %s", len(compact_docs), idx_path)

        shot_report = build_shot_caption_report(
            cfg.video_id, shot_index, shot_caption_records,
        )
        srpt_json_path = Path(cfg.reports_dir) / "shot_caption_report.json"
        write_json(srpt_json_path, shot_report)
        logger.info("Wrote shot caption report JSON -> %s", srpt_json_path)

        srpt_md_path = Path(cfg.reports_dir) / "shot_caption_report.md"
        srpt_md = render_shot_caption_report_markdown(shot_report)
        write_text(srpt_md_path, srpt_md)
        logger.info("Wrote shot caption report MD -> %s", srpt_md_path)

    # ---- Phase 4: TRAKE event steps ----
    if cfg.enable_trake_events:
        logger.info("--- Phase 4: Generating TRAKE event steps ---")
        event_records = build_trake_event_steps(shot_index, shot_evidence, cfg)

        logger.info("--- Phase 4: Building event-step index ---")
        event_step_index = build_event_step_index(
            shot_index, shot_evidence, event_records,
        )

        es_path = Path(cfg.captions_dir) / "event_step_index.jsonl"
        write_jsonl(es_path, event_step_index)
        logger.info("Wrote %d event-step index -> %s", len(event_step_index), es_path)

        if cfg.frame_caption.rebuild_compact_with_captions:
            logger.info("--- Phase 4: Rebuilding compact index with TRAKE text ---")
            compact_docs = rebuild_compact_with_trake_text(compact_docs, event_records)
            write_jsonl(idx_path, compact_docs)
            logger.info("Rebuilt %d search docs -> %s", len(compact_docs), idx_path)

        event_report = build_event_step_report(
            cfg.video_id, event_step_index, event_records,
        )
        erpt_json_path = Path(cfg.reports_dir) / "event_step_report.json"
        write_json(erpt_json_path, event_report)
        logger.info("Wrote event-step report JSON -> %s", erpt_json_path)

        erpt_md_path = Path(cfg.reports_dir) / "event_step_report.md"
        erpt_md = render_event_step_report_markdown(event_report)
        write_text(erpt_md_path, erpt_md)
        logger.info("Wrote event-step report MD -> %s", erpt_md_path)

    elapsed = time.monotonic() - t0

    # ---- Summary ----
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Frames:  %d", len(frame_evidence))
    logger.info("  Shots:   %d", len(shot_evidence))
    logger.info("  Index:   %d docs", len(compact_docs))
    logger.info("  Warnings: %d", len(report.get("warnings", [])))
    if cfg.enable_frame_captions:
        num_cap = sum(1 for fi in frame_index if fi.get("caption_text"))
        logger.info("  Captions: %d/%d", num_cap, len(frame_index))
    if cfg.enable_shot_captions:
        num_shot_cap = sum(1 for si in shot_index if si.get("caption_text"))
        logger.info("  Shot captions: %d/%d", num_shot_cap, len(shot_index))
    if cfg.enable_trake_events:
        num_trake = sum(1 for es in event_step_index if es.get("trake_text"))
        logger.info("  Event steps: %d/%d", num_trake, len(event_step_index))

    # Print quick acceptance
    problems = []
    if report["num_timestamp_mismatch"] > 0:
        problems.append(f"{report['num_timestamp_mismatch']} timestamp mismatches")
    if report["num_frames_missing_ocr"] > 0:
        problems.append(f"{report['num_frames_missing_ocr']} frames missing OCR")
    if report["num_frame_evidence"] != report["num_joined_frames"]:
        problems.append("frame evidence count != joined frames")

    if cfg.enable_frame_captions:
        num_empty = sum(1 for fi in frame_index if not fi.get("caption_text"))
        if num_empty > 0:
            problems.append(f"{num_empty} empty captions")
    if cfg.enable_shot_captions:
        num_empty_shots = sum(1 for si in shot_index if not si.get("caption_text"))
        num_empty_temporal = sum(1 for si in shot_index if not si.get("temporal_caption"))
        if num_empty_shots > 0:
            problems.append(f"{num_empty_shots} empty shot captions")
        if num_empty_temporal > 0:
            problems.append(f"{num_empty_temporal} empty temporal captions")
    if cfg.enable_trake_events:
        num_empty_events = sum(1 for es in event_step_index if not es.get("event_caption"))
        num_empty_trake = sum(1 for es in event_step_index if not es.get("trake_text"))
        duplicate_event_doc_ids = _count_duplicates(event_step_index, "document_id")
        duplicate_event_ids = _count_duplicates(event_step_index, "event_id")
        duplicate_event_shot_ids = _count_duplicates(event_step_index, "shot_id")
        if len(event_step_index) != len(shot_index):
            problems.append(
                f"event steps ({len(event_step_index)}) != shots ({len(shot_index)})"
            )
        if num_empty_events > 0:
            problems.append(f"{num_empty_events} empty event captions")
        if num_empty_trake > 0:
            problems.append(f"{num_empty_trake} empty TRAKE texts")
        if duplicate_event_doc_ids > 0:
            problems.append(f"{duplicate_event_doc_ids} duplicate event document_ids")
        if duplicate_event_ids > 0:
            problems.append(f"{duplicate_event_ids} duplicate event_ids")
        if duplicate_event_shot_ids > 0:
            problems.append(f"{duplicate_event_shot_ids} duplicate event shot_ids")

    if problems:
        logger.warning("Acceptance issues: %s", "; ".join(problems))
        return 1
    else:
        logger.info("All acceptance checks passed.")
        return 0


def _count_duplicates(rows: list[dict], key: str) -> int:
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        value = row.get(key, "")
        if value in seen:
            duplicates += 1
        seen.add(value)
    return duplicates


if __name__ == "__main__":
    sys.exit(main())
