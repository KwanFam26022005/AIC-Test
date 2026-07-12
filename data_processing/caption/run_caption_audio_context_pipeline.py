#!/usr/bin/env python3
"""Run caption pipeline v2: visual frame captions + audio-context shot ReCap."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from caption_pipeline.runtime import load_env_file
from caption_pipeline_v2.audio_context_adapter import align_audio_to_shots
from caption_pipeline_v2.config import load_pipeline_config
from caption_pipeline_v2.frame_propagator import propagate_captions
from caption_pipeline_v2.frame_selector import select_all_shots
from caption_pipeline_v2.input_contracts import (
    load_audio_features,
    load_keyframes,
    resolve_images,
)
from caption_pipeline_v2.io_utils import read_jsonl, utc_now_iso, write_json, write_jsonl, write_text
from caption_pipeline_v2.report import build_markdown_report, build_run_manifest
from caption_pipeline_v2.shot_builder import build_shots
from caption_pipeline_v2.shot_recap import generate_shot_captions
from caption_pipeline_v2.timeline import sort_frames, validate_shot_membership, validate_timeline
from caption_pipeline_v2.validators import validate_outputs
from caption_pipeline_v2.vlm_frame_captioner import VLMFrameCaptioner

logger = logging.getLogger("caption_audio_context_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run caption audio-context pipeline v2.")
    parser.add_argument("--config", required=True, help="Pipeline v2 YAML config")
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--env-override", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true", help="Reuse existing VLM runtime rows")
    parser.add_argument("--skip-shot-recap", action="store_true", help="Reuse existing shot captions")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    start = time.time()

    for env_file in args.env_file:
        load_env_file(env_file, override=args.env_override)

    cfg = load_pipeline_config(args.config)
    if args.no_resume:
        cfg.checkpoint.resume = False
    if args.strict:
        cfg.strict = True
    if not cfg.run_id:
        cfg.run_id = f"{cfg.video_id}_{int(time.time())}"
    cfg.ensure_dirs()

    logger.info("=== Caption audio-context pipeline v2: %s ===", cfg.video_id)
    logger.info("Output: %s", Path(cfg.output_dir) / cfg.video_id)

    # Phase 0
    logger.info("--- Phase 0: Preflight and canonical timeline ---")
    frames = load_keyframes(cfg.keyframe_map, cfg.keyframes_root, cfg.video_id)
    frames, image_warnings = resolve_images(frames, cfg.keyframes_root)
    frames = sort_frames(frames)
    timeline_warnings = validate_timeline(frames)
    audio_segments, audio_stats = load_audio_features(cfg.audio_features_jsonl, cfg.video_id)

    # Phase 1
    logger.info("--- Phase 1: Shot building and frame selection ---")
    shots = build_shots(
        frames,
        cfg.video_id,
        max_gap_sec=cfg.shot_max_gap_sec,
        max_duration_sec=cfg.shot_max_duration_sec,
    )
    membership_warnings = validate_shot_membership(frames, shots)
    selections = select_all_shots(shots, cfg.frame_selection, cfg.run_id, cfg.video_id)
    write_jsonl(Path(cfg.selection_dir) / "frame_selection.jsonl", selections)

    # Phase 2
    logger.info("--- Phase 2: VLM captions for selected frames ---")
    frame_by_id = {frame["canonical_frame_id"]: frame for frame in frames}
    selected_ids = [
        row["canonical_frame_id"]
        for row in selections
        if row.get("selected_for_vlm")
    ]
    selected_frames = [frame_by_id[fid] for fid in selected_ids if fid in frame_by_id]
    vlm_result_path = Path(cfg.runtime_dir) / "frame_vlm_results.jsonl"
    vlm_results = _load_vlm_results(vlm_result_path) if cfg.checkpoint.resume else {}
    missing_frames = [
        frame
        for frame in selected_frames
        if frame["canonical_frame_id"] not in vlm_results
    ]

    if missing_frames:
        if args.skip_vlm:
            raise RuntimeError(
                f"--skip-vlm requested but {len(missing_frames)} selected frames are missing"
            )
        captioner = VLMFrameCaptioner(cfg.vlm)
        captioner.load_model()
        try:
            new_rows = captioner.caption_batch(missing_frames, batch_id_prefix=f"{cfg.run_id}_vlm")
        finally:
            if cfg.vlm.unload_after_stage:
                captioner.unload_model()
        for row in new_rows:
            vlm_results[row["canonical_frame_id"]] = row
        write_jsonl(vlm_result_path, vlm_results.values())
    else:
        logger.info("Reusing %d existing VLM rows", len(vlm_results))

    # Phase 3
    logger.info("--- Phase 3: Frame caption propagation ---")
    frame_captions = propagate_captions(
        frames=frames,
        selections=selections,
        vlm_results=vlm_results,
        run_id=cfg.run_id,
        video_id=cfg.video_id,
        vlm_config=asdict(cfg.vlm),
        prompt_version=cfg.vlm.prompt_version,
        prompt_hash=_prompt_hash_for_vlm(cfg),
    )
    created_at = utc_now_iso()
    for row in frame_captions:
        row["created_at"] = created_at
    write_jsonl(Path(cfg.captions_dir) / "frame_captions.jsonl", frame_captions)

    # Phase 4
    logger.info("--- Phase 4: ASR alignment by shot ---")
    audio_contexts = align_audio_to_shots(shots, audio_segments, cfg.audio_context, cfg.video_id)
    write_jsonl(Path(cfg.evidence_dir) / "shot_audio_context.jsonl", audio_contexts)

    # Phase 5
    logger.info("--- Phase 5: Shot ReCap with audio context ---")
    shot_caption_path = Path(cfg.captions_dir) / "shot_captions.jsonl"
    memory_path = Path(cfg.state_dir) / "recap_memory.jsonl"
    if args.skip_shot_recap:
        shot_captions = read_jsonl(shot_caption_path)
        memory_rows = read_jsonl(memory_path)
        if not shot_captions:
            raise RuntimeError("--skip-shot-recap requested but no shot_captions.jsonl exists")
    else:
        shot_captions, memory_rows = generate_shot_captions(
            shots=shots,
            frame_captions=frame_captions,
            audio_contexts=audio_contexts,
            cfg=cfg.shot_recap,
            run_id=cfg.run_id,
            video_id=cfg.video_id,
        )
        write_jsonl(shot_caption_path, shot_captions)
        write_jsonl(memory_path, memory_rows)

    # Phase 7
    logger.info("--- Phase 7: Reports and acceptance checks ---")
    stats, issues = validate_outputs(
        frames=frames,
        shots=shots,
        selections=selections,
        frame_captions=frame_captions,
        audio_contexts=audio_contexts,
        shot_captions=shot_captions,
        max_fallback_rate=0.15,
    )
    stats.update(audio_stats)
    stats["num_image_warnings"] = len(image_warnings)
    stats["num_timeline_warnings"] = len(timeline_warnings)
    stats["num_membership_warnings"] = len(membership_warnings)
    stats["elapsed_sec"] = round(time.time() - start, 1)
    issues.extend(image_warnings)
    issues.extend(timeline_warnings)
    issues.extend(membership_warnings)

    manifest = build_run_manifest(
        video_id=cfg.video_id,
        run_id=cfg.run_id,
        config_path=str(Path(args.config).resolve()),
        stats=stats,
        issues=issues,
        model_info={
            "vlm_model_id": cfg.vlm.model_id,
            "shot_recap_model_id": cfg.shot_recap.model_id,
        },
    )
    write_json(Path(cfg.manifest_dir) / "run_manifest.json", manifest)
    write_json(Path(cfg.reports_dir) / "caption_v2_report.json", manifest)
    write_text(Path(cfg.reports_dir) / "caption_v2_report.md", build_markdown_report(manifest))

    logger.info("=== Done in %.1fs ===", time.time() - start)
    logger.info("  Frames:       %d", stats["num_frames"])
    logger.info("  Shots:        %d", stats["num_shots"])
    logger.info("  VLM selected: %d", stats["num_selected_for_vlm"])
    logger.info("  Issues:       %d", len(issues))

    if issues:
        for issue in issues[:20]:
            logger.warning("Acceptance issue: %s", issue)
        if cfg.strict:
            return 1
    else:
        logger.info("All caption v2 acceptance checks passed.")
    return 0


def _load_vlm_results(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    return {
        str(row.get("canonical_frame_id")): row
        for row in rows
        if row.get("canonical_frame_id") and str(row.get("text", "")).strip()
    }


def _prompt_hash_for_vlm(cfg) -> str:
    captioner = VLMFrameCaptioner(cfg.vlm)
    return captioner.prompt_hash


if __name__ == "__main__":
    sys.exit(main())
