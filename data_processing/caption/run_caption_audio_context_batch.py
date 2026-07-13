#!/usr/bin/env python3
"""Batch runner for caption v2 demos.

This wrapper keeps server commands short:
  - discover videos from a keyframe CSV folder,
  - optionally run the audio pipeline,
  - generate per-video caption v2 configs,
  - run caption v2,
  - export visualization HTML.

Use two terminals with different ``--shard-index`` and GPU values to process a
larger dataset on two GPUs.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("caption_audio_context_batch")


@dataclass(frozen=True)
class VideoItem:
    video_id: str
    csv_path: Path
    video_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio + caption v2 for a batch of videos.")
    parser.add_argument("--dataset", default="L25", help="Dataset prefix, e.g. L25")
    parser.add_argument("--aic-root", default=os.environ.get("AIC_ROOT", "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test"))
    parser.add_argument("--keyframes-dir", default="", help="Directory containing <video_id>.csv files")
    parser.add_argument("--keyframes-root", default="", help="Directory containing frame image folders/files")
    parser.add_argument("--video-root", default="", help="Directory containing source .mp4 videos")
    parser.add_argument("--video-id", action="append", default=[], help="Limit to one or more video IDs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of discovered videos")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--stage",
        choices=("all", "audio", "caption", "visualize", "audio-caption"),
        default="all",
    )
    parser.add_argument("--audio-gpu", default="0")
    parser.add_argument("--caption-gpu", default="0")
    parser.add_argument("--audio-env-dir", default="")
    parser.add_argument("--audio-model", default="medium")
    parser.add_argument("--audio-compute-type", default="float16")
    parser.add_argument("--force-audio", action="store_true")
    parser.add_argument("--caption-env-file", default="")
    parser.add_argument("--caption-output-root", default="")
    parser.add_argument("--audio-output-root", default="")
    parser.add_argument("--caption-config-dir", default="")
    parser.add_argument("--resume-caption", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-existing-audio", action="store_true")
    parser.add_argument("--skip-existing-caption", action="store_true")
    parser.add_argument("--image-mode", choices=("copy", "link", "embed"), default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    aic_root = Path(args.aic_root)
    caption_dir = Path(__file__).resolve().parent
    keyframes_dir = Path(args.keyframes_dir) if args.keyframes_dir else aic_root / "keyframe_test" / args.dataset
    keyframes_root = Path(args.keyframes_root) if args.keyframes_root else keyframes_dir
    video_root = Path(args.video_root) if args.video_root else Path(f"/tmp2/maitanha/vgu/ttn/data/AIC2025/videos/Videos_{args.dataset}")
    audio_output_root = Path(args.audio_output_root) if args.audio_output_root else aic_root / "outputs" / "audio"
    caption_output_root = Path(args.caption_output_root) if args.caption_output_root else aic_root / "caption_v2"
    config_dir = Path(args.caption_config_dir) if args.caption_config_dir else aic_root / "data_processing" / "caption" / "configs" / f"{args.dataset.lower()}_batch"
    audio_env_dir = Path(args.audio_env_dir) if args.audio_env_dir else aic_root / ".venv_audio"

    items = discover_items(
        keyframes_dir=keyframes_dir,
        video_root=video_root,
        wanted=set(args.video_id),
    )
    if args.limit > 0:
        items = items[: args.limit]
    items = shard_items(items, args.shard_index, args.num_shards)
    if not items:
        logger.error("No videos selected for this shard.")
        return 1

    logger.info("Selected %d videos for shard %d/%d", len(items), args.shard_index, args.num_shards)
    logger.info("Keyframe CSV dir: %s", keyframes_dir)
    logger.info("Keyframe image root: %s", keyframes_root)
    logger.info("Video root: %s", video_root)

    failures: list[str] = []
    for index, item in enumerate(items, start=1):
        logger.info("=== [%d/%d] %s ===", index, len(items), item.video_id)
        try:
            if args.stage in {"all", "audio", "audio-caption"}:
                audio_features = audio_output_root / item.video_id / "features" / "audio_features.jsonl"
                if args.skip_existing_audio and audio_features.exists():
                    logger.info("Audio exists, skipping: %s", audio_features)
                else:
                    run_audio(item, aic_root, audio_env_dir, audio_output_root, args)

            if args.stage in {"all", "caption", "audio-caption"}:
                manifest = caption_output_root / item.video_id / "manifest" / "run_manifest.json"
                if args.skip_existing_caption and manifest.exists():
                    logger.info("Caption exists, skipping: %s", manifest)
                else:
                    cfg_path = write_caption_config(
                        item=item,
                        keyframes_root=keyframes_root,
                        audio_output_root=audio_output_root,
                        caption_output_root=caption_output_root,
                        config_dir=config_dir,
                    )
                    run_caption(cfg_path, args, caption_dir=caption_dir, aic_root=aic_root)

            if args.stage in {"all", "visualize"}:
                run_visualize(item, keyframes_root, caption_output_root, args, caption_dir=caption_dir)
        except subprocess.CalledProcessError as exc:
            logger.error("Video %s failed with exit code %s", item.video_id, exc.returncode)
            failures.append(item.video_id)
            if args.strict:
                break

    if failures:
        logger.error("Failed videos: %s", ", ".join(failures))
        return 1
    logger.info("Batch completed successfully.")
    return 0


def discover_items(keyframes_dir: Path, video_root: Path, wanted: set[str]) -> list[VideoItem]:
    if not keyframes_dir.exists():
        raise FileNotFoundError(f"Keyframe CSV directory not found: {keyframes_dir}")
    items: list[VideoItem] = []
    csv_paths = sorted(keyframes_dir.glob("*.csv"))
    if not csv_paths:
        csv_paths = sorted(keyframes_dir.glob("*/*.csv"))
    for csv_path in csv_paths:
        video_id = csv_path.stem
        if wanted and video_id not in wanted:
            continue
        video_path = find_video(video_root, video_id)
        if not video_path:
            logger.warning("Missing video for %s under %s", video_id, video_root)
            continue
        items.append(VideoItem(video_id=video_id, csv_path=csv_path, video_path=video_path))
    return items


def find_video(video_root: Path, video_id: str) -> Path | None:
    candidates = [
        video_root / f"{video_id}.mp4",
        video_root / "video" / f"{video_id}.mp4",
        video_root / video_id / f"{video_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(video_root.rglob(f"{video_id}.mp4")) if video_root.exists() else []
    return matches[0] if matches else None


def shard_items(items: list[VideoItem], shard_index: int, num_shards: int) -> list[VideoItem]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def run_audio(
    item: VideoItem,
    aic_root: Path,
    audio_env_dir: Path,
    audio_output_root: Path,
    args: argparse.Namespace,
) -> None:
    script = aic_root / "data_processing" / "audio_extraction" / "scripts" / "run_l22_v012_smoke.sh"
    if not script.exists():
        raise FileNotFoundError(f"Audio script not found: {script}")
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.audio_gpu),
        "AUDIO_ENV_DIR": str(audio_env_dir),
        "TEST_VIDEO": str(item.video_path),
        "OUTPUT_DIR": str(audio_output_root / item.video_id),
    })
    cmd = [
        "bash",
        str(script),
        "--video_id",
        item.video_id,
        "--model",
        args.audio_model,
        "--compute_type",
        args.audio_compute_type,
    ]
    if args.force_audio:
        cmd.append("--force")
    run_command(cmd, env=env, dry_run=args.dry_run, cwd=aic_root)


def write_caption_config(
    *,
    item: VideoItem,
    keyframes_root: Path,
    audio_output_root: Path,
    caption_output_root: Path,
    config_dir: Path,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"{item.video_id}.yaml"
    cfg = f"""pipeline:
  version: caption_audio_context_v1
  video_id: {item.video_id}

inputs:
  keyframe_map: {item.csv_path}
  keyframes_root: {keyframes_root}
  audio_features: {audio_output_root / item.video_id / "features" / "audio_features.jsonl"}

paths:
  output_root: {caption_output_root}
  prompt_dir: ""

models:
  vlm:
    provider: qwen2_5_vl
    model_name: Qwen/Qwen2.5-VL-7B-Instruct
    dtype: bfloat16
    device_map: auto
    attn_implementation: flash_attention_2
    batch_size: 2
    max_visual_tokens: 1280
    max_new_tokens: 64
    unload_after_stage: true

  text_llm:
    provider: transformers
    model_name: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    device_map: auto
    attn_implementation: flash_attention_2
    max_input_tokens: 4096
    max_new_tokens: 512
    unload_after_stage: true

frame_selection:
  similarity_method: phash
  novelty_threshold: 0.30
  min_selected_per_shot: 1
  max_selected_per_shot: 1
  force_first_last_for_long_shots: false
  long_shot_min_frames: 6

audio_context:
  padding_before_sec: 1.5
  padding_after_sec: 1.5
  max_audio_context_chars: 1200

generation:
  shot_max_gap_sec: 5.0
  shot_max_duration_sec: 20.0

checkpoint:
  resume: true
  write_every: 10

strict: true
"""
    cfg_path.write_text(cfg, encoding="utf-8")
    return cfg_path


def run_caption(cfg_path: Path, args: argparse.Namespace, *, caption_dir: Path, aic_root: Path) -> None:
    env = dict(os.environ)
    env.update({
        "AIC_ROOT": str(aic_root),
        "CUDA_VISIBLE_DEVICES": str(args.caption_gpu),
        "PYTHONIOENCODING": "utf-8",
        "CAPTION_VLM_ATTN": os.environ.get("CAPTION_VLM_ATTN", "flash_attention_2"),
        "CAPTION_LLM_ATTN": os.environ.get("CAPTION_LLM_ATTN", "flash_attention_2"),
    })
    cmd = [
        sys.executable,
        "run_caption_audio_context_pipeline.py",
        "--config",
        str(cfg_path),
    ]
    if args.caption_env_file:
        cmd.extend(["--env-file", args.caption_env_file, "--env-override"])
    if not args.resume_caption:
        cmd.append("--no-resume")
    if args.strict:
        cmd.append("--strict")
    run_command(cmd, env=env, dry_run=args.dry_run, cwd=caption_dir)


def run_visualize(
    item: VideoItem,
    keyframes_root: Path,
    caption_output_root: Path,
    args: argparse.Namespace,
    caption_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "run_caption_v2_visualize.py",
        "--video-id",
        item.video_id,
        "--caption-dir",
        str(caption_output_root),
        "--keyframes-root",
        str(keyframes_root),
        "--image-mode",
        args.image_mode,
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run_command(cmd, env=env, dry_run=args.dry_run, cwd=caption_dir)


def run_command(cmd: Iterable[str], *, env: dict[str, str], dry_run: bool, cwd: Path | None = None) -> None:
    command = list(cmd)
    if cwd:
        logger.info("Working dir: %s", cwd)
    logger.info("Command: %s", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, env=env, cwd=str(cwd) if cwd else None)


if __name__ == "__main__":
    sys.exit(main())
