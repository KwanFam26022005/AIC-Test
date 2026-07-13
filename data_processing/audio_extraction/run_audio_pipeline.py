from __future__ import annotations

import argparse
import logging
from pathlib import Path

from audio_pipeline.config import apply_overrides, load_config
from audio_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract timestamped audio features from videos.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to a single .mp4 video.")
    input_group.add_argument("--videos_dir", help="Directory containing videos.")
    parser.add_argument("--pattern", default="*.mp4", help="Video glob when using --videos_dir.")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect videos_dir.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of videos.")
    parser.add_argument("--video_id", default=None, help="Override video_id for single --video input.")
    parser.add_argument("--output_dir", default="outputs/audio", help="Audio pipeline output root.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--force", action="store_true", help="Recreate generated ASR/audio outputs for this run.")
    parser.add_argument("--skip_asr", action="store_true", help="Run only manifest, audio extraction, and VAD.")
    parser.add_argument("--model", default=None, help="Override ASR model, e.g. large-v3, turbo, or local CT2 path.")
    parser.add_argument("--device", default=None, help="Override ASR device, e.g. cuda or cpu.")
    parser.add_argument("--compute_type", default=None, help="Override ASR compute type, e.g. float16.")
    parser.add_argument("--beam_size", type=int, default=None, help="Override ASR beam size.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def collect_videos(args: argparse.Namespace) -> list[Path]:
    if args.video:
        return [Path(args.video)]

    root = Path(args.videos_dir)
    iterator = root.rglob(args.pattern) if args.recursive else root.glob(args.pattern)
    videos = sorted(path for path in iterator if path.is_file())
    if args.limit:
        videos = videos[: args.limit]
    return videos


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    apply_overrides(
        cfg,
        {
            "asr.model": args.model,
            "asr.device": args.device,
            "asr.compute_type": args.compute_type,
            "asr.beam_size": args.beam_size,
        },
    )

    videos = collect_videos(args)
    summary = run_pipeline(
        videos=videos,
        output_dir=args.output_dir,
        cfg=cfg,
        video_id_override=args.video_id,
        force=args.force,
        skip_asr=args.skip_asr,
    )

    print("\nAudio pipeline complete")
    print(f"Videos:          {summary['num_videos']}")
    print(f"Audio success:   {summary['num_audio_success']}")
    print(f"ASR jobs:        {summary['num_asr_jobs']}")
    print(f"ASR rows:        {summary['num_asr_rows']}")
    print(f"Audio features:  {summary['num_audio_features']}")
    print(f"Output root:     {summary['output_root']}")
    print(f"Audio features:  {summary['audio_features']}")
    print(f"ASR failed:      {summary['asr_failed']}")


if __name__ == "__main__":
    main()

