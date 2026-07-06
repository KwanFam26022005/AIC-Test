# -*- coding: utf-8 -*-
"""Phase A: build RAM++ tag cache JSONL for keyframes."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ram_locate_common import (
    discover_frames,
    discover_video_dirs,
    extract_frame_idx,
    get_torch_device,
    load_existing_frame_ids,
    log_device,
    make_frame_id,
    setup_logging,
    write_jsonl_record,
)
from tag_filter import filter_tags_for_locate


RAM_CHECKPOINT_FILENAME = "ram_plus_swin_large_14m.pth"
RAM_CHECKPOINT_URL = (
    "https://huggingface.co/xinyu1205/recognize-anything-plus-model"
    "/resolve/main/ram_plus_swin_large_14m.pth"
)


def find_ram_checkpoint(cli_path: str | None = None) -> str:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))

    env_path = os.environ.get("RAM_CHECKPOINT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve().parent
    candidates.extend([
        Path("pretrained") / RAM_CHECKPOINT_FILENAME,
        here / "pretrained" / RAM_CHECKPOINT_FILENAME,
        here.parent / "pretrained" / RAM_CHECKPOINT_FILENAME,
    ])

    for path in candidates:
        if path.is_file():
            return str(path)

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"RAM++ checkpoint not found: {RAM_CHECKPOINT_FILENAME}\n"
        f"Searched:\n{searched}\n"
        f"Download: {RAM_CHECKPOINT_URL}"
    )


def load_ram_model(device, checkpoint_path: str | None, image_size: int):
    from ram import get_transform
    from ram.models import ram_plus

    checkpoint = find_ram_checkpoint(checkpoint_path)
    logging.info("Loading RAM++ checkpoint: %s", checkpoint)
    model = ram_plus(pretrained=checkpoint, image_size=image_size, vit="swin_l")
    model.eval().to(device)
    transform = get_transform(image_size=image_size)
    logging.info("RAM++ ready.")
    return model, transform


def _split_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in str(raw).split("|") if tag.strip()]


def _extract_english_outputs(result, expected: int) -> list[str] | None:
    english = result[0] if isinstance(result, (tuple, list)) and result else result
    if isinstance(english, str):
        return [english] if expected == 1 else None
    if isinstance(english, (list, tuple)):
        outputs = [str(x) for x in english]
        return outputs if len(outputs) == expected else None
    return None


def infer_ram_tags_batch(images: list[Image.Image], model, transform, device) -> list[list[str]]:
    import torch
    from ram import inference_ram as inference

    if not images:
        return []

    tensors = torch.stack([transform(img) for img in images], dim=0).to(device)
    with torch.inference_mode():
        try:
            result = inference(tensors, model)
            outputs = _extract_english_outputs(result, len(images))
            if outputs is not None:
                return [_split_tags(raw) for raw in outputs]
        except Exception as exc:
            if len(images) == 1:
                raise
            logging.warning("RAM++ batch inference failed; falling back to per-image: %s", exc)

    del tensors
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()

    tags: list[list[str]] = []
    for image in images:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = inference(tensor, model)
        outputs = _extract_english_outputs(result, 1)
        tags.append(_split_tags(outputs[0] if outputs else ""))
    return tags


def process_video(
    frames_dir: Path,
    output_path: Path,
    video_id: str,
    model,
    transform,
    device,
    args,
) -> dict[str, int | float | str]:
    frames = discover_frames(frames_dir, pattern=args.pattern, limit=args.limit)
    existing = load_existing_frame_ids(output_path) if args.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.resume and existing else "w"

    processed = 0
    skipped = 0
    errors = 0
    start = time.time()

    pending: list[Path] = []
    with open(output_path, write_mode, encoding="utf-8") as f_out:
        pbar = tqdm(frames, desc=f"RAM++ {video_id}", unit="frame")
        for frame_path in pbar:
            frame_id = make_frame_id(video_id, frame_path)
            if frame_id in existing:
                skipped += 1
                continue

            pending.append(frame_path)
            if len(pending) < args.ram_batch_size:
                continue

            ok, err = flush_batch(pending, f_out, video_id, model, transform, device, args)
            processed += ok
            errors += err
            pending = []
            elapsed = max(time.time() - start, 1e-6)
            pbar.set_postfix(ok=processed, skip=skipped, err=errors, fps=f"{processed / elapsed:.2f}")

        if pending:
            ok, err = flush_batch(pending, f_out, video_id, model, transform, device, args)
            processed += ok
            errors += err

    elapsed = time.time() - start
    logging.info(
        "[%s] RAM tag cache done: %d processed, %d skipped, %d errors, %.1fs",
        video_id,
        processed,
        skipped,
        errors,
        elapsed,
    )
    return {
        "video_id": video_id,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 3),
        "output": str(output_path),
    }


def flush_batch(
    frame_paths: list[Path],
    f_out,
    video_id: str,
    model,
    transform,
    device,
    args,
) -> tuple[int, int]:
    images: list[Image.Image] = []
    loaded_paths: list[Path] = []
    errors = 0

    for frame_path in frame_paths:
        try:
            images.append(Image.open(frame_path).convert("RGB"))
            loaded_paths.append(frame_path)
        except Exception as exc:
            logging.error("Failed to load image %s: %s", frame_path, exc)
            errors += 1

    if not images:
        return 0, errors

    start = time.time()
    try:
        batch_tags = infer_ram_tags_batch(images, model, transform, device)
    except Exception as exc:
        logging.error("RAM++ inference failed for batch: %s", exc)
        if args.debug:
            raise
        return 0, errors + len(images)
    elapsed = time.time() - start
    per_image_sec = elapsed / max(1, len(images))

    processed = 0
    for frame_path, image, raw_tags in zip(loaded_paths, images, batch_tags):
        frame_id = make_frame_id(video_id, frame_path)
        prompt_tags = filter_tags_for_locate(raw_tags, max_tags=args.max_prompt_tags)
        doc = {
            "frame_id": frame_id,
            "video_id": video_id,
            "frame_idx": extract_frame_idx(frame_path),
            "frame_name": frame_path.stem,
            "image_path": str(frame_path.resolve()),
            "image_size": list(image.size),
            "raw_tags": raw_tags,
            "tags": [t.lower().strip() for t in raw_tags if str(t).strip()],
            "object_prompt_tags": prompt_tags,
            "quality": {
                "num_raw_tags": len(raw_tags),
                "num_prompt_tags": len(prompt_tags),
            },
            "timing": {
                "ram_sec": round(per_image_sec, 4),
            },
        }
        write_jsonl_record(f_out, doc, flush=args.flush_every == 1)
        processed += 1

    if args.flush_every > 1:
        f_out.flush()
    return processed, errors


def run_single(args) -> None:
    device = get_torch_device(args.device)
    log_device(device)
    model, transform = load_ram_model(device, args.ram_checkpoint, args.ram_image_size)

    input_dir = Path(args.input)
    video_id = args.video_id or input_dir.name
    output_path = Path(args.output)
    process_video(input_dir, output_path, video_id, model, transform, device, args)


def run_batch(args) -> None:
    device = get_torch_device(args.device)
    log_device(device)
    model, transform = load_ram_model(device, args.ram_checkpoint, args.ram_image_size)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_id, frames_dir in discover_video_dirs(args.input, pattern=args.pattern):
        output_path = output_dir / f"{video_id}_ram_tags.jsonl"
        process_video(frames_dir, output_path, video_id, model, transform, device, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase A: RAM++ tag cache builder for RAM++ + LocateAnything.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Frame directory, or parent directory with --batch.")
    parser.add_argument("--output", required=True, help="Output JSONL path, or output directory with --batch.")
    parser.add_argument("--video-id", default=None, help="Video ID for single-directory mode.")
    parser.add_argument("--batch", action="store_true", help="Process child video directories under --input.")
    parser.add_argument("--pattern", default="*", help="Glob pattern for frames.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames per video.")
    parser.add_argument("--resume", action="store_true", help="Skip frame IDs already present in output JSONL.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cuda:0, cpu.")
    parser.add_argument("--ram-checkpoint", default=None, help="Path to ram_plus_swin_large_14m.pth.")
    parser.add_argument("--ram-image-size", type=int, default=384, help="RAM++ input image size.")
    parser.add_argument("--ram-batch-size", type=int, default=16, help="RAM++ batch size.")
    parser.add_argument("--max-prompt-tags", type=int, default=20, help="Max tags to send to LocateAnything.")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush JSONL every N records.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs.")
    parser.add_argument("--debug", action="store_true", help="Raise errors for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ram_batch_size = max(1, args.ram_batch_size)
    args.flush_every = max(1, args.flush_every)
    setup_logging(quiet=args.quiet, debug=args.debug)
    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
