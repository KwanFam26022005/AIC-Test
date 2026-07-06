# -*- coding: utf-8 -*-
"""Phase B: LocateAnything detection/count from RAM++ tag cache."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from locate_worker import LocateAnythingWorker
from postprocess import PostprocessConfig, postprocess_detections, summarize_objects
from ram_locate_common import (
    IMAGE_EXTENSIONS,
    get_torch_device,
    iter_jsonl,
    load_existing_frame_ids,
    log_device,
    setup_logging,
    unique_preserve_order,
    write_jsonl_record,
)


def resolve_image_path(
    doc: dict[str, Any],
    frames_dir: Path | None = None,
    frames_root: Path | None = None,
) -> Path | None:
    raw = doc.get("image_path")
    if raw:
        path = Path(str(raw))
        if path.is_file():
            return path

    frame_name = str(doc.get("frame_name") or Path(str(raw or "")).stem or doc.get("frame_id", ""))
    video_id = str(doc.get("video_id") or "")
    candidates: list[Path] = []

    if frames_dir:
        candidates.extend(frames_dir / f"{frame_name}{ext}" for ext in IMAGE_EXTENSIONS)
    if frames_root and video_id:
        candidates.extend(frames_root / video_id / f"{frame_name}{ext}" for ext in IMAGE_EXTENSIONS)

    for path in candidates:
        if path.is_file():
            return path
    return None


def build_output_doc(
    tag_doc: dict[str, Any],
    image_path: Path,
    image_size: tuple[int, int],
    prompt_tags: list[str],
    raw_detections: list[dict[str, Any]],
    final_objects: list[dict[str, Any]],
    scene_labels: list[str],
    post_stats: dict[str, int],
    timing: dict[str, float],
    raw_answer: str | None,
) -> dict[str, Any]:
    object_summary, object_counts = summarize_objects(final_objects)
    raw_tags = tag_doc.get("raw_tags") or tag_doc.get("tags") or []
    all_tags = unique_preserve_order([str(t).lower() for t in raw_tags] + scene_labels)

    quality = {
        "num_raw_tags": len(raw_tags),
        "num_prompt_tags": len(prompt_tags),
        **post_stats,
    }

    doc = {
        "frame_id": tag_doc.get("frame_id"),
        "video_id": tag_doc.get("video_id"),
        "frame_idx": tag_doc.get("frame_idx"),
        "frame_name": tag_doc.get("frame_name") or image_path.stem,
        "image_path": str(image_path.resolve()),
        "image_size": [image_size[0], image_size[1]],
        "tags": all_tags,
        "object_prompt_tags": prompt_tags,
        "objects": final_objects,
        "object_summary": object_summary,
        "object_counts": object_counts,
        "quality": quality,
        "timing": timing,
    }

    if raw_answer is not None:
        doc["locate_raw_answer"] = raw_answer
    if raw_detections:
        doc["raw_detection_count"] = len(raw_detections)
    return doc


def process_tag_cache(
    tag_cache_path: Path,
    output_path: Path,
    worker: LocateAnythingWorker,
    args,
    frames_dir: Path | None = None,
    frames_root: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_frame_ids(output_path) if args.resume else set()
    write_mode = "a" if args.resume and existing else "w"

    cfg = PostprocessConfig(
        min_box_width_px=args.min_box_width,
        min_box_height_px=args.min_box_height,
        min_box_area_px=args.min_box_area,
        scene_area_threshold=args.scene_area_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        repeated_box_iou_threshold=args.repeated_box_iou_threshold,
    )

    processed = 0
    skipped = 0
    errors = 0
    total_locate_sec = 0.0
    total_start = time.time()

    with open(output_path, write_mode, encoding="utf-8") as f_out:
        records = iter_jsonl(tag_cache_path)
        if args.limit is not None:
            records = _limit_iter(records, args.limit)

        pbar = tqdm(records, desc=f"Locate {tag_cache_path.stem}", unit="frame")
        for tag_doc in pbar:
            frame_id = tag_doc.get("frame_id")
            if frame_id and frame_id in existing:
                skipped += 1
                continue

            image_path = resolve_image_path(tag_doc, frames_dir=frames_dir, frames_root=frames_root)
            if not image_path:
                errors += 1
                logging.error("Cannot resolve image path for frame_id=%s", frame_id)
                continue

            try:
                result = process_one_record(tag_doc, image_path, worker, cfg, args)
                next_processed = processed + 1
                flush_now = args.flush_every == 1 or next_processed % args.flush_every == 0
                write_jsonl_record(f_out, result, flush=flush_now)
                processed = next_processed
                total_locate_sec += result["timing"].get("locate_sec", 0.0)
            except Exception as exc:
                errors += 1
                logging.error("Failed frame_id=%s image=%s: %s", frame_id, image_path, exc)
                if args.debug:
                    raise

            elapsed = max(time.time() - total_start, 1e-6)
            avg = elapsed / max(1, processed)
            pbar.set_postfix(ok=processed, skip=skipped, err=errors, avg=f"{avg:.2f}s")

        if args.flush_every > 1:
            f_out.flush()

    elapsed = time.time() - total_start
    summary = {
        "tag_cache": str(tag_cache_path),
        "output": str(output_path),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(elapsed, 3),
        "avg_sec_per_processed_frame": round(elapsed / processed, 4) if processed else None,
        "avg_locate_sec": round(total_locate_sec / processed, 4) if processed else None,
    }
    logging.info(
        "Locate done: %d processed, %d skipped, %d errors, %.1fs",
        processed,
        skipped,
        errors,
        elapsed,
    )
    return summary


def _limit_iter(records, limit: int):
    for idx, doc in enumerate(records):
        if idx >= limit:
            break
        yield doc


def process_one_record(
    tag_doc: dict[str, Any],
    image_path: Path,
    worker: LocateAnythingWorker,
    cfg: PostprocessConfig,
    args,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    prompt_tags = tag_doc.get("object_prompt_tags") or tag_doc.get("filtered_tags") or []
    prompt_tags = [str(t).strip() for t in prompt_tags if str(t).strip()]
    if args.max_prompt_tags and args.max_prompt_tags > 0:
        prompt_tags = prompt_tags[:args.max_prompt_tags]

    raw_answer: str | None = None
    raw_detections: list[dict[str, Any]] = []
    locate_sec = 0.0

    if prompt_tags:
        start = time.time()
        raw_detections, answer = worker.detect(
            image,
            prompt_tags,
            generation_mode=args.generation_mode,
            max_new_tokens=args.max_new_tokens,
            image_max_side=args.image_max_side,
            verbose=args.verbose_generate,
        )
        locate_sec = time.time() - start
        if args.save_raw_answer:
            raw_answer = answer

    post_start = time.time()
    final_objects, scene_labels, post_stats = postprocess_detections(
        raw_detections,
        img_w,
        img_h,
        cfg,
    )
    postprocess_sec = time.time() - post_start

    timing = {
        "ram_sec": float((tag_doc.get("timing") or {}).get("ram_sec", 0.0)),
        "locate_sec": round(locate_sec, 4),
        "postprocess_sec": round(postprocess_sec, 4),
    }

    return build_output_doc(
        tag_doc=tag_doc,
        image_path=image_path,
        image_size=(img_w, img_h),
        prompt_tags=prompt_tags,
        raw_detections=raw_detections,
        final_objects=final_objects,
        scene_labels=scene_labels,
        post_stats=post_stats,
        timing=timing,
        raw_answer=raw_answer,
    )


def infer_video_id_from_tag_cache(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_ram_tags"):
        return stem[: -len("_ram_tags")]
    return stem


def write_summary(summary_path: Path, summaries: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": summaries,
        "total_processed": sum(int(s.get("processed") or 0) for s in summaries),
        "total_errors": sum(int(s.get("errors") or 0) for s in summaries),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_single(args) -> None:
    device = get_torch_device(args.device)
    log_device(device)
    worker = LocateAnythingWorker(
        model_path=args.model_path,
        device=str(device),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    summary = process_tag_cache(
        tag_cache_path=Path(args.tag_cache),
        output_path=Path(args.output),
        worker=worker,
        args=args,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        frames_root=Path(args.frames_root) if args.frames_root else None,
    )
    if args.summary_output:
        write_summary(Path(args.summary_output), [summary])


def run_batch(args) -> None:
    device = get_torch_device(args.device)
    log_device(device)
    worker = LocateAnythingWorker(
        model_path=args.model_path,
        device=str(device),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    tag_dir = Path(args.tag_cache)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    tag_files = sorted(tag_dir.glob(args.tag_cache_pattern))
    if not tag_files:
        raise FileNotFoundError(f"No tag cache files found in {tag_dir} with pattern {args.tag_cache_pattern}")

    for tag_cache_path in tag_files:
        video_id = infer_video_id_from_tag_cache(tag_cache_path)
        output_path = output_dir / f"{video_id}_objects.jsonl"
        frames_dir = Path(args.frames_root) / video_id if args.frames_root else None
        summary = process_tag_cache(
            tag_cache_path=tag_cache_path,
            output_path=output_path,
            worker=worker,
            args=args,
            frames_dir=frames_dir,
            frames_root=Path(args.frames_root) if args.frames_root else None,
        )
        summaries.append(summary)

    if args.summary_output:
        write_summary(Path(args.summary_output), summaries)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase B: LocateAnything detection/count from RAM++ tag cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag-cache", required=True, help="Tag cache JSONL file, or directory with --batch.")
    parser.add_argument("--output", required=True, help="Output JSONL file, or output directory with --batch.")
    parser.add_argument("--batch", action="store_true", help="Process all tag cache JSONLs under --tag-cache.")
    parser.add_argument("--tag-cache-pattern", default="*_ram_tags.jsonl", help="Pattern for batch tag cache files.")
    parser.add_argument("--frames-dir", default=None, help="Frame directory fallback for single mode.")
    parser.add_argument("--frames-root", default=None, help="Parent frame root fallback for batch mode.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per tag cache.")
    parser.add_argument("--resume", action="store_true", help="Skip frame IDs already present in output JSONL.")
    parser.add_argument("--summary-output", default=None, help="Optional timing summary JSON path.")

    parser.add_argument("--model-path", default="nvidia/LocateAnything-3B", help="HF model path or local model dir.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cuda:0, cpu.")
    parser.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "bf16", "fp32"], help="Model dtype.")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation.")
    parser.add_argument("--generation-mode", default="hybrid", choices=["hybrid", "fast", "slow"], help="LocateAnything generation mode.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation token cap.")
    parser.add_argument("--image-max-side", type=int, default=1280, help="Resize longest side before LocateAnything; <=0 disables.")
    parser.add_argument("--max-prompt-tags", type=int, default=20, help="Max tags per LocateAnything prompt.")

    parser.add_argument("--min-box-width", type=float, default=4.0, help="Drop boxes narrower than this.")
    parser.add_argument("--min-box-height", type=float, default=4.0, help="Drop boxes shorter than this.")
    parser.add_argument("--min-box-area", type=float, default=16.0, help="Drop boxes below this pixel area.")
    parser.add_argument("--scene-area-threshold", type=float, default=0.65, help="Move large boxes to tags instead of object counts.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.75, help="Class-agnostic NMS IoU threshold.")
    parser.add_argument("--repeated-box-iou-threshold", type=float, default=0.98, help="Same-label repeated box IoU threshold.")

    parser.add_argument("--save-raw-answer", action="store_true", help="Store raw LocateAnything answer in JSONL.")
    parser.add_argument("--verbose-generate", action="store_true", help="Enable model verbose generation if supported.")
    parser.add_argument("--flush-every", type=int, default=1, help="Flush JSONL every N records.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs.")
    parser.add_argument("--debug", action="store_true", help="Raise errors for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.flush_every = max(1, args.flush_every)
    setup_logging(quiet=args.quiet, debug=args.debug)
    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
