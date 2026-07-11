#!/usr/bin/env python3
"""Run full-frame VLM captioning over a frame folder.

The script is intentionally provider-lazy:
- provider=vintern imports the OCR Vintern stack only.
- provider=qwen imports qwen-vl-utils/Qwen only.

This lets the same demo run in two different conda environments.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("vlm_frame_caption_demo")
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPT = (
    "Describe this video keyframe for retrieval in concise English.\n"
    "Mention only clearly visible scene, people, objects, actions, layout, and prominent readable text.\n"
    "Do not guess identities, locations, causes, or events outside the image.\n"
    "Return plain caption text only, under 320 characters."
)


@dataclass
class CaptionResult:
    caption: str
    raw_response: str
    elapsed_seconds: float
    num_tiles: int | None = None
    warning: str = ""


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 10**18


def discover_frames(frames_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_dir.glob(pattern), key=lambda p: (frame_number(p), p.name))
    return [p for p in frames if p.is_file()]


def canonical_frame_id(video_id: str, image_path: Path) -> str:
    stem = image_path.stem
    if stem.startswith(video_id + "_"):
        return stem
    return f"{video_id}_{stem}"


def read_done_frame_ids(output_jsonl: Path) -> set[str]:
    done = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_id = doc.get("frame_id")
            if frame_id:
                done.add(str(frame_id))
    return done


def clean_caption(text: str, max_chars: int) -> tuple[str, str]:
    raw = (text or "").strip()
    warning = ""
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("caption_en", "caption", "text", "description"):
                if parsed.get(key):
                    raw = str(parsed[key]).strip()
                    break
    except Exception:
        pass
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > max_chars:
        raw = raw[:max_chars].rsplit(" ", 1)[0].strip()
        warning = "truncated"
    if not raw:
        warning = "empty_caption"
    elif len(raw.split()) < 4:
        warning = "very_short_caption"
    return raw, warning


def utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        import flash_attn.flash_attn_interface  # noqa: F401
        return True
    except Exception as exc:
        LOGGER.warning("flash_attn is unavailable or ABI-incompatible: %s", exc)
        return False


def has_sdpa() -> bool:
    try:
        import torch.nn.functional as F
        return hasattr(F, "scaled_dot_product_attention")
    except Exception:
        return False


def attention_candidates(requested: str | None) -> list[str | None]:
    requested = (requested or "").strip()
    if requested in {"", "auto", "none", "None"}:
        if has_flash_attn():
            requested = "flash_attention_2"
        elif has_sdpa():
            requested = "sdpa"
        else:
            requested = "eager"
    order: list[str | None] = []
    if requested == "flash_attention_2" and not has_flash_attn():
        LOGGER.warning("Requested flash_attention_2 but it is not usable; falling back.")
    elif requested:
        order.append(requested)
    for candidate in ("sdpa", "eager", None):
        if candidate not in order:
            if candidate == "sdpa" and not has_sdpa():
                continue
            order.append(candidate)
    return order


class VinternFrameCaptioner:
    def __init__(self, args: argparse.Namespace) -> None:
        ocr_root = REPO_ROOT / "data_processing" / "OCR" / "ocr_vlm_pipeline_v2"
        sys.path.insert(0, str(ocr_root))
        from ocr_pipeline.config import make_config
        from ocr_pipeline.recognizers.vintern_recognizer import (
            _build_generation_config,
            _preprocess_for_vintern,
            load_vintern_model,
            vintern_ocr_from_pixel_values,
        )

        self.torch = __import__("torch")
        self._preprocess_for_vintern = _preprocess_for_vintern
        self._build_generation_config = _build_generation_config
        self.vintern_from_pixel_values = vintern_ocr_from_pixel_values
        self.cfg = make_config(
            vintern_model_id=args.model_id,
            vintern_device=args.device,
            vintern_quantization=None if args.quantization == "none" else args.quantization,
            vintern_attn_implementation=args.attn,
            vintern_input_size=args.input_size,
            vintern_max_new_tokens=args.max_new_tokens,
            vintern_do_sample=False,
        )
        LOGGER.info("Loading Vintern frame captioner: %s", args.model_id)
        self.model, self.tokenizer = load_vintern_model(self.cfg)
        self.generation_config = self._build_generation_config(self.tokenizer, self.cfg)
        self.prompt = args.prompt
        self.input_size = args.input_size
        self.max_tiles = args.max_tiles

    def caption(self, image_path: Path) -> CaptionResult:
        started = time.perf_counter()
        pixel_values = self._preprocess_for_vintern(
            image_path,
            input_size=self.input_size,
            max_tiles=self.max_tiles,
        )
        num_tiles = int(pixel_values.shape[0])
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=self.torch.float16)
        raw = self.vintern_from_pixel_values(
            self.model,
            self.tokenizer,
            pixel_values,
            self.prompt,
            self.generation_config,
        )
        return CaptionResult(
            caption=raw.strip(),
            raw_response=raw.strip(),
            elapsed_seconds=time.perf_counter() - started,
            num_tiles=num_tiles,
        )


class QwenFrameCaptioner:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        self.torch = torch
        self.process_vision_info = process_vision_info
        dtype = getattr(torch, args.dtype)
        attn = args.attn or os.environ.get("CAPTION_VLM_ATTN") or None
        LOGGER.info("Loading Qwen frame captioner: %s", args.model_id)
        self.model = None
        last_error = None
        for candidate in attention_candidates(attn):
            model_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": args.device_map,
            }
            if candidate:
                model_kwargs["attn_implementation"] = candidate
            LOGGER.info("Trying Qwen attention=%s", candidate or "default")
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_id,
                    **model_kwargs,
                ).eval()
                LOGGER.info("Qwen loaded with attention=%s", candidate or "default")
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Qwen load failed with attention=%s: %s", candidate or "default", exc)
        if self.model is None:
            raise RuntimeError(f"Could not load Qwen model {args.model_id}") from last_error
        self.processor = AutoProcessor.from_pretrained(args.model_id)
        self.prompt = args.prompt
        self.max_new_tokens = args.max_new_tokens

    def caption(self, image_path: Path) -> CaptionResult:
        started = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": self.prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return CaptionResult(
            caption=raw,
            raw_response=raw,
            elapsed_seconds=time.perf_counter() - started,
            num_tiles=None,
        )


class MockFrameCaptioner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.prompt = args.prompt

    def caption(self, image_path: Path) -> CaptionResult:
        started = time.perf_counter()
        raw = f"Mock caption for {image_path.name}."
        return CaptionResult(raw, raw, time.perf_counter() - started, num_tiles=0)


def create_captioner(args: argparse.Namespace):
    if args.provider == "vintern":
        return VinternFrameCaptioner(args)
    if args.provider == "qwen":
        return QwenFrameCaptioner(args)
    if args.provider == "mock":
        return MockFrameCaptioner(args)
    raise ValueError(f"Unsupported provider: {args.provider}")


def write_report(output_dir: Path, docs: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> None:
    if docs:
        avg = sum(float(doc["elapsed_seconds"]) for doc in docs) / len(docs)
        warnings = sum(1 for doc in docs if doc.get("warning"))
        empty = sum(1 for doc in docs if not doc.get("caption"))
    else:
        avg = 0.0
        warnings = 0
        empty = 0
    report = {
        "created_at": utc_now_iso(),
        "video_id": args.video_id,
        "provider": args.provider,
        "model_id": args.model_id,
        "num_written_this_run": len(docs),
        "avg_seconds_per_frame_this_run": round(avg, 4),
        "elapsed_seconds_this_run": round(elapsed, 3),
        "warnings_this_run": warnings,
        "empty_this_run": empty,
        "prompt": args.prompt,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM frame caption demo over a frame folder.")
    parser.add_argument("--provider", choices=["vintern", "qwen", "mock"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-caption-chars", type=int, default=420)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--write-every", type=int, default=20)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output-name", default="frame_caption_vlm.jsonl")
    parser.add_argument("--device", default=None, help="Vintern torch device, e.g. cuda:0.")
    parser.add_argument("--device-map", default="auto", help="Qwen device_map.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn", default=None, help="Attention implementation.")
    parser.add_argument("--quantization", default="4bit_nf4", choices=["4bit_nf4", "none"])
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-tiles", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if not args.frames_dir.exists():
        raise FileNotFoundError(args.frames_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / args.output_name
    all_frames = discover_frames(args.frames_dir, args.pattern)
    frames = [p for idx, p in enumerate(all_frames) if idx % args.num_shards == args.shard_index]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    done = read_done_frame_ids(output_jsonl) if args.resume else set()
    pending = [p for p in frames if canonical_frame_id(args.video_id, p) not in done]
    LOGGER.info("Frames discovered=%d shard_frames=%d done=%d pending=%d", len(all_frames), len(frames), len(done), len(pending))
    LOGGER.info("Output JSONL: %s", output_jsonl)
    if not pending:
        write_report(output_dir, [], args, 0.0)
        LOGGER.info("Nothing to do.")
        return 0

    captioner = create_captioner(args)
    run_started = time.perf_counter()
    written_docs: list[dict[str, Any]] = []
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for idx, image_path in enumerate(pending, start=1):
            frame_id = canonical_frame_id(args.video_id, image_path)
            LOGGER.info("[%d/%d] frame=%s image=%s", idx, len(pending), frame_id, image_path.name)
            try:
                result = captioner.caption(image_path)
                caption, warning = clean_caption(result.caption, args.max_caption_chars)
                if result.warning:
                    warning = ";".join([w for w in (warning, result.warning) if w])
                doc = {
                    "schema_version": "vlm_frame_caption_demo.v1",
                    "created_at": utc_now_iso(),
                    "video_id": args.video_id,
                    "frame_id": frame_id,
                    "frame_name": image_path.stem,
                    "frame_number": frame_number(image_path),
                    "image_path": str(image_path),
                    "provider": args.provider,
                    "model_id": args.model_id,
                    "caption": caption,
                    "raw_response": result.raw_response,
                    "elapsed_seconds": round(result.elapsed_seconds, 4),
                    "num_tiles": result.num_tiles,
                    "warning": warning,
                }
            except Exception as exc:
                LOGGER.exception("Caption failed for %s", frame_id)
                doc = {
                    "schema_version": "vlm_frame_caption_demo.v1",
                    "created_at": utc_now_iso(),
                    "video_id": args.video_id,
                    "frame_id": frame_id,
                    "frame_name": image_path.stem,
                    "frame_number": frame_number(image_path),
                    "image_path": str(image_path),
                    "provider": args.provider,
                    "model_id": args.model_id,
                    "caption": "",
                    "raw_response": "",
                    "elapsed_seconds": 0.0,
                    "num_tiles": None,
                    "warning": f"exception:{type(exc).__name__}:{exc}",
                }
            handle.write(json.dumps(doc, ensure_ascii=False))
            handle.write("\n")
            written_docs.append(doc)
            if args.write_every > 0 and idx % args.write_every == 0:
                handle.flush()
                LOGGER.info("Flushed %d docs", idx)

    elapsed = time.perf_counter() - run_started
    write_report(output_dir, written_docs, args, elapsed)
    LOGGER.info("Done. Wrote %d docs in %.1fs -> %s", len(written_docs), elapsed, output_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
