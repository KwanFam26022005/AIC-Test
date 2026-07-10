"""VLM frame-caption experiment runner.

This module intentionally writes only frame-caption override JSONL files.
The baseline caption outputs stay untouched; use
``run_caption_experiment_materialize.py`` to create a comparable experiment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, utc_now_iso, write_json, write_jsonl
from .text_utils import normalize_whitespace, truncate

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = (
    "Describe this video frame for video retrieval. "
    "Mention visible people, objects, scene, actions, and on-screen text if clear. "
    "Be factual, concise, and avoid guessing."
)



def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE lines from a .env file into os.environ."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Missing env file: {target}")
    loaded: dict[str, str] = {}
    with target.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                raise ValueError(f"Invalid .env line {target}:{line_no}: {raw_line.rstrip()}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                raise ValueError(f"Invalid empty .env key at {target}:{line_no}")
            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = os.environ.get(key, value)
    return loaded


def load_vlm_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and expand environment variables in string values."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read VLM experiment configs.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return _expand_env(cfg)


def run_vlm_frame_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one VLM frame-caption experiment and write override JSONL."""
    experiment = cfg.get("experiment") or {}
    paths = cfg.get("paths") or {}
    batch = cfg.get("batch") or {}
    model_cfg = cfg.get("model") or {}
    generation = cfg.get("generation") or {}

    video_id = experiment.get("video_id") or cfg.get("video_id")
    if not video_id:
        raise ValueError("experiment.video_id is required")
    experiment_name = experiment.get("name", "vlm_frame_caption")
    baseline_dir = Path(paths["baseline_dir"])
    keyframes_root = Path(paths["keyframes_root"])
    raw_output_root = Path(paths["raw_output_root"])

    baseline_video_dir = _resolve_video_dir(baseline_dir, video_id)
    frame_index = read_jsonl(baseline_video_dir / "captions" / "frame_index.jsonl")
    shot_index = read_jsonl(baseline_video_dir / "captions" / "shot_index.jsonl")
    selected = select_frame_docs(frame_index, shot_index, batch)

    shard_index = int(batch.get("shard_index", 0))
    num_shards = max(int(batch.get("num_shards", 1)), 1)
    selected = [
        doc for idx, doc in enumerate(selected)
        if idx % num_shards == shard_index
    ]

    output_dir = raw_output_root / experiment_name / video_id
    output_name = batch.get("output_name") or f"frame_caption_overrides.shard_{shard_index:03d}.jsonl"
    output_path = output_dir / output_name
    report_path = output_dir / f"frame_caption_overrides.shard_{shard_index:03d}.report.json"
    existing = read_jsonl(output_path) if batch.get("resume", True) else []
    done_ids = {
        row.get("canonical_frame_id") or row.get("frame_id")
        for row in existing
    }

    model = create_vlm_captioner(model_cfg, generation)
    rows = list(existing)
    failures: list[dict[str, str]] = []
    write_every = max(int(batch.get("write_every", 20)), 1)
    max_frames = int(batch.get("max_frames", 0) or 0)
    processed = 0

    logger.info(
        "VLM frame experiment: %s/%s selected=%d existing=%d shard=%d/%d",
        experiment_name, video_id, len(selected), len(existing), shard_index, num_shards,
    )
    for doc in selected:
        frame_id = doc.get("canonical_frame_id") or doc.get("frame_id")
        if not frame_id or frame_id in done_ids:
            continue
        if max_frames and processed >= max_frames:
            break
        image_path = resolve_image_path(doc, keyframes_root, video_id)
        if not image_path.exists():
            failures.append({
                "frame_id": frame_id,
                "error": f"missing image: {image_path}",
            })
            continue
        try:
            caption_text = model.caption_image(image_path)
            row = build_override_row(
                video_id=video_id,
                frame_doc=doc,
                image_path=image_path,
                caption_text=caption_text,
                model_cfg=model_cfg,
                generation=generation,
            )
            rows.append(row)
            done_ids.add(frame_id)
            processed += 1
        except Exception as exc:  # noqa: BLE001 - keep long batch alive and report frame-level failures.
            logger.exception("Frame %s failed", frame_id)
            failures.append({"frame_id": frame_id, "error": str(exc)})
        if processed and processed % write_every == 0:
            write_jsonl(output_path, rows)
            logger.info("Checkpoint: wrote %d rows -> %s", len(rows), output_path)

    write_jsonl(output_path, rows)
    report = {
        "schema_version": "caption_vlm_frame_experiment_report_v1",
        "created_at": utc_now_iso(),
        "video_id": video_id,
        "experiment_name": experiment_name,
        "provider": model_cfg.get("provider", ""),
        "model_name": model_cfg.get("model_name", ""),
        "frame_selection": batch.get("frame_selection", "representative"),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_selected_for_shard": len(selected),
        "num_existing_before": len(existing),
        "num_processed_this_run": processed,
        "num_output_rows": len(rows),
        "num_failures": len(failures),
        "failures": failures[:100],
        "output_path": str(output_path),
    }
    write_json(report_path, report)
    logger.info("Wrote VLM overrides -> %s", output_path)
    logger.info("Wrote VLM report -> %s", report_path)
    return report


def select_frame_docs(
    frame_index: list[dict],
    shot_index: list[dict],
    batch_cfg: dict[str, Any],
) -> list[dict]:
    """Select frame docs for VLM captioning."""
    strategy = batch_cfg.get("frame_selection", "representative")
    frame_map = {
        row.get("canonical_frame_id") or row.get("frame_id"): row
        for row in frame_index
    }
    if strategy == "all":
        selected = list(frame_index)
    elif strategy == "sampled":
        stride = max(int(batch_cfg.get("sample_stride", 5)), 1)
        selected = list(frame_index)[::stride]
    elif strategy == "representative":
        ids: list[str] = []
        seen: set[str] = set()
        for shot in shot_index:
            for frame_id in shot.get("representative_frame_ids") or []:
                if frame_id and frame_id not in seen:
                    seen.add(frame_id)
                    ids.append(frame_id)
        selected = [frame_map[frame_id] for frame_id in ids if frame_id in frame_map]
    else:
        raise ValueError(
            f"Unknown frame_selection '{strategy}'. Use representative, sampled, or all."
        )

    max_total = int(batch_cfg.get("max_total_frames", 0) or 0)
    if max_total:
        selected = selected[:max_total]
    return selected


def resolve_image_path(frame_doc: dict, keyframes_root: Path, video_id: str) -> Path:
    """Resolve a frame image path from frame_index metadata."""
    rel = frame_doc.get("image_relpath", "") or ""
    candidates = []
    if rel:
        candidates.append(keyframes_root / rel)
        candidates.append(keyframes_root / video_id / Path(rel).name)
    frame_id = frame_doc.get("canonical_frame_id") or frame_doc.get("frame_id") or ""
    if frame_id:
        candidates.append(keyframes_root / video_id / f"{frame_id}.jpg")
        candidates.append(keyframes_root / video_id / f"{Path(frame_id).name}.jpg")
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else keyframes_root / video_id / f"{frame_id}.jpg"


def build_override_row(
    video_id: str,
    frame_doc: dict,
    image_path: Path,
    caption_text: str,
    model_cfg: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    caption_text = normalize_whitespace(caption_text)
    caption_text = truncate(caption_text, int(generation.get("max_caption_chars", 420)))
    return {
        "schema_version": "caption_frame_override_v1",
        "video_id": video_id,
        "canonical_frame_id": frame_doc.get("canonical_frame_id") or frame_doc.get("frame_id", ""),
        "frame_id": frame_doc.get("frame_id", ""),
        "timestamp_sec": frame_doc.get("timestamp_sec", 0.0),
        "image_relpath": frame_doc.get("image_relpath", ""),
        "image_path": str(image_path),
        "caption_text": caption_text,
        "caption_mode": "vlm",
        "caption_model": model_cfg.get("model_name", ""),
        "provider": model_cfg.get("provider", ""),
        "prompt_version": generation.get("prompt_version", "vlm_frame_caption_v1"),
        "created_at": utc_now_iso(),
    }


def create_vlm_captioner(model_cfg: dict[str, Any], generation: dict[str, Any]):
    provider = model_cfg.get("provider", "qwen2_5_vl")
    if provider == "qwen2_5_vl":
        return Qwen25VLCaptioner(model_cfg, generation)
    if provider == "internvl":
        return InternVLCaptioner(model_cfg, generation)
    if provider == "mock":
        return MockCaptioner(model_cfg, generation)
    raise ValueError(f"Unsupported VLM provider: {provider}")


class MockCaptioner:
    """Tiny provider for smoke tests without GPU/model dependencies."""

    def __init__(self, model_cfg: dict[str, Any], generation: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.generation = generation

    def caption_image(self, image_path: Path) -> str:
        return f"Mock VLM caption for frame image {image_path.name}."


class Qwen25VLCaptioner:
    """Qwen2.5-VL captioner using Hugging Face Transformers."""

    def __init__(self, model_cfg: dict[str, Any], generation: dict[str, Any]) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Installed transformers does not expose Qwen2_5_VLForConditionalGeneration."
            ) from exc
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required for provider=qwen2_5_vl.") from exc

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.model_cfg = model_cfg
        self.generation = generation
        model_name = model_cfg["model_name"]
        dtype = _torch_dtype(torch, model_cfg.get("dtype", "bfloat16"))
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": model_cfg.get("device_map", "auto"),
        }
        if model_cfg.get("attn_implementation"):
            model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt = generation.get("prompt", DEFAULT_PROMPT)

    def caption_image(self, image_path: Path) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_first_model_device(self.model))
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(self.generation.get("max_new_tokens", 96)),
            do_sample=bool(self.generation.get("do_sample", False)),
            temperature=float(self.generation.get("temperature", 0.0) or 1.0),
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()


class InternVLCaptioner:
    """InternVL captioner using the model's chat API."""

    def __init__(self, model_cfg: dict[str, Any], generation: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_cfg = model_cfg
        self.generation = generation
        model_name = model_cfg["model_name"]
        dtype = _torch_dtype(torch, model_cfg.get("dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=model_cfg.get("device_map", "auto"),
        ).eval()
        self.prompt = generation.get("prompt", DEFAULT_PROMPT)

    def caption_image(self, image_path: Path) -> str:
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        image = Image.open(image_path).convert("RGB")
        image_size = int(self.model_cfg.get("image_size", 448))
        transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        pixel_values = transform(image).unsqueeze(0)
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=next(self.model.parameters()).dtype)
        gen_config = {
            "max_new_tokens": int(self.generation.get("max_new_tokens", 96)),
            "do_sample": bool(self.generation.get("do_sample", False)),
        }
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            self.prompt,
            gen_config,
        ).strip()


def _torch_dtype(torch_module, name: str):
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def _first_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cuda"

def _resolve_video_dir(root: Path, video_id: str) -> Path:
    if (root / "captions").exists() and (root / "indexes").exists():
        return root
    return root / video_id


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value



