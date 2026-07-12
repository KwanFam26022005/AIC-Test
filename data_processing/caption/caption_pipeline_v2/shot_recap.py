"""Shot-level ReCap generation with audio context.

Phase 5 creates one structured text-LLM request per shot. It consumes only:
  - visual frame captions from Phase 3,
  - aligned ASR text from Phase 4,
  - compact memory from previous shots.

It does not consume OCR rows or object-detection rows.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from caption_pipeline.runtime.json_output import parse_json_object

from .config import ShotRecapConfig
from .recap_memory import (
    build_memory_ledger_row,
    empty_memory,
    should_reset_memory,
    truncate_memory,
)
from .schemas import (
    ACTION_STATE_ALIASES,
    PIPELINE_VERSION,
    SHOT_CAPTION_SCHEMA,
    TEMPORAL_ROLE_ALIASES,
    VALID_ACTION_STATES,
    VALID_TEMPORAL_ROLES,
    normalize_action_state,
    normalize_temporal_role,
)

logger = logging.getLogger(__name__)


DEFAULT_SHOT_RECAP_PROMPT = """You are generating a video-understanding caption for one shot.

Use only the provided visual captions, audio transcript context, and memory.
Do not invent OCR, object detector labels, people names, locations, or causes that are not supported.
If audio and visual evidence disagree, state the visually supported action and keep audio as topic/context.

Shot:
{shot_json}

Memory before:
{memory_before_json}

Visual frame captions:
{visual_captions_json}

Audio context:
{audio_context_json}

Return exactly one valid JSON object with these fields:
{
  "shot_caption": "one concise English caption for the shot",
  "temporal_caption": "how the shot fits the local sequence",
  "event_caption": "event-level summary grounded in visual + audio context",
  "action_state": "start|middle|end|transition|result|unknown",
  "temporal_role": "beginning|continuation|change|completion|unknown",
  "actors": ["short entity strings"],
  "actions": ["short action strings"],
  "objects_involved": ["semantic items from visual/audio evidence only"],
  "scene": "short scene description",
  "trake_text": "search-friendly event text combining action, entities, and context",
  "memory_after": {
    "active_entities": ["short entity strings"],
    "setting": "short setting",
    "ongoing_topic": "short topic",
    "ongoing_action": "short action",
    "last_event": "short event"
  }
}
"""


REQUIRED_FIELDS = (
    "shot_caption",
    "temporal_caption",
    "event_caption",
    "action_state",
    "temporal_role",
    "actors",
    "actions",
    "objects_involved",
    "scene",
    "trake_text",
    "memory_after",
)


class ShotRecapRuntime:
    """Small text runtime tailored for the v2 shot ReCap task."""

    def __init__(self, cfg: ShotRecapConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.torch = None
        self._prompt_text = ""
        self._prompt_hash = ""
        self._loaded = False

    @property
    def prompt_text(self) -> str:
        if not self._prompt_text:
            self._prompt_text = self._load_prompt()
            import hashlib

            self._prompt_hash = hashlib.sha256(self._prompt_text.encode("utf-8")).hexdigest()
        return self._prompt_text

    @property
    def prompt_hash(self) -> str:
        _ = self.prompt_text
        return self._prompt_hash

    def load_model(self) -> None:
        if self.cfg.provider == "mock" or self._loaded:
            return
        if self.cfg.provider != "transformers":
            raise ValueError(f"Unsupported shot recap provider: {self.cfg.provider}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = _torch_dtype(torch, self.cfg.dtype)
        requested_attn = os.environ.get("CAPTION_LLM_ATTN") or self.cfg.attn_implementation
        last_error: Exception | None = None

        for attention in _attention_fallback_order(requested_attn):
            kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": self.cfg.device_map,
                "low_cpu_mem_usage": True,
            }
            if attention:
                kwargs["attn_implementation"] = attention
            try:
                logger.info("Loading shot ReCap LLM %s with attention=%s", self.cfg.model_id, attention)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id,
                    revision=self.cfg.model_revision or None,
                    **kwargs,
                ).eval()
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Shot ReCap load failed with attention=%s: %s", attention, exc)

        if self.model is None:
            raise RuntimeError(f"Could not load shot ReCap LLM {self.cfg.model_id}") from last_error
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            revision=self.cfg.model_revision or None,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._loaded = True

    def unload_model(self) -> None:
        if not self._loaded:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, values: dict[str, Any]) -> dict[str, Any]:
        if self.cfg.provider == "mock":
            return {
                "data": _mock_payload(values),
                "attempts": 1,
                "raw_output": "mock",
                "elapsed_ms": 0,
                "input_tokens": None,
                "output_tokens": None,
            }
        if not self._loaded:
            raise RuntimeError("Shot ReCap model not loaded")

        prompt = _render_prompt(self.prompt_text, values)
        last_error: Exception | None = None
        raw_output = ""
        t0 = time.time()

        for attempt in range(1, self.cfg.max_retries + 2):
            attempt_tokens = min(int(self.cfg.max_new_tokens) * (2 ** (attempt - 1)), 2048)
            raw_output, input_tokens, output_tokens = self._generate_raw(prompt, attempt_tokens)
            try:
                data = parse_json_object(
                    raw_output,
                    required_fields=REQUIRED_FIELDS,
                    validator=validate_shot_payload,
                )
                return {
                    "data": data,
                    "attempts": attempt,
                    "raw_output": raw_output,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "shot_recap JSON attempt %d failed at max_new_tokens=%d: %s; tail=%r",
                    attempt,
                    attempt_tokens,
                    exc,
                    raw_output[-400:],
                )
                prompt = (
                    _render_prompt(self.prompt_text, values)
                    + "\n\nThe previous response was invalid. Return exactly one complete JSON object."
                    + f"\nValidation error: {exc}"
                    + f"\nPrevious response: {raw_output[:1200]}"
                )

        raise RuntimeError(f"shot_recap generation failed after retries: {last_error}")

    def _generate_raw(self, prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        messages = [
            {"role": "system", "content": "Return exactly one valid JSON object and no other text."},
            {"role": "user", "content": prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        json_prefix = "{"
        chat_text += json_prefix
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.cfg.max_input_tokens),
        )
        device = _first_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(self.cfg.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.cfg.do_sample:
            generation["temperature"] = float(self.cfg.temperature)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]
        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return (json_prefix + continuation).strip(), int(prompt_length), int(new_tokens.shape[0])

    def _load_prompt(self) -> str:
        if self.cfg.prompt_file:
            path = Path(self.cfg.prompt_file)
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
            logger.warning("Shot ReCap prompt file not found: %s; using default", path)
        return DEFAULT_SHOT_RECAP_PROMPT


def generate_shot_captions(
    shots: list[dict[str, Any]],
    frame_captions: list[dict[str, Any]],
    audio_contexts: list[dict[str, Any]],
    cfg: ShotRecapConfig,
    run_id: str,
    video_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate shot caption rows and memory ledger rows."""
    frame_by_id = {row["canonical_frame_id"]: row for row in frame_captions}
    audio_by_shot = {row["shot_id"]: row for row in audio_contexts}
    runtime = ShotRecapRuntime(cfg)
    runtime.load_model()

    rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    memory = empty_memory()
    prev_shot: dict[str, Any] | None = None

    try:
        for index, shot in enumerate(shots):
            reset, reason = should_reset_memory(prev_shot, shot)
            memory_before = empty_memory() if reset else dict(memory)
            visual_rows = [
                frame_by_id[fid]
                for fid in shot.get("frame_ids", [])
                if fid in frame_by_id
            ]
            visual_payload = _visual_payload(visual_rows)
            audio_context = audio_by_shot.get(shot["shot_id"], {})
            values = {
                "shot_json": _json({
                    "video_id": video_id,
                    "shot_id": shot["shot_id"],
                    "shot_index": index,
                    "start_sec": shot["start_sec"],
                    "end_sec": shot["end_sec"],
                    "frame_ids": shot.get("frame_ids", []),
                }),
                "memory_before_json": _json(memory_before),
                "visual_captions_json": _json(visual_payload),
                "audio_context_json": _json({
                    "text": audio_context.get("text", ""),
                    "language": audio_context.get("language", ""),
                    "feature_ids": audio_context.get("feature_ids", []),
                    "quality_levels": audio_context.get("quality_levels", []),
                }),
            }
            outcome = runtime.generate(values)
            data = outcome["data"]
            memory_after = truncate_memory(data["memory_after"], cfg.max_memory_chars)
            memory = dict(memory_after)

            row = _build_shot_row(
                shot=shot,
                data=data,
                memory_before=memory_before,
                memory_after=memory_after,
                visual_rows=visual_rows,
                audio_context=audio_context,
                outcome=outcome,
                cfg=cfg,
                run_id=run_id,
                video_id=video_id,
                prompt_hash=runtime.prompt_hash,
            )
            rows.append(row)
            memory_rows.append(
                build_memory_ledger_row(
                    video_id=video_id,
                    shot_id=shot["shot_id"],
                    memory_before=memory_before,
                    memory_after=memory_after,
                    reset_applied=reset,
                    reset_reason=reason,
                    source_generation_mode="text_recap",
                    input_signature=row["provenance"]["input_signature"],
                )
            )
            prev_shot = shot
            logger.info(
                "Shot ReCap progress: %d/%d (%s)",
                index + 1,
                len(shots),
                shot["shot_id"],
            )
    finally:
        if cfg.unload_after_stage:
            runtime.unload_model()

    logger.info("Shot ReCap: %d shot rows built", len(rows))
    return rows, memory_rows


def validate_shot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    memory_after = payload.get("memory_after")
    if not isinstance(memory_after, dict):
        memory_after = empty_memory()
    clean = {
        "shot_caption": _required_text(payload, "shot_caption", 520),
        "temporal_caption": _required_text(payload, "temporal_caption", 620),
        "event_caption": _required_text(payload, "event_caption", 520),
        "action_state": normalize_action_state(
            _enum(payload, "action_state", VALID_ACTION_STATES, ACTION_STATE_ALIASES)
        ),
        "temporal_role": normalize_temporal_role(
            _enum(payload, "temporal_role", VALID_TEMPORAL_ROLES, TEMPORAL_ROLE_ALIASES)
        ),
        "actors": _string_list(payload.get("actors"), 12),
        "actions": _string_list(payload.get("actions"), 12),
        "objects_involved": _string_list(payload.get("objects_involved"), 16),
        "scene": _optional_text(payload, "scene", 240),
        "trake_text": _required_text(payload, "trake_text", 900),
        "memory_after": {
            "active_entities": _string_list(memory_after.get("active_entities"), 12),
            "setting": _clean_text(memory_after.get("setting", ""), 160),
            "ongoing_topic": _clean_text(memory_after.get("ongoing_topic", ""), 180),
            "ongoing_action": _clean_text(memory_after.get("ongoing_action", ""), 180),
            "last_event": _clean_text(memory_after.get("last_event", ""), 220),
        },
    }
    return clean


def _build_shot_row(
    shot: dict[str, Any],
    data: dict[str, Any],
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
    visual_rows: list[dict[str, Any]],
    audio_context: dict[str, Any],
    outcome: dict[str, Any],
    cfg: ShotRecapConfig,
    run_id: str,
    video_id: str,
    prompt_hash: str,
) -> dict[str, Any]:
    import hashlib

    visual_ids = [row["canonical_frame_id"] for row in visual_rows]
    audio_ids = audio_context.get("feature_ids", [])
    signature_input = {
        "shot_id": shot["shot_id"],
        "visual_ids": visual_ids,
        "audio_ids": audio_ids,
        "memory_before": memory_before,
        "model_id": cfg.model_id,
        "prompt_hash": prompt_hash,
        "schema_version": SHOT_CAPTION_SCHEMA,
    }
    input_signature = hashlib.sha256(
        json.dumps(signature_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SHOT_CAPTION_SCHEMA,
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "video_id": video_id,
        "shot_id": shot["shot_id"],
        "start_sec": shot["start_sec"],
        "end_sec": shot["end_sec"],
        "frame_ids": shot.get("frame_ids", []),
        "visual_evidence": {
            "frame_ids": visual_ids,
            "captions": [
                {
                    "canonical_frame_id": row["canonical_frame_id"],
                    "caption_text": row.get("caption", {}).get("text", ""),
                    "mode": row.get("caption", {}).get("mode", ""),
                }
                for row in visual_rows
            ],
        },
        "audio_context": {
            "feature_ids": audio_ids,
            "text": audio_context.get("text", ""),
            "language": audio_context.get("language", ""),
            "has_usable_audio": bool(audio_context.get("has_usable_audio", False)),
        },
        "caption": {
            "shot_caption": data["shot_caption"],
            "temporal_caption": data["temporal_caption"],
            "event_caption": data["event_caption"],
            "trake_text": data["trake_text"],
            "language": "en",
            "mode": "text_recap",
        },
        "structure": {
            "action_state": data["action_state"],
            "temporal_role": data["temporal_role"],
            "actors": data["actors"],
            "actions": data["actions"],
            "objects_involved": data["objects_involved"],
            "scene": data["scene"],
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
        },
        "generation": {
            "provider": cfg.provider,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "prompt_version": cfg.prompt_version,
            "prompt_hash": f"sha256:{prompt_hash}",
            "attempts": outcome.get("attempts", 1),
            "max_new_tokens": cfg.max_new_tokens,
            "input_tokens": outcome.get("input_tokens"),
            "output_tokens": outcome.get("output_tokens"),
            "elapsed_ms": outcome.get("elapsed_ms", 0),
        },
        "quality": {
            "valid": True,
            "warnings": [],
        },
        "provenance": {
            "input_signature": input_signature,
        },
    }


def _mock_payload(values: dict[str, Any]) -> dict[str, Any]:
    visual = json.loads(values.get("visual_captions_json") or "[]")
    audio = json.loads(values.get("audio_context_json") or "{}")
    memory = json.loads(values.get("memory_before_json") or "{}")
    first_caption = ""
    if visual:
        first_caption = str(visual[0].get("caption_text", ""))
    audio_text = str(audio.get("text", ""))
    shot_caption = first_caption or "A video shot is shown."
    if audio_text:
        event_caption = f"{shot_caption} The audio context discusses: {audio_text[:180]}"
    else:
        event_caption = shot_caption
    return validate_shot_payload({
        "shot_caption": shot_caption,
        "temporal_caption": event_caption,
        "event_caption": event_caption,
        "action_state": "unknown",
        "temporal_role": "continuation",
        "actors": [],
        "actions": [],
        "objects_involved": [],
        "scene": shot_caption[:160],
        "trake_text": event_caption,
        "memory_after": {
            "active_entities": memory.get("active_entities", []),
            "setting": memory.get("setting", ""),
            "ongoing_topic": audio_text[:120],
            "ongoing_action": "",
            "last_event": event_caption[:180],
        },
    })


def _visual_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_frame_id": row["canonical_frame_id"],
            "timestamp_sec": row.get("timestamp_sec", 0.0),
            "caption_text": row.get("caption", {}).get("text", ""),
            "caption_mode": row.get("caption", {}).get("mode", ""),
            "selection_role": row.get("selection", {}).get("role", ""),
        }
        for row in rows
    ]


def _render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _required_text(payload: dict[str, Any], field: str, max_chars: int) -> str:
    value = _clean_text(payload.get(field, ""), max_chars)
    if not value:
        raise ValueError(f"Field '{field}' must be a non-empty string")
    return value


def _optional_text(payload: dict[str, Any], field: str, max_chars: int) -> str:
    return _clean_text(payload.get(field, ""), max_chars)


def _clean_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = " ".join(value.split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() or text[:max_chars]
    return text


def _string_list(value: Any, max_items: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item, 80)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _enum(payload: dict[str, Any], field: str, allowed: frozenset[str], aliases: dict[str, str]) -> str:
    value = _clean_text(payload.get(field, "unknown"), 40).lower() or "unknown"
    value = aliases.get(value, value)
    return value if value in allowed else "unknown"


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        import flash_attn.flash_attn_interface  # noqa: F401

        return True
    except Exception as exc:
        logger.warning("flash_attn is unavailable or ABI-incompatible: %s", exc)
        return False


def _has_sdpa() -> bool:
    try:
        import torch.nn.functional as functional

        return hasattr(functional, "scaled_dot_product_attention")
    except Exception:
        return False


def _attention_fallback_order(requested: str | None) -> list[str]:
    requested = (requested or "auto").strip()
    order: list[str] = []
    if requested in {"", "auto", "none", "None"}:
        requested = "flash_attention_2" if _has_flash_attn() else "sdpa"
    if requested == "flash_attention_2" and not _has_flash_attn():
        logger.warning("Requested flash_attention_2 is unusable; falling back")
    elif requested:
        order.append(requested)
    if _has_sdpa() and "sdpa" not in order:
        order.append("sdpa")
    if "eager" not in order:
        order.append("eager")
    return order


def _torch_dtype(torch_module: Any, name: str):
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def _first_model_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cuda"
