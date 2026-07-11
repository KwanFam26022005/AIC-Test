"""Qwen text-generation runtime shared by frame, shot, and TRAKE stages."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from .json_output import Validator, parse_json_object
from .prompt_loader import PromptRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationOutcome:
    data: dict[str, Any]
    attempts: int
    raw_output: str
    prompt_hash: str
    prompt_path: str


class TextLLMRuntime:
    """Load one text model and execute all caption JSON tasks sequentially."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.prompt_version = config.prompt_version
        self.prompts = PromptRegistry(config.prompt_dir)
        self.model = None
        self.tokenizer = None
        self.torch = None
        if self.provider == "transformers":
            self._load_transformers()
        elif self.provider != "mock":
            raise ValueError(f"Unsupported text LLM provider: {self.provider}")

    def task_signature(self, task: str) -> str:
        value = "|".join((
            self.provider,
            self.model_name,
            self.prompt_version,
            self.prompts.fingerprint(task),
        ))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def generate_task(
        self,
        task: str,
        values: dict[str, Any],
        required_fields: tuple[str, ...],
        max_new_tokens: int,
        validator: Validator | None = None,
    ) -> GenerationOutcome:
        rendered = self.prompts.render(task, values)
        if self.provider == "mock":
            data = _mock_payload(task, values)
            if validator:
                data = validator(data)
            return GenerationOutcome(
                data=data,
                attempts=1,
                raw_output="mock",
                prompt_hash=rendered.sha256,
                prompt_path=rendered.path,
            )

        last_error: Exception | None = None
        raw_output = ""
        prompt = rendered.text
        for attempt in range(1, self.config.max_retries + 2):
            raw_output = self._generate(prompt, max_new_tokens=max_new_tokens)
            try:
                data = parse_json_object(
                    raw_output,
                    required_fields=required_fields,
                    validator=validator,
                )
                return GenerationOutcome(
                    data=data,
                    attempts=attempt,
                    raw_output=raw_output,
                    prompt_hash=rendered.sha256,
                    prompt_path=rendered.path,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("%s JSON attempt %d failed: %s", task, attempt, exc)
                prompt = (
                    rendered.text
                    + "\n\nYour previous response was invalid. Return one valid JSON object only."
                    + f"\nValidation error: {exc}"
                    + f"\nPrevious response: {raw_output[:1200]}"
                )
        raise RuntimeError(f"{task} generation failed after retries: {last_error}")

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = _torch_dtype(torch, self.config.dtype)
        requested_attn = os.environ.get("CAPTION_LLM_ATTN") or self.config.attn_implementation
        last_error: Exception | None = None
        for attention in _attention_fallback_order(requested_attn):
            kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": self.config.device_map,
                "low_cpu_mem_usage": True,
            }
            if attention:
                kwargs["attn_implementation"] = attention
            try:
                logger.info(
                    "Loading text LLM %s with attention=%s",
                    self.model_name,
                    attention or "default",
                )
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).eval()
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Text LLM load failed with attention=%s: %s", attention, exc)
        if self.model is None:
            raise RuntimeError(f"Could not load text LLM {self.model_name}") from last_error
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        device = _first_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation["temperature"] = float(self.config.temperature)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _mock_payload(task: str, values: dict[str, Any]) -> dict[str, Any]:
    if task == "frame":
        source = values.get("initial_caption") or values.get("scene_tags") or "A video frame."
        return {"caption_text": str(source)[:300]}
    if task == "shot":
        source = values.get("representative_frame_captions") or "A video shot."
        return {
            "caption_text": str(source)[:520],
            "temporal_caption": str(source)[:620],
            "memory_after": str(source)[:500],
        }
    return {
        "event_caption": str(values.get("caption_text") or "An event occurs.")[:420],
        "action_state": "unknown",
        "temporal_role": "continuation",
        "actors": [],
        "actions": [],
        "objects_involved": [],
        "scene": str(values.get("scene_text") or "")[:200],
        "before_context": str(values.get("before_context") or "")[:300],
        "current_observation": str(values.get("caption_text") or "")[:420],
        "after_context": str(values.get("after_context") or "")[:300],
        "trake_text": str(values.get("temporal_caption") or values.get("caption_text") or "event")[:900],
    }


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
