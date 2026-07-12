"""VLM frame captioner — Qwen2.5-VL visual-only captioning.

Phase 2 (§8): Generate visual-only captions for selected frames.
- Receives only one image and one prompt per frame.
- Does NOT receive ASR, OCR, objects, neighboring captions, or memory.
- Returns plain text captions wrapped by the runtime into canonical JSON.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import VLMConfig

logger = logging.getLogger(__name__)

# Default prompt from design §8.3
DEFAULT_VLM_PROMPT = (
    "Inspect the scene, people, visible actions, spatial layout, and prominent "
    "readable text. Resolve obvious ambiguities internally. Describe only what is "
    "visually supported in the target frame. Do not infer identities, causes, "
    "locations, or events. Return one concise English caption only."
)

# Patterns that indicate audio/metadata leakage
_AUDIO_LEAK_PATTERNS = [
    re.compile(r"\bthe speaker says\b", re.IGNORECASE),
    re.compile(r"\bthe narrator\b", re.IGNORECASE),
    re.compile(r"\bvoice[- ]?over\b", re.IGNORECASE),
    re.compile(r"\baudio\b", re.IGNORECASE),
    re.compile(r"\bconfidence[:\s]*\d", re.IGNORECASE),
    re.compile(r"\bbounding[- ]?box\b", re.IGNORECASE),
    re.compile(r"\bjson\b", re.IGNORECASE),
    re.compile(r"\bmarkdown\b", re.IGNORECASE),
]

# Pattern for repeated phrases
_REPEAT_PATTERN = re.compile(r"(.{10,}?)\1{2,}")


class VLMFrameCaptioner:
    """Manages VLM model lifecycle and batch inference for frame captioning."""

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.processor = None
        self._prompt_text = ""
        self._prompt_hash = ""
        self._loaded = False

    @property
    def prompt_text(self) -> str:
        if not self._prompt_text:
            self._prompt_text = self._load_prompt()
            self._prompt_hash = hashlib.sha256(
                self._prompt_text.encode("utf-8")
            ).hexdigest()
        return self._prompt_text

    @property
    def prompt_hash(self) -> str:
        _ = self.prompt_text  # ensure loaded
        return self._prompt_hash

    def _load_prompt(self) -> str:
        """Load prompt from file or use default."""
        if self.cfg.prompt_file:
            p = Path(self.cfg.prompt_file)
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
            logger.warning("Prompt file not found: %s, using default", p)
        return DEFAULT_VLM_PROMPT

    def load_model(self) -> None:
        """Load the VLM model and processor onto GPU."""
        if self._loaded:
            return
        if self.cfg.provider == "mock":
            logger.info("Using mock VLM provider")
            self._loaded = True
            return
        if self.cfg.provider not in {"transformers", "qwen2_5_vl"}:
            raise ValueError(f"Unsupported VLM provider: {self.cfg.provider}")

        logger.info("Loading VLM model: %s", self.cfg.model_id)
        t0 = time.time()

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

            processor_kwargs: dict[str, Any] = {}
            if self.cfg.max_visual_tokens:
                # Qwen-VL visual tokens are roughly 28x28-pixel patches.
                processor_kwargs["max_pixels"] = int(self.cfg.max_visual_tokens) * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                self.cfg.model_id,
                revision=self.cfg.model_revision or None,
                **processor_kwargs,
            )

            requested_attn = os.environ.get("CAPTION_VLM_ATTN") or self.cfg.attn_implementation
            last_error: Exception | None = None
            for attention in _attention_fallback_order(requested_attn):
                model_kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    "device_map": self.cfg.device_map,
                }
                if attention:
                    model_kwargs["attn_implementation"] = attention
                try:
                    logger.info("Loading VLM with attention=%s", attention or "default")
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.cfg.model_id,
                        revision=self.cfg.model_revision or None,
                        **model_kwargs,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("VLM load failed with attention=%s: %s", attention, exc)
            if self.model is None:
                raise RuntimeError(f"Could not load VLM {self.cfg.model_id}") from last_error
            self.model.eval()
            self._loaded = True

            elapsed = time.time() - t0
            logger.info("VLM loaded in %.1fs", elapsed)

        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for VLM inference"
            ) from exc

    def unload_model(self) -> None:
        """Release GPU memory."""
        if not self._loaded:
            return
        if self.cfg.provider == "mock":
            self._loaded = False
            return

        logger.info("Unloading VLM model")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def caption_frame(
        self,
        image_path: str,
        canonical_frame_id: str,
    ) -> dict[str, Any]:
        """Generate a visual-only caption for a single frame.

        Returns a dict with: text, attempts, input_tokens, output_tokens, elapsed_ms
        """
        if not self._loaded:
            raise RuntimeError("VLM model not loaded — call load_model() first")
        if self.cfg.provider == "mock":
            return {
                "text": f"A representative visual view of frame {canonical_frame_id}.",
                "attempts": 1,
                "input_tokens": None,
                "output_tokens": None,
                "elapsed_ms": 0,
                "validation_warnings": [],
            }

        prompt = self.prompt_text
        result = self._generate_one(image_path, prompt, canonical_frame_id)

        # Validation + retry
        text = result["text"]
        warnings = validate_frame_caption(text, self.cfg)

        if warnings and self.cfg.max_retries > 0:
            logger.debug(
                "Frame %s: caption validation failed (%s), retrying",
                canonical_frame_id, warnings,
            )
            correction_prompt = (
                "The previous caption had issues. "
                "Describe only what is visible in the image. "
                "Return one concise English sentence under 320 characters."
            )
            retry_result = self._generate_one(
                image_path, correction_prompt, canonical_frame_id,
            )
            retry_warnings = validate_frame_caption(
                retry_result["text"], self.cfg,
            )

            if len(retry_warnings) < len(warnings):
                result = retry_result
                result["attempts"] = 2
                warnings = retry_warnings

        result["validation_warnings"] = warnings
        return result

    def _generate_one(
        self,
        image_path: str,
        prompt: str,
        canonical_frame_id: str,
    ) -> dict[str, Any]:
        """Run a single VLM generation."""
        import torch
        from qwen_vl_utils import process_vision_info

        t0 = time.time()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generation: dict[str, Any] = {
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.do_sample,
            }
            if self.cfg.do_sample:
                generation["temperature"] = self.cfg.temperature
            output_ids = self.model.generate(
                **inputs,
                **generation,
            )

        # Decode only new tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True,
        )[0].strip()

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "text": text,
            "attempts": 1,
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": generated.shape[1],
            "elapsed_ms": elapsed_ms,
        }

    def caption_batch(
        self,
        frames: list[dict],
        batch_id_prefix: str = "vlm_batch",
    ) -> list[dict[str, Any]]:
        """Caption a batch of frames sequentially.

        Each frame dict must have: image_path, canonical_frame_id
        Returns a list of result dicts.
        """
        results: list[dict[str, Any]] = []
        for i, frame in enumerate(frames):
            batch_id = f"{batch_id_prefix}_{i:06d}"
            try:
                result = self.caption_frame(
                    frame["image_path"],
                    frame["canonical_frame_id"],
                )
                result["batch_id"] = batch_id
                result["canonical_frame_id"] = frame["canonical_frame_id"]
                results.append(result)
            except Exception as exc:
                logger.error(
                    "VLM failed for frame %s: %s",
                    frame["canonical_frame_id"], exc,
                )
                results.append({
                    "text": "",
                    "attempts": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "elapsed_ms": 0,
                    "batch_id": batch_id,
                    "canonical_frame_id": frame["canonical_frame_id"],
                    "validation_warnings": ["vlm_generation_error"],
                    "error": str(exc),
                })
        return results


# ---------------------------------------------------------------------------
# Caption validation  (§8.4)
# ---------------------------------------------------------------------------

def validate_frame_caption(text: str, cfg: VLMConfig) -> list[str]:
    """Validate a frame caption against the design rules.

    Returns a list of warning strings (empty = valid).
    """
    warnings: list[str] = []

    # Non-empty after normalization
    normalized = " ".join(text.split()).strip()
    if not normalized:
        warnings.append("empty_caption")
        return warnings

    # Max characters
    if len(normalized) > cfg.max_caption_chars:
        warnings.append(f"exceeds_max_chars_{len(normalized)}")

    # Word count range
    words = normalized.split()
    if len(words) < cfg.min_caption_words:
        warnings.append(f"too_few_words_{len(words)}")
    if len(words) > cfg.max_caption_words:
        warnings.append(f"too_many_words_{len(words)}")

    # No repeated phrase loop
    if _REPEAT_PATTERN.search(normalized):
        warnings.append("repeated_phrase_loop")

    # No audio/metadata language
    for pattern in _AUDIO_LEAK_PATTERNS:
        if pattern.search(normalized):
            warnings.append(f"audio_metadata_leak_{pattern.pattern[:20]}")
            break

    return warnings


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
