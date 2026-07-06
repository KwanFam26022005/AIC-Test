# -*- coding: utf-8 -*-
"""Official Transformers worker for NVIDIA LocateAnything."""

from __future__ import annotations

import logging
import re
from typing import Any

from PIL import Image

from tag_filter import build_locate_prompt


BOX_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>")
REF_BOX_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>",
    re.DOTALL,
)


def _clean_label(label: str) -> str:
    label = re.sub(r"<.*?>", " ", str(label))
    label = " ".join(label.replace("</c>", " ").split())
    return label.strip() or "object"


def select_dtype(dtype_name: str):
    import torch

    name = dtype_name.lower()
    if name in {"auto", "default"}:
        return "auto"
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resize_for_locate(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0

    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image, 1.0

    scale = max_side / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC), scale


def parse_detections(
    answer: str,
    image_width: int,
    image_height: int,
    default_label: str = "object",
) -> list[dict[str, Any]]:
    compact_answer = re.sub(r"\s+", "", answer.lower()) if answer else ""
    if not answer or "<box>none</box>" in compact_answer:
        return []

    detections: list[dict[str, Any]] = []
    for idx, match in enumerate(REF_BOX_RE.finditer(answer)):
        label = _clean_label(match.group(1))
        x1, y1, x2, y2 = [int(g) for g in match.groups()[1:]]
        detections.append({
            "label": label,
            "box": [
                x1 / 1000.0 * image_width,
                y1 / 1000.0 * image_height,
                x2 / 1000.0 * image_width,
                y2 / 1000.0 * image_height,
            ],
            "source": "locateanything",
            "raw_index": idx,
        })

    if detections:
        return detections

    for idx, match in enumerate(BOX_RE.finditer(answer)):
        x1, y1, x2, y2 = [int(g) for g in match.groups()]
        detections.append({
            "label": default_label,
            "box": [
                x1 / 1000.0 * image_width,
                y1 / 1000.0 * image_height,
                x2 / 1000.0 * image_width,
                y2 / 1000.0 * image_height,
            ],
            "source": "locateanything",
            "raw_index": idx,
        })
    return detections


class LocateAnythingWorker:
    """Stateful LocateAnything worker. Load once, call many times."""

    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        dtype: str = "fp16",
        attn_implementation: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        if str(device).startswith("cpu") and dtype.lower() in {"fp16", "float16", "half", "bf16", "bfloat16"}:
            logging.warning("CPU execution requested with %s; switching LocateAnything dtype to fp32.", dtype)
            dtype = "fp32"
        self.dtype = select_dtype(dtype)
        self.model_path = model_path
        self.attn_implementation = attn_implementation

        logging.info("Loading LocateAnything tokenizer/processor: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.dtype != "auto":
            kwargs["torch_dtype"] = self.dtype
        else:
            kwargs["torch_dtype"] = "auto"
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        logging.info("Loading LocateAnything model: %s", model_path)
        try:
            self.model = AutoModel.from_pretrained(model_path, **kwargs).to(device).eval()
        except Exception as exc:
            if not attn_implementation:
                raise
            logging.warning(
                "LocateAnything load failed with attn_implementation=%s; retrying default attention: %s",
                attn_implementation,
                exc,
            )
            kwargs.pop("attn_implementation", None)
            self.model = AutoModel.from_pretrained(model_path, **kwargs).to(device).eval()
        torch.set_grad_enabled(False)
        logging.info("LocateAnything ready on %s with dtype=%s", device, dtype)

    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 1024,
        verbose: bool = False,
    ) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        if hasattr(self.processor, "py_apply_chat_template"):
            text = self.processor.py_apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if hasattr(self.processor, "process_vision_info"):
            images, videos = self.processor.process_vision_info(messages)
        else:
            images, videos = [image], None

        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": images,
            "return_tensors": "pt",
        }
        if videos is not None:
            processor_kwargs["videos"] = videos

        inputs = self.processor(**processor_kwargs).to(self.device)
        if "pixel_values" in inputs and self.dtype != "auto":
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        generate_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "generation_mode": generation_mode,
            "do_sample": False,
            "verbose": verbose,
        }
        if "pixel_values" in inputs:
            generate_kwargs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_hws" in inputs:
            generate_kwargs["image_grid_hws"] = inputs["image_grid_hws"]

        with torch.inference_mode():
            response = self._generate_with_fallback(generate_kwargs)

        return self._response_to_text(response)

    def _generate_with_fallback(self, kwargs: dict[str, Any]):
        unsupported_keys = ["verbose", "do_sample", "tokenizer", "generation_mode"]
        current = dict(kwargs)
        while True:
            try:
                return self.model.generate(**current)
            except TypeError:
                if not unsupported_keys:
                    raise
                key = unsupported_keys.pop(0)
                if key in current:
                    logging.warning("Retrying LocateAnything generate without unsupported key: %s", key)
                    current.pop(key, None)
                else:
                    continue

    def _response_to_text(self, response) -> str:
        import torch

        if isinstance(response, tuple) and response:
            response = response[0]
        if isinstance(response, list) and response:
            response = response[0]
        if isinstance(response, str):
            return response
        if torch.is_tensor(response):
            if response.ndim == 1:
                response = response.unsqueeze(0)
            return self.tokenizer.batch_decode(response, skip_special_tokens=False)[0]
        return str(response)

    def detect(
        self,
        image: Image.Image,
        categories: list[str],
        generation_mode: str = "hybrid",
        max_new_tokens: int = 1024,
        image_max_side: int = 1280,
        verbose: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = build_locate_prompt(categories)
        if not prompt:
            return [], ""

        original_w, original_h = image.size
        locate_image, scale = resize_for_locate(image, image_max_side)
        answer = self.predict(
            locate_image,
            prompt,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

        default_label = categories[0] if len(categories) == 1 else "object"
        raw_detections = parse_detections(
            answer,
            locate_image.size[0],
            locate_image.size[1],
            default_label=default_label,
        )

        if scale != 1.0:
            for det in raw_detections:
                det["box"] = [coord / scale for coord in det["box"]]

        for det in raw_detections:
            det["image_size"] = [original_w, original_h]

        return raw_detections, answer
