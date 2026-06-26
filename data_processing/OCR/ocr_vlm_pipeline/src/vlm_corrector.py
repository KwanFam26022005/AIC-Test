from __future__ import annotations

import time
from pathlib import Path

from .vlm_loader import load_transformers_vlm
from .vlm_parser import parse_vlm_json
from .vlm_prompt import build_prompt, max_new_tokens


class VLMCorrector:
    def __init__(self, model_id: str, cfg):
        self.model_id = model_id
        self.cfg = cfg
        self.model, self.tokenizer = load_transformers_vlm(model_id, cfg)

    def correct_group(self, crop_path: str, raw_lines: list[dict], metadata: dict) -> dict:
        from PIL import Image

        image = Image.open(Path(crop_path)).convert("RGB")
        prompt = build_prompt(raw_lines)
        start = time.perf_counter()
        response = self._chat(image, prompt, len(raw_lines))
        parsed = parse_vlm_json(response, [line["line_idx"] for line in raw_lines])
        if parsed["parse_status"] not in ["ok", "partial"]:
            response = self._chat(image, build_prompt(raw_lines, short=True), len(raw_lines))
            parsed = parse_vlm_json(response, [line["line_idx"] for line in raw_lines])
        parsed.update(
            {
                "model_id": self.model_id,
                "latency_sec": time.perf_counter() - start,
                "raw_response": response,
                "metadata": metadata,
            }
        )
        return parsed

    def _chat(self, image, prompt: str, n_lines: int) -> str:
        import torch

        do_sample = bool(self.cfg.vlm.do_sample)
        generation_config = {
            "max_new_tokens": max_new_tokens(n_lines, self.cfg),
            "do_sample": do_sample,
            "num_beams": 1,
        }
        if do_sample:
            generation_config["temperature"] = float(self.cfg.vlm.temperature) or 0.7
        if hasattr(self.model, "chat"):
            pixel_values = preprocess_internvl_image(
                image,
                input_size=int(self.cfg.vlm.get("image_input_size", 448)),
                max_num=int(self.cfg.vlm.get("image_max_tiles", 6)),
            )
            dtype = next(self.model.parameters()).dtype
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
            pixel_values = pixel_values.to(dtype=dtype, device=device)
            with torch.no_grad():
                return self.model.chat(self.tokenizer, pixel_values, prompt, generation_config=generation_config)
        raise RuntimeError(f"Model {self.model_id} does not expose a chat() API; add adapter in vlm_corrector.py")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_internvl_image(image, input_size: int = 448, max_num: int = 6):
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def dynamic_preprocess(image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = True):
    width, height = image.size
    aspect_ratio = width / max(1, height)
    ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    target_ratio = _best_aspect_ratio(aspect_ratio, ratios, width, height, image_size)
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    cols = target_width // image_size
    resized = image.resize((target_width, target_height))
    tiles = [
        resized.crop(
            (
                (idx % cols) * image_size,
                (idx // cols) * image_size,
                (idx % cols + 1) * image_size,
                (idx // cols + 1) * image_size,
            )
        )
        for idx in range(target_ratio[0] * target_ratio[1])
    ]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _best_aspect_ratio(aspect_ratio: float, ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect_ratio)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_diff = diff
            best_ratio = ratio
    return best_ratio
