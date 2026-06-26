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
        generation_config = {
            "max_new_tokens": max_new_tokens(n_lines, self.cfg),
            "do_sample": bool(self.cfg.vlm.do_sample),
            "temperature": float(self.cfg.vlm.temperature),
            "num_beams": 1,
        }
        if hasattr(self.model, "chat"):
            return self.model.chat(self.tokenizer, image, prompt, generation_config=generation_config)
        raise RuntimeError(f"Model {self.model_id} does not expose a chat() API; add adapter in vlm_corrector.py")

