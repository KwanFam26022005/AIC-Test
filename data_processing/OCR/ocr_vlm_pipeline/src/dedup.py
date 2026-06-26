from __future__ import annotations

import hashlib

from .normalize_text import normalize_for_cache


def vlm_cache_key(raw_group_text: str, crop_phash: str | None, bbox_layout_signature: str, model_id: str, prompt_version: str) -> str:
    payload = "|".join(
        [
            normalize_for_cache(raw_group_text),
            crop_phash or "",
            bbox_layout_signature,
            model_id,
            prompt_version,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

