from __future__ import annotations

from collections import Counter
from typing import Any


def score_quality(clean_text: str, duration_sec: float, cfg: dict[str, Any]) -> dict[str, Any]:
    text = (clean_text or "").strip()
    tokens = text.split()
    text_length = len(text)
    repetition_ratio = _repetition_ratio(tokens)
    min_text_length = int(cfg.get("min_text_length") or 5)
    max_repetition_ratio = float(cfg.get("max_repetition_ratio") or 0.35)
    min_duration_sec = float(cfg.get("min_duration_sec") or 0.5)

    has_text = bool(text)
    is_too_short = text_length < min_text_length
    is_repeated = repetition_ratio > max_repetition_ratio if len(tokens) >= 4 else False
    is_duration_too_short = float(duration_sec) < min_duration_sec

    if not has_text:
        quality_level = "empty"
    elif is_too_short or is_repeated or is_duration_too_short:
        quality_level = "bad"
    elif text_length < min_text_length * 3:
        quality_level = "medium"
    else:
        quality_level = "good"

    usable = quality_level in {"good", "medium"}
    return {
        "has_text": has_text,
        "text_length": text_length,
        "token_count": len(tokens),
        "repetition_ratio": round(repetition_ratio, 3),
        "is_repeated": is_repeated,
        "is_too_short": is_too_short,
        "is_duration_too_short": is_duration_too_short,
        "usable_for_caption": usable,
        "need_fallback": quality_level in {"bad", "empty"},
        "quality_level": quality_level,
    }


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(token.lower() for token in tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(tokens))

