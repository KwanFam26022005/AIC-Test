"""
Feature computation and composite scoring.

Computes:
- charset_penalty: ratio of disallowed characters
- repetition_penalty: penalizes repeated tokens/characters
- composite_score: weighted combination of all features
- priority: ranking score for VLM candidates
"""
from __future__ import annotations

import re
from collections import Counter

# Allowed character set for Vietnamese text + common symbols
_ALLOWED_CHARSET = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "àáảãạăắằẳẵặâấầẩẫậ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "đĐ"
    "èéẻẽẹêếềểễệ"
    "ÈÉẺẼẸÊẾỀỂỄỆ"
    "ìíỉĩị"
    "ÌÍỈĨỊ"
    "òóỏõọôốồổỗộơớờởỡợ"
    "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ùúủũụưứừửữự"
    "ÙÚỦŨỤƯỨỪỬỮỰ"
    "ỳýỷỹỵ"
    "ỲÝỶỸỴ"
    " .,;:!?-–—()[]{}\"'/&@#%+*=<>_~`^$|\\°₫"
    "\n\t\r"
)


def compute_charset_penalty(text: str) -> float:
    """Compute ratio of characters outside the allowed charset.

    Args:
        text: OCR text

    Returns:
        Penalty in [0, 1], higher = more bad characters
    """
    if not text:
        return 0.0

    bad_count = sum(1 for ch in text if ch not in _ALLOWED_CHARSET)
    return bad_count / len(text) if len(text) > 0 else 0.0


def compute_repetition_penalty(text: str, tokens: list[str]) -> float:
    """Compute penalty for repeated tokens or character runs.

    Examples that should be penalized:
    - "PP PP PP" (repeated tokens)
    - "aaaaaa" (repeated characters)

    Args:
        text: OCR text
        tokens: Pre-extracted tokens

    Returns:
        Penalty in [0, 1]
    """
    if not text or not tokens:
        return 0.0

    penalties = []

    # 1. Token repetition: same token appearing many times
    if len(tokens) >= 2:
        counter = Counter(t.lower() for t in tokens)
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count >= 3:
            penalties.append(min(1.0, most_common_count / len(tokens)))
        elif most_common_count >= 2 and len(tokens) <= 3:
            penalties.append(0.5)

    # 2. Character run repetition: same char repeated 4+ times
    char_runs = re.findall(r"(.)\1{3,}", text)
    if char_runs:
        total_run_len = sum(len(m) + 3 for m in char_runs)  # approximate
        penalties.append(min(1.0, total_run_len / max(1, len(text))))

    # 3. Full text is just repeated pattern
    clean = text.strip()
    if len(clean) >= 4:
        for plen in range(1, len(clean) // 2 + 1):
            pattern = clean[:plen]
            if pattern * (len(clean) // plen) == clean[:plen * (len(clean) // plen)]:
                if len(clean) // plen >= 3:
                    penalties.append(0.8)
                    break

    return max(penalties) if penalties else 0.0


def compute_composite_score(
    rec_conf: float,
    det_score: float,
    lex_ratio: float,
    diacritic_susp: float,
    charset_penalty: float,
    repetition_penalty: float,
    cfg: dict,
) -> tuple[float, float]:
    """Compute composite quality score.

    Formula from plan section 11.1:
        score = bias
              + 0.42 * rec_conf
              + 0.18 * det_score
              + 0.22 * lex_ratio
              + -0.12 * diacritic_susp
              + -0.06 * charset_penalty
              + -0.08 * repetition_penalty

    Args:
        All feature values and config

    Returns:
        (composite_score, quality_score) where quality_score = composite_score * 100
    """
    score = cfg.get("score_bias", 0.0)
    score += cfg.get("score_w_rec_conf", 0.42) * rec_conf
    score += cfg.get("score_w_det_score", 0.18) * det_score
    score += cfg.get("score_w_lex_ratio", 0.22) * lex_ratio
    score += cfg.get("score_w_diacritic_susp", -0.12) * diacritic_susp
    score += cfg.get("score_w_charset_penalty", -0.06) * charset_penalty
    score += cfg.get("score_w_repetition_penalty", -0.08) * repetition_penalty

    # Clip to [0, 1]
    composite = max(0.0, min(1.0, score))
    quality = composite * 100.0

    return composite, quality


def compute_priority(
    composite_score: float,
    text: str,
    tokens: list[str],
    cfg: dict,
) -> float:
    """Compute VLM priority score (higher = more urgent for Vintern).

    Formula from plan section 11.2:
        priority = (1 - composite_score) + content_value_weight * content_value

    Content value increases slightly with text length/token count.

    Args:
        composite_score: Composite quality score
        text: OCR text
        tokens: Pre-extracted tokens
        cfg: Config dict

    Returns:
        Priority score (higher = more likely to be sent to VLM)
    """
    # Content value: slight boost for longer text
    text_len = len(text.strip()) if text else 0
    token_count = len(tokens)
    content_value = min(1.0, (text_len / 50.0 + token_count / 10.0) / 2.0)

    weight = cfg.get("content_value_weight", 0.10)
    priority = (1.0 - composite_score) + weight * content_value

    return priority
