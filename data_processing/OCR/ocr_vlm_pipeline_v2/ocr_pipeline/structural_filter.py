"""
Structural filter to remove noise text before OCR scoring/gating.

Filters:
- Empty text
- Timestamps (HH:MM:SS, HH.MM.SS, HH:MM)
- Top-right logos (short text in top-right corner)
- Bottom counters / "60 giây" variants
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ── Regex patterns ────────────────────────────────────────────────────

TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*$"
)

# "60 giây" and common OCR misreads
_60_GIAY_TOKENS = {
    "60", "69", "6o",
    "giây", "giay", "giấy", "gầy", "(gầy", "giy",
}

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-'][0-9A-Za-zÀ-ỹĐđ]+)*")

_VOWELS = set(
    "aeiouy"
    "àáảãạăắằẳẵặâấầẩẫậ"
    "èéẻẽẹêếềểễệ"
    "ìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữự"
    "ỳýỷỹỵ"
)


def is_empty(text: str) -> bool:
    """Check if text is empty or whitespace-only."""
    return not text or not text.strip()


def is_timestamp(text: str) -> bool:
    """Check if text matches timestamp pattern."""
    return bool(TIMESTAMP_RE.match(text.strip()))


def is_top_right_logo(
    bbox_xyxy: list[int],
    text: str,
    W: int,
    H: int,
    cfg: dict,
) -> bool:
    """Check if text is a top-right logo/watermark.

    Conditions:
    - x1 > W * logo_x_ratio (default 0.72)
    - y2 < H * logo_y_ratio (default 0.22)
    - len(text) <= logo_max_text_len (default 10)
    """
    x1, y1, x2, y2 = bbox_xyxy
    if x1 <= W * cfg.get("logo_x_ratio", 0.72):
        return False
    if y2 >= H * cfg.get("logo_y_ratio", 0.22):
        return False
    if len(text.strip()) > cfg.get("logo_max_text_len", 10):
        return False
    return True


def is_60_giay_variant(
    text: str,
    bbox_xyxy: list[int],
    H: int,
    cfg: dict,
) -> bool:
    """Check if text is a '60 giây' variant in the lower third.

    Only filters if y1 >= H * bottom_filter_y_ratio (default 0.70).
    """
    _, y1, _, _ = bbox_xyxy
    if y1 < H * cfg.get("bottom_filter_y_ratio", 0.70):
        return False

    text_lower = text.strip().lower()
    tokens = text_lower.split()

    # Check each token
    for token in tokens:
        # Strip common surrounding chars
        clean = token.strip("()[]{}.,;:!?\"'")
        if clean in _60_GIAY_TOKENS:
            return True

    # Also check the full string for short matches
    clean_full = re.sub(r"[^a-zà-ỹđ0-9]", "", text_lower)
    if clean_full in _60_GIAY_TOKENS:
        return True

    return False


def is_bottom_counter(
    text: str,
    bbox_xyxy: list[int],
    H: int,
    cfg: dict,
) -> bool:
    """Check if text is a bottom counter (pure numbers in lower area)."""
    _, y1, _, _ = bbox_xyxy
    if y1 < H * cfg.get("bottom_filter_y_ratio", 0.70):
        return False

    clean = text.strip()
    # Pure short numbers at bottom
    if re.fullmatch(r"\d{1,3}", clean):
        return True

    return False


def _normalize_alnum_char(ch: str) -> str:
    normalized = unicodedata.normalize("NFD", ch)
    without_marks = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return without_marks.casefold()


def classify_noise_text(text: str, cfg: dict | None = None) -> str | None:
    """Detect OCR hallucination/noise text that should not be indexed.

    This targets long low-diversity strings such as:
    - AIIIAAIAIIII...
    - 1B0CBC2BBC...
    - NN1(N1N([N...

    These are common OCR/VLM false positives from texture, borders, or signs.
    Returns a short reason string when the text should be dropped.
    """
    cfg = cfg or {}
    if not cfg.get("noise_filter_enabled", True):
        return None

    stripped = (text or "").strip()
    min_text_len = int(cfg.get("noise_filter_min_text_len", 18))
    if len(stripped) < min_text_len:
        return None

    chars = [ch for ch in stripped if not ch.isspace()]
    if not chars:
        return "empty"

    alnum = [ch for ch in stripped if ch.isalnum()]
    min_alnum = int(cfg.get("noise_filter_min_alnum_len", 14))
    if len(alnum) < min_alnum:
        symbol_ratio = sum(1 for ch in chars if not ch.isalnum()) / max(1, len(chars))
        if symbol_ratio >= float(cfg.get("noise_filter_max_symbol_ratio", 0.38)):
            return "symbol_heavy_noise"
        return None

    normalized_alnum = [_normalize_alnum_char(ch) for ch in alnum]
    unique_ratio = len(set(normalized_alnum)) / max(1, len(normalized_alnum))
    top4_ratio = sum(count for _, count in Counter(normalized_alnum).most_common(4)) / len(normalized_alnum)

    tokens = TOKEN_RE.findall(stripped)
    avg_token_len = sum(len(t) for t in tokens) / max(1, len(tokens))
    long_token_len = int(cfg.get("noise_filter_long_token_len", 18))
    long_token_unique_ratio = float(cfg.get("noise_filter_long_token_unique_ratio", 0.35))

    for token in tokens:
        token_alnum = [_normalize_alnum_char(ch) for ch in token if ch.isalnum()]
        if len(token_alnum) < long_token_len:
            continue
        token_unique_ratio = len(set(token_alnum)) / max(1, len(token_alnum))
        if token_unique_ratio <= long_token_unique_ratio:
            return "low_diversity_long_token"

    max_unique_ratio = float(cfg.get("noise_filter_max_unique_alnum_ratio", 0.28))
    min_top4_ratio = float(cfg.get("noise_filter_min_top4_alnum_ratio", 0.82))
    max_fragment_avg = float(cfg.get("noise_filter_max_avg_token_len_for_fragmented", 3.5))
    if unique_ratio <= max_unique_ratio and top4_ratio >= min_top4_ratio:
        if len(tokens) <= 2 or avg_token_len <= max_fragment_avg:
            return "low_diversity_fragmented_text"

    alpha = [_normalize_alnum_char(ch) for ch in alnum if ch.isalpha()]
    if len(alpha) >= int(cfg.get("noise_filter_min_alpha_for_no_vowel", 8)):
        vowel_ratio = sum(1 for ch in alpha if ch in _VOWELS) / max(1, len(alpha))
        if vowel_ratio <= float(cfg.get("noise_filter_max_no_vowel_ratio", 0.08)):
            return "no_vowel_alpha_noise"

    symbol_ratio = sum(1 for ch in chars if not ch.isalnum()) / max(1, len(chars))
    if symbol_ratio >= float(cfg.get("noise_filter_max_symbol_ratio", 0.38)):
        if unique_ratio <= float(cfg.get("noise_filter_symbol_max_unique_ratio", 0.40)):
            return "symbol_heavy_noise"

    return None


def classify_filter_status(
    line_item: dict,
    W: int,
    H: int,
    cfg: dict,
) -> str | None:
    """Classify a line for structural filtering.

    Args:
        line_item: Dict with vietocr_text, bbox_xyxy
        W: Image width
        H: Image height
        cfg: Config dict

    Returns:
        "structural_filter" if the line should be filtered, None otherwise.
    """
    text = line_item.get("vietocr_text", "") or ""

    if is_empty(text):
        return "structural_filter"

    if is_timestamp(text):
        return "structural_filter"

    noise_reason = classify_noise_text(text, cfg)
    if noise_reason:
        line_item["noise_reason"] = noise_reason
        return "structural_filter"

    bbox = line_item.get("bbox_xyxy", [0, 0, 0, 0])

    if is_top_right_logo(bbox, text, W, H, cfg):
        return "structural_filter"

    if is_60_giay_variant(text, bbox, H, cfg):
        return "structural_filter"

    if is_bottom_counter(text, bbox, H, cfg):
        return "structural_filter"

    return None
