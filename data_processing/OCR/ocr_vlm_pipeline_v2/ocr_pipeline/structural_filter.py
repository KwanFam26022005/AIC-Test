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

# ── Regex patterns ────────────────────────────────────────────────────

TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\s*$"
)

# "60 giây" and common OCR misreads
_60_GIAY_TOKENS = {
    "60", "69", "6o",
    "giây", "giay", "giấy", "gầy", "(gầy", "giy",
}


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

    bbox = line_item.get("bbox_xyxy", [0, 0, 0, 0])

    if is_top_right_logo(bbox, text, W, H, cfg):
        return "structural_filter"

    if is_60_giay_variant(text, bbox, H, cfg):
        return "structural_filter"

    if is_bottom_counter(text, bbox, H, cfg):
        return "structural_filter"

    return None
