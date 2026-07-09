"""Text normalization utilities for caption evidence."""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters, on a word boundary."""
    if not text or len(text) <= max_chars:
        return text
    # Try to break at a space before max_chars
    cut = text[:max_chars].rfind(" ")
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip()


def dedup_tokens(text: str) -> str:
    """Remove duplicate whitespace-separated tokens while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for tok in text.split():
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            result.append(tok)
    return " ".join(result)


def remove_accents(text: str) -> str:
    """Strip combining marks (diacritics) from *text*."""
    text = text.replace("Đ", "D").replace("đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def join_texts(*parts: str, sep: str = " ") -> str:
    """Join non-empty text parts with *sep* and normalize whitespace."""
    return normalize_whitespace(sep.join(p for p in parts if p))


def dedup_lines(texts: list[str]) -> list[str]:
    """Deduplicate a list of text snippets (normalized, case-insensitive)."""
    seen: set[str] = set()
    result: list[str] = []
    for t in texts:
        key = normalize_whitespace(t).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(t)
    return result
