from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_search(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_cache(text: str) -> str:
    return normalize_for_search(text)

