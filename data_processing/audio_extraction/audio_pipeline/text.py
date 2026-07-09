from __future__ import annotations

import re
from collections import Counter
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_PUNCT_END = {".", "?", "!", "..."}
_STOPWORDS = {
    "cua",
    "cho",
    "voi",
    "mot",
    "nhung",
    "cac",
    "la",
    "va",
    "thi",
    "trong",
    "tren",
    "duoi",
    "nay",
    "kia",
    "do",
    "da",
    "dang",
    "duoc",
    "se",
    "toi",
    "ban",
    "anh",
    "chi",
    "em",
    "minh",
    "chung",
    "ta",
}


def clean_transcript(text: str, cfg: dict[str, Any]) -> str:
    value = text or ""
    if cfg.get("normalize_whitespace", True):
        value = _SPACE_RE.sub(" ", value).strip()
    if cfg.get("fix_simple_repetition", True):
        value = collapse_repeated_tokens(value)
    if value and cfg.get("add_basic_punctuation", True):
        value = value[0].upper() + value[1:]
        if not any(value.endswith(p) for p in _PUNCT_END):
            value += "."
    return value


def collapse_repeated_tokens(text: str, max_repeat: int = 2) -> str:
    tokens = text.split()
    if not tokens:
        return text
    output: list[str] = []
    last_lower = None
    repeat_count = 0
    for token in tokens:
        key = token.lower()
        if key == last_lower:
            repeat_count += 1
        else:
            last_lower = key
            repeat_count = 1
        if repeat_count <= max_repeat:
            output.append(token)
    return " ".join(output)


def summarize_transcript(clean_text: str, cfg: dict[str, Any]) -> str | None:
    mode = cfg.get("summary_mode", "passthrough")
    if mode == "none":
        return None
    if not clean_text:
        return None
    return clean_text


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    raw_tokens = _TOKEN_RE.findall((text or "").lower())
    tokens = [
        token
        for token in raw_tokens
        if len(strip_accents(token)) >= 3
        and strip_accents(token) not in _STOPWORDS
        and not token.isdigit()
    ]
    counts = Counter(tokens)
    keywords = [token for token, _ in counts.most_common(max_keywords)]
    return keywords


def strip_accents(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_for_quality(text: str) -> str:
    """Normalize text for deduplication and boilerplate matching.

    Lowercases, strips accents, collapses whitespace, removes trailing
    punctuation.  Two transcripts that are semantically identical but
    differ only in case/accents/spacing will produce the same output.
    """
    value = strip_accents((text or "").lower().strip())
    value = _SPACE_RE.sub(" ", value)
    # Remove trailing punctuation for matching
    value = re.sub(r"[.?!,;:]+$", "", value).strip()
    return value


def is_boilerplate_transcript(text: str, patterns: list[str]) -> bool:
    """Return True if text matches any of the boilerplate patterns.

    Matching is case-insensitive and accent-insensitive.
    """
    if not text or not patterns:
        return False
    normalized = normalize_for_quality(text)
    if not normalized:
        return False
    for pattern in patterns:
        normalized_pattern = normalize_for_quality(pattern)
        if normalized_pattern and normalized_pattern in normalized:
            return True
    return False
