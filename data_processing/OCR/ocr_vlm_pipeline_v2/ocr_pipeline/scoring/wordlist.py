"""
Vietnamese wordlist loader and lexical scoring.

Loads VN_WORDSET and VN_BASE_TO_VARIANTS from dictionary files.
Computes lex_ratio and diacritic suspicion score for OCR text.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# Token regex from plan section 10.1
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-'][0-9A-Za-zÀ-ỹĐđ]+)*")

# Vietnamese diacritic characters for stripping
_VN_DIACRITICS = str.maketrans({
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "đ": "d",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
})


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics, keeping base Latin characters."""
    return text.lower().translate(_VN_DIACRITICS)


def extract_tokens(text: str) -> list[str]:
    """Extract word tokens from OCR text.

    Uses TOKEN_RE pattern from plan section 10.1.
    """
    return TOKEN_RE.findall(text or "")


def get_eval_tokens(tokens: list[str]) -> list[str]:
    """Get tokens suitable for evaluation (length >= 2, not pure digits)."""
    return [t for t in tokens if len(t) >= 2 and not t.isdigit()]


def load_wordlist(
    paths: list[str],
    min_entries: int = 1000,
) -> tuple[set[str], dict[str, set[str]], bool]:
    """Load Vietnamese wordlist from dictionary files.

    Args:
        paths: List of file paths to try loading
        min_entries: Minimum number of entries to consider wordlist valid

    Returns:
        (VN_WORDSET, VN_BASE_TO_VARIANTS, wordlist_enabled)
    """
    wordset: set[str] = set()

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Wordlist not found: {path}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if not word:
                        continue
                    # Clean: remove quotes, punctuation at start/end
                    word = word.strip("\"'()[]{}.,;:!?-– ")
                    if word and len(word) >= 1:
                        # Add lowercase version
                        wordset.add(word.lower())
            logger.info(f"Loaded {path.name}: {len(wordset)} cumulative entries")
        except Exception as e:
            logger.warning(f"Error loading wordlist {path}: {e}")

    # Check if wordlist is large enough
    enabled = len(wordset) >= min_entries
    if not enabled:
        logger.warning(f"Wordlist too small ({len(wordset)} < {min_entries}), disabling lex scoring")

    # Build base-to-variants mapping
    base_to_variants: dict[str, set[str]] = {}
    for word in wordset:
        base = strip_diacritics(word)
        if base != word:  # Only map if there are diacritics
            base_to_variants.setdefault(base, set()).add(word)
        # Also add the base itself as valid
        base_to_variants.setdefault(base, set()).add(word)

    logger.info(f"Wordlist: {len(wordset)} entries, {len(base_to_variants)} base forms, enabled={enabled}")
    return wordset, base_to_variants, enabled


def compute_lex_ratio(
    eval_tokens: list[str],
    wordset: set[str],
    wordlist_enabled: bool,
    neutral_ratio: float = 0.50,
) -> float:
    """Compute lexical ratio: fraction of tokens found in wordlist.

    Args:
        eval_tokens: Tokens to evaluate
        wordset: Set of valid words
        wordlist_enabled: Whether wordlist is active
        neutral_ratio: Fallback ratio when wordlist is disabled

    Returns:
        lex_ratio in [0, 1]
    """
    if not wordlist_enabled or not eval_tokens:
        return neutral_ratio

    in_wordset = sum(1 for t in eval_tokens if t.lower() in wordset)
    return in_wordset / len(eval_tokens)


def compute_diacritic_susp(
    eval_tokens: list[str],
    base_to_variants: dict[str, set[str]],
    wordset: set[str],
    wordlist_enabled: bool,
) -> tuple[float, list[str]]:
    """Compute diacritic suspicion score.

    A token is suspicious if:
    - It's NOT in the wordset
    - But its base (no-diacritic) form EXISTS in base_to_variants
    - → Likely a diacritic error from OCR

    Args:
        eval_tokens: Tokens to check
        base_to_variants: Mapping base → {valid variants}
        wordset: Set of valid words
        wordlist_enabled: Whether wordlist is active

    Returns:
        (diacritic_susp, suspicious_tokens_list)
    """
    if not wordlist_enabled or not eval_tokens:
        return 0.0, []

    suspicious = []
    for token in eval_tokens:
        token_lower = token.lower()
        if token_lower in wordset:
            continue  # Token is valid, skip

        base = strip_diacritics(token_lower)
        if base in base_to_variants:
            # Base exists but this specific form is not in wordset
            # Check if it's actually a diacritic issue (not just a different word)
            variants = base_to_variants[base]
            if token_lower not in variants:
                suspicious.append(token)

    susp_ratio = len(suspicious) / len(eval_tokens) if eval_tokens else 0.0
    return susp_ratio, suspicious


def compute_oov_tokens(
    eval_tokens: list[str],
    wordset: set[str],
    wordlist_enabled: bool,
) -> list[str]:
    """Find out-of-vocabulary tokens."""
    if not wordlist_enabled:
        return []
    return [t for t in eval_tokens if t.lower() not in wordset]
