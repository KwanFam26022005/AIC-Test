"""
Gating decision logic for OCR pipeline.

Determines for each line:
- auto_accept: confident enough, no VLM needed
- vlm_candidate: suspicious, should be sent to Vintern
- structural_filter: noise/timestamp/logo, skip entirely
- review_only: not confident enough to index

Two rule sets:
1. Standard rule: when rec_conf is reliable
2. rec_missing rule: when rec_conf is flat/missing
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .wordlist import (
    extract_tokens,
    get_eval_tokens,
    compute_lex_ratio,
    compute_diacritic_susp,
    compute_oov_tokens,
)
from .features import (
    compute_charset_penalty,
    compute_repetition_penalty,
    compute_composite_score,
    compute_priority,
)
from ..structural_filter import classify_filter_status

logger = logging.getLogger(__name__)


def score_and_gate_line(
    line_item: dict,
    cfg: dict,
    wordset: set[str],
    base_to_variants: dict[str, set[str]],
    wordlist_enabled: bool,
    W: int,
    H: int,
    use_rec_missing_rule: bool = False,
) -> None:
    """Compute all features, composite score, and gating decision for a line.

    Modifies line_item in place with:
    - tokens, eval_tokens
    - lex_ratio, diacritic_susp, charset_penalty, repetition_penalty
    - composite_score, quality_score, priority
    - filter_status, send_to_vintern, is_filtered
    - keep_for_index, need_review

    Args:
        line_item: Line dict to score (modified in place)
        cfg: Config dict
        wordset: Vietnamese wordset
        base_to_variants: Base → variants mapping
        wordlist_enabled: Whether wordlist is active
        W: Image width
        H: Image height
        use_rec_missing_rule: Force rec_missing gating rule
    """
    text = line_item.get("vietocr_text", "") or ""

    # ── Structural filter (check first) ─────────────────────────────
    structural = classify_filter_status(line_item, W, H, cfg)
    if structural:
        line_item["filter_status"] = structural
        line_item["send_to_vintern"] = False
        line_item["is_filtered"] = True
        line_item["keep_for_index"] = False
        line_item["need_review"] = False
        line_item["final_text"] = ""
        line_item["final_source"] = "structural_filter"
        # Still compute tokens for reporting
        tokens = extract_tokens(text)
        line_item["tokens"] = tokens
        line_item["eval_tokens"] = get_eval_tokens(tokens)
        line_item["lex_ratio"] = 0.0
        line_item["diacritic_susp"] = 0.0
        line_item["diacritic_suspicious_tokens"] = []
        line_item["oov_tokens"] = []
        line_item["charset_penalty"] = 0.0
        line_item["repetition_penalty"] = 0.0
        line_item["composite_score"] = 0.0
        line_item["quality_score"] = 0.0
        line_item["priority"] = 0.0
        line_item["det_score_norm"] = line_item.get("det_score", 0.0)
        return

    # ── Tokenization ────────────────────────────────────────────────
    tokens = extract_tokens(text)
    eval_tokens = get_eval_tokens(tokens)
    line_item["tokens"] = tokens
    line_item["eval_tokens"] = eval_tokens

    # ── Lexical features ────────────────────────────────────────────
    neutral = cfg.get("neutral_lex_ratio_if_no_wordlist", 0.50)
    lex_ratio = compute_lex_ratio(eval_tokens, wordset, wordlist_enabled, neutral)
    diacritic_susp, susp_tokens = compute_diacritic_susp(
        eval_tokens, base_to_variants, wordset, wordlist_enabled
    )
    oov_tokens = compute_oov_tokens(eval_tokens, wordset, wordlist_enabled)

    line_item["lex_ratio"] = lex_ratio
    line_item["diacritic_susp"] = diacritic_susp
    line_item["diacritic_suspicious_tokens"] = susp_tokens
    line_item["oov_tokens"] = oov_tokens

    # ── Character features ──────────────────────────────────────────
    charset_penalty = compute_charset_penalty(text)
    repetition_penalty = compute_repetition_penalty(text, tokens)
    line_item["charset_penalty"] = charset_penalty
    line_item["repetition_penalty"] = repetition_penalty

    # ── Confidence normalization ────────────────────────────────────
    rec_conf = line_item.get("rec_conf")
    rec_conf_source = line_item.get("rec_conf_source", "")

    if rec_conf is None:
        rec_conf_norm = cfg.get("rec_conf_fallback_when_missing", 0.50)
    else:
        rec_conf_norm = float(rec_conf)

    det_score = line_item.get("det_score", 0.0)
    line_item["det_score_norm"] = det_score

    # ── Composite score ─────────────────────────────────────────────
    composite, quality = compute_composite_score(
        rec_conf_norm, det_score, lex_ratio, diacritic_susp,
        charset_penalty, repetition_penalty, cfg
    )
    line_item["composite_score"] = composite
    line_item["quality_score"] = quality

    # ── Priority ────────────────────────────────────────────────────
    priority = compute_priority(composite, text, tokens, cfg)
    line_item["priority"] = priority

    # ── Gating decision ─────────────────────────────────────────────
    is_rec_missing = (
        use_rec_missing_rule
        or line_item.get("rec_conf_flat_run", False)
        or rec_conf is None
        or rec_conf_source in ("return_prob_missing", "text_only", "return_prob_no_conf")
    )

    if is_rec_missing:
        _gate_rec_missing(line_item, cfg, eval_tokens, det_score, lex_ratio,
                          diacritic_susp, charset_penalty, composite)
    else:
        _gate_standard(line_item, cfg, rec_conf_norm, det_score, lex_ratio,
                       diacritic_susp, charset_penalty, composite)


def _gate_standard(
    line_item: dict,
    cfg: dict,
    rec_conf: float,
    det_score: float,
    lex_ratio: float,
    diacritic_susp: float,
    charset_penalty: float,
    composite: float,
) -> None:
    """Standard gating rule when rec_conf is reliable (plan section 13.1)."""
    # Auto accept conditions
    auto_accept = (
        rec_conf >= cfg.get("auto_accept_min_rec_conf", 0.90)
        and det_score >= cfg.get("auto_accept_min_det_score", 0.70)
        and lex_ratio >= cfg.get("auto_accept_min_lex_ratio", 0.50)
        and diacritic_susp <= cfg.get("auto_accept_max_diacritic_susp", 0.00)
        and charset_penalty <= cfg.get("auto_accept_max_charset_penalty", 0.05)
    )

    if auto_accept:
        line_item["filter_status"] = "auto_accept"
        line_item["send_to_vintern"] = False
        line_item["is_filtered"] = False
        line_item["keep_for_index"] = True
        line_item["need_review"] = False
        line_item["final_text"] = line_item.get("vietocr_text", "")
        line_item["final_source"] = "vietocr_auto_accept"
        return

    # Weak detection check
    weak_detection = det_score < cfg.get("auto_accept_min_det_score", 0.70)

    # Escalation conditions
    escalate = (
        weak_detection
        or composite < cfg.get("escalate_composite_below", 0.68)
        or rec_conf < cfg.get("escalate_rec_conf_below", 0.85)
        or lex_ratio < cfg.get("escalate_lex_ratio_below", 0.45)
        or diacritic_susp > cfg.get("escalate_diacritic_susp_above", 0.30)
    )

    if escalate:
        line_item["filter_status"] = "vlm_candidate"
        line_item["send_to_vintern"] = True
        line_item["is_filtered"] = False
        line_item["keep_for_index"] = False  # pending Vintern
        line_item["need_review"] = True
        line_item["final_text"] = line_item.get("vietocr_text", "")
        line_item["final_source"] = "vietocr_pending_vlm"
    else:
        # Between auto_accept and escalate → review
        line_item["filter_status"] = "review_only"
        line_item["send_to_vintern"] = False
        line_item["is_filtered"] = False
        line_item["keep_for_index"] = True
        line_item["need_review"] = True
        line_item["final_text"] = line_item.get("vietocr_text", "")
        line_item["final_source"] = "vietocr_review"


def _gate_rec_missing(
    line_item: dict,
    cfg: dict,
    eval_tokens: list[str],
    det_score: float,
    lex_ratio: float,
    diacritic_susp: float,
    charset_penalty: float,
    composite: float,
) -> None:
    """Gating rule when rec_conf is missing/flat (plan section 13.2).

    More conservative: only auto-accept long, clear, high-lex text.
    """
    token_count = len(eval_tokens)
    weak_detection = det_score < cfg.get("rec_missing_auto_accept_min_det_score", 0.78)

    auto_accept = (
        token_count >= cfg.get("rec_missing_auto_accept_min_tokens", 5)
        and det_score >= cfg.get("rec_missing_auto_accept_min_det_score", 0.78)
        and lex_ratio >= cfg.get("rec_missing_auto_accept_min_lex_ratio", 0.70)
        and diacritic_susp <= cfg.get("rec_missing_auto_accept_max_diacritic_susp", 0.05)
        and charset_penalty <= cfg.get("rec_missing_auto_accept_max_charset_penalty", 0.02)
        and not weak_detection
    )

    if auto_accept:
        line_item["filter_status"] = "auto_accept"
        line_item["send_to_vintern"] = False
        line_item["is_filtered"] = False
        line_item["keep_for_index"] = True
        line_item["need_review"] = False
        line_item["final_text"] = line_item.get("vietocr_text", "")
        line_item["final_source"] = "vietocr_auto_accept_rec_missing"
    else:
        line_item["filter_status"] = "vlm_candidate"
        line_item["send_to_vintern"] = True
        line_item["is_filtered"] = False
        line_item["keep_for_index"] = False
        line_item["need_review"] = True
        line_item["final_text"] = line_item.get("vietocr_text", "")
        line_item["final_source"] = "vietocr_pending_vlm_rec_missing"


def detect_rec_conf_flat(line_items: list[dict], cfg: dict) -> bool:
    """Detect if rec_conf values are flat/unreliable across all lines.

    Returns True if:
    - std(rec_confs) <= threshold
    - OR unique_ratio <= threshold

    Args:
        line_items: All line items with rec_conf values
        cfg: Config dict

    Returns:
        True if rec_conf is considered flat
    """
    confs = []
    for item in line_items:
        conf = item.get("rec_conf")
        if conf is not None and not item.get("is_filtered", False):
            confs.append(float(conf))

    if len(confs) < 2:
        return False

    conf_arr = np.array(confs)
    std_val = float(conf_arr.std())
    unique_ratio = len(set(round(c, 4) for c in confs)) / len(confs)

    std_thresh = cfg.get("rec_conf_flat_std_threshold", 0.02)
    unique_thresh = cfg.get("rec_conf_flat_unique_ratio_threshold", 0.10)

    is_flat = std_val <= std_thresh or unique_ratio <= unique_thresh

    if is_flat:
        logger.info(f"rec_conf FLAT detected: std={std_val:.4f}, unique_ratio={unique_ratio:.4f}")

    return is_flat
