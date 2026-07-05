"""
Configuration for OCR VLM Pipeline v2.

All thresholds and parameters from the workflow plan document.
"""
from __future__ import annotations

import os
from pathlib import Path


def get_default_config() -> dict:
    """Return the full default configuration dictionary."""
    # Base directory of this package (ocr_pipeline/)
    _pkg_dir = Path(__file__).resolve().parent
    # Root of ocr_vlm_pipeline_v2/
    _root_dir = _pkg_dir.parent

    return {
        # ── Detector (PP-OCRv6 TextDetection via PaddleX) ──────────────
        "det_model_name": "PP-OCRv6_medium_det",
        "det_limit_type": "min",
        "det_limit_side_len": 960,
        "det_thresh": 0.30,
        "det_box_thresh": 0.50,
        "det_unclip_ratio": 1.8,
        "paddle_device": None,  # auto-detect: gpu:0 if available, else cpu
        "paddle_engine": "paddle_static",  # override with OCR_V2_PADDLE_ENGINE

        # ── Line box filtering ─────────────────────────────────────────
        "drop_low_det_score_below": 0.0,
        "min_box_width": 10,
        "min_box_height": 8,
        "min_box_aspect_ratio": 0.25,

        # ── Perspective crop ───────────────────────────────────────────
        "perspective_padding": 12,
        "crop_min_height_for_ocr": 72,
        "crop_upscale_max_factor": 3.5,
        "add_white_border": 10,
        "contrast_factor": 1.25,

        # ── Grouping ──────────────────────────────────────────────────
        "group_min_x_overlap_ratio": 0.12,
        "group_max_vertical_gap_ratio": 1.80,
        "group_min_lines": 2,
        "group_crop_padding": 28,

        # ── VietOCR ───────────────────────────────────────────────────
        "vietocr_config": "vgg_transformer",
        "vietocr_beamsearch": True,
        "use_vietocr_return_prob": True,
        "rec_conf_fallback_when_missing": 0.50,
        "vietocr_device": None,  # auto-detect

        # ── rec_conf flat detection ───────────────────────────────────
        "rec_conf_flat_std_threshold": 0.02,
        "rec_conf_flat_unique_ratio_threshold": 0.10,

        # ── Wordlist ──────────────────────────────────────────────────
        "wordlist_paths": [
            str(_root_dir / "word_list" / "vn_dictionary.txt"),
            str(_root_dir / "word_list" / "general_dict.txt"),
        ],
        "wordlist_min_entries_to_enable": 1000,
        "neutral_lex_ratio_if_no_wordlist": 0.50,

        # ── Composite score weights ──────────────────────────────────
        "score_bias": 0.0,
        "score_w_rec_conf": 0.42,
        "score_w_det_score": 0.18,
        "score_w_lex_ratio": 0.22,
        "score_w_diacritic_susp": -0.12,
        "score_w_charset_penalty": -0.06,
        "score_w_repetition_penalty": -0.08,

        # ── Priority for VLM ─────────────────────────────────────────
        "content_value_weight": 0.10,

        # ── Gating — standard rule (rec_conf reliable) ───────────────
        "auto_accept_min_rec_conf": 0.90,
        "auto_accept_min_det_score": 0.70,
        "auto_accept_min_lex_ratio": 0.50,
        "auto_accept_max_diacritic_susp": 0.00,
        "auto_accept_max_charset_penalty": 0.05,

        "escalate_composite_below": 0.68,
        "escalate_rec_conf_below": 0.85,
        "escalate_lex_ratio_below": 0.45,
        "escalate_diacritic_susp_above": 0.30,

        # ── Gating — rec_missing rule (rec_conf flat/missing) ────────
        "rec_missing_auto_accept_min_tokens": 5,
        "rec_missing_auto_accept_min_det_score": 0.78,
        "rec_missing_auto_accept_min_lex_ratio": 0.70,
        "rec_missing_auto_accept_max_diacritic_susp": 0.05,
        "rec_missing_auto_accept_max_charset_penalty": 0.02,

        # ── Structural filter ─────────────────────────────────────────
        "logo_x_ratio": 0.72,
        "logo_y_ratio": 0.22,
        "logo_max_text_len": 10,
        "bottom_filter_y_ratio": 0.70,

        # ── Vintern line fallback ─────────────────────────────────────
        "vintern_model_id": "5CD-AI/Vintern-1B-v3_5",
        "vintern_device": None,  # auto-detect
        "vintern_quantization": "4bit_nf4",
        "vintern_attn_implementation": None,  # auto: flash_attention_2 or eager
        "vintern_max_new_tokens": 256,
        "vintern_do_sample": False,
        "vintern_temperature": 0.0,
        "vintern_input_size": 448,
        "vintern_max_tiles": 6,
        "vintern_progress": True,

        "use_vintern_line_fallback": True,
        "vintern_max_candidates": 8,
        "vintern_min_composite_accept": 0.55,
        "vintern_override_margin": 0.05,
        "vintern_agreement_threshold": 0.82,

        # ── Vintern group fallback ────────────────────────────────────
        "use_vintern_group_fallback": True,
        "vintern_group_max_candidates": 4,
        "vintern_group_min_lines": 2,
        "group_vintern_min_composite_accept": 0.55,
        "group_vintern_min_text_len_ratio": 0.6,
        "group_vintern_min_text_len_abs": 6,
        "group_vintern_accept_lex_ratio": 0.45,
        "group_vintern_accept_max_charset": 0.05,
        "group_vintern_agreement_threshold": 0.82,

        # Group VLM priority bonus
        "group_priority_bonus_multiline": 0.25,
        "group_priority_bonus_vlm_candidate": 0.20,
        "group_priority_bonus_need_review": 0.15,

        # ── Vintern prompts ───────────────────────────────────────────
        "vintern_line_prompt": (
            "Hãy đọc chính xác toàn bộ chữ trong ảnh crop này.\n"
            "Chỉ trả về nội dung OCR, không giải thích.\n"
            "Giữ nguyên tiếng Việt có dấu nếu có."
        ),
        "vintern_group_prompt": (
            "Hãy đọc chính xác toàn bộ chữ trong ảnh crop này theo đúng từng dòng.\n"
            "Nếu ảnh có nhiều dòng chữ, hãy xuống dòng giữa các dòng.\n"
            "Chỉ trả về nội dung OCR, không giải thích."
        ),

        # ── Output ────────────────────────────────────────────────────
        "output_dir": str(_root_dir / "output"),
        "line_crops_subdir": "ppocr_line_perspective_crops",
        "group_crops_subdir": "ppocr_stacked_group_crops",
        "output_suffix": "wordlist_gating_v2",
    }


def make_config(**overrides) -> dict:
    """Create config with optional overrides."""
    cfg = get_default_config()
    cfg.update(overrides)
    return cfg
