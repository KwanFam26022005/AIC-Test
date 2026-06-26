from __future__ import annotations

import json

PROMPT_VERSION = "vi_ocr_json_v3"


def build_prompt(raw_lines: list[dict], short: bool = False) -> str:
    normalized_lines = [
        {
            "line_idx": int(line["line_idx"]),
            "raw_text": str(line.get("raw_text") or line.get("ocr_text") or ""),
            "corrected_text": str(line.get("raw_text") or line.get("ocr_text") or ""),
            "confidence_note": "medium",
        }
        for line in raw_lines
    ]
    lines = "\n".join(f"[{line['line_idx']}] {line['raw_text']}" for line in normalized_lines)
    json_template = json.dumps({"lines": normalized_lines}, ensure_ascii=False, separators=(",", ":"))
    if short:
        return (
            "OCR correction task. Return ONLY valid JSON. Do not describe the image. "
            "Keep exactly the same line_idx values. Fix Vietnamese OCR text only when visible.\n"
            f"JSON template to fill:\n{json_template}\n"
            f"Raw OCR lines:\n{lines}"
        )
    return (
        "You are an OCR text correction engine for Vietnamese video frames.\n"
        "Use the image crop only to correct the raw OCR lines below.\n"
        "Return ONLY valid minified JSON. Do not explain. Do not describe the image. Do not use Markdown.\n"
        "Keep exactly the same number of lines and exactly the same line_idx values.\n"
        "For each item, keep raw_text unchanged and write the fixed text in corrected_text.\n"
        "If the text is unreadable or not visible, copy raw_text to corrected_text.\n"
        f"JSON template to fill:\n{json_template}\n"
        f"Raw OCR lines:\n{lines}"
    )


def max_new_tokens(n_lines: int, cfg) -> int:
    return min(
        int(cfg.vlm.max_new_tokens_cap),
        int(cfg.vlm.max_new_tokens_base) + int(n_lines) * int(cfg.vlm.max_new_tokens_per_line),
    )
