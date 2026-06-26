from __future__ import annotations

PROMPT_VERSION = "vi_ocr_json_v1"


def build_prompt(raw_lines: list[dict], short: bool = False) -> str:
    lines = "\n".join(f"[{line['line_idx']}] {line.get('raw_text') or line.get('ocr_text') or ''}" for line in raw_lines)
    if short:
        return (
            "Sua loi OCR tieng Viet tu anh crop. Chi tra JSON hop le, khong giai thich. "
            '{"lines":[{"line_idx":0,"raw_text":"...","corrected_text":"...","confidence_note":"low|medium|high"}]}\n'
            f"Raw OCR lines:\n{lines}"
        )
    return (
        "Ban la he thong sua loi OCR tieng Viet.\n"
        "Nhiem vu: dua vao anh crop va danh sach raw OCR ben duoi, sua loi dau, chinh ta va ky tu OCR.\n"
        "Khong them thong tin moi ngoai noi dung nhin thay trong anh. Giu so dong output bang so dong input.\n"
        "Chi tra ve JSON hop le theo schema:\n"
        '{"lines":[{"line_idx":0,"raw_text":"...","corrected_text":"...","confidence_note":"low|medium|high"}]}\n'
        f"Raw OCR lines:\n{lines}"
    )


def max_new_tokens(n_lines: int, cfg) -> int:
    return min(
        int(cfg.vlm.max_new_tokens_cap),
        int(cfg.vlm.max_new_tokens_base) + int(n_lines) * int(cfg.vlm.max_new_tokens_per_line),
    )

