from __future__ import annotations

import re


VIETNAMESE_HINTS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def has_suspicious_chars(text: str, suspicious_chars: list[str]) -> bool:
    return any(ch in (text or "") for ch in suspicious_chars)


def is_noise_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    if len(text) <= 2 and not text.isalnum():
        return True
    return bool(re.fullmatch(r"[\W_]+", text))


def has_vietnamese_spelling_risk(text: str) -> bool:
    text = text or ""
    if not text.strip():
        return False
    words = re.findall(r"[A-Za-zÀ-ỹ]+", text)
    if not words:
        return False
    has_letters = any(ch.isalpha() for ch in text)
    has_vietnamese = any(ch.lower() in VIETNAMESE_HINTS for ch in text)
    likely_vietnamese_plain = any(len(w) >= 4 for w in words) and not has_vietnamese
    odd_mojibake = bool(re.search(r"[ÄËÅÂÃ][A-Za-zÀ-ỹ]?", text))
    return has_letters and (likely_vietnamese_plain or odd_mojibake)


def compute_ocr_risk(row: dict, cfg) -> tuple[float, list[str]]:
    text = row.get("ocr_text") or row.get("raw_text") or ""
    conf = float(row.get("confidence") or 0.0)
    reasons: list[str] = []
    risk = 0.0
    if conf < cfg.risk.confidence_low:
        risk += 0.45
        reasons.append("low_confidence")
    elif conf < cfg.risk.confidence_medium:
        risk += 0.20
        reasons.append("medium_confidence")
    if has_suspicious_chars(text, list(cfg.risk.suspicious_chars)):
        risk += 0.35
        reasons.append("suspicious_chars")
    if cfg.risk.get("enable_vietnamese_spell_rules", True) and has_vietnamese_spelling_risk(text):
        risk += 0.25
        reasons.append("vietnamese_spelling_risk")
    if is_noise_text(text):
        risk += 0.15
        reasons.append("noise_text")
    return min(risk, 1.0), reasons


def score_ocr_dataframe(df, cfg):
    scored = df.copy()
    risks = [compute_ocr_risk(row, cfg) for row in scored.to_dict("records")]
    scored["risk_score"] = [item[0] for item in risks]
    scored["risk_reasons"] = [item[1] for item in risks]
    scored["need_vlm_line"] = scored["risk_score"] >= float(cfg.risk.vlm_risk_threshold)
    return scored

