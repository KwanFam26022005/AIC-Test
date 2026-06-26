from src.config import to_config
from src.risk_scoring import compute_ocr_risk


def _cfg():
    return to_config(
        {
            "risk": {
                "confidence_low": 0.75,
                "confidence_medium": 0.90,
                "suspicious_chars": ["Ä", "Ë", "�"],
                "enable_vietnamese_spell_rules": True,
                "vlm_risk_threshold": 0.45,
            }
        }
    )


def test_high_confidence_mojibake_is_flagged():
    risk, reasons = compute_ocr_risk({"ocr_text": "LUYÊN DÊ NGH! LUÂN VÄN HOC", "confidence": 0.96}, _cfg())
    assert risk >= 0.35
    assert "suspicious_chars" in reasons


def test_low_confidence_is_flagged():
    risk, reasons = compute_ocr_risk({"ocr_text": "abc", "confidence": 0.5}, _cfg())
    assert risk >= 0.45
    assert "low_confidence" in reasons

