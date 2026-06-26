from __future__ import annotations


def choose_vlm_model(group: dict, cfg) -> str:
    if not cfg.vlm.get("allow_model_switch", True):
        return cfg.vlm.model_id
    if group.get("region_type") in ["table", "document_block", "complex_signboard"]:
        return cfg.vlm.alternative_model_id
    if float(group.get("max_risk_score") or 0.0) >= 0.85:
        return cfg.vlm.alternative_model_id
    return cfg.vlm.model_id

