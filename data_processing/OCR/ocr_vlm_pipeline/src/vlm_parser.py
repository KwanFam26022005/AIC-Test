from __future__ import annotations

import json
import re


def parse_vlm_json(text: str, expected_line_indices: list[int]) -> dict:
    raw = text or ""
    candidate = raw.strip()
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        candidate = match.group(0) if match else candidate
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"parse_status": "json_error", "parse_error": str(exc), "lines": []}
    lines = payload.get("lines")
    if not isinstance(lines, list):
        return {"parse_status": "schema_error", "parse_error": "missing lines list", "lines": []}
    parsed = []
    expected = set(int(idx) for idx in expected_line_indices)
    for item in lines:
        if not isinstance(item, dict) or "line_idx" not in item:
            continue
        idx = int(item["line_idx"])
        if idx not in expected:
            continue
        parsed.append(
            {
                "line_idx": idx,
                "raw_text": str(item.get("raw_text") or ""),
                "corrected_text": str(item.get("corrected_text") or item.get("raw_text") or ""),
                "confidence_note": str(item.get("confidence_note") or "medium"),
            }
        )
    status = "ok" if len(parsed) == len(expected) else "partial"
    return {"parse_status": status, "parse_error": None, "lines": parsed}

