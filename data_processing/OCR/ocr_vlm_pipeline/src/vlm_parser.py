from __future__ import annotations

import ast
import json
import re


def parse_vlm_json(text: str, expected_line_indices: list[int]) -> dict:
    raw = text or ""
    try:
        payload = _load_payload(raw)
    except ValueError as exc:
        return {"parse_status": "json_error", "parse_error": str(exc), "lines": []}
    if isinstance(payload, list):
        payload = {"lines": payload}
    if isinstance(payload, dict) and "line_idx" in payload:
        payload = {"lines": [payload]}
    if not isinstance(payload, dict):
        return {"parse_status": "schema_error", "parse_error": "payload is not an object", "lines": []}
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


def _load_payload(text: str):
    candidates = _json_candidates(text)
    errors = []
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(normalized)
            except Exception as exc:
                errors.append(f"{parser.__name__}: {exc}")
    message = "; ".join(errors[-3:]) if errors else "no JSON object or array found"
    raise ValueError(message)


def _json_candidates(text: str) -> list[str]:
    cleaned = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    candidates = [item.strip() for item in fenced if item.strip()]
    candidates.append(cleaned)
    for opener, closer in [("{", "}"), ("[", "]")]:
        extracted = _extract_balanced(cleaned, opener, closer)
        if extracted:
            candidates.append(extracted)
    # Keep order but remove duplicates.
    unique = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in ["'", '"']:
            in_string = True
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_candidate(candidate: str) -> str:
    normalized = candidate.strip()
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    return normalized
