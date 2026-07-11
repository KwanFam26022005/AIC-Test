"""Robust JSON extraction and lightweight schema validation for LLM output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable


Validator = Callable[[dict[str, Any]], dict[str, Any]]


def parse_json_object(
    text: str,
    required_fields: tuple[str, ...] = (),
    validator: Validator | None = None,
) -> dict[str, Any]:
    """Extract the first JSON object, check required fields, then normalize it."""
    payload = _first_json_object(text)
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"JSON output is missing fields: {', '.join(missing)}")
    if validator:
        payload = validator(payload)
    return payload


def require_text(payload: dict[str, Any], field: str, max_chars: int) -> str:
    value = payload.get(field, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field}' must be a non-empty string")
    value = " ".join(value.split())
    if len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].rstrip() or value[:max_chars]
    return value


def optional_text(payload: dict[str, Any], field: str, max_chars: int) -> str:
    value = payload.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Field '{field}' must be a string")
    value = " ".join(value.split())
    if len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].rstrip() or value[:max_chars]
    return value


def string_list(payload: dict[str, Any], field: str, max_items: int = 12) -> list[str]:
    value = payload.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field '{field}' must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        clean = " ".join(item.split())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
        if len(result) >= max_items:
            break
    return result


def enum_value(
    payload: dict[str, Any],
    field: str,
    allowed: set[str],
    default: str = "unknown",
    aliases: Mapping[str, str] | None = None,
) -> str:
    value = str(payload.get(field, default) or default).strip().lower()
    if aliases:
        value = aliases.get(value, value)
    if value not in allowed:
        raise ValueError(f"Field '{field}' has invalid value: {value}")
    return value


def _first_json_object(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(clean):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(clean[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a valid JSON object")
