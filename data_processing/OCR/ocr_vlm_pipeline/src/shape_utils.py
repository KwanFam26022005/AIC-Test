from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def to_plain_list(value: Any, default: list | None = None) -> list:
    if value is None:
        return list(default or [])
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        import pandas as pd

        if pd.isna(value):
            return list(default or [])
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return list(default or [])


def normalize_bbox(value: Any, default: list[int] | None = None) -> list[int]:
    default = [0, 0, 0, 0] if default is None else default
    bbox = to_plain_list(value, default)
    if len(bbox) < 4:
        return list(default)
    return [int(float(coord)) for coord in bbox[:4]]


def normalize_poly(value: Any) -> list[list[int]]:
    poly = to_plain_list(value, [])
    normalized = []
    for point in poly:
        coords = to_plain_list(point, [])
        if len(coords) >= 2:
            normalized.append([int(float(coords[0])), int(float(coords[1]))])
    return normalized
