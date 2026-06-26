from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return to_config(value)


def to_config(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config({k: to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_config(v) for v in value]
    return value


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text)


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    cfg = load_yaml(cfg_path)
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(cfg_path.parent / parent)
        cfg = deep_update(base, cfg)
    return to_config(cfg)


def add_config_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    return parser

