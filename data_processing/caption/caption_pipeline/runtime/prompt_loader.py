"""Load, render, and fingerprint caption prompt templates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class RenderedPrompt:
    task: str
    text: str
    path: str
    sha256: str


class PromptRegistry:
    """Resolve task prompt files from a single prompt directory."""

    DEFAULT_FILES = {
        "frame": "frame_caption_prompt.txt",
        "shot": "shot_caption_prompt.txt",
        "trake": "trake_event_step_prompt.txt",
    }

    def __init__(self, prompt_dir: str | Path, files: dict[str, str] | None = None) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.files = dict(self.DEFAULT_FILES)
        self.files.update(files or {})
        self._cache: dict[str, tuple[str, str, str]] = {}

    def render(self, task: str, values: dict[str, Any]) -> RenderedPrompt:
        text, path, digest = self._load(task)
        missing = sorted(set(_PLACEHOLDER.findall(text)) - set(values))
        if missing:
            raise ValueError(f"Prompt '{task}' is missing values for: {', '.join(missing)}")

        def replace(match: re.Match[str]) -> str:
            return str(values.get(match.group(1), ""))

        rendered = _PLACEHOLDER.sub(replace, text)
        return RenderedPrompt(task=task, text=rendered, path=path, sha256=digest)

    def fingerprint(self, task: str) -> str:
        return self._load(task)[2]

    def _load(self, task: str) -> tuple[str, str, str]:
        if task in self._cache:
            return self._cache[task]
        if task not in self.files:
            raise ValueError(f"Unknown prompt task: {task}")
        path = self.prompt_dir / self.files[task]
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        item = (text, str(path), digest)
        self._cache[task] = item
        return item
