from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path):
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return path


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

