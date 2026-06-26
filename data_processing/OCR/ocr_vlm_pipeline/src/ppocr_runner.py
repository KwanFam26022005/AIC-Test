from __future__ import annotations

import time
from pathlib import Path

from .io_utils import append_jsonl, read_table, write_table
from .ppocr_engine import create_ppocr_engine, normalize_ppocr_result


def run_ppocr(cfg) -> Path:
    import pandas as pd

    out = Path(cfg.project.output_dir) / "ppocr_raw.parquet"
    if out.exists() and cfg.project.get("resume", True):
        return out
    registry = read_table(Path(cfg.project.output_dir) / "frame_registry.parquet")
    engine = create_ppocr_engine(cfg)
    rows = []
    failed = []
    for frame in registry[registry["source_status"] == "ok"].to_dict("records"):
        start = time.perf_counter()
        try:
            result = engine.predict(frame["frame_path"])
            runtime_ms = (time.perf_counter() - start) * 1000
            for line in normalize_ppocr_result(result):
                rows.append({**frame, **line, "ocr_engine": cfg.ppocr.ocr_version, "runtime_ms": runtime_ms})
        except Exception as exc:
            failed.append({"frame_id": frame["frame_id"], "frame_path": frame["frame_path"], "error": repr(exc)})
    if failed:
        append_jsonl(Path(cfg.project.output_dir) / "logs" / "failed_jobs.jsonl", failed)
    return write_table(pd.DataFrame(rows), out)

