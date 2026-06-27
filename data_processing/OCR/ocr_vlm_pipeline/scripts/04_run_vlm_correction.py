from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm.auto import tqdm
from src.config import add_config_arg, load_config
from src.dedup import vlm_cache_key
from src.io_utils import read_table, write_table
from src.shape_utils import json_safe, to_plain_list
from src.vlm_cache import VLMCache
from src.vlm_corrector import VLMCorrector
from src.vlm_prompt import PROMPT_VERSION
from src.vlm_router import choose_vlm_model


def _chunks(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _prepare_job(group: dict, model_id: str) -> dict:
    line_indices = [int(x) for x in to_plain_list(group.get("line_indices"), [])]
    raw_texts = str(group["raw_group_text"]).splitlines()
    raw_lines = [{"line_idx": idx, "raw_text": raw_texts[pos] if pos < len(raw_texts) else ""} for pos, idx in enumerate(line_indices)]
    key = vlm_cache_key(group["raw_group_text"], group.get("crop_phash"), str(group["merged_bbox"]), model_id, PROMPT_VERSION)
    return {"group": group, "model_id": model_id, "raw_lines": raw_lines, "cache_key": key}


def _append_result_rows(rows: list[dict], job: dict, result: dict, parse_status: str) -> None:
    group = job["group"]
    by_idx = {int(line["line_idx"]): line for line in result.get("lines", [])}
    for raw in job["raw_lines"]:
        fixed = by_idx.get(raw["line_idx"], raw)
        rows.append(
            {
                "frame_id": group["frame_id"],
                "group_id": int(group["group_id"]),
                "line_idx": int(raw["line_idx"]),
                "raw_text": raw["raw_text"],
                "corrected_text": fixed.get("corrected_text") or raw["raw_text"],
                "vlm_model": job["model_id"],
                "parse_status": parse_status,
                "parse_error": result.get("parse_error"),
                "raw_response": str(result.get("raw_response") or "")[:4000],
                "cache_key": job["cache_key"],
            }
        )


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    jobs = read_table(Path(cfg.project.output_dir) / "vlm_jobs.parquet")
    if args.limit:
        jobs = jobs.head(args.limit)
    cache = VLMCache(cfg.vlm.cache_path) if cfg.vlm.cache_enabled else None
    correctors = {}
    rows = []
    job_records = jobs.to_dict("records")
    batch_size = max(1, int(cfg.vlm.get("batch_size", 1)))
    print(f"Stage 4 VLM jobs: {len(job_records)}")
    print(f"Stage 4 batch_size: {batch_size}")
    started = time.perf_counter()
    progress = tqdm(total=len(job_records), desc="VLM correction", unit="group")
    for batch in _chunks(job_records, batch_size):
        uncached_by_model = {}
        prepared_jobs = [_prepare_job(group, choose_vlm_model(group, cfg)) for group in batch]
        for job in prepared_jobs:
            result = cache.get(job["cache_key"]) if cache else None
            if result and result.get("parse_status") not in ["ok", "partial"]:
                result = None
            if result:
                _append_result_rows(rows, job, result, "cache_hit")
                progress.update(1)
                continue
            uncached_by_model.setdefault(job["model_id"], []).append(job)

        for model_id, model_jobs in uncached_by_model.items():
            if model_id not in correctors:
                correctors[model_id] = VLMCorrector(model_id, cfg)
            results = correctors[model_id].correct_batch(
                [
                    {"crop_path": job["group"]["crop_path"], "raw_lines": job["raw_lines"], "metadata": job["group"]}
                    for job in model_jobs
                ]
            )
            for job, result in zip(model_jobs, results):
                corrected_text = "\n".join(line.get("corrected_text", "") for line in result.get("lines", []))
                if cache:
                    cache.put(job["cache_key"], job["group"]["raw_group_text"], corrected_text, json_safe(result), model_id, PROMPT_VERSION)
                _append_result_rows(rows, job, result, result.get("parse_status"))
                progress.update(1)
    progress.close()
    import pandas as pd

    print(write_table(pd.DataFrame(rows), Path(cfg.project.output_dir) / "vlm_corrected.parquet"))
    elapsed = time.perf_counter() - started
    print(f"Stage 4 elapsed: {elapsed / 60:.2f} min ({elapsed:.1f} sec)")


if __name__ == "__main__":
    main()
