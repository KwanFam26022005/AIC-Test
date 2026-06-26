from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.dedup import vlm_cache_key
from src.io_utils import read_table, write_table
from src.vlm_cache import VLMCache
from src.vlm_corrector import VLMCorrector
from src.vlm_prompt import PROMPT_VERSION
from src.vlm_router import choose_vlm_model


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
    for group in jobs.to_dict("records"):
        model_id = choose_vlm_model(group, cfg)
        line_indices = [int(x) for x in group["line_indices"]]
        raw_texts = str(group["raw_group_text"]).splitlines()
        raw_lines = [{"line_idx": idx, "raw_text": raw_texts[pos] if pos < len(raw_texts) else ""} for pos, idx in enumerate(line_indices)]
        key = vlm_cache_key(group["raw_group_text"], group.get("crop_phash"), str(group["merged_bbox"]), model_id, PROMPT_VERSION)
        result = cache.get(key) if cache else None
        parse_status = "cache_hit" if result else None
        if result is None:
            correctors.setdefault(model_id, VLMCorrector(model_id, cfg))
            result = correctors[model_id].correct_group(group["crop_path"], raw_lines, group)
            corrected_text = "\n".join(line.get("corrected_text", "") for line in result.get("lines", []))
            if cache:
                cache.put(key, group["raw_group_text"], corrected_text, result, model_id, PROMPT_VERSION)
            parse_status = result.get("parse_status")
        by_idx = {int(line["line_idx"]): line for line in result.get("lines", [])}
        for raw in raw_lines:
            fixed = by_idx.get(raw["line_idx"], raw)
            rows.append(
                {
                    "frame_id": group["frame_id"],
                    "group_id": int(group["group_id"]),
                    "line_idx": int(raw["line_idx"]),
                    "raw_text": raw["raw_text"],
                    "corrected_text": fixed.get("corrected_text") or raw["raw_text"],
                    "vlm_model": model_id,
                    "parse_status": parse_status,
                    "cache_key": key,
                }
            )
    import pandas as pd

    print(write_table(pd.DataFrame(rows), Path(cfg.project.output_dir) / "vlm_corrected.parquet"))


if __name__ == "__main__":
    main()

