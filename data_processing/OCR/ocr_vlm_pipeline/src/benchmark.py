from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from .io_utils import read_table


DEFAULT_BENCHMARK_MODELS = [
    "5CD-AI/Vintern-3B-beta",
    "5CD-AI/Vintern-3B-R-beta",
    "erax-ai/EraX-VL-2B-V1.5",
]


def benchmark_models(cfg, models: list[str], sample_groups: int, output: str | Path) -> Path:
    from .vlm_corrector import VLMCorrector

    groups = read_table(Path(cfg.project.output_dir) / "ocr_groups.parquet")
    sample = groups[groups["need_vlm_group"]].head(sample_groups).to_dict("records")
    rows = []
    for model_id in models:
        corrector = VLMCorrector(model_id, cfg)
        latencies = []
        parse_ok = 0
        text_changed = 0
        for group in sample:
            raw_lines = [{"line_idx": idx, "raw_text": text} for idx, text in zip(group["line_indices"], str(group["raw_group_text"]).splitlines())]
            start = time.perf_counter()
            result = corrector.correct_group(group["crop_path"], raw_lines, group)
            latencies.append(time.perf_counter() - start)
            parse_ok += int(result["parse_status"] in ["ok", "partial"])
            text_changed += int(any(line["raw_text"] != line["corrected_text"] for line in result.get("lines", [])))
        rows.append(_metric_row(model_id, latencies, len(sample), parse_ok, text_changed))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model_id"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def _metric_row(model_id: str, latencies: list[float], num_groups: int, parse_ok: int, text_changed: int) -> dict:
    if not latencies:
        latencies = [0.0]
    return {
        "model_id": model_id,
        "num_groups": num_groups,
        "avg_latency_sec_per_group": sum(latencies) / len(latencies),
        "p50_latency": statistics.median(latencies),
        "p95_latency": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "gpu_memory_peak_gb": None,
        "parse_success_rate": parse_ok / max(1, num_groups),
        "json_valid_rate": parse_ok / max(1, num_groups),
        "text_change_rate": text_changed / max(1, num_groups),
        "manual_accuracy_score": None,
    }

