from __future__ import annotations

import csv
import gc
import statistics
import time
from pathlib import Path

from .io_utils import read_table
from .shape_utils import to_plain_list


DEFAULT_BENCHMARK_MODELS = [
    "5CD-AI/Vintern-3B-beta",
    "5CD-AI/Vintern-3B-R-beta",
    "erax-ai/EraX-VL-2B-V1.5",
]


def benchmark_models(cfg, models: list[str], sample_groups: int, output: str | Path) -> Path:
    from .vlm_corrector import VLMCorrector

    groups = read_table(Path(cfg.project.output_dir) / "ocr_groups.parquet")
    sample_df = groups[groups["need_vlm_group"]].copy()
    if "crop_path" in sample_df:
        sample_df = sample_df[sample_df["crop_path"].notna()]
        sample_df = sample_df[sample_df["crop_path"].map(lambda path: Path(str(path)).exists())]
    sample = sample_df.head(sample_groups).to_dict("records")
    rows = []
    for model_id in models:
        latencies = []
        parse_ok = 0
        text_changed = 0
        try:
            _reset_gpu_peak_memory()
            corrector = VLMCorrector(model_id, cfg)
            for group in sample:
                raw_lines = _raw_lines_for_group(group)
                start = time.perf_counter()
                result = corrector.correct_group(group["crop_path"], raw_lines, group)
                latencies.append(time.perf_counter() - start)
                parse_ok += int(result["parse_status"] in ["ok", "partial"])
                text_changed += int(any(line["raw_text"] != line["corrected_text"] for line in result.get("lines", [])))
            rows.append(_metric_row(model_id, latencies, len(sample), parse_ok, text_changed, "ok", None, _gpu_peak_gb()))
        except Exception as exc:
            rows.append(_metric_row(model_id, latencies, len(sample), parse_ok, text_changed, "error", repr(exc), _gpu_peak_gb()))
        finally:
            try:
                del corrector
            except UnboundLocalError:
                pass
            gc.collect()
            _empty_cuda_cache()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model_id"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def _raw_lines_for_group(group: dict) -> list[dict]:
    line_indices = [int(idx) for idx in to_plain_list(group.get("line_indices"), [])]
    raw_texts = str(group["raw_group_text"]).splitlines()
    return [
        {"line_idx": idx, "raw_text": raw_texts[pos] if pos < len(raw_texts) else ""}
        for pos, idx in enumerate(line_indices)
    ]


def _metric_row(
    model_id: str,
    latencies: list[float],
    num_groups: int,
    parse_ok: int,
    text_changed: int,
    status: str,
    error_message: str | None,
    gpu_memory_peak_gb: float | None,
) -> dict:
    if not latencies:
        latencies = [0.0]
    return {
        "model_id": model_id,
        "status": status,
        "num_groups": num_groups,
        "avg_latency_sec_per_group": sum(latencies) / len(latencies),
        "p50_latency": statistics.median(latencies),
        "p95_latency": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "gpu_memory_peak_gb": gpu_memory_peak_gb,
        "parse_success_rate": parse_ok / max(1, num_groups),
        "json_valid_rate": parse_ok / max(1, num_groups),
        "text_change_rate": text_changed / max(1, num_groups),
        "manual_accuracy_score": None,
        "error_message": error_message,
    }


def _reset_gpu_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _gpu_peak_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        return None
    return None


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
