from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.io_utils import ensure_dir, read_table
from src.shape_utils import json_safe


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser(description="Summarize OCR/VLM results after stage 5."))
    parser.add_argument("--summary-json", default=None, help="Where to save summary stats as JSON.")
    parser.add_argument("--top", type=int, default=15, help="Number of example rows to print.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.project.output_dir)
    merged_path = out_dir / "ocr_merged_lines.parquet"
    summary_path = out_dir / "ocr_frame_summary.parquet"
    if not merged_path.exists() or not summary_path.exists():
        raise SystemExit(
            "Stage 5 outputs are missing. Run scripts/05_merge_features.py first. "
            f"Expected {merged_path} and {summary_path}."
        )

    merged = read_table(merged_path)
    summary = read_table(summary_path)
    stats = build_stats(out_dir, merged, summary)
    print_stats(stats)
    print_examples(merged, args.top)

    summary_json = Path(args.summary_json) if args.summary_json else out_dir / "qa_summary.json"
    ensure_dir(summary_json.parent)
    summary_json.write_text(json.dumps(json_safe(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON: {summary_json}")


def build_stats(out_dir: Path, merged, summary) -> dict:
    stats = {
        "output_dir": str(out_dir),
        "num_frames_with_ocr": int(summary["frame_id"].nunique()) if "frame_id" in summary else int(len(summary)),
        "num_ocr_lines": int(len(merged)),
        "avg_lines_per_frame": float(summary["line_count"].mean()) if len(summary) else 0.0,
        "avg_confidence": float(merged["confidence"].mean()) if "confidence" in merged and len(merged) else 0.0,
        "min_confidence": float(merged["confidence"].min()) if "confidence" in merged and len(merged) else 0.0,
        "vlm_corrected_lines": int(merged.get("vlm_corrected", []).sum()) if "vlm_corrected" in merged else 0,
        "text_changed_lines": int(merged.get("text_changed", []).sum()) if "text_changed" in merged else 0,
    }
    stats["text_changed_rate"] = stats["text_changed_lines"] / max(1, stats["num_ocr_lines"])
    if "parse_status" in merged:
        stats["parse_status_counts"] = _value_counts(merged["parse_status"])
    if "vlm_model" in merged:
        stats["vlm_model_counts"] = _value_counts(merged["vlm_model"])
    if "region_type" in merged:
        stats["region_type_counts"] = _value_counts(merged["region_type"])
    if "risk_score" in merged:
        stats["avg_risk_score"] = float(merged["risk_score"].fillna(0).mean())
        stats["max_risk_score"] = float(merged["risk_score"].fillna(0).max())

    for name in [
        "frame_registry.parquet",
        "ppocr_raw.parquet",
        "ocr_risk.parquet",
        "ocr_groups.parquet",
        "vlm_jobs.parquet",
        "vlm_corrected.parquet",
    ]:
        path = out_dir / name
        if path.exists():
            try:
                stats[f"rows_{path.stem}"] = int(len(read_table(path)))
            except Exception:
                stats[f"rows_{path.stem}"] = None
    return stats


def print_stats(stats: dict) -> None:
    print("\n=== OCR/VLM QA Summary ===")
    print(f"Output dir:          {stats['output_dir']}")
    print(f"Frames with OCR:     {stats['num_frames_with_ocr']:,}")
    print(f"OCR lines:           {stats['num_ocr_lines']:,}")
    print(f"Avg lines/frame:     {stats['avg_lines_per_frame']:.2f}")
    print(f"Confidence avg/min:  {stats['avg_confidence']:.3f} / {stats['min_confidence']:.3f}")
    print(f"VLM corrected lines: {stats['vlm_corrected_lines']:,}")
    print(f"Text changed lines:  {stats['text_changed_lines']:,} ({stats['text_changed_rate'] * 100:.1f}%)")
    if "avg_risk_score" in stats:
        print(f"Risk avg/max:        {stats['avg_risk_score']:.3f} / {stats['max_risk_score']:.3f}")
    _print_counts("Parse status", stats.get("parse_status_counts"))
    _print_counts("VLM models", stats.get("vlm_model_counts"))
    _print_counts("Region types", stats.get("region_type_counts"))


def print_examples(merged, top: int) -> None:
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1400)
    cols = [
        col
        for col in [
            "frame_id",
            "line_idx",
            "ocr_text",
            "corrected_text",
            "confidence",
            "risk_score",
            "vlm_corrected",
            "text_changed",
            "parse_status",
        ]
        if col in merged.columns
    ]
    if "text_changed" in merged:
        changed = merged[merged["text_changed"].fillna(False)]
        if len(changed):
            print("\n=== Changed Text Examples ===")
            print(changed[cols].head(top).to_string(index=False))
    if "confidence" in merged and len(merged):
        print("\n=== Lowest Confidence Lines ===")
        print(merged.nsmallest(min(top, len(merged)), "confidence")[cols].to_string(index=False))


def _value_counts(series) -> dict:
    return {str(key): int(value) for key, value in series.fillna("NA").value_counts(dropna=False).to_dict().items()}


def _print_counts(title: str, counts: dict | None) -> None:
    if not counts:
        return
    print(f"\n{title}:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()
