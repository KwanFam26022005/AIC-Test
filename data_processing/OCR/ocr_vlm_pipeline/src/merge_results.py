from __future__ import annotations

from pathlib import Path

from .grouping import line_to_group_map, line_to_region_map
from .io_utils import read_table, write_table
from .normalize_text import normalize_for_search


def merge_ocr_results(raw_df, risk_df, groups_df, corrected_df=None):
    merged = raw_df.copy()
    risk_cols = ["frame_id", "line_idx", "risk_score", "need_vlm_line", "risk_reasons"]
    merged = merged.merge(risk_df[risk_cols], on=["frame_id", "line_idx"], how="left")
    group_map = line_to_group_map(groups_df)
    region_map = line_to_region_map(groups_df)
    merged["group_id"] = [group_map.get((row.frame_id, int(row.line_idx))) for row in merged.itertuples()]
    merged["region_type"] = [region_map.get((row.frame_id, int(row.line_idx)), "unknown") for row in merged.itertuples()]
    merged["corrected_text"] = merged["ocr_text"]
    merged["vlm_corrected"] = False
    merged["vlm_model"] = None
    merged["parse_status"] = None
    if corrected_df is not None and len(corrected_df):
        for row in corrected_df.to_dict("records"):
            mask = (merged["frame_id"] == row["frame_id"]) & (merged["line_idx"] == int(row["line_idx"]))
            merged.loc[mask, "corrected_text"] = row.get("corrected_text") or merged.loc[mask, "ocr_text"]
            merged.loc[mask, "vlm_corrected"] = row.get("parse_status") in ["ok", "partial", "cache_hit"]
            merged.loc[mask, "vlm_model"] = row.get("vlm_model")
            merged.loc[mask, "parse_status"] = row.get("parse_status")
    merged["normalized_text"] = merged["corrected_text"].map(normalize_for_search)
    merged["text_changed"] = merged["corrected_text"].fillna("") != merged["ocr_text"].fillna("")
    return merged


def build_frame_summary(line_df):
    import pandas as pd

    rows = []
    for frame_id, frame_df in line_df.groupby("frame_id", sort=False):
        first = frame_df.iloc[0].to_dict()
        rows.append(
            {
                "video_id": first["video_id"],
                "frame_id": frame_id,
                "frame_number": int(first["frame_number"]),
                "frame_path": first["frame_path"],
                "ocr_raw_text": "\n".join(frame_df["ocr_text"].fillna("").astype(str)),
                "ocr_corrected_text": "\n".join(frame_df["corrected_text"].fillna("").astype(str)),
                "ocr_normalized_text": "\n".join(frame_df["normalized_text"].fillna("").astype(str)),
                "line_count": int(len(frame_df)),
                "min_confidence": float(frame_df["confidence"].min()),
                "avg_confidence": float(frame_df["confidence"].mean()),
                "vlm_corrected": bool(frame_df["vlm_corrected"].any()),
                "vlm_model_ids": sorted(set(x for x in frame_df["vlm_model"].dropna().astype(str))),
            }
        )
    return pd.DataFrame(rows)


def run_merge(cfg) -> Path:
    raw = read_table(Path(cfg.project.output_dir) / "ppocr_raw.parquet")
    risk = read_table(Path(cfg.project.output_dir) / "ocr_risk.parquet")
    groups = read_table(Path(cfg.project.output_dir) / "ocr_groups.parquet")
    corrected_path = Path(cfg.project.output_dir) / "vlm_corrected.parquet"
    corrected = read_table(corrected_path) if corrected_path.exists() else None
    merged = merge_ocr_results(raw, risk, groups, corrected)
    write_table(merged, Path(cfg.project.output_dir) / "ocr_merged_lines.parquet")
    summary = build_frame_summary(merged)
    return write_table(summary, Path(cfg.project.output_dir) / "ocr_frame_summary.parquet")
