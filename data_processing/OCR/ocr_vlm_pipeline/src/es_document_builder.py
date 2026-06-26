from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .io_utils import read_table
from .shape_utils import normalize_bbox, normalize_poly


def build_documents(frame_summary, line_df):
    docs = []
    for frame in frame_summary.to_dict("records"):
        frame_id = frame["frame_id"]
        lines = line_df[line_df["frame_id"] == frame_id]
        docs.append(
            {
                "video_id": frame["video_id"],
                "frame_id": frame_id,
                "frame_number": int(frame["frame_number"]),
                "timestamp_ms": int(frame["frame_number"]) * 40,
                "frame_path": frame["frame_path"],
                "image": {"width": None, "height": None, "phash": None, "scene_id": None},
                "ocr": {
                    "raw_text": frame["ocr_raw_text"],
                    "corrected_text": frame["ocr_corrected_text"],
                    "normalized_text": frame["ocr_normalized_text"],
                    "has_text": bool(frame["line_count"]),
                    "line_count": int(frame["line_count"]),
                    "min_confidence": float(frame["min_confidence"]),
                    "avg_confidence": float(frame["avg_confidence"]),
                    "vlm_corrected": bool(frame["vlm_corrected"]),
                    "vlm_model_ids": frame.get("vlm_model_ids") or [],
                    "ocr_engine": "PP-OCRv6",
                    "lines": [_line_doc(row) for row in lines.to_dict("records")],
                },
                "quality": {"blur_score": None, "brightness": None, "is_duplicate": False},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return docs


def _line_doc(row: dict) -> dict:
    return {
        "line_id": int(row["line_idx"]),
        "group_id": int(row["group_id"]) if row.get("group_id") is not None else None,
        "region_type": row.get("region_type", "unknown"),
        "bbox": normalize_bbox(row.get("bbox"), []),
        "poly": normalize_poly(row.get("poly")),
        "raw_text": row.get("ocr_text") or "",
        "corrected_text": row.get("corrected_text") or row.get("ocr_text") or "",
        "normalized_text": row.get("normalized_text") or "",
        "confidence": float(row.get("confidence") or 0.0),
        "risk_score": float(row.get("risk_score") or 0.0),
        "need_vlm": bool(row.get("need_vlm_line")),
        "vlm_corrected": bool(row.get("vlm_corrected")),
        "text_changed": bool(row.get("text_changed")),
    }


def run_build_es_documents(cfg) -> Path:
    import json

    out = Path(cfg.project.output_dir) / "es_documents.jsonl"
    frame_summary = read_table(Path(cfg.project.output_dir) / "ocr_frame_summary.parquet")
    lines = read_table(Path(cfg.project.output_dir) / "ocr_merged_lines.parquet")
    docs = build_documents(frame_summary, lines)
    with out.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return out
