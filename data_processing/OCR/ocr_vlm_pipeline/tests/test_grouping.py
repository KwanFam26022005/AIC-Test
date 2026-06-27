from src.config import to_config
from src.grouping import group_ocr_lines


def _grouping_cfg(**overrides):
    grouping = {
        "vertical_gap": 15,
        "horizontal_gap": 15,
        "iou_threshold": 0.05,
        "min_horizontal_overlap": 0.25,
        "min_vertical_overlap": 0.25,
    }
    grouping.update(overrides)
    return to_config({"grouping": grouping, "risk": {"vlm_risk_threshold": 0.45}})


def test_grouping_keeps_nearby_lines_together():
    import pandas as pd

    cfg = _grouping_cfg()
    df = pd.DataFrame(
        [
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 0, "bbox": [0, 0, 100, 20], "ocr_text": "A", "confidence": 0.9, "risk_score": 0.5},
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 1, "bbox": [0, 25, 100, 45], "ocr_text": "B", "confidence": 0.9, "risk_score": 0.5},
        ]
    )
    groups = group_ocr_lines(df, cfg)
    assert len(groups) == 1
    assert groups.iloc[0]["line_indices"] == [0, 1]


def test_grouping_accepts_numpy_bbox_from_parquet():
    import numpy as np
    import pandas as pd

    cfg = _grouping_cfg()
    df = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": "001",
                "frame_number": 1,
                "frame_path": "x.jpg",
                "line_idx": 0,
                "bbox": np.array([0, 0, 100, 20]),
                "ocr_text": "A",
                "confidence": 0.9,
                "risk_score": 0.5,
            }
        ]
    )
    groups = group_ocr_lines(df, cfg)
    assert groups.iloc[0]["merged_bbox"] == [0, 0, 100, 20]


def test_grouping_links_stacked_boxes_by_vertical_gap_and_x_overlap():
    import pandas as pd

    cfg = _grouping_cfg(vertical_gap=12)
    df = pd.DataFrame(
        [
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 0, "bbox": [100, 10, 220, 30], "ocr_text": "A", "confidence": 0.9, "risk_score": 0.5},
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 1, "bbox": [102, 39, 218, 60], "ocr_text": "B", "confidence": 0.9, "risk_score": 0.5},
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 2, "bbox": [360, 10, 450, 30], "ocr_text": "C", "confidence": 0.9, "risk_score": 0.5},
        ]
    )
    groups = group_ocr_lines(df, cfg)
    assert len(groups) == 2
    assert groups.iloc[0]["line_indices"] == [0, 1]
    assert groups.iloc[0]["merged_bbox"] == [100, 10, 220, 60]


def test_grouping_links_overlapping_boxes_by_iou():
    import pandas as pd

    cfg = _grouping_cfg(vertical_gap=0, horizontal_gap=0, min_horizontal_overlap=0.9, min_vertical_overlap=0.9)
    df = pd.DataFrame(
        [
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 0, "bbox": [10, 10, 100, 50], "ocr_text": "A", "confidence": 0.9, "risk_score": 0.5},
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 1, "bbox": [20, 20, 110, 60], "ocr_text": "B", "confidence": 0.9, "risk_score": 0.5},
        ]
    )
    groups = group_ocr_lines(df, cfg)
    assert len(groups) == 1
    assert groups.iloc[0]["merged_bbox"] == [10, 10, 110, 60]


def test_grouping_on_requested_frame_from_parquet():
    """Optional manual test.

    Example:
        OCR_GROUPING_CONFIG=configs/server_keyframe_test.yaml \
        OCR_GROUPING_FRAME_ID=017 \
        python -m pytest tests/test_grouping.py::test_grouping_on_requested_frame_from_parquet -s

    Optional:
        OCR_GROUPING_INPUT=outputs/L21_V001/ocr_risk.parquet
    """
    import os
    from pathlib import Path

    import pandas as pd
    import pytest

    from src.config import load_config

    frame_id = os.environ.get("OCR_GROUPING_FRAME_ID")
    frame_number = os.environ.get("OCR_GROUPING_FRAME_NUMBER")
    config_path = os.environ.get("OCR_GROUPING_CONFIG")
    if not config_path or (not frame_id and not frame_number):
        pytest.skip("Set OCR_GROUPING_CONFIG and OCR_GROUPING_FRAME_ID or OCR_GROUPING_FRAME_NUMBER to run this manual frame test.")

    cfg = load_config(config_path)
    input_path = os.environ.get("OCR_GROUPING_INPUT")
    if input_path is None:
        out_dir = Path(cfg.project.output_dir)
        input_path = out_dir / "ocr_risk.parquet"
        if not input_path.exists():
            input_path = out_dir / "ppocr_raw.parquet"

    df = pd.read_parquet(input_path)
    if frame_id:
        frame_df = df[df["frame_id"].astype(str) == str(frame_id)].copy()
    else:
        frame_df = df[df["frame_number"].astype(int) == int(frame_number)].copy()

    assert not frame_df.empty, f"No OCR rows found for frame_id={frame_id!r}, frame_number={frame_number!r} in {input_path}"

    groups = group_ocr_lines(frame_df, cfg)
    assert not groups.empty

    print(f"\nFrame rows: {len(frame_df)}")
    print(f"Groups: {len(groups)}")
    print(groups[["frame_id", "group_id", "line_indices", "merged_bbox", "raw_group_text", "need_vlm_group", "region_type"]].to_string(index=False))
