from src.config import to_config
from src.grouping import classify_region, group_ocr_lines


def _grouping_cfg(**overrides):
    grouping = {
        "vertical_gap": 15,
        "horizontal_gap": 15,
        "iou_threshold": 0.05,
        "min_horizontal_overlap": 0.25,
        "min_vertical_overlap": 0.25,
        "lower_third_y_start_ratio": 0.55,
        "lower_third_horizontal_gap": 180,
        "lower_third_vertical_gap": 50,
        "lower_third_min_vertical_overlap": 0.12,
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


def test_classify_region_uses_broadcast_overlay_geometry():
    image_width = 1280
    image_height = 720

    assert classify_region(
        {"raw_group_text": "BBo", "merged_bbox": [281, 26, 457, 136], "line_indices": [0]},
        image_width=image_width,
        image_height=image_height,
    ) == "logo"
    assert classify_region(
        {"raw_group_text": "HTV9D\nHD\n06:50:05", "merged_bbox": [1039, 48, 1179, 115], "line_indices": [1, 2, 3]},
        image_width=image_width,
        image_height=image_height,
    ) == "timestamp"
    assert classify_region(
        {
            "raw_group_text": "DAN MACH: KHUYEN KHICH DU KHACH\nTHAM GIA LAM SACH MOI TRUONG",
            "merged_bbox": [386, 581, 1238, 694],
            "line_indices": [5, 7],
        },
        image_width=image_width,
        image_height=image_height,
    ) == "lower_third"


def test_lower_third_groups_are_sent_to_vlm_but_channel_overlays_are_not():
    import pandas as pd

    cfg = _grouping_cfg()
    df = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 0,
                "bbox": [281, 26, 457, 136],
                "ocr_text": "BBo",
                "confidence": 0.49,
                "risk_score": 0.45,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 1,
                "bbox": [1039, 48, 1179, 115],
                "ocr_text": "06:50:05",
                "confidence": 0.99,
                "risk_score": 0.0,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 2,
                "bbox": [386, 581, 1238, 612],
                "ocr_text": "DAN MACH: KHUYEN KHICH DU KHACH",
                "confidence": 0.99,
                "risk_score": 0.0,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 3,
                "bbox": [386, 622, 1238, 650],
                "ocr_text": "THAM GIA LAM SACH MOI TRUONG",
                "confidence": 0.99,
                "risk_score": 0.0,
            },
        ]
    )

    groups = group_ocr_lines(df, cfg)
    by_region = {row["region_type"]: row for row in groups.to_dict("records")}
    assert not by_region["logo"]["need_vlm_group"]
    assert not by_region["timestamp"]["need_vlm_group"]
    assert by_region["lower_third"]["need_vlm_group"]


def test_lower_third_side_by_side_blocks_are_merged_before_crop():
    import pandas as pd

    cfg = _grouping_cfg()
    df = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 5,
                "bbox": [386, 581, 1238, 612],
                "ocr_text": "DAN MACH: KHUYEN KHICH DU KHACH",
                "confidence": 0.99,
                "risk_score": 0.0,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 6,
                "bbox": [0, 606, 266, 642],
                "ocr_text": "giay",
                "confidence": 0.88,
                "risk_score": 0.2,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 7,
                "bbox": [386, 622, 1238, 650],
                "ocr_text": "THAM GIA LAM SACH MOI TRUONG",
                "confidence": 0.99,
                "risk_score": 0.0,
            },
            {
                "video_id": "V",
                "frame_id": "236",
                "frame_number": 236,
                "frame_path": "x.jpg",
                "width": 1280,
                "height": 720,
                "line_idx": 8,
                "bbox": [0, 656, 266, 692],
                "ocr_text": "n xe chay qua toc do",
                "confidence": 0.88,
                "risk_score": 0.2,
            },
        ]
    )

    groups = group_ocr_lines(df, cfg)
    assert len(groups) == 1
    group = groups.iloc[0]
    assert group["region_type"] == "lower_third"
    assert group["need_vlm_group"]
    assert group["line_indices"] == [5, 6, 7, 8]
    assert group["merged_bbox"] == [0, 581, 1238, 692]


def test_grouping_on_requested_frame_from_parquet():
    """Optional manual test.

    Example:
        OCR_GROUPING_CONFIG=configs/server_keyframe_test.yaml \
        OCR_GROUPING_FRAME_ID=017 \
        python -m pytest tests/test_grouping.py::test_grouping_on_requested_frame_from_parquet -s

    Optional:
        OCR_GROUPING_INPUT=outputs/L21_V001/ocr_risk.parquet
        OCR_GROUPING_CROP_DIR=outputs/L21_V001/debug_grouping_crops/frame_017
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

    crop_index = _save_debug_group_crops(groups, cfg)
    print(f"\nDebug crops: {crop_index['crop_dir']}")
    print(crop_index["rows"][["group_id", "line_indices", "need_vlm_group", "merged_bbox", "crop_box", "crop_path"]].to_string(index=False))


def _save_debug_group_crops(groups, cfg):
    import json
    import os
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from src.io_utils import ensure_dir
    from src.shape_utils import json_safe, normalize_bbox

    frame_id = str(groups.iloc[0]["frame_id"])
    crop_dir_env = os.environ.get("OCR_GROUPING_CROP_DIR")
    if crop_dir_env:
        crop_dir = ensure_dir(crop_dir_env)
    else:
        crop_dir = ensure_dir(Path(cfg.project.output_dir) / "debug_grouping_crops" / f"frame_{frame_id}")

    padding = int(cfg.grouping.get("crop_padding", 20))
    rows = []
    for row in groups.to_dict("records"):
        frame_path = Path(str(row["frame_path"]))
        if not frame_path.exists():
            frame_path = Path.cwd() / frame_path
        if not frame_path.exists():
            rows.append({
                **row,
                "crop_box": None,
                "crop_path": None,
                "crop_size": None,
                "crop_error": f"missing frame image: {row['frame_path']}",
            })
            continue

        with Image.open(frame_path) as img:
            width, height = img.size
            x1, y1, x2, y2 = normalize_bbox(row.get("merged_bbox"))
            crop_box = [
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(width, x2 + padding),
                min(height, y2 + padding),
            ]
            crop = img.crop(tuple(crop_box))
            crop_path = crop_dir / f"{frame_id}_g{int(row['group_id']):03d}.jpg"
            crop.save(crop_path, quality=95)
            rows.append({**row, "crop_box": crop_box, "crop_path": str(crop_path), "crop_size": list(crop.size)})

    index_df = pd.DataFrame(rows)
    index_path = crop_dir / "crops_index.json"
    index_path.write_text(json.dumps(json_safe(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"crop_dir": crop_dir, "index_path": index_path, "rows": index_df}
