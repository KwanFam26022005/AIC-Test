from src.config import to_config
from src.grouping import group_ocr_lines


def test_grouping_keeps_nearby_lines_together():
    import pandas as pd

    cfg = to_config({"grouping": {"vertical_gap": 15, "horizontal_gap": 15}, "risk": {"vlm_risk_threshold": 0.45}})
    df = pd.DataFrame(
        [
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 0, "bbox": [0, 0, 100, 20], "ocr_text": "A", "confidence": 0.9, "risk_score": 0.5},
            {"video_id": "V", "frame_id": "001", "frame_number": 1, "frame_path": "x.jpg", "line_idx": 1, "bbox": [0, 25, 100, 45], "ocr_text": "B", "confidence": 0.9, "risk_score": 0.5},
        ]
    )
    groups = group_ocr_lines(df, cfg)
    assert len(groups) == 1
    assert groups.iloc[0]["line_indices"] == [0, 1]

