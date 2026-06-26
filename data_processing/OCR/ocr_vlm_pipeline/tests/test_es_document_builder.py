from src.es_document_builder import build_documents


def test_build_documents_has_nested_ocr_lines():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "video_id": "V",
                "frame_id": "001",
                "frame_number": 1,
                "frame_path": "001.jpg",
                "ocr_raw_text": "Nguyn",
                "ocr_corrected_text": "Nguyễn",
                "ocr_normalized_text": "nguyen",
                "line_count": 1,
                "min_confidence": 0.9,
                "avg_confidence": 0.9,
                "vlm_corrected": True,
                "vlm_model_ids": ["m"],
            }
        ]
    )
    lines = pd.DataFrame(
        [
            {
                "frame_id": "001",
                "line_idx": 0,
                "group_id": 0,
                "region_type": "scene_text",
                "bbox": [0, 0, 10, 10],
                "poly": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "ocr_text": "Nguyn",
                "corrected_text": "Nguyễn",
                "normalized_text": "nguyen",
                "confidence": 0.9,
                "risk_score": 0.5,
                "need_vlm_line": True,
                "vlm_corrected": True,
                "text_changed": True,
            }
        ]
    )
    docs = build_documents(frame, lines)
    assert docs[0]["ocr"]["lines"][0]["corrected_text"] == "Nguyễn"
    assert "objects" not in docs[0]
    assert "colors" not in docs[0]
