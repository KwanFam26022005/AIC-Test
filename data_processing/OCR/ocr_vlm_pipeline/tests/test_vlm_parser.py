from src.vlm_parser import parse_vlm_json


def test_parse_json_inside_model_text():
    result = parse_vlm_json('extra {"lines":[{"line_idx":0,"raw_text":"Nguyn","corrected_text":"Nguyễn"}]}', [0])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "Nguyễn"


def test_invalid_json_returns_error():
    result = parse_vlm_json("not json", [0])
    assert result["parse_status"] == "json_error"

