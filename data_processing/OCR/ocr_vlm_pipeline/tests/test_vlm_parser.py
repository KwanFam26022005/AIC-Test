from src.vlm_parser import parse_vlm_json


def test_parse_json_inside_model_text():
    result = parse_vlm_json('extra {"lines":[{"line_idx":0,"raw_text":"Nguyn","corrected_text":"Nguyễn"}]}', [0])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "Nguyễn"


def test_invalid_json_returns_error():
    result = parse_vlm_json("not json", [0])
    assert result["parse_status"] == "json_error"


def test_parse_json_code_fence():
    result = parse_vlm_json('```json\n{"lines":[{"line_idx":1,"corrected_text":"giây"}]}\n```', [1])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "giây"


def test_parse_root_list():
    result = parse_vlm_json('[{"line_idx":3,"raw_text":"SAT LO","corrected_text":"SẠT LỞ"}]', [3])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "SẠT LỞ"


def test_parse_python_style_dict():
    result = parse_vlm_json("{'lines': [{'line_idx': 4, 'corrected_text': 'giây',},]}", [4])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "giây"
