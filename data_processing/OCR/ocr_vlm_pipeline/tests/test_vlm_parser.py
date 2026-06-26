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


def test_parse_line_key_dict_from_model():
    result = parse_vlm_json("{'line_5': 'TÌNH TRẠNG SỤT LÚN ĐBSCL ĐANG DIỄN RA RẤT NHANH'}", [5])
    assert result["parse_status"] == "ok"
    assert result["lines"][0]["corrected_text"] == "TÌNH TRẠNG SỤT LÚN ĐBSCL ĐANG DIỄN RA RẤT NHANH"


def test_parse_bracket_line_output_from_model():
    response = "[3] CÁNH BÁO\n[4] SẠT LỞ NGUY HIỂM\n[5] TẠM DỪNG LƯU THÔNG"
    result = parse_vlm_json(response, [3, 4, 5])
    assert result["parse_status"] == "ok"
    assert result["lines"][1]["corrected_text"] == "SẠT LỞ NGUY HIỂM"


def test_bracket_number_without_text_is_not_partial():
    result = parse_vlm_json("[3]", [3])
    assert result["parse_status"] in ["json_error", "schema_error"]
    assert result["lines"] == []


def test_parse_complete_objects_from_truncated_json_array():
    response = (
        '{"lines":['
        '{"line_idx":3,"raw_text":"CÁNH BÁO","corrected_text":"CÁNH BÁO"},'
        '{"line_idx":4,"raw_text":"SAT L NGUY HIÊM","corrected_text":"SẠT LỞ NGUY HIỂM"},'
        '{"line_idx":5,"raw_text":"TAM DÜNG'
    )
    result = parse_vlm_json(response, [3, 4, 5])
    assert result["parse_status"] == "partial"
    assert len(result["lines"]) == 2
    assert result["lines"][1]["corrected_text"] == "SẠT LỞ NGUY HIỂM"
