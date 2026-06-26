from src.normalize_text import normalize_for_search


def test_normalize_for_search_strips_vietnamese_accents():
    assert normalize_for_search("Nguyễn Minh Châu!") == "nguyen minh chau"

