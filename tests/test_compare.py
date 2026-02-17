from app.table_pipeline import compare_json


def test_compare_json_exact_match() -> None:
    score, matches, total, details = compare_json({"a": "1"}, {"a": "1"})
    assert score == 100.0
    assert matches == 1
    assert total == 1
    assert details == []
