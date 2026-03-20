from app.executor import normalize_result_shape
from app.llm import _extract_json
from app.main import _extract_time_scope, _normalize_history, _time_scope_covered_by


def test_normalize_history_filters_blank_items():
    history = [
        {"role": "user", "content": "How many tickets last week?"},
        {"role": "", "content": "ignored"},
        {"role": "assistant", "content": "  "},
    ]

    assert _normalize_history(history) == [
        {"role": "user", "content": "How many tickets last week?"},
    ]


def test_extract_time_scope_handles_common_ranges():
    assert _extract_time_scope("Top 5 companies in the last 30 days") == ("last_days", 30)
    assert _extract_time_scope("What happened yesterday?") == ("yesterday", 1)
    assert _extract_time_scope("Hours worked this month") == ("this_month", None)


def test_time_scope_covered_by_allows_superset_windows():
    assert _time_scope_covered_by(("last_days", 30), ("last_days", 7)) is True
    assert _time_scope_covered_by(("this_month", None), ("today", 0)) is True
    assert _time_scope_covered_by(("last_days", 3), ("last_days", 7)) is False


def test_normalize_result_shape_accepts_dict_rows_and_tuple_columns():
    result = {
        "columns": ("Company", "Count"),
        "rows": [{"company": "Acme", "count": 4}],
        "notes": {"source": "cached"},
    }

    normalized = normalize_result_shape(result)

    assert normalized["columns"] == ["Company", "Count"]
    assert normalized["rows"] == [["Acme", 4]]
    assert normalized["notes"] == ["source: cached"]


def test_extract_json_reads_embedded_json_object():
    payload = _extract_json("Here is the result:\n{\"reuse_existing_result\": true, \"rationale\": \"cached\"}\n")

    assert payload == {"reuse_existing_result": True, "rationale": "cached"}
