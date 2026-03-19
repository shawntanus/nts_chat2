from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import atws

from app.autotask import build_helpers, enrich_result_labels


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_result_shape(result: dict[str, Any]) -> dict[str, Any]:
    columns = result.get("columns", [])
    rows = result.get("rows", [])

    if isinstance(columns, tuple):
        columns = list(columns)
    if not isinstance(columns, list):
        columns = [str(columns)] if columns not in (None, "") else []

    if isinstance(rows, dict):
        rows = [rows]
    elif not isinstance(rows, list):
        rows = [rows] if rows not in (None, "") else []

    normalized_rows: list[list[Any]] = []

    for row in rows:
        if isinstance(row, dict):
            if not columns:
                columns = list(row.keys())
            normalized_rows.append([row.get(column) for column in columns])
            continue
        if isinstance(row, (list, tuple)):
            normalized_rows.append(list(row))
            continue
        normalized_rows.append([row])

    if normalized_rows and not columns:
        width = max(len(row) for row in normalized_rows)
        columns = [f"Column {index + 1}" for index in range(width)]

    if columns:
        width = len(columns)
        normalized_rows = [
            row[:width] + [None] * max(0, width - len(row))
            for row in normalized_rows
        ]

    result["columns"] = columns
    result["rows"] = normalized_rows
    notes = result.get("notes", [])
    if isinstance(notes, tuple):
        notes = list(notes)
    elif isinstance(notes, dict):
        notes = [f"{key}: {value}" for key, value in notes.items()]
    elif not isinstance(notes, list):
        notes = [notes] if notes not in (None, "") else []
    result["notes"] = notes
    return result


def execute_generated_code(code: str, at_client) -> dict[str, Any]:
    sandbox_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "atws": atws,
        "Query": atws.Query,
        "Counter": Counter,
        "defaultdict": defaultdict,
    }
    sandbox_locals: dict[str, Any] = {}
    exec(code, sandbox_globals, sandbox_locals)

    answer_fn = sandbox_locals.get("answer_question") or sandbox_globals.get("answer_question")
    if not callable(answer_fn):
        raise ValueError("Generated program did not define answer_question(at, helpers).")

    result = answer_fn(at_client, build_helpers())
    if not isinstance(result, dict):
        raise ValueError("Generated program must return a dictionary.")

    result.setdefault("summary", "")
    result.setdefault("columns", [])
    result.setdefault("rows", [])
    result.setdefault("notes", [])
    normalized = normalize_result_shape(result)
    enriched = enrich_result_labels(at_client, normalized)
    return make_json_safe(enriched)
