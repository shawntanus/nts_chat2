from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
import math
import statistics
from typing import Any, Callable

import atws

from app.autotask import build_helpers, enrich_result_labels


SAFE_IMPORTS = {
    "collections": __import__("collections"),
    "math": math,
    "statistics": statistics,
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are not allowed.")
    root_name = name.split(".")[0]
    module = SAFE_IMPORTS.get(root_name)
    if module is None:
        raise ImportError(f"Import of '{name}' is not allowed.")
    return module


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "callable": callable,
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
    "RuntimeError": RuntimeError,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
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

    def canonical_key(value: Any) -> str:
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

    for row in rows:
        if isinstance(row, dict):
            if not columns:
                columns = list(row.keys())
            row_key_map = {canonical_key(key): key for key in row.keys()}
            resolved_values: list[Any] = []
            for column in columns:
                value = row.get(column)
                if value is None:
                    matched_key = row_key_map.get(canonical_key(column))
                    if matched_key is not None:
                        value = row.get(matched_key)
                resolved_values.append(value)
            normalized_rows.append(resolved_values)
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


def _preview_value(value: Any) -> str:
    safe = make_json_safe(value)
    if isinstance(safe, list):
        if not safe:
            return "[]"
        if len(safe) <= 3:
            return json_like(safe)
        return f"{json_like(safe[:3])} ... ({len(safe)} items)"
    if isinstance(safe, dict):
        items = list(safe.items())
        if len(items) <= 4:
            return json_like(dict(items))
        preview = dict(items[:4])
        return f"{json_like(preview)} ... ({len(items)} keys)"
    return str(safe)


def json_like(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def summarize_context(context: dict[str, Any]) -> str:
    if not context:
        return "No intermediate values captured."

    checkpoint_keys = context.get("checkpoint_keys")
    checkpoint = context.get("checkpoint")
    selected_keys: list[str] = []

    if isinstance(checkpoint_keys, (list, tuple, set)):
        selected_keys = [str(key) for key in checkpoint_keys if str(key) in context]
    elif checkpoint not in (None, ""):
        checkpoint_key = str(checkpoint)
        if checkpoint_key in context:
            selected_keys = [checkpoint_key]

    preview_keys = selected_keys or [
        key for key in sorted(context.keys())
        if not str(key).startswith("__") and str(key) not in {"checkpoint", "checkpoint_keys"}
    ]

    lines: list[str] = []
    for key in preview_keys:
        value = context[key]
        if isinstance(value, list):
            lines.append(f"{key}: {len(value)} items")
            if value:
                lines.append(f"{key} sample: {_preview_value(value[:2])}")
            continue
        if isinstance(value, dict):
            lines.append(f"{key}: {len(value)} keys")
            if value:
                sample_items = list(value.items())[:3]
                lines.append(f"{key} sample: {_preview_value(dict(sample_items))}")
            continue
        lines.append(f"{key}: {_preview_value(value)}")
    return "\n".join(lines[:8]) or "No intermediate values captured."


def execute_generated_code(
    code: str,
    at_client,
    cell_start_callback: Callable[[int, str, str], None] | None = None,
    cell_result_callback: Callable[[int, str, str, str], None] | None = None,
    cached_context: dict[str, Any] | None = None,
    return_context: bool = False,
) -> dict[str, Any]:
    latest_context: dict[str, Any] = {}

    def emit_cell_start(cell_index: int, cell_name: str, cell_purpose: str) -> None:
        if cell_start_callback:
            cell_start_callback(cell_index, cell_name, cell_purpose)

    def emit_cell_result(cell_index: int, cell_name: str, cell_purpose: str, context: dict[str, Any]) -> None:
        nonlocal latest_context
        latest_context = dict(context)
        if cell_result_callback:
            cell_result_callback(cell_index, cell_name, cell_purpose, summarize_context(context))

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

    result = answer_fn(
        at_client,
        build_helpers(
            {
                "emit_cell_start": emit_cell_start,
                "emit_cell_result": emit_cell_result,
                "cached_context": cached_context or {},
            }
        ),
    )
    if not isinstance(result, dict):
        raise ValueError("Generated program must return a dictionary.")

    result.setdefault("summary", "")
    result.setdefault("columns", [])
    result.setdefault("rows", [])
    result.setdefault("notes", [])
    normalized = normalize_result_shape(result)
    enriched = enrich_result_labels(at_client, normalized)
    safe_result = make_json_safe(enriched)
    if return_context:
        return safe_result, latest_context
    return safe_result
