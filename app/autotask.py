from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from typing import Any

import atws

from app.config import AutotaskConfig


@lru_cache(maxsize=1)
def get_client(
    user: str,
    password: str,
    integration_code: str,
    api_version: float,
):
    return atws.connect(
        username=user,
        password=password,
        integrationcode=integration_code,
        apiversion=api_version,
    )


def connect_autotask(config: AutotaskConfig):
    return get_client(config.user, config.password, config.integration_code, config.api_version)


def start_of_day(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), time.min)


def end_of_day(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), time.max)


def start_of_week(reference: datetime | None = None) -> datetime:
    ref = reference or datetime.now()
    return start_of_day(ref - timedelta(days=ref.weekday()))


def start_of_last_week(reference: datetime | None = None) -> datetime:
    return start_of_week(reference) - timedelta(days=7)


def start_of_month(reference: datetime | None = None) -> datetime:
    ref = reference or datetime.now()
    return datetime(ref.year, ref.month, 1)


def days_ago(days: int, reference: datetime | None = None) -> datetime:
    ref = reference or datetime.now()
    return ref - timedelta(days=days)


def fetch_all(at, query):
    return at.query(query).fetch_all()


def fetch_one_by_id(at, entity_type: str, entity_id: Any):
    query = atws.Query(entity_type)
    query.WHERE("id", query.Equals, entity_id)
    return at.query(query).fetch_one()


def attr(entity: Any, name: str, default: Any = None) -> Any:
    return getattr(entity, name, default)


def reverse_picklist(at, entity_type: str, field_name: str, value: Any) -> Any:
    if value is None:
        return None
    try:
        return at.picklist[entity_type][field_name].reverse_lookup(value)
    except Exception:
        return value


def safe_name(value: Any, fallback: str = "Unknown") -> str:
    if value in (None, ""):
        return fallback
    return str(value)


_RESOURCE_CACHE: dict[int, str] = {}
_COMPANY_CACHE: dict[int, str] = {}
_CONTACT_CACHE: dict[int, str] = {}


def _coerce_int_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def _full_name(first: Any, last: Any, fallback: str) -> str:
    parts = [str(part).strip() for part in (first, last) if part not in (None, "")]
    return " ".join(parts) if parts else fallback


def resolve_resource_name(at, value: Any) -> Any:
    resource_id = _coerce_int_id(value)
    if resource_id is None:
        return value
    if resource_id in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[resource_id]
    try:
        entity = fetch_one_by_id(at, "Resource", resource_id)
        if entity:
            label = _full_name(
                attr(entity, "FirstName"),
                attr(entity, "LastName"),
                safe_name(attr(entity, "UserName", resource_id)),
            )
            _RESOURCE_CACHE[resource_id] = label
            return label
    except Exception:
        pass
    return value


def resolve_company_name(at, value: Any) -> Any:
    company_id = _coerce_int_id(value)
    if company_id is None:
        return value
    if company_id in _COMPANY_CACHE:
        return _COMPANY_CACHE[company_id]
    try:
        entity = fetch_one_by_id(at, "Company", company_id)
        if entity:
            label = safe_name(attr(entity, "CompanyName", company_id))
            _COMPANY_CACHE[company_id] = label
            return label
    except Exception:
        pass
    return value


def resolve_contact_name(at, value: Any) -> Any:
    contact_id = _coerce_int_id(value)
    if contact_id is None:
        return value
    if contact_id in _CONTACT_CACHE:
        return _CONTACT_CACHE[contact_id]
    try:
        entity = fetch_one_by_id(at, "Contact", contact_id)
        if entity:
            label = _full_name(
                attr(entity, "FirstName"),
                attr(entity, "LastName"),
                safe_name(attr(entity, "EMailAddress", contact_id)),
            )
            _CONTACT_CACHE[contact_id] = label
            return label
    except Exception:
        pass
    return value


def resolve_display_value(at, column_name: str, value: Any) -> tuple[str, Any]:
    label = column_name or ""
    lowered = label.lower()

    if "status" in lowered:
        friendly = reverse_picklist(at, "Ticket", "Status", value)
        return label.replace(" ID", "").replace("Id", ""), friendly

    if any(keyword in lowered for keyword in ("technician", "tech", "resource", "owner")):
        return label.replace(" ID", "").replace("Id", ""), resolve_resource_name(at, value)

    if any(keyword in lowered for keyword in ("company", "client", "account", "customer")):
        return label.replace(" ID", "").replace("Id", ""), resolve_company_name(at, value)

    if "contact" in lowered:
        return label.replace(" ID", "").replace("Id", ""), resolve_contact_name(at, value)

    return label, value


def enrich_result_labels(at, result: dict[str, Any]) -> dict[str, Any]:
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    if not columns or not rows:
        return result

    enriched_columns = list(columns)
    enriched_rows: list[list[Any]] = []

    for row in rows:
        enriched_row = list(row)
        for index, value in enumerate(enriched_row):
            column_name = enriched_columns[index] if index < len(enriched_columns) else f"Column {index + 1}"
            new_column_name, new_value = resolve_display_value(at, column_name, value)
            enriched_columns[index] = new_column_name
            enriched_row[index] = new_value
        enriched_rows.append(enriched_row)

    result["columns"] = enriched_columns
    result["rows"] = enriched_rows
    return result


def rows_from_mapping(mapping: dict[str, Any], label: str, value_label: str) -> dict[str, Any]:
    sorted_rows = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    return {
        "summary": "",
        "columns": [label, value_label],
        "rows": [[key, value] for key, value in sorted_rows],
        "notes": [],
    }


def build_helpers() -> dict[str, Any]:
    return {
        "Query": atws.Query,
        "datetime": datetime,
        "timedelta": timedelta,
        "days_ago": days_ago,
        "start_of_day": start_of_day,
        "end_of_day": end_of_day,
        "start_of_week": start_of_week,
        "start_of_last_week": start_of_last_week,
        "start_of_month": start_of_month,
        "fetch_all": fetch_all,
        "fetch_one_by_id": fetch_one_by_id,
        "attr": attr,
        "reverse_picklist": reverse_picklist,
        "resolve_resource_name": resolve_resource_name,
        "resolve_company_name": resolve_company_name,
        "resolve_contact_name": resolve_contact_name,
        "safe_name": safe_name,
        "rows_from_mapping": rows_from_mapping,
    }
