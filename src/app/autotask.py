from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from typing import Any

import atws
from atws.query import get_queries_for_entities_by_id

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


def reverse_picklist_values(at, entity_type: str, field_name: str, values: list[Any]) -> dict[Any, Any]:
    seen: set[str] = set()
    resolved: dict[Any, Any] = {}
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        resolved[value] = reverse_picklist(at, entity_type, field_name, value)
    return resolved


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


def _unique_int_ids(values: list[Any]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        coerced = _coerce_int_id(value)
        if coerced is None or coerced in seen:
            continue
        seen.add(coerced)
        result.append(coerced)
    return result


def fetch_many_by_ids(at, entity_type: str, entity_ids: list[Any]) -> list[Any]:
    ids = _unique_int_ids(entity_ids)
    if not ids:
        return []

    entities: list[Any] = []
    for query in get_queries_for_entities_by_id(entity_type, ids):
        try:
            entities.extend(at.query(query).fetch_all())
        except Exception:
            continue
    return entities


def _build_id_to_label_map(
    at,
    entity_candidates: list[tuple[str, tuple[str, ...]]],
    values: list[Any],
    cache: dict[int, str],
) -> dict[int, Any]:
    ids = _unique_int_ids(values)
    unresolved = [entity_id for entity_id in ids if entity_id not in cache]
    if not unresolved:
        return {entity_id: cache.get(entity_id, entity_id) for entity_id in ids}

    for entity_type, field_names in entity_candidates:
        still_unresolved = [entity_id for entity_id in unresolved if entity_id not in cache]
        if not still_unresolved:
            break
        for entity in fetch_many_by_ids(at, entity_type, still_unresolved):
            entity_id = _coerce_int_id(attr(entity, "id"))
            if entity_id is None:
                continue
            for field_name in field_names:
                resolved = attr(entity, field_name)
                if resolved not in (None, ""):
                    cache[entity_id] = safe_name(resolved, entity_id)
                    break

    return {entity_id: cache.get(entity_id, entity_id) for entity_id in ids}


def _build_people_label_map(
    at,
    entity_type: str,
    values: list[Any],
    cache: dict[int, str],
    fallback_field: str,
) -> dict[int, Any]:
    ids = _unique_int_ids(values)
    unresolved = [entity_id for entity_id in ids if entity_id not in cache]
    if unresolved:
        for entity in fetch_many_by_ids(at, entity_type, unresolved):
            entity_id = _coerce_int_id(attr(entity, "id"))
            if entity_id is None:
                continue
            cache[entity_id] = _full_name(
                attr(entity, "FirstName"),
                attr(entity, "LastName"),
                safe_name(attr(entity, fallback_field, entity_id)),
            )
    return {entity_id: cache.get(entity_id, entity_id) for entity_id in ids}


def resolve_resource_name(at, value: Any) -> Any:
    resource_id = _coerce_int_id(value)
    if resource_id is None:
        return value
    return resolve_resource_names(at, [resource_id]).get(resource_id, value)


def resolve_company_name(at, value: Any) -> Any:
    company_id = _coerce_int_id(value)
    if company_id is None:
        return value
    return resolve_company_names(at, [company_id]).get(company_id, value)


def resolve_contact_name(at, value: Any) -> Any:
    contact_id = _coerce_int_id(value)
    if contact_id is None:
        return value
    return resolve_contact_names(at, [contact_id]).get(contact_id, value)


def resolve_company_names(at, values: list[Any]) -> dict[int, Any]:
    candidates = [
        ("Account", ("AccountName",)),
        ("Company", ("CompanyName", "Name", "Company")),
        ("Client", ("ClientName", "Name")),
    ]
    return _build_id_to_label_map(at, candidates, values, _COMPANY_CACHE)


def resolve_resource_names(at, values: list[Any]) -> dict[int, Any]:
    return _build_people_label_map(at, "Resource", values, _RESOURCE_CACHE, "UserName")


def resolve_contact_names(at, values: list[Any]) -> dict[int, Any]:
    return _build_people_label_map(at, "Contact", values, _CONTACT_CACHE, "EMailAddress")


def resolve_display_value(at, column_name: str, value: Any) -> tuple[str, Any]:
    label = column_name or ""
    lowered = label.lower()

    if "status" in lowered:
        friendly = reverse_picklist(at, "Ticket", "Status", value)
        return label.replace(" ID", "").replace("Id", ""), friendly

    if any(keyword in lowered for keyword in ("technician", "tech", "resource", "owner")):
        return label.replace(" ID", "").replace("Id", ""), resolve_resource_name(at, value)

    if "queue" in lowered:
        friendly = reverse_picklist(at, "Ticket", "QueueID", value)
        return label.replace(" ID", "").replace("Id", ""), friendly

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


def build_helpers(extra_helpers: dict[str, Any] | None = None) -> dict[str, Any]:
    helpers = {
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
        "reverse_picklist_values": reverse_picklist_values,
        "resolve_resource_name": resolve_resource_name,
        "resolve_resource_names": resolve_resource_names,
        "resolve_company_name": resolve_company_name,
        "resolve_company_names": resolve_company_names,
        "resolve_contact_name": resolve_contact_name,
        "resolve_contact_names": resolve_contact_names,
        "safe_name": safe_name,
        "rows_from_mapping": rows_from_mapping,
        "fetch_many_by_ids": fetch_many_by_ids,
    }
    if extra_helpers:
        helpers.update(extra_helpers)
    return helpers
