from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Generator

from anthropic import Anthropic
from openai import OpenAI

from app.config import LLMConfig


STRATEGY_SYSTEM_PROMPT = """You narrate visible reasoning for an analytics assistant.
Do not reveal hidden chain-of-thought.
Provide short, user-friendly process updates about what the assistant is doing.
Focus on steps like understanding the question, choosing Autotask entities, deciding time filters, planning aggregation, and validating output shape.
Keep the tone concise and concrete."""


REUSE_DECISION_SYSTEM_PROMPT = """You decide whether a follow-up user request can be answered from an existing structured result without querying Autotask again.

Return strict JSON with:
- reuse_existing_result: true or false
- rationale: short sentence

Reuse the existing result when the user is asking to:
- reformat the answer
- show as a table
- summarize differently
- sort/filter/restate only using data already present

Do not reuse when the user is asking for:
- new live data
- a different date range
- additional fields not present in the existing result
- a materially different query."""


RESULT_RENDER_SYSTEM_PROMPT = """You turn an existing structured analytics result into user-facing Markdown.

Rules:
- Use only the provided structured result and conversation context.
- Do not invent data.
- If the user asks for a table, return a Markdown table.
- If the user asks for a summary, return concise Markdown prose or bullets.
- Preserve key notes and assumptions when helpful.
- Return Markdown only."""


CACHED_CONTEXT_DECISION_SYSTEM_PROMPT = """You decide whether a follow-up user request can be answered from cached context from the previous turn instead of re-querying Autotask.

Return strict JSON with:
- reuse_cached_context: true or false
- rationale: short sentence

Choose true when:
- the cached data already covers the needed timeframe or a superset of it
- the user wants a new grouping, filtering, or aggregation over that cached data
- the user is asking a follow-up based on the immediately previous data

Choose false when:
- the cached data clearly does not cover the needed timeframe or entity set
- fresh live data is required
- the follow-up needs fields that are not available in the cached context

Use the cached context manifest carefully:
- Prefer full artifacts over partial ones.
- A top-N list or small name map is usually partial coverage.
- Aggregations like `ticket_counts` are strong candidates for ranking expansions.
- Entity lists like `tickets` are strong candidates for regrouping or filtering follow-ups."""


CODE_SYSTEM_PROMPT = """You generate Python programs for Autotask analytics using the atws library.

Return strict JSON with these keys:
- title: short title for the answer
- reasoning_summary: array of 3-6 short strings
- assumptions: array of short strings
- cells: array of 3-5 notebook-like code cells

Each cell object must have:
- name: short snake_case name
- purpose: short human description
- python_code: Python statements for that stage only

The server will stitch the cells together into answer_question(at, helpers).
Think like Jupyter:
- Cell 1 should usually fetch the base records and validate the entity/filter.
- Cell 2 should usually transform or aggregate and validate key fields.
- Cell 3+ should enrich labels, shape rows, and return the final dict.
- Prefer repairing only the failing cell conceptually instead of rewriting everything.
- Every non-final cell must write at least one previewable value into `context`, for example:
  - `context["tickets"] = tickets`
  - `context["ticket_counts"] = ticket_counts`
  - `context["company_ticket_counts"] = company_ticket_counts`
- Every cell should explicitly mark which context values are safe and relevant to preview:
  - Prefer `context["checkpoint_keys"] = ["ticket_counts"]`
  - For a single value, `context["checkpoint"] = "tickets"` is acceptable
- Do not preview the entire accumulated context unless truly necessary.
- Every non-final cell must validate its output and raise `ValueError(...)` with a specific message if the data is empty, missing, or obviously wrong.
- The final cell should also save the final rows into `context["rows"]` before returning when practical.

Hard rules for python_code:
- No markdown fences.
- Do not import anything.
- `Counter` and `defaultdict` are already available; never write `import` or `from ... import ...`.
- Use helpers["Query"] or Query for queries.
- The final cell must return a JSON-serializable dict with keys: summary, columns, rows, notes.
- Use at.query(query).fetch_all() or helpers["fetch_all"](at, query) to retrieve records.
- Prefer datetime filters for "last week", "this month", and "last 30 days".
- Be defensive with missing attributes.
- Add validation checkpoints with clear ValueError messages when a likely field guess is wrong.
- Keep the answer focused on the user's question.
- When a result row contains a company/client/account ID, resolve it to a display name before returning rows whenever possible.
- Use variables created by earlier cells instead of re-querying the same data.
- Optimize for the least data fetched from Autotask.
- Avoid broad historical ticket pulls when a smaller candidate set can be derived first.
- For high-volume questions, prefer a staged plan like:
  - query the smallest recent window needed to identify active entities
  - fetch the entity list or candidate IDs
  - only then fetch older or per-entity history for the reduced inactive set
- If the question asks for companies with no tickets in the last N days, do not start by fetching all tickets older than N days.
  - First identify companies with recent tickets.
  - Then determine inactive companies from the company list.
  - Then fetch only the minimal older ticket history needed to find the latest old ticket date for those inactive companies.
- Prefer query filters that reduce row count early.
- Prefer company/account-level candidate reduction before ticket-level historical scans.

Useful atws patterns:
- query = Query("Ticket")
- query.WHERE("CreateDate", query.GreaterThanorEquals, some_datetime)
- query.AND("CreateDate", query.LessThanOrEquals, some_datetime)
- query.open_bracket("AND")
- query.OR("Status", query.Equals, at.picklist["Ticket"]["Status"]["Complete"])
- query.close_bracket()
- query.AND("ResolvedDateTime", query.LessThanOrEquals, some_datetime)
- records = at.query(query).fetch_all()

Documented query syntax from atws usage docs:
- Use `Query("EntityName")` or `atws.Query("EntityName")`.
- Use comparison constants from the query object itself, such as:
  - `query.Equals`
  - `query.NotEqual`
  - `query.GreaterThan`
  - `query.LessThan`
  - `query.GreaterThanorEquals`
  - `query.LessThanOrEquals`
- Do not invent alternate operator spellings, raw operators like `">="`, or attributes like `query.Op`.
- For grouped boolean logic, use `open_bracket("AND")`, `OR(...)`, and `close_bracket()`.
- To execute a query, use `at.query(query)` and then:
  - `.fetch_all()` for a list
  - `.fetch_one()` for a single result
- Remember that the wrapper fetches query results in batches up to 500 entities, so broad queries can still be expensive.

Helper functions available inside answer_question:
- helpers["days_ago"](n)
- helpers["start_of_week"]()
- helpers["start_of_last_week"]()
- helpers["start_of_month"]()
- helpers["start_of_day"](dt)
- helpers["end_of_day"](dt)
- helpers["fetch_all"](at, query)
- helpers["fetch_one_by_id"](at, entity_type, entity_id)
- helpers["attr"](entity, field_name, default=None)
- helpers["reverse_picklist"](at, entity_type, field_name, value)
- helpers["reverse_picklist_values"](at, entity_type, field_name, values)
- helpers["resolve_resource_name"](at, value)
- helpers["resolve_resource_names"](at, values)
- helpers["resolve_company_name"](at, value)
- helpers["resolve_company_names"](at, values)
- helpers["resolve_contact_name"](at, value)
- helpers["resolve_contact_names"](at, values)
- helpers["safe_name"](value, fallback="Unknown")
- helpers["fetch_many_by_ids"](at, entity_type, entity_ids)
- helpers["cached_context"]: reusable context values from the previous successful turn

Query syntax safety rules:
- Always use the documented query constants from the `query` instance.
- Never use `query.Op.*`.
- Never use raw operator strings such as `"="`, `">"`, `">="`, `"<"`, or `"<="`.
- Never use SQL syntax.
- For follow-up questions, prefer reusing `helpers["cached_context"]` before querying Autotask again when that cached data is sufficient.
- Reusing cached data is allowed even if you still need additional filtering, grouping, sorting, or picklist resolution.
- Follow-up cells start with prior cached values already seeded into `context`, so previously computed values like `context["tickets"]` or `context["ticket_counts"]` may already exist.
- You may also read the same prior values from `context["cached_context"]` when you want to be explicit.
- When reusing cached Autotask entities, treat them like objects and access fields with `helpers["attr"](entity, field_name, default)` rather than `entity["FieldName"]` or `.get(...)`.
- When the user asks to expand a ranking, such as changing top 5 to top 10, do not rely on a previously resolved top-N name map if it only covers the smaller set.
- For top-N expansions, prefer this order:
  - reuse `context["ticket_counts"]` if available
  - otherwise recompute counts from cached `context["tickets"]`
  - then resolve names in bulk only for the final requested IDs
- Never build expanded results from `context["company_names"]` alone unless it clearly covers every ID needed for the new ranking.
- For Ticket company grouping, ignore falsey or zero AccountID values when they represent missing companies.
- The cached context input is a manifest, not raw data. Read the manifest to choose the best artifact, then write code against the actual seeded `context` values.
- Prefer artifacts with `coverage = "full"` over artifacts with `coverage = "partial"`.
- If the manifest says a name map or top-N artifact is partial, reuse it only for display of that exact same subset, not for expansions or regrouping.
- When resolving many IDs to names, do not call a single-ID resolver repeatedly inside a loop if you can avoid it.
- Prefer this pattern for grouped results:
  - aggregate first into IDs and counts
  - collect the unique IDs or picklist values
  - call a bulk helper like `helpers["resolve_company_names"](at, ids)`, `helpers["resolve_resource_names"](at, ids)`, `helpers["resolve_contact_names"](at, ids)`, or `helpers["reverse_picklist_values"](at, entity_type, field_name, values)`
  - build final rows from the returned mapping
- Single-ID resolvers like `helpers["resolve_company_name"](at, value)` are fine for one-off values, but bulk helpers are preferred for grouped reports, top-N lists, or any repeated lookup because Autotask is slow.
- For queue reports specifically:
  - first produce `queue_counts` keyed by the Ticket field `QueueID`
  - then resolve queue labels in bulk from the Ticket `QueueID` picklist, via `helpers["reverse_picklist_values"](at, "Ticket", "QueueID", list(queue_counts.keys()))`
  - then build rows from `queue_counts` plus the resolved name map
- Do not mix queue-name resolution into the same loop that aggregates queue counts.
- Do not treat a partial queue-name map as sufficient for expanded rankings or regrouping.
- Apply the same pattern to all repeated lookups:
  - technician/resource IDs -> `helpers["resolve_resource_names"](at, ids)`
  - contact IDs -> `helpers["resolve_contact_names"](at, ids)`
  - company/account/client IDs -> `helpers["resolve_company_names"](at, ids)`
  - queue IDs -> `helpers["reverse_picklist_values"](at, "Ticket", "QueueID", ids)`
  - picklist values like status or issue type -> `helpers["reverse_picklist_values"](at, entity_type, field_name, values)`
- Do not resolve names or picklist labels inside the same loop that performs the main aggregation when the lookup will run multiple times.

When the exact field or entity name is uncertain, make the best Autotask PSA assumption and note it in assumptions.
Prefer returning human-readable labels instead of raw IDs for technicians, companies/clients, contacts, and ticket status values.
For ticket queue grouping, prefer `QueueID` and resolve it with the Ticket `QueueID` picklist.
Never leave company/account/client IDs in the final rows when `helpers["resolve_company_name"](at, value)` can be used.
On the `Ticket` entity, company linkage is often `AccountID`. Prefer `AccountID` for company/customer grouping unless there is strong evidence that another field is correct.
For company/customer names in the SOAP API, prefer the `Account` entity and the `AccountName` field.
Common question patterns include:
- Top 5 companies by ticket volume in last 30 days
- How many tickets last week?
- Hours worked by each tech this month

For example, for "Top 5 companies by ticket volume in last 7 days":
- Cell 1 fetches tickets, validates `tickets`, stores `context["tickets"] = tickets`, and sets `context["checkpoint"] = "tickets"`
- Cell 2 groups into `ticket_counts`, validates it, stores `context["ticket_counts"] = ticket_counts`, and sets `context["checkpoint"] = "ticket_counts"`
- Cell 3 resolves names in bulk into `company_ticket_counts`, for example by calling `helpers["resolve_company_names"](at, list(ticket_counts.keys()))`, validates it, stores `context["company_ticket_counts"] = company_ticket_counts`, and sets `context["checkpoint"] = "company_ticket_counts"`
- Cell 4 builds `rows`, stores `context["rows"] = rows`, and returns the final dict

For "ticket volume by queue name":
- first build `queue_counts`
- then bulk resolve `queue_names = helpers["reverse_picklist_values"](at, "Ticket", "QueueID", list(queue_counts.keys()))`
- then build rows from those two structures

Checkpoint preview rules:
- Prefer showing only the most relevant variable for the current cell.
- Use `context["checkpoint_keys"]` when you want to preview 2-3 related values together.
- Avoid leaving old checkpoint selections in place when a new cell should show something narrower."""


CELL_REPAIR_SYSTEM_PROMPT = """You repair one notebook-like Python cell for an Autotask analytics workflow.

Return strict JSON with these keys:
- name: short snake_case name
- purpose: short human description
- python_code: Python statements for that one cell only

Rules:
- Return only the repaired cell, not the whole program.
- Do not import anything.
- Preserve the surrounding workflow and variable names whenever possible.
- Add clear validation checks if the prior version failed due to empty data, missing fields, or wrong ID/name mapping.
- Ensure the repaired cell stores at least one useful preview value in `context` unless it is the final return cell.
- Ensure the repaired cell also sets `context["checkpoint"]` or `context["checkpoint_keys"]` so the preview stays focused.
- Preserve or improve data-fetch efficiency; do not repair a cell by broadening it into a much larger data pull unless unavoidable.
- For ticket-to-company grouping failures, prefer `AccountID` over `CompanyID` unless the surrounding code proves otherwise.
- The final answer contract remains summary, columns, rows, notes."""


@dataclass(slots=True)
class GeneratedCell:
    name: str
    purpose: str
    python_code: str


@dataclass(slots=True)
class GeneratedProgram:
    title: str
    reasoning_summary: list[str]
    assumptions: list[str]
    cells: list[GeneratedCell]
    python_code: str


def _build_program_from_cells(cells: list[GeneratedCell]) -> str:
    lines = [
        "def answer_question(at, helpers):",
        "    context = dict(helpers.get('cached_context', {}))",
        "    context['cached_context'] = dict(helpers.get('cached_context', {}))",
    ]
    total_cells = len(cells)
    for index, cell in enumerate(cells, start=1):
        label = f"Cell {index} ({cell.name})"
        lines.append(f"    # {label}: {cell.purpose}")
        lines.append("    try:")
        lines.append("        emit_cell_start = helpers.get('emit_cell_start')")
        lines.append("        if callable(emit_cell_start):")
        lines.append(f"            emit_cell_start({index}, {cell.name!r}, {cell.purpose!r})")
        lines.append("        __context_keys_before = set(context.keys())")
        body = cell.python_code.strip() or "pass"
        lines.extend(indent(body, "        ").splitlines())
        lines.append("        emit_cell_result = helpers.get('emit_cell_result')")
        lines.append("        if callable(emit_cell_result):")
        lines.append(f"            emit_cell_result({index}, {cell.name!r}, {cell.purpose!r}, context)")
        if index < total_cells:
            lines.append("        if len(context.keys() - __context_keys_before) == 0:")
            lines.append(f'            raise ValueError("{label} did not store any previewable value in context.")')
        lines.append("    except Exception as exc:")
        lines.append(f'        raise RuntimeError("{label} failed: " + str(exc)) from exc')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_text_from_openai_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(text)
    return "".join(pieces).strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model did not return JSON.")
        return json.loads(text[start : end + 1])


class LLMService:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider == "openai":
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        elif config.provider == "anthropic":
            self.client = Anthropic(api_key=config.api_key, base_url=config.base_url)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def _history_block(self, history: list[dict[str, str]] | None) -> str:
        if not history:
            return "No prior conversation."
        trimmed = history[-8:]
        lines = [f"{item.get('role', 'unknown')}: {item.get('content', '').strip()}" for item in trimmed]
        return "\n".join(lines)

    def _create_response(
        self,
        *,
        system_prompt: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
        stream: bool = False,
    ):
        if self.config.provider == "openai":
            return self.client.responses.create(
                model=self.config.model,
                instructions=system_prompt,
                input=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=stream,
            )

        return self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=temperature,
            stream=stream,
        )

    def _request_json(
        self,
        *,
        system_prompt: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        response = self._create_response(
            system_prompt=system_prompt,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if self.config.provider == "openai":
            return _extract_json(_extract_text_from_openai_response(response))

        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return _extract_json(text)

    def _request_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        response = self._create_response(
            system_prompt=system_prompt,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if self.config.provider == "openai":
            return _extract_text_from_openai_response(response).strip()

        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()

    def stream_strategy(self, question: str, history: list[dict[str, str]] | None = None) -> Generator[str, None, None]:
        prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Latest question: {question}\n\n"
            "Write 4-6 short realtime updates."
        )
        stream = self._create_response(
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=min(self.config.max_tokens, 300),
            temperature=0.6,
            stream=True,
        )
        if self.config.provider == "openai":
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
            return

        for event in stream:
            if getattr(event, "type", "") == "content_block_delta":
                delta = getattr(getattr(event, "delta", None), "text", "")
                if delta:
                    yield delta

    def should_reuse_existing_result(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        last_result: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if not last_result:
            return False, "No existing structured result is available."
        prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Latest user question: {question}\n\n"
            f"Existing structured result:\n{json.dumps(last_result, ensure_ascii=False)}\n\n"
            "Decide whether the existing result is enough. Return only JSON."
        )
        payload = self._request_json(
            system_prompt=REUSE_DECISION_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=min(self.config.max_tokens, 200),
            temperature=0,
        )
        return bool(payload.get("reuse_existing_result")), str(payload.get("rationale", "")).strip()

    def render_result_markdown(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        last_result: dict[str, Any],
    ) -> str:
        prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Latest user question: {question}\n\n"
            f"Structured result to use:\n{json.dumps(last_result, ensure_ascii=False)}\n\n"
            "Return Markdown only."
        )
        return self._request_text(
            system_prompt=RESULT_RENDER_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=min(self.config.max_tokens, 800),
            temperature=0.2,
        )

    def generate_program_with_context(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        cached_context_summary: str | None = None,
    ) -> GeneratedProgram:
        user_prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Question: {question}\n\n"
            f"Available cached context manifest from the previous turn:\n{cached_context_summary or 'None.'}\n\n"
            "Generate the JSON payload now. Return only JSON.\n"
            "Be especially careful to minimize data fetched from Autotask when the question suggests large ticket volume."
        )
        return self._request_program(user_prompt)

    def should_reuse_cached_context(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        cached_context_summary: str | None,
    ) -> tuple[bool, str]:
        if not cached_context_summary:
            return False, "No cached context is available."
        prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Latest user question: {question}\n\n"
            f"Cached context manifest:\n{cached_context_summary}\n\n"
            "Decide whether the cached context is enough. Return only JSON."
        )
        payload = self._request_json(
            system_prompt=CACHED_CONTEXT_DECISION_SYSTEM_PROMPT,
            prompt=prompt,
            max_output_tokens=min(self.config.max_tokens, 200),
            temperature=0,
        )
        return bool(payload.get("reuse_cached_context")), str(payload.get("rationale", "")).strip()

    def repair_program(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        broken_code: str,
        error_message: str,
    ) -> GeneratedProgram:
        user_prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Question: {question}\n\n"
            "The previous generated Python failed during execution or output handling.\n"
            f"Error:\n{error_message}\n\n"
            "Fix the Python so it still answers the question. Make it resilient and keep the same output contract.\n"
            "If numeric values may be Decimal-like, convert them to float or int before returning.\n\n"
            "Keep or improve efficiency. Avoid broadening the query into a larger ticket pull unless absolutely necessary.\n\n"
            "Do not use import statements. Counter and defaultdict are already available.\n\n"
            f"Previous python_code:\n{broken_code}\n\n"
            "Return only JSON."
        )
        return self._request_program(user_prompt)

    def repair_program_cell(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        program: GeneratedProgram,
        cell_index: int,
        error_message: str,
    ) -> GeneratedProgram:
        current_cell = program.cells[cell_index]
        previous_cells = program.cells[:cell_index]
        later_cells = program.cells[cell_index + 1 :]
        user_prompt = (
            f"Conversation so far:\n{self._history_block(history)}\n\n"
            f"Question: {question}\n\n"
            f"Failing cell index: {cell_index + 1}\n"
            f"Failing cell name: {current_cell.name}\n"
            f"Failing cell purpose: {current_cell.purpose}\n\n"
            f"Error:\n{error_message}\n\n"
            f"Earlier cells already considered valid:\n{self._cells_block(previous_cells)}\n\n"
            f"Current failing cell:\n{current_cell.python_code}\n\n"
            f"Later cells that should keep working after the fix:\n{self._cells_block(later_cells)}\n\n"
            "Repair only the failing cell. Return only JSON.\n"
            "Keep or improve efficiency and avoid increasing the volume of fetched Autotask records."
        )
        repaired_cell = self._request_cell(user_prompt)
        repaired_cells = list(program.cells)
        repaired_cells[cell_index] = repaired_cell
        return GeneratedProgram(
            title=program.title,
            reasoning_summary=program.reasoning_summary,
            assumptions=program.assumptions,
            cells=repaired_cells,
            python_code=_build_program_from_cells(repaired_cells),
        )

    def _cells_block(self, cells: list[GeneratedCell]) -> str:
        if not cells:
            return "None."
        blocks = []
        for index, cell in enumerate(cells, start=1):
            blocks.append(f"[{index}] {cell.name} - {cell.purpose}\n{cell.python_code}")
        return "\n\n".join(blocks)

    def _request_program(self, user_prompt: str) -> GeneratedProgram:
        payload = self._request_json(
            system_prompt=CODE_SYSTEM_PROMPT,
            prompt=user_prompt,
            max_output_tokens=self.config.max_tokens,
            temperature=0.2,
        )

        raw_cells = payload.get("cells") or []
        cells: list[GeneratedCell] = []
        for index, item in enumerate(raw_cells, start=1):
            if not isinstance(item, dict):
                continue
            python_code = str(item.get("python_code", "")).strip()
            if not python_code:
                continue
            cells.append(
                GeneratedCell(
                    name=str(item.get("name", f"cell_{index}")).strip() or f"cell_{index}",
                    purpose=str(item.get("purpose", "Program step")).strip() or "Program step",
                    python_code=python_code,
                )
            )

        fallback_code = str(payload.get("python_code", "")).strip()
        python_code = _build_program_from_cells(cells) if cells else fallback_code

        return GeneratedProgram(
            title=str(payload.get("title", "Autotask Answer")).strip() or "Autotask Answer",
            reasoning_summary=[str(item).strip() for item in payload.get("reasoning_summary", []) if str(item).strip()],
            assumptions=[str(item).strip() for item in payload.get("assumptions", []) if str(item).strip()],
            cells=cells,
            python_code=python_code,
        )

    def _request_cell(self, user_prompt: str) -> GeneratedCell:
        payload = self._request_json(
            system_prompt=CELL_REPAIR_SYSTEM_PROMPT,
            prompt=user_prompt,
            max_output_tokens=self.config.max_tokens,
            temperature=0.2,
        )

        return GeneratedCell(
            name=str(payload.get("name", "repaired_cell")).strip() or "repaired_cell",
            purpose=str(payload.get("purpose", "Repaired program step")).strip() or "Repaired program step",
            python_code=str(payload.get("python_code", "")).strip(),
        )
