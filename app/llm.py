from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generator

from anthropic import Anthropic
from openai import OpenAI

from app.config import LLMConfig


STRATEGY_SYSTEM_PROMPT = """You narrate visible reasoning for an analytics assistant.
Do not reveal hidden chain-of-thought.
Provide short, user-friendly process updates about what the assistant is doing.
Focus on steps like understanding the question, choosing Autotask entities, deciding time filters, planning aggregation, and validating output shape.
Keep the tone concise and concrete."""


CODE_SYSTEM_PROMPT = """You generate Python programs for Autotask analytics using the atws library.

Return strict JSON with these keys:
- title: short title for the answer
- reasoning_summary: array of 3-6 short strings
- assumptions: array of short strings
- python_code: a Python program that defines answer_question(at, helpers)

Hard rules for python_code:
- No markdown fences.
- Do not import anything.
- Use helpers["Query"] or Query for queries.
- Return a JSON-serializable dict with keys: summary, columns, rows, notes.
- Use at.query(query).fetch_all() or helpers["fetch_all"](at, query) to retrieve records.
- Prefer datetime filters for "last week", "this month", and "last 30 days".
- Be defensive with missing attributes.
- Keep the answer focused on the user's question.

Useful atws patterns:
- query = Query("Ticket")
- query.WHERE("CreateDate", query.GreaterThanorEquals, some_datetime)
- query.AND("ResolvedDateTime", query.LessThanOrEquals, some_datetime)
- records = at.query(query).fetch_all()

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
- helpers["resolve_resource_name"](at, value)
- helpers["resolve_company_name"](at, value)
- helpers["resolve_contact_name"](at, value)
- helpers["safe_name"](value, fallback="Unknown")

When the exact field or entity name is uncertain, make the best Autotask PSA assumption and note it in assumptions.
Prefer returning human-readable labels instead of raw IDs for technicians, companies/clients, contacts, and ticket status values.
Common question patterns include:
- Top 5 companies by ticket volume in last 30 days
- How many tickets last week?
- Hours worked by each tech this month"""


@dataclass(slots=True)
class GeneratedProgram:
    title: str
    reasoning_summary: list[str]
    assumptions: list[str]
    python_code: str


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

    def stream_strategy(self, question: str) -> Generator[str, None, None]:
        if self.config.provider == "openai":
            stream = self.client.responses.create(
                model=self.config.model,
                instructions=STRATEGY_SYSTEM_PROMPT,
                input=f"Question: {question}\n\nWrite 4-6 short realtime updates.",
                max_output_tokens=min(self.config.max_tokens, 300),
                temperature=0.6,
                stream=True,
            )
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
            return

        stream = self.client.messages.create(
            model=self.config.model,
            system=STRATEGY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Question: {question}\n\nWrite 4-6 short realtime updates."}],
            max_tokens=min(self.config.max_tokens, 300),
            temperature=0.6,
            stream=True,
        )
        for event in stream:
            if getattr(event, "type", "") == "content_block_delta":
                delta = getattr(getattr(event, "delta", None), "text", "")
                if delta:
                    yield delta

    def generate_program(self, question: str) -> GeneratedProgram:
        user_prompt = f"Question: {question}\n\nGenerate the JSON payload now. Return only JSON."
        return self._request_program(user_prompt)

    def repair_program(self, question: str, broken_code: str, error_message: str) -> GeneratedProgram:
        user_prompt = (
            f"Question: {question}\n\n"
            "The previous generated Python failed during execution or output handling.\n"
            f"Error:\n{error_message}\n\n"
            "Fix the Python so it still answers the question. Make it resilient and keep the same output contract.\n"
            "If numeric values may be Decimal-like, convert them to float or int before returning.\n\n"
            f"Previous python_code:\n{broken_code}\n\n"
            "Return only JSON."
        )
        return self._request_program(user_prompt)

    def _request_program(self, user_prompt: str) -> GeneratedProgram:
        if self.config.provider == "openai":
            response = self.client.responses.create(
                model=self.config.model,
                instructions=CODE_SYSTEM_PROMPT,
                input=user_prompt,
                max_output_tokens=self.config.max_tokens,
                temperature=0.2,
            )
            payload = _extract_json(_extract_text_from_openai_response(response))
        else:
            response = self.client.messages.create(
                model=self.config.model,
                system=CODE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=0.2,
            )
            text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
            payload = _extract_json(text)

        return GeneratedProgram(
            title=str(payload.get("title", "Autotask Answer")).strip() or "Autotask Answer",
            reasoning_summary=[str(item).strip() for item in payload.get("reasoning_summary", []) if str(item).strip()],
            assumptions=[str(item).strip() for item in payload.get("assumptions", []) if str(item).strip()],
            python_code=str(payload.get("python_code", "")).strip(),
        )
