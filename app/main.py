from __future__ import annotations

import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.autotask import connect_autotask
from app.config import AppConfig, load_config
from app.executor import execute_generated_code, make_json_safe, summarize_context
from app.llm import GeneratedProgram, LLMService


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MANIFEST_SAMPLE_FIELDS = ("AccountID", "CompanyID", "QueueID", "IssueType", "CreateDate", "Status")


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatHistoryItem] = []
    last_result: dict | None = None


def _event(payload: dict) -> str:
    return json.dumps(make_json_safe(payload), ensure_ascii=False) + "\n"


def _normalize_history(history: list[ChatHistoryItem] | list[dict] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, ChatHistoryItem):
            role = item.role
            content = item.content
        else:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def _chunk_text(text: str, chunk_size: int = 32) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [""]


def _cell_failure_index(error_message: str, program: GeneratedProgram) -> int | None:
    match = re.search(r"Cell\s+(\d+)\s+\([^)]+\)\s+failed:", error_message)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(program.cells):
            return index
    return None


def _issue_cell_index(issue: str, program: GeneratedProgram) -> int | None:
    if not program.cells:
        return None
    lowered = issue.lower()
    if "no rows" in lowered or "no structured result" in lowered:
        return 1 if len(program.cells) > 1 else 0
    return len(program.cells) - 1


def _repair_hint(question: str, error_message: str, program: GeneratedProgram | None = None) -> str:
    lowered_question = question.lower()
    lowered_error = error_message.lower()
    hints: list[str] = []

    if "company" in lowered_question and "ticket" in lowered_question:
        hints.append(
            "For Ticket company grouping, prefer AccountID. Many Ticket records use AccountID rather than CompanyID."
        )

    if "no ticket counts found" in lowered_error:
        hints.append(
            "If tickets were fetched successfully but no grouped company counts were produced, the grouping field is likely wrong."
        )

    if "companyid" in lowered_error:
        hints.append("Try AccountID instead of CompanyID.")

    return "\n".join(hints).strip()


def _extract_time_scope(text: str) -> tuple[str, int | None]:
    lowered = text.lower()
    match = re.search(r"last\s+(\d+)\s+days?", lowered)
    if match:
        return ("last_days", int(match.group(1)))
    if "yesterday" in lowered:
        return ("yesterday", 1)
    if "today" in lowered:
        return ("today", 0)
    if "last week" in lowered:
        return ("last_week", 7)
    if "this month" in lowered:
        return ("this_month", None)
    return ("unknown", None)


def _time_scope_covered_by(cached_scope: tuple[str, int | None], requested_scope: tuple[str, int | None]) -> bool:
    cached_kind, cached_value = cached_scope
    requested_kind, requested_value = requested_scope
    if requested_kind == "unknown":
        return False
    if cached_scope == requested_scope:
        return True
    if cached_kind == "last_days":
        if requested_kind == "last_days" and cached_value is not None and requested_value is not None:
            return cached_value >= requested_value
        if requested_kind in {"today", "yesterday"} and cached_value is not None:
            return cached_value >= 1
    if cached_kind == "this_month" and requested_kind in {"today", "yesterday", "last_days"}:
        return True
    return False


def _should_reuse_ticket_context(question: str, cached_question: str | None, cached_context: dict | None) -> tuple[bool, str]:
    if not cached_context or "tickets" not in cached_context:
        return False, ""

    lowered_question = question.lower()
    cached_text = (cached_question or "").lower()
    follow_up_keywords = (
        "issue type",
        "volume",
        "group",
        "status",
        "count",
        "by ",
        "show ",
        "table",
        "chart",
        "breakdown",
    )
    looks_like_ticket_follow_up = (
        "ticket" in lowered_question
        or any(keyword in lowered_question for keyword in follow_up_keywords)
    )
    if "ticket" not in cached_text or not looks_like_ticket_follow_up:
        return False, ""

    requested_scope = _extract_time_scope(question)
    cached_scope = _extract_time_scope(cached_question or "")
    if _time_scope_covered_by(cached_scope, requested_scope):
        return True, "Reusing cached ticket data because the previous turn already covers this timeframe."

    if requested_scope[0] == "unknown" and "ticket" in cached_text:
        return True, "Reusing cached ticket data because this looks like a follow-up transformation of the previous ticket result."

    return False, ""


async def _emit_program_cells(program: GeneratedProgram, only_index: int | None = None) -> AsyncIterator[dict]:
    if program.cells:
        cells = list(enumerate(program.cells, start=1))
        if only_index is not None:
            cells = [cells[only_index]]
        for index, cell in cells:
            yield {"type": "step", "step": "coding", "text": f"Drafting cell {index}: {cell.purpose}."}
            yield {
                "type": "code_start",
                "title": f"Cell {index}: {cell.purpose}",
                "cell_index": index,
                "cell_name": cell.name,
                "cell_purpose": cell.purpose,
            }
            for chunk in _chunk_text(cell.python_code, 36):
                yield {"type": "code_delta", "content": chunk}
            yield {"type": "code_end", "cell_index": index}
        return

    yield {"type": "code_start", "title": "Generated Python"}
    for chunk in _chunk_text(program.python_code, 36):
        yield {"type": "code_delta", "content": chunk}
    yield {"type": "code_end"}


def _result_markdown(title: str, result: dict | None, assumptions: list[str] | None = None) -> str:
    result = result or {}
    assumptions = assumptions or []
    parts: list[str] = [f"### {title or 'Answer'}"]

    summary = str(result.get("summary", "")).strip()
    if summary:
        parts.append(summary)

    columns = result.get("columns") or []
    rows = result.get("rows") or []
    if columns and rows:
        bullet_lines = []
        for row in rows:
            pairs = [f"**{columns[index]}**: {row[index] if index < len(row) else ''}" for index in range(len(columns))]
            bullet_lines.append(f"- {', '.join(pairs)}")
        parts.append("\n".join(bullet_lines))

    notes = result.get("notes") or []
    if notes:
        parts.append("\n".join(f"- {note}" for note in notes))

    if assumptions:
        parts.append("\n".join(f"- Assumption: {item}" for item in assumptions))

    return "\n\n".join(part for part in parts if part.strip())


def _result_preview(result: dict | None) -> str:
    if not result:
        return "No structured result was returned."
    rows = result.get("rows") or []
    columns = result.get("columns") or []
    summary = str(result.get("summary", "")).strip()
    preview_lines = []
    if summary:
        preview_lines.append(summary)
    preview_lines.append(f"Rows returned: {len(rows)}")
    if columns:
        preview_lines.append(f"Columns: {', '.join(str(column) for column in columns)}")
    return "\n".join(preview_lines)


def _result_issue(question: str, result: dict | None) -> str | None:
    if not result:
        return "The query returned no structured result."

    rows = result.get("rows") or []
    lowered = question.lower()
    expects_ranked_rows = any(
        keyword in lowered
        for keyword in ("top ", "hours worked", "each tech", "by ticket volume", "companies", "company")
    )

    if expects_ranked_rows and not rows:
        return (
            "The query executed but returned no rows for a question that should produce ranked or grouped results. "
            "Likely causes are an incorrect entity field or filter."
        )

    return None


async def _stream_sync_generator(generator_factory) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for chunk in generator_factory():
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(asyncio.to_thread(worker))

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _execute_with_live_events_and_context(code: str, at_client, cached_context: dict | None) -> AsyncIterator[dict]:
    queue: asyncio.Queue[dict | Exception | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(payload: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def worker() -> None:
        try:
            result, latest_context = execute_generated_code(
                code,
                at_client,
                lambda cell_index, cell_name, cell_purpose: emit(
                    {
                        "type": "cell_running",
                        "cell_index": cell_index,
                        "cell_name": cell_name,
                        "cell_purpose": cell_purpose,
                    }
                ),
                lambda cell_index, cell_name, cell_purpose, preview: emit(
                    {
                        "type": "cell_result",
                        "cell_index": cell_index,
                        "cell_name": cell_name,
                        "cell_purpose": cell_purpose,
                        "content": preview,
                    }
                ),
                cached_context=cached_context,
                return_context=True,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "__final_result__",
                    "result": result,
                    "context": {**(cached_context or {}), **latest_context},
                },
            )
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(worker))

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            await task
            raise item
        yield item
    await task


def _cached_context_summary(cached_context: dict | None, cached_question: str | None) -> str | None:
    if not cached_context:
        return None
    return json.dumps(_cached_context_manifest(cached_context, cached_question), ensure_ascii=False, indent=2)


def _cached_context_manifest(cached_context: dict | None, cached_question: str | None) -> dict:
    manifest: dict[str, object] = {
        "source_question": cached_question,
        "time_scope": None,
        "artifacts": [],
    }
    if not cached_context:
        return manifest

    manifest["time_scope"] = _extract_time_scope(cached_question or "")
    artifacts: list[dict[str, object]] = []

    for key in sorted(cached_context.keys()):
        if key in {"cached_context", "checkpoint", "checkpoint_keys", "rows"} or str(key).startswith("__"):
            continue
        value = cached_context[key]
        artifact: dict[str, object] = {
            "key": key,
            "kind": "value",
            "preview": None,
            "reusable_for": [],
            "coverage": "full",
        }

        if isinstance(value, list):
            artifact["count"] = len(value)
            artifact["kind"] = "entities" if value and not isinstance(value[0], (str, int, float, bool)) else "list"
            artifact["preview"] = summarize_context({key: value})
            if value and not isinstance(value[0], (str, int, float, bool)):
                sample = value[0]
                fields = []
                for field in MANIFEST_SAMPLE_FIELDS:
                    attr_value = getattr(sample, field, None)
                    if attr_value is not None:
                        fields.append(field)
                if fields:
                    artifact["fields_seen"] = fields
                artifact["reusable_for"] = ["filtering", "grouping", "aggregation", "top_n_expansion"]
            elif key.startswith("top_"):
                artifact["coverage"] = "partial"
                artifact["reusable_for"] = ["display", "reformatting"]
        elif isinstance(value, dict):
            artifact["count"] = len(value)
            artifact["kind"] = "mapping"
            artifact["preview"] = summarize_context({key: value})
            keys_lower = key.lower()
            if "count" in keys_lower:
                artifact["kind"] = "aggregation"
                artifact["reusable_for"] = ["top_n_expansion", "sorting", "name_resolution"]
            elif "name" in keys_lower:
                artifact["kind"] = "lookup_map"
                artifact["reusable_for"] = ["label_rendering"]
                source_counts_key = key.replace("names", "counts").replace("name", "count")
                if source_counts_key not in cached_context:
                    artifact["coverage"] = "partial"
            if key.startswith("top_"):
                artifact["coverage"] = "partial"
        else:
            artifact["preview"] = summarize_context({key: value})
            if "count" in key.lower():
                artifact["reusable_for"] = ["display", "reformatting"]

        artifacts.append(artifact)

    manifest["artifacts"] = artifacts
    manifest["human_summary"] = summarize_context(cached_context)
    return manifest


async def _chat_events(
    question: str,
    history: list[dict[str, str]],
    llm_service: LLMService,
    config: AppConfig,
    last_result: dict | None = None,
    cached_context: dict | None = None,
    cached_question: str | None = None,
) -> AsyncIterator[dict]:
    reuse_existing, rationale = await asyncio.to_thread(
        llm_service.should_reuse_existing_result,
        question,
        history,
        last_result,
    )
    if reuse_existing and last_result:
        yield {"type": "step", "step": "analyzing", "text": rationale or "Reusing the previous structured result."}
        markdown = await asyncio.to_thread(llm_service.render_result_markdown, question, history, last_result)
        yield {"type": "final_result_data", "result": last_result}
        for chunk in _chunk_text(markdown, 28):
            yield {"type": "text_delta", "content": chunk}
        yield {"type": "step", "step": "done", "text": "Answer ready."}
        yield {"type": "done"}
        return

    cached_context_summary = _cached_context_summary(cached_context, cached_question)
    reuse_cached_context, cached_rationale = _should_reuse_ticket_context(question, cached_question, cached_context)
    if not reuse_cached_context:
        reuse_cached_context, cached_rationale = await asyncio.to_thread(
            llm_service.should_reuse_cached_context,
            question,
            history,
            cached_context_summary,
        )

    yield {"type": "step", "step": "analyzing", "text": "Planning the Autotask query and aggregation."}

    strategy_buffer = ""
    async for chunk in _stream_sync_generator(lambda: llm_service.stream_strategy(question, history)):
        strategy_buffer += chunk
        while "\n" in strategy_buffer:
            line, strategy_buffer = strategy_buffer.split("\n", 1)
            line = line.strip(" -\t")
            if line:
                yield {"type": "step", "step": "analyzing", "text": line}

    trailing = strategy_buffer.strip(" -\t")
    if trailing:
        yield {"type": "step", "step": "analyzing", "text": trailing}

    yield {"type": "step", "step": "coding", "text": "Generating Python code for the query."}
    if reuse_cached_context and cached_context_summary:
        yield {"type": "step", "step": "analyzing", "text": cached_rationale or "Reusing cached context from the previous turn."}
        program = await asyncio.to_thread(
            llm_service.generate_program_with_context,
            question,
            history,
            cached_context_summary,
        )
    else:
        program = await asyncio.to_thread(llm_service.generate_program_with_context, question, history, None)
    if not program.python_code:
        raise ValueError("The LLM did not return Python code.")

    async for event in _emit_program_cells(program):
        yield event

    yield {"type": "step", "step": "executing", "text": "Connecting to Autotask and running the generated program."}
    at_client = await asyncio.to_thread(connect_autotask, config.autotask)
    max_attempts = 20
    result = None
    latest_context = cached_context or {}
    current_program = program

    for attempt in range(1, max_attempts + 1):
        try:
            async for event in _execute_with_live_events_and_context(current_program.python_code, at_client, cached_context):
                if event["type"] == "__final_result__":
                    result = event["result"]
                    latest_context = event.get("context") or {}
                    continue
                if event["type"] == "cell_running":
                    yield {
                        "type": "step",
                        "step": "executing",
                        "text": f"Running cell {event['cell_index']}: {event['cell_purpose']}.",
                    }
                    yield event
                    continue
                if event["type"] == "cell_result":
                    yield {
                        "type": "result",
                        "cell_index": event["cell_index"],
                        "cell_name": event["cell_name"],
                        "cell_purpose": event["cell_purpose"],
                        "content": event["content"],
                    }
            issue = _result_issue(question, result)
            if not issue:
                break
            if attempt == max_attempts:
                raise ValueError(issue)

            repair_index = _issue_cell_index(issue, current_program)
            yield {"type": "step", "step": "coding", "text": "The result looked incomplete. Repairing the most likely failing cell and trying again."}
            if repair_index is None:
                current_program = await asyncio.to_thread(
                    llm_service.repair_program,
                    question,
                    history,
                    current_program.python_code,
                    "\n".join(filter(None, [issue, _repair_hint(question, issue, current_program)])),
                )
            else:
                current_program = await asyncio.to_thread(
                    llm_service.repair_program_cell,
                    question,
                    history,
                    current_program,
                    repair_index,
                    "\n".join(filter(None, [issue, _repair_hint(question, issue, current_program)])),
                )
            async for event in _emit_program_cells(current_program, repair_index):
                yield event
            yield {"type": "step", "step": "executing", "text": "Retrying the repaired Autotask query."}
        except Exception as run_error:
            error_message = "".join(traceback.format_exception_only(type(run_error), run_error)).strip()
            repair_index = _cell_failure_index(error_message, current_program)
            if repair_index is not None:
                failing_cell = current_program.cells[repair_index]
                yield {
                    "type": "cell_error",
                    "cell_index": repair_index + 1,
                    "cell_name": failing_cell.name,
                    "cell_purpose": failing_cell.purpose,
                    "content": error_message,
                }
            if attempt == max_attempts:
                raise
            yield {"type": "step", "step": "coding", "text": "The query failed. Repairing the failing cell and retrying."}
            if repair_index is None:
                current_program = await asyncio.to_thread(
                    llm_service.repair_program,
                    question,
                    history,
                    current_program.python_code,
                    "\n".join(filter(None, [error_message, _repair_hint(question, error_message, current_program)])),
                )
            else:
                current_program = await asyncio.to_thread(
                    llm_service.repair_program_cell,
                    question,
                    history,
                    current_program,
                    repair_index,
                    "\n".join(filter(None, [error_message, _repair_hint(question, error_message, current_program)])),
                )
            async for event in _emit_program_cells(current_program, repair_index):
                yield event
            yield {"type": "step", "step": "executing", "text": "Retrying after the repair."}

    yield {
        "type": "result",
        "cell_index": len(current_program.cells) if current_program.cells else None,
        "cell_name": current_program.cells[-1].name if current_program.cells else None,
        "cell_purpose": current_program.cells[-1].purpose if current_program.cells else None,
        "content": _result_preview(result),
    }
    yield {"type": "final_result_data", "result": result}
    yield {
        "type": "final_context_data",
        "context": make_json_safe(latest_context),
        "context_raw": latest_context,
    }

    answer_markdown = _result_markdown(current_program.title, result, current_program.assumptions)
    for chunk in _chunk_text(answer_markdown, 28):
        yield {"type": "text_delta", "content": chunk}

    yield {"type": "step", "step": "done", "text": "Answer ready."}
    yield {"type": "done"}


def create_app() -> FastAPI:
    config = load_config(BASE_DIR / "config.yaml")
    app = FastAPI(title="NTS AI Data Assistant")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    llm_service = LLMService(config.llm)

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        question = request.question.strip()
        history = _normalize_history(request.history)
        last_result = make_json_safe(request.last_result) if request.last_result else None
        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")

        async def stream():
            sent_done = False
            try:
                async for payload in _chat_events(question, history, llm_service, config, last_result, None, None):
                    if payload.get("type") == "done":
                        sent_done = True
                    yield _event(payload)
            except Exception as exc:
                yield _event(
                    {
                        "type": "error",
                        "content": str(exc),
                        "details": traceback.format_exc(limit=3),
                    }
                )
            finally:
                if not sent_done:
                    yield _event({"type": "done"})

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.websocket("/ws")
    async def websocket_chat(websocket: WebSocket):
        await websocket.accept()
        session_last_result: dict | None = None
        session_last_context: dict | None = None
        session_last_question: str | None = None
        try:
            while True:
                payload = await websocket.receive_json()
                question = str(payload.get("question", "")).strip()
                history = _normalize_history(payload.get("history"))
                last_result = make_json_safe(payload.get("last_result")) if payload.get("last_result") else None
                if not question:
                    await websocket.send_json({"type": "error", "content": "Question is required."})
                    await websocket.send_json({"type": "done"})
                    continue
                try:
                    async for event in _chat_events(
                        question,
                        history,
                        llm_service,
                        config,
                        last_result or session_last_result,
                        session_last_context,
                        session_last_question,
                    ):
                        if event.get("type") == "final_context_data":
                            session_last_context = event.get("context_raw") or event.get("context")
                        safe_event = make_json_safe(event)
                        if event.get("type") == "final_result_data":
                            session_last_result = safe_event.get("result")
                        if event.get("type") == "done":
                            session_last_question = question
                        if event.get("type") == "final_context_data":
                            safe_event.pop("context_raw", None)
                        await websocket.send_json(safe_event)
                except Exception as exc:
                    await websocket.send_json(
                        make_json_safe(
                            {
                                "type": "error",
                                "content": str(exc),
                                "details": traceback.format_exc(limit=3),
                            }
                        )
                    )
                    await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app


app = create_app()


def run() -> None:
    import uvicorn

    config: AppConfig = load_config(BASE_DIR / "config.yaml")
    uvicorn.run("app.main:app", host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    run()
