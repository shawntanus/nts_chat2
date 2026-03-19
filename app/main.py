from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.autotask import connect_autotask
from app.config import AppConfig, load_config
from app.executor import execute_generated_code, make_json_safe
from app.llm import LLMService


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


class ChatRequest(BaseModel):
    question: str


def _event(payload: dict) -> str:
    return json.dumps(make_json_safe(payload), ensure_ascii=False) + "\n"


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
        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")

        async def stream():
            try:
                yield _event({"type": "status", "phase": "received", "message": "Question received. Warming up the analyst."})

                yield _event({"type": "status", "phase": "thinking", "message": "Planning the Autotask query and aggregation."})
                async for chunk in _stream_sync_generator(lambda: llm_service.stream_strategy(question)):
                    yield _event({"type": "thinking_delta", "delta": chunk})

                yield _event({"type": "status", "phase": "coding", "message": "Generating Python code for the query."})
                program = await asyncio.to_thread(llm_service.generate_program, question)

                if not program.python_code:
                    raise ValueError("The LLM did not return Python code.")

                yield _event(
                    {
                        "type": "plan",
                        "title": program.title,
                        "reasoning_summary": program.reasoning_summary,
                        "assumptions": program.assumptions,
                    }
                )
                yield _event({"type": "code", "code": program.python_code})

                yield _event({"type": "status", "phase": "autotask", "message": "Connecting to Autotask and running the generated program."})
                at_client = await asyncio.to_thread(connect_autotask, config.autotask)
                max_attempts = 2
                result = None
                current_program = program
                for attempt in range(1, max_attempts + 1):
                    try:
                        result = await asyncio.to_thread(execute_generated_code, current_program.python_code, at_client)
                        break
                    except Exception as run_error:
                        if attempt == max_attempts:
                            raise
                        yield _event(
                            {
                                "type": "status",
                                "phase": "coding",
                                "message": "The first query failed. Repairing the generated code and retrying.",
                            }
                        )
                        current_program = await asyncio.to_thread(
                            llm_service.repair_program,
                            question,
                            current_program.python_code,
                            "".join(traceback.format_exception_only(type(run_error), run_error)).strip(),
                        )
                        yield _event(
                            {
                                "type": "plan",
                                "title": current_program.title,
                                "reasoning_summary": current_program.reasoning_summary,
                                "assumptions": current_program.assumptions,
                            }
                        )
                        yield _event({"type": "code", "code": current_program.python_code})
                        yield _event(
                            {
                                "type": "status",
                                "phase": "autotask",
                                "message": "Retrying the repaired Autotask query.",
                            }
                        )

                yield _event(
                    {
                        "type": "result",
                        "title": current_program.title,
                        "result": result,
                    }
                )
            except Exception as exc:
                yield _event(
                    {
                        "type": "error",
                        "message": str(exc),
                        "details": traceback.format_exc(limit=3),
                    }
                )
            finally:
                yield _event({"type": "done"})

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    config: AppConfig = load_config(BASE_DIR / "config.yaml")
    uvicorn.run("app.main:app", host=config.server.host, port=config.server.port, reload=False)


if __name__ == "__main__":
    run()
