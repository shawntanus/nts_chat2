from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_tokens: int
    base_url: str | None = None


@dataclass(slots=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(slots=True)
class AutotaskConfig:
    user: str
    password: str
    integration_code: str
    api_version: float = 1.6


@dataclass(slots=True)
class AppConfig:
    llm: LLMConfig
    server: ServerConfig
    autotask: AutotaskConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return data


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    data = _read_yaml(config_path)

    llm_data = data.get("llm", {})
    openai_data = data.get("openai", {})
    server_data = data.get("server", {})
    autotask_data = data.get("autotask", {})

    llm = LLMConfig(
        provider=str(llm_data.get("provider", "openai")).strip().lower(),
        model=str(llm_data.get("model", "gpt-4o")).strip(),
        api_key=str(os.getenv("LLM_API_KEY") or llm_data.get("api_key", "")).strip(),
        max_tokens=int(llm_data.get("max_tokens", 4096)),
        base_url=str(openai_data.get("base_url", "")).strip() or None,
    )
    server = ServerConfig(
        host=str(server_data.get("host", "0.0.0.0")).strip(),
        port=int(server_data.get("port", 8000)),
    )
    autotask = AutotaskConfig(
        user=str(os.getenv("AUTOTASK_USER") or autotask_data.get("user", "")).strip(),
        password=str(os.getenv("AUTOTASK_PASSWORD") or autotask_data.get("password", "")).strip(),
        integration_code=str(
            os.getenv("AUTOTASK_INTEGRATION_CODE") or autotask_data.get("integration_code", "")
        ).strip(),
        api_version=float(autotask_data.get("api_version", 1.6)),
    )

    if not llm.api_key:
        raise ValueError("LLM API key is missing. Set llm.api_key in config.yaml or LLM_API_KEY.")
    if not autotask.user or not autotask.password or not autotask.integration_code:
        raise ValueError(
            "Autotask credentials are missing. Populate autotask.user/password/integration_code "
            "in config.yaml or environment variables."
        )

    return AppConfig(llm=llm, server=server, autotask=autotask)
