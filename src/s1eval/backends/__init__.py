"""Backend registry."""

from __future__ import annotations

import os
from typing import Any

from .base import Backend, BackendError, Decision
from .llamacpp import LlamaCppLogprobBackend
from .openai_decisions import OpenAIDecisionsBackend
from .scripted import ScriptedBackend
from .systemone import SystemOneBackend

__all__ = [
    "Backend",
    "BackendError",
    "Decision",
    "LlamaCppLogprobBackend",
    "OpenAIDecisionsBackend",
    "ScriptedBackend",
    "SystemOneBackend",
    "make_backend",
]

# Known System One-compatible endpoints (base URLs; the client appends /v1/systemone).
PRESETS: dict[str, dict[str, str]] = {
    "typesafe": {"base_url": "https://api.typesafe.ai", "model": "jev-latest", "key_env": "TYPESAFE_API_KEY"},
    "langsmith-semif": {
        "base_url": "https://gateway.smith.langchain.com",
        "model": "semif-qwen3.5-4b",
        "key_env": "LANGSMITH_API_KEY",
    },
    "langsmith-jev": {
        "base_url": "https://gateway.smith.langchain.com",
        "model": "typesafe/jev-1.13.0",
        "key_env": "LANGSMITH_API_KEY",
    },
    "openrouter": {"base_url": "https://openrouter.ai/api", "model": "~typesafe/jev-latest", "key_env": "OPENROUTER_API_KEY"},
}


def make_backend(kind: str, **kw: Any) -> Any:
    if kind == "local":
        return LlamaCppLogprobBackend(
            base_url=kw.get("base_url") or os.environ.get("S1EVAL_LLAMA_URL", "http://127.0.0.1:8081"),
            choice_permutations=kw.get("choice_permutations", 1),
            min_valid_mass=kw.get("min_valid_mass", 0.5),
        )
    if kind == "openai":
        key = os.environ.get(kw.get("key_env") or "OPENAI_API_KEY")
        if not key:
            raise BackendError("OPENAI_API_KEY is not set")
        return OpenAIDecisionsBackend(model=kw.get("model") or "gpt-6-luna", api_key=key,
                                      base_url=kw.get("base_url") or "https://api.openai.com")
    if kind == "systemone" or kind in PRESETS:
        preset = PRESETS.get(kind, {})
        base = kw.get("base_url") or preset.get("base_url")
        model = kw.get("model") or preset.get("model")
        if not base or not model:
            raise BackendError("systemone backend needs --base-url and --model (or a preset)")
        key_env = kw.get("key_env") or preset.get("key_env") or "S1EVAL_API_KEY"
        key = os.environ.get(key_env)
        if not key and not base.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
            raise BackendError(f"{key_env} is not set")
        return SystemOneBackend(base, model, key, max_questions_per_request=kw.get("max_questions", 32))
    if kind == "scripted":
        return ScriptedBackend()
    raise BackendError(f"unknown backend {kind!r}; choose local, systemone, openai, scripted or {sorted(PRESETS)}")
