"""LiteLLM custom provider ``s1``: System One-style categorical judging for ASSERT (C21, C25).

Model strings: ``s1/<backend>/<model>`` with backend in ``llamacpp`` (alias ``local``),
``openai_decisions`` (alias ``openai``), ``systemone``, a System One preset (``typesafe``,
``openrouter``), or ``scripted``. Examples::

    s1/llamacpp/qwen3.5-4b          # llama-server at $CI_S1_LLAMA_URL (default :8081), logprob codes
    s1/scripted/default             # offline, deterministic
    s1/typesafe/jev-latest          # TypeSafe System One ($TYPESAFE_API_KEY)

The handler parses ASSERT's transcript-judge request (system prompt + strict json_schema),
asks one constrained categorical question per custom dimension and per taxonomy behaviour,
and returns exactly the judge JSON the schema requires. Built-in ``policy_violation`` /
``overrefusal`` are derived from the node judgments the way ASSERT's normaliser enforces.
Per-decision probabilities/confidence appear in the justification strings, in
``response._hidden_params["s1"]`` and (optionally) in the JSONL sidecar ``$CI_S1_JUDGE_LOG``.

If the request is not a transcript-judge call, a decision abstains, or the backend fails, the
handler falls back to plain chat judging with ``$CI_S1_FALLBACK_MODEL`` (default
``openai/local``) using the original messages and parameters (C25).

Integration (ASSERT wrapper process, once before ``assert_ai.cli`` runs)::

    from ci_lab.judge.provider import register; register()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import litellm
from litellm import CustomLLM

from ci_lab.judge.assert_contract import (
    Abstained,
    JudgeRequest,
    Unsupported,
    build_questions,
    build_verdict,
    parse_request,
)
from ci_lab.judge.backends import BackendError, make_backend

log = logging.getLogger(__name__)

PROVIDER = "s1"
FALLBACK_ENV = "CI_S1_FALLBACK_MODEL"
DEFAULT_FALLBACK = "openai/local"
SIDECAR_ENV = "CI_S1_JUDGE_LOG"
RUBRICS_ENV = "CI_S1_RUBRICS"  # optional JSON/YAML {dimension: rubric} override (candidate rubrics)


class S1Fallback(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def split_model(model: str) -> tuple[str, str]:
    """``[s1/]<backend>/<model>`` -> (backend, model)."""
    name = model[len(PROVIDER) + 1:] if model.startswith(PROVIDER + "/") else model
    backend, _, rest = name.partition("/")
    if not backend:
        raise BackendError(f"bad s1 model string {model!r}; expected s1/<backend>/<model>")
    return backend, rest


def _load_rubrics(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        import yaml

        data = yaml.safe_load(text)
    rubrics = data.get("rubrics", data) if isinstance(data, dict) else None
    if not isinstance(rubrics, dict):
        raise TypeError(f"{path}: expected a mapping of dimension -> rubric")
    return {str(k): str(v) for k, v in rubrics.items()}


class S1JudgeLLM(CustomLLM):
    """LiteLLM handler for ``s1/...`` model strings (sync + async)."""

    def __init__(self, *, fallback_model: str | None = None, sidecar: str | None = None,
                 rubrics: dict[str, str] | None = None) -> None:
        super().__init__()
        self.fallback_model = fallback_model
        self.sidecar = sidecar
        self.rubrics = rubrics
        self._backends: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self.stats = {"s1": 0, "fallback": 0}

    # ------------------------------------------------------------ helpers
    def _fallback_model(self) -> str:
        m = self.fallback_model or os.environ.get(FALLBACK_ENV) or DEFAULT_FALLBACK
        if m.startswith(PROVIDER + "/"):
            raise BackendError(f"fallback model {m!r} must not be an s1/ model")
        return m

    def _backend(self, backend: str, model: str, api_base: str | None, api_key: str | None) -> Any:
        key = (backend, model, api_base, hashlib.sha256((api_key or "").encode()).hexdigest())
        with self._lock:
            if key not in self._backends:
                self._backends[key] = make_backend(backend, model, api_base=api_base, api_key=api_key)
            return self._backends[key]

    def _rubrics(self) -> dict[str, str] | None:
        if self.rubrics is None and os.environ.get(RUBRICS_ENV):
            self.rubrics = _load_rubrics(os.environ[RUBRICS_ENV])
        return self.rubrics

    def judge(self, model: str, messages: list[dict[str, Any]], optional_params: dict[str, Any],
              api_base: str | None = None, api_key: str | None = None) -> tuple[str, dict[str, Any]]:
        """Return (verdict JSON text, s1 metadata) or raise :class:`S1Fallback`."""
        try:
            req: JudgeRequest = parse_request(messages, optional_params or {})
            backend_name, backend_model = split_model(model)
            backend = self._backend(backend_name, backend_model, api_base, api_key)
            questions = build_questions(req, self._rubrics())
            t0 = time.perf_counter()
            decision = backend.decide(req.transcript, questions)
            verdict, meta = build_verdict(req, questions, decision.answers, label=backend_name)
        except (Unsupported, Abstained, BackendError) as e:
            raise S1Fallback(f"{type(e).__name__}: {e}") from e
        meta.update({
            "backend": backend_name,
            "model": decision.model,
            "questions": len(questions),
            "model_calls": decision.model_calls,
            "usage": decision.usage,
            "latency_s": round(time.perf_counter() - t0, 4),
            "provenance": backend.provenance(),
        })
        self._log_sidecar(model, req, verdict, meta)
        return json.dumps(verdict, ensure_ascii=False), meta

    def _log_sidecar(self, model: str, req: JudgeRequest, verdict: dict[str, Any], meta: dict[str, Any]) -> None:
        path = self.sidecar or os.environ.get(SIDECAR_ENV)
        if not path:
            return
        row = {"ts": time.time(), "model": model,
               "transcript_sha256": hashlib.sha256(req.transcript.encode()).hexdigest(),
               "dimensions": verdict["dimensions"], "s1": meta}
        with self._lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _response(self, model: str, content: str, meta: dict[str, Any]) -> litellm.ModelResponse:
        usage = meta.get("usage") or {}
        pt = int(usage.get("input_tokens") or 0)
        ct = int(usage.get("output_tokens") or 0)
        resp = litellm.ModelResponse(
            model=f"{PROVIDER}/{model}",
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            usage=litellm.Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        )
        resp._hidden_params["s1"] = meta
        return resp

    def _fallback_kwargs(self, messages: list, optional_params: dict[str, Any], timeout: Any) -> dict[str, Any]:
        kw = {k: v for k, v in (optional_params or {}).items() if v is not None}
        kw.update(model=self._fallback_model(), messages=messages)
        if timeout is not None and not hasattr(timeout, "connect"):
            kw["timeout"] = timeout
        return kw

    def _note_fallback(self, resp: Any, reason: str) -> Any:
        self.stats["fallback"] += 1
        log.warning("s1 judge falling back to %s: %s", self._fallback_model(), reason)
        hidden = getattr(resp, "_hidden_params", None)
        if isinstance(hidden, dict):
            hidden["s1_fallback"] = reason
        return resp

    # ------------------------------------------------------------ LiteLLM entry points
    def completion(self, model: str, messages: list, **kwargs: Any) -> Any:  # type: ignore[override]
        optional_params = kwargs.get("optional_params") or {}
        try:
            content, meta = self.judge(model, messages, optional_params, kwargs.get("api_base"), kwargs.get("api_key"))
        except S1Fallback as fb:
            resp = litellm.completion(**self._fallback_kwargs(messages, optional_params, kwargs.get("timeout")))
            return self._note_fallback(resp, fb.reason)
        self.stats["s1"] += 1
        return self._response(model, content, meta)

    async def acompletion(self, model: str, messages: list, **kwargs: Any) -> Any:  # type: ignore[override]
        optional_params = kwargs.get("optional_params") or {}
        try:
            content, meta = await asyncio.to_thread(self.judge, model, messages, optional_params,
                                                    kwargs.get("api_base"), kwargs.get("api_key"))
        except S1Fallback as fb:
            resp = await litellm.acompletion(**self._fallback_kwargs(messages, optional_params, kwargs.get("timeout")))
            return self._note_fallback(resp, fb.reason)
        self.stats["s1"] += 1
        return self._response(model, content, meta)


_REGISTER_LOCK = threading.Lock()


def register(*, force: bool = False, **handler_kwargs: Any) -> S1JudgeLLM:
    """Register the ``s1`` provider in ``litellm.custom_provider_map`` (idempotent).

    Returns the active handler. ``force=True`` replaces an existing ``s1`` handler.
    """
    with _REGISTER_LOCK:
        current = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == PROVIDER]
        if current and not force:
            return current[0]["custom_handler"]
        handler = S1JudgeLLM(**handler_kwargs)
        others = [e for e in (litellm.custom_provider_map or []) if e.get("provider") != PROVIDER]
        litellm.custom_provider_map = [*others, {"provider": PROVIDER, "custom_handler": handler}]
        litellm.utils.custom_llm_setup()
        return handler


def unregister() -> None:
    with _REGISTER_LOCK:
        litellm.custom_provider_map = [e for e in (litellm.custom_provider_map or [])
                                       if e.get("provider") != PROVIDER]
