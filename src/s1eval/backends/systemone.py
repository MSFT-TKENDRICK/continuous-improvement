"""Client for any TypeSafe System One-compatible endpoint (POST {base}/v1/systemone).

Works against: TypeSafe (https://api.typesafe.ai, model ``jev-latest``), the LangSmith LLM
Gateway (https://gateway.smith.langchain.com, e.g. ``semif-qwen3.5-4b`` or
``typesafe/jev-1.13.0``), OpenRouter (https://openrouter.ai/api, ``~typesafe/jev-latest``), and
``s1eval serve`` (local). Uses raw httpx so one code path serves every compatible server and
tests can inject ``httpx.MockTransport``.
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

from ..types import Answer, Question, WireError, questions_to_wire
from .base import BackendError, Decision

RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class SystemOneBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        max_questions_per_request: int = 32,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_q = max_questions_per_request
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.name = name or f"systemone:{model}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(extra_headers or {})
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)
        self._secrets = [api_key] if api_key else []
        self._served_model: str | None = None

    def close(self) -> None:
        self._client.close()

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                r = self._client.post("/v1/systemone", json=body)
            except httpx.TransportError as e:
                if attempts > self.max_retries:
                    raise BackendError(f"transport error after {attempts} attempts: {type(e).__name__}") from e
                self._sleep(attempts, None)
                continue
            if r.status_code in RETRY_STATUSES and attempts <= self.max_retries:
                self._sleep(attempts, r.headers.get("retry-after"))
                continue
            if r.status_code != 200:
                # Never echo request bodies or auth headers; truncate server text.
                text = r.text[:300]
                for secret in self._secrets:
                    text = text.replace(secret, "[redacted]")
                raise BackendError(f"HTTP {r.status_code} from {self.base_url}/v1/systemone: {text}")
            try:
                return r.json(), attempts
            except ValueError as e:
                raise BackendError("response was not JSON") from e

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        delay = self.backoff_base * (2 ** (attempt - 1)) * (0.5 + random.random())
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(min(delay, 60.0))

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision:
        t0 = time.perf_counter()
        names = list(questions)
        answers: dict[str, Answer] = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        http_calls = 0
        for i in range(0, len(names), self.max_q):
            chunk = {k: questions[k] for k in names[i : i + self.max_q]}
            body = {"model": self.model, "state": state, "questions": questions_to_wire(chunk)}
            data, attempts = self._post(body)
            http_calls += attempts
            if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                raise BackendError("response missing 'answers' object")
            self._served_model = data.get("model", self._served_model)
            for k, q in chunk.items():
                if k not in data["answers"]:
                    raise BackendError(f"response missing answer for question {k!r}")
                try:
                    answers[k] = Answer.from_wire(data["answers"][k], q, where=f"answers.{k}")
                except WireError as e:
                    raise BackendError(str(e)) from e
            u = data.get("usage") or {}
            for key in usage:
                if isinstance(u.get(key), int):
                    usage[key] += u[key]
        return Decision(
            answers=answers,
            model=self._served_model or self.model,
            usage=usage,
            http_calls=http_calls,
            model_calls=http_calls,
            latency_s=time.perf_counter() - t0,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "systemone",
            "base_url": self.base_url,
            "requested_model": self.model,
            "served_model": self._served_model,
        }
