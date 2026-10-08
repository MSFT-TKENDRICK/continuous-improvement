"""Adapter for the OpenAI Decisions API (POST {base}/v1/decisions, e.g. model ``gpt-6-luna``).

Mapping (TypeSafe -> OpenAI):
  noul   -> {type: "predicate", name, instructions}   (noul criteria folded into instructions)
  choice -> {type: "choice", name, instructions, choices: [{value, description}]}
  score  -> {type: "score", name, instructions, levels: [{label, description}]}
Answers come back as a list in question order; ``refusal`` answers become status="refusal"
(never mapped to false / 0 / the first option).

Note: third-party "OpenAI Decisions" sites document a different, incorrect schema; this
follows the official platform.openai.com guide and the openai-python reference.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..types import Answer, Question, WireError, render_text
from .base import BackendError, Decision
from .systemone import RETRY_STATUSES


def _instructions(q: Question) -> str:
    text = render_text(q.instructions)
    if q.type == "noul" and q.criteria:
        parts = [text] if text else []
        if q.criteria.get("true") is not None:
            parts.append(f"True when: {render_text(q.criteria['true'])}")
        if q.criteria.get("false") is not None:
            parts.append(f"False when: {render_text(q.criteria['false'])}")
        text = "\n".join(parts)
    return text


def level_label(i: int) -> str:
    return f"level_{i}"


def question_to_openai(name: str, q: Question) -> dict[str, Any]:
    d: dict[str, Any] = {"name": name, "instructions": _instructions(q)}
    if q.type == "noul":
        d["type"] = "predicate"
    elif q.type == "choice":
        d["type"] = "choice"
        d["choices"] = [
            {"value": opt, **({"description": render_text(desc)} if desc is not None else {})}
            for opt, desc in q.criteria.items()
        ]
    else:
        d["type"] = "score"
        d["levels"] = [{"label": level_label(i), "description": render_text(desc)} for i, desc in enumerate(q.criteria)]
    return d


def answer_from_openai(a: dict[str, Any], q: Question) -> Answer:
    t = a.get("type")
    if t == "refusal":
        return Answer.non_answer(q.type, "refusal", provider="openai")
    if q.type == "noul":
        if t != "predicate":
            raise WireError(f"expected predicate answer, got {t!r}")
        return Answer.from_noul_probability(a.get("probability"))
    if t != q.type:
        raise WireError(f"expected {q.type} answer, got {t!r}")
    plist = a.get("probabilities")
    if not isinstance(plist, list):
        raise WireError("probabilities must be a list")
    if q.type == "choice":
        probs = {str(p["value"]): float(p["probability"]) for p in plist}
        if set(probs) != set(q.criteria):
            raise WireError(f"choice probabilities {sorted(probs)} != options {sorted(q.criteria)}")
        if abs(sum(probs.values()) - 1) > 0.02:
            raise WireError("choice probabilities do not sum to ~1")
        ordered = {k: probs[k] for k in q.criteria}
        ans = Answer.from_choice_distribution(ordered, provider="openai")
        if a.get("choice") is not None and str(a["choice"]) != ans.choice:
            ans.diagnostics["provider_choice"] = a["choice"]
        if isinstance(a.get("confidence"), (int, float)):
            ans.confidence = float(a["confidence"])
        return ans
    n = len(q.criteria)
    by_label = {level_label(i): i for i in range(n)}
    probs_l = [0.0] * n
    seen = set()
    for p in plist:
        idx = by_label.get(p.get("label"))
        if idx is None:
            idx = p.get("value")
        if not isinstance(idx, int) or not 0 <= idx < n or idx in seen:
            raise WireError(f"bad score probability entry {p!r}")
        seen.add(idx)
        probs_l[idx] = float(p["probability"])
    if len(seen) != n or abs(sum(probs_l) - 1) > 0.02:
        raise WireError("score probabilities incomplete or do not sum to ~1")
    ans = Answer.from_score_distribution(probs_l, list(q.criteria), provider="openai")
    if isinstance(a.get("score"), (int, float)):
        ans.score = float(a["score"])
    if isinstance(a.get("confidence"), (int, float)):
        ans.confidence = float(a["confidence"])
    return ans


class OpenAIDecisionsBackend:
    def __init__(
        self,
        model: str = "gpt-6-luna",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        *,
        timeout: float = 120.0,
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.name = f"openai-decisions:{model}"
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._api_key = api_key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision:
        t0 = time.perf_counter()
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        body = {
            "model": self.model,
            "input": text,
            "questions": [question_to_openai(k, q) for k, q in questions.items()],
        }
        attempts = 0
        while True:
            attempts += 1
            try:
                r = self._client.post("/v1/decisions", json=body)
            except httpx.TransportError as e:
                if attempts > self.max_retries:
                    raise BackendError(f"transport error after {attempts} attempts: {type(e).__name__}") from e
                time.sleep(min(2 ** (attempts - 1), 30))
                continue
            if r.status_code in RETRY_STATUSES and attempts <= self.max_retries:
                time.sleep(min(2 ** (attempts - 1), 30))
                continue
            break
        if r.status_code != 200:
            text = r.text
            if self._api_key:
                text = text.replace(self._api_key, "[redacted]")
            raise BackendError(f"HTTP {r.status_code} from OpenAI decisions: {text[:300]}")
        try:
            data = r.json()
        except ValueError as e:
            raise BackendError("response was not JSON") from e
        if not isinstance(data, dict):
            raise BackendError("response was not a JSON object")
        raw = data.get("answers")
        if not isinstance(raw, list):
            raise BackendError("response missing answers list")
        by_name = {a.get("name"): a for a in raw if isinstance(a, dict)}
        answers = {}
        for k, q in questions.items():
            if k not in by_name:
                raise BackendError(f"missing answer for {k!r}")
            try:
                answers[k] = answer_from_openai(by_name[k], q)
            except (WireError, KeyError, TypeError, ValueError) as e:
                raise BackendError(f"answers.{k}: {e}") from e
        u = data.get("usage") or {}
        return Decision(
            answers=answers,
            model=data.get("model", self.model),
            usage={"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")},
            http_calls=attempts,
            model_calls=attempts,
            latency_s=time.perf_counter() - t0,
        )

    def provenance(self) -> dict[str, Any]:
        return {"backend": "openai-decisions", "base_url": self.base_url, "model": self.model}
