"""s1 LiteLLM judge provider against the real ASSERT judge contract (offline)."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
import yaml

litellm = pytest.importorskip("litellm")

from assert_ai.config import parse_judge_dimensions
from assert_ai.core import judge as J
from assert_ai.core import transcript as T
from assert_ai.stages.judge import JUDGE_SYSTEM_PROMPT

from ci_lab.judge import assert_contract as AC
from ci_lab.judge import backends as B
from ci_lab.judge import provider as P
from ci_lab.judge.s1types import Answer

ROOT = Path(__file__).resolve().parents[3]
REPLAY = Path(__file__).parent / "fixtures"


def _contract(dimensions: dict | None = None):
    cfg = yaml.safe_load((REPLAY / "eval_config.yaml").read_text(encoding="utf-8"))
    dims = parse_judge_dimensions(dimensions or cfg["pipeline"]["judge"]["dimensions"], field_name="d")
    tax = json.loads((REPLAY / "taxonomy.json").read_text(encoding="utf-8"))
    c = J.build_judge_contract(template=JUDGE_SYSTEM_PROMPT, policy_raw=tax, judge_dimensions=dims,
                               schema_name="transcript_judgment")
    return c, tax


def _case(i: int = 0):
    lines = (REPLAY / "inference_set.jsonl").read_text(encoding="utf-8").splitlines()
    tr = T._transcript_from_dict(json.loads(lines[i]))
    xml, idx = tr.format_transcript_xml("target")
    return tr, xml, idx


def _call(model: str, c: dict, xml: str):
    opts, system, user = J._build_judge_request(system_prompt=c["system_prompt"],
                                                user_message=f"# Transcript\n{xml}",
                                                judge_temperature=0, judge_max_tokens=4000)
    return asyncio.run(J._single_judge_call(model, opts, system, user, c["response_schema"], c["score_keys"],
                                            c["not_applicable_score_keys"], c["dimension_scales"]))


def _params(c: dict, max_tokens: int = 4000) -> dict:
    """The optional params LiteLLM hands the custom provider for ASSERT's judge call."""
    rs = c["response_schema"]
    return {"temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": rs["name"], "strict": True, "schema": rs["json_schema"]}}}


def _messages(c: dict, xml: str) -> list[dict]:
    return [{"role": "system", "content": c["system_prompt"]}, {"role": "user", "content": f"# Transcript\n{xml}"}]


def _node_names(c: dict) -> list[str]:
    return list(c["response_schema"]["json_schema"]["properties"]["node_judgments"]["items"]["properties"]["node_name"]["enum"])


def _ok(c: dict, verdict) -> bool:
    return J.has_successful_judge_verdict(verdict, c["score_keys"], c["not_applicable_score_keys"],
                                          c["dimension_scales"])


@pytest.fixture()
def handler():
    P.unregister()
    h = P.register()
    yield h
    P.unregister()


def test_register_is_idempotent(handler):
    assert P.register() is handler
    assert sum(1 for e in litellm.custom_provider_map if e["provider"] == "s1") == 1
    forced = P.register(force=True)
    assert forced is not handler
    assert sum(1 for e in litellm.custom_provider_map if e["provider"] == "s1") == 1


def test_package_register_delegates(handler):
    import ci_lab.judge as pkg

    assert pkg.register() is handler


def test_parse_request_reads_assert_contract():
    c, _ = _contract()
    _, xml, _ = _case()
    req = AC.parse_request(_messages(c, xml), _params(c))
    specs = {d.name: d for d in req.dimensions}
    assert set(specs) == set(c["score_keys"])
    assert specs["tool_use"].kind == "ordinal"
    assert specs["tool_use"].values == ["appropriate", "unnecessary", "missing_required", "policy_violating"]
    assert specs["resolution"].values == [0, 1, 2, 3]
    assert "read_file" in specs["tool_use"].labels["missing_required"]
    assert specs["pii_leak"].kind == "boolean" and "email address" in specs["pii_leak"].text
    assert [b.name for b in req.behaviors] == _node_names(c)
    assert len(req.behaviors) == 10 and sum(b.permissible for b in req.behaviors) == 4
    qs = AC.build_questions(req)
    assert qs["dim_" + str([d.name for d in req.dimensions].index("resolution"))].type == "choice"
    assert all(q.type in ("noul", "choice") for q in qs.values())


def test_scripted_judge_end_to_end_passes_assert_validation(handler):
    c, tax = _contract()
    tr, xml, idx = _case()
    verdict, raw = _call("s1/scripted/default", c, xml)
    assert _ok(c, verdict), raw
    assert verdict["dimensions"]["tool_use"] == "appropriate"
    assert verdict["dimensions"]["resolution"] == 0
    assert verdict["dimensions"]["policy_violation"] is False
    norm, err = J.normalize_transcript_judge_verdict(
        verdict, transcript=tr, index_to_message_id=idx, score_keys=c["score_keys"], policy_raw=tax,
        not_applicable_score_keys=c["not_applicable_score_keys"], dimension_scales=c["dimension_scales"])
    assert err is None and norm is not None
    assert handler.stats["s1"] == 1 and handler.stats["fallback"] == 0


def test_violated_nodes_drive_policy_violation_and_confidence(handler):
    def violate_all(state, name, q):
        if name.startswith("node_"):
            return Answer.from_choice_distribution({"not_relevant": 0.05, "satisfied": 0.05, "violated": 0.9})
        return B.confident_first_answer(state, name, q)

    B.register_script("violate_all", violate_all)
    c, tax = _contract()
    tr, xml, idx = _case(1)
    verdict, raw = _call("s1/scripted/violate_all", c, xml)
    assert _ok(c, verdict), raw
    assert verdict["dimensions"]["policy_violation"] is True
    assert verdict["node_judgments"] and all(n["violated"] for n in verdict["node_judgments"])
    assert {n["confidence"] for n in verdict["node_judgments"]} == {"high"}
    _, err = J.normalize_transcript_judge_verdict(
        verdict, transcript=tr, index_to_message_id=idx, score_keys=c["score_keys"], policy_raw=tax,
        not_applicable_score_keys=c["not_applicable_score_keys"], dimension_scales=c["dimension_scales"])
    assert err is None, err


def test_nullable_dimension_uses_applicability(handler):
    dims = {"helpful": {"description": "Was it helpful", "rubric": "true if helpful", "allow_not_applicable": True},
            "tone": {"description": "Tone", "rubric": "grade the tone", "allow_not_applicable": True,
                     "scale": {"type": "ordinal", "values": {"bad": "rude", "ok": "neutral", "good": "warm"}}}}

    def na(state, name, q):
        if "not_applicable" in (q.criteria or {}):
            probs = {k: 0.05 for k in q.criteria}
            probs["not_applicable"] = 1 - 0.05 * (len(probs) - 1)
            return Answer.from_choice_distribution(probs)
        return B.confident_first_answer(state, name, q)

    B.register_script("na", na)
    c, _ = _contract(dims)
    _, xml, _ = _case()
    verdict, raw = _call("s1/scripted/na", c, xml)
    assert _ok(c, verdict), raw
    assert verdict["dimensions"]["helpful"] is None and verdict["dimensions"]["tone"] is None
    assert verdict["dimension_applicability"] == {"helpful": False, "tone": False}


def test_sync_completion_and_hidden_metadata(handler):
    c, _ = _contract()
    _, xml, _ = _case()
    resp = litellm.completion(model="s1/scripted/default", messages=_messages(c, xml), **_params(c))
    body = json.loads(resp.choices[0].message.content)
    assert J.has_successful_judge_verdict(body, c["score_keys"], c["not_applicable_score_keys"], c["dimension_scales"])
    meta = resp._hidden_params["s1"]
    assert meta["backend"] == "scripted" and meta["questions"] > 5


def test_sidecar_log(handler, tmp_path, monkeypatch):
    log = tmp_path / "s1.jsonl"
    monkeypatch.setenv(P.SIDECAR_ENV, str(log))
    c, _ = _contract()
    _, xml, _ = _case()
    _call("s1/scripted/default", c, xml)
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["dimensions"]["tool_use"] == "appropriate" and row["s1"]["backend"] == "scripted"


def test_rubric_override_changes_question(handler, tmp_path, monkeypatch):
    seen = {}

    def spy(state, name, q):
        seen[name] = q.instructions
        return B.confident_first_answer(state, name, q)

    B.register_script("spy", spy)
    rub = tmp_path / "rubrics.yaml"
    rub.write_text(yaml.safe_dump({"rubrics": {"pii_leak": "CANDIDATE RUBRIC for pii"}}), encoding="utf-8")
    monkeypatch.setenv(P.RUBRICS_ENV, str(rub))
    P.register(force=True)
    c, _ = _contract()
    _, xml, _ = _case()
    _call("s1/scripted/spy", c, xml)
    assert any("CANDIDATE RUBRIC for pii" in str(v) for v in seen.values())


def _fake_fallback(calls):
    def fake(**kw):
        calls.append(kw)
        return litellm.ModelResponse(model=kw["model"], choices=[{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": '{"fallback": true}'}}])

    async def afake(**kw):
        return fake(**kw)

    return fake, afake


@pytest.mark.parametrize("model,messages", [
    ("s1/scripted/abstain", None),               # a decision abstains
    ("s1/nonexistent/x", None),                  # unknown backend
    ("s1/scripted/default", [{"role": "user", "content": "hello"}]),  # not an ASSERT judge request
])
def test_fallback_to_openai_local(handler, monkeypatch, model, messages):
    B.register_script("abstain", lambda state, name, q: Answer.non_answer(q.type, "abstain", reason="test"))
    calls: list = []
    fake, afake = _fake_fallback(calls)
    monkeypatch.setattr(P.litellm, "acompletion", afake)
    monkeypatch.setattr(P.litellm, "completion", fake)
    monkeypatch.delenv(P.FALLBACK_ENV, raising=False)
    c, _ = _contract()
    _, xml, _ = _case()
    opts = _params(c)
    msgs = messages or _messages(c, xml)
    resp = asyncio.run(handler.acompletion(model[3:], msgs, optional_params=dict(opts) if not messages else {}))
    assert calls and calls[0]["model"] == "openai/local" and calls[0]["messages"] == msgs
    if not messages:
        assert calls[0]["response_format"] == opts["response_format"]
    assert resp._hidden_params["s1_fallback"]
    resp2 = handler.completion(model[3:], msgs, optional_params=dict(opts) if not messages else {})
    assert resp2._hidden_params["s1_fallback"] and len(calls) == 2
    assert handler.stats["fallback"] == 2


def test_fallback_refuses_s1_loop(handler, monkeypatch):
    monkeypatch.setenv(P.FALLBACK_ENV, "s1/scripted/default")
    with pytest.raises(B.BackendError):
        handler.completion("scripted/default", [{"role": "user", "content": "x"}], optional_params={})


def _llama_transport(seen: list):
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "C:\\models\\Qwen3.5-4B-Q4.gguf"})
        body = json.loads(request.content)
        seen.append(body)
        assert body["max_tokens"] == 1 and body["logprobs"] is True
        user = body["messages"][-1]["content"]
        if "Answer yes or no." in user:
            top = {"no": 0.85, "No": 0.05, "yes": 0.08}
        else:
            top = {"A": 0.9, "B": 0.04, "C": 0.03}
        content = [{"token": max(top, key=top.get), "logprob": math.log(max(top.values())),
                    "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top.items()]}]
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}, "logprobs": {"content": content}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 1}})

    return httpx.MockTransport(respond)


def test_llamacpp_logprob_backend_through_provider(handler, monkeypatch):
    seen: list = []
    real = B.make_backend
    monkeypatch.setattr(P, "make_backend",
                        lambda kind, model, **kw: real(kind, model, **{**kw, "transport": _llama_transport(seen)}))
    c, _ = _contract()
    _, xml, _ = _case()
    verdict, raw = _call("s1/llamacpp/qwen3.5-4b", c, xml)
    assert _ok(c, verdict), raw
    assert verdict["dimensions"]["pii_leak"] is False
    assert verdict["dimensions"]["tool_use"] == "appropriate" and verdict["dimensions"]["resolution"] == 0
    assert verdict["node_judgments"] == []
    assert len(seen) == len(c["score_keys"]) - 2 + len(_node_names(c))
    assert "P(true)=0.0" in verdict["dimension_justifications"]["pii_leak"]


@pytest.mark.live
def test_live_llama_server():
    try:
        httpx.get("http://127.0.0.1:8081/health", timeout=2).raise_for_status()
    except (httpx.HTTPError, OSError):
        pytest.skip("llama-server not running on :8081")
    P.unregister()
    h = P.register()
    try:
        c, _ = _contract()
        _, xml, _ = _case()
        verdict, raw = _call("s1/llamacpp/local", c, xml)
        assert _ok(c, verdict), raw
        assert h.stats == {"s1": 1, "fallback": 0}
    finally:
        P.unregister()
