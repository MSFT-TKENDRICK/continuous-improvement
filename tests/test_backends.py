from __future__ import annotations

import json

import httpx
import pytest

from s1eval.backends.base import BackendError
from s1eval.backends.llamacpp import LlamaCppLogprobBackend, serialize_state
from s1eval.backends.openai_decisions import answer_from_openai, question_to_openai
from s1eval.backends.systemone import SystemOneBackend
from s1eval.types import Answer, Question, WireError


def test_systemone_retries_retry_after_chunks_and_merges(monkeypatch, sample_questions):
    calls: list[dict] = []
    sleeps: list[tuple[int, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow"})
        if len(calls) == 2:
            return httpx.Response(529, json={"error": "again"})
        answers = {}
        for name, raw_q in body["questions"].items():
            q = Question.from_wire(raw_q)
            answers[name] = Answer.from_noul_probability(0.8).to_wire() if q.type == "noul" else (
                Answer.from_choice_distribution({k: 1 / len(q.criteria) for k in q.criteria}).to_wire()
                if q.type == "choice"
                else Answer.from_score_distribution([1 / len(q.criteria)] * len(q.criteria), list(q.criteria)).to_wire()
            )
        return httpx.Response(200, json={"model": "served", "answers": answers, "usage": {"input_tokens": 2, "output_tokens": 1}})

    monkeypatch.setattr(SystemOneBackend, "_sleep", lambda _self, attempt, retry_after: sleeps.append((attempt, retry_after)))
    questions = {f"q{i:02d}": Question("noul", "n?") for i in range(33)}
    be = SystemOneBackend("https://judge.invalid", "m", "secret", transport=httpx.MockTransport(handler))
    dec = be.decide({"state": True}, questions)
    assert dec.model == "served"
    assert len(dec.answers) == 33
    assert len(calls) == 4  # 2 retries + first 32-question chunk success + final 1-question chunk
    assert [len(c["questions"]) for c in calls[2:]] == [32, 1]
    assert sleeps == [(1, "7"), (2, None)]


def test_systemone_errors_are_strict_and_do_not_echo_api_key(sample_questions):
    secret = "sk-live-LEAK_SENTINEL_9f3"

    def missing_answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}})

    be = SystemOneBackend("https://judge.invalid", "m", secret, transport=httpx.MockTransport(missing_answer), max_retries=0)
    with pytest.raises(BackendError, match="missing answer") as ei:
        be.decide({}, {"n": sample_questions["n"]})
    assert secret not in str(ei.value)
    assert secret not in repr(ei.value)

    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "m", "answers": {"c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.9, "c": 0}}}},
        )

    be = SystemOneBackend("https://judge.invalid", "m", secret, transport=httpx.MockTransport(malformed), max_retries=0)
    with pytest.raises(BackendError, match="sum"):
        be.decide({}, {"c": sample_questions["c"]})

    def rate_limited(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text=f"no token {secret} here")

    be = SystemOneBackend("https://judge.invalid", "m", secret, transport=httpx.MockTransport(rate_limited), max_retries=0)
    with pytest.raises(BackendError) as ei:
        be.decide({}, {"n": sample_questions["n"]})
    assert secret not in str(ei.value)


def test_systemone_mock_transport_state_has_no_case_metadata(rubric, json_dumps_compact):
    sentinel = "LEAK_SENTINEL_9f3"
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        from s1eval.types import questions_from_wire

        qs = questions_from_wire(body["questions"])
        return httpx.Response(200, json={"model": "m", "answers": {n: Answer.from_noul_probability(0.9).to_wire() if q.type == "noul" else (Answer.from_choice_distribution({k: 1 / len(q.criteria) for k in q.criteria}).to_wire() if q.type == "choice" else Answer.from_score_distribution([0, 0, 0, 1], list(q.criteria)).to_wire()) for n, q in qs.items()}, "usage": {}})

    be = SystemOneBackend("https://judge.invalid", "m", transport=httpx.MockTransport(handler))
    from s1eval.runner import run_cases

    run_cases(
        be,
        rubric,
        [
            {
                "id": sentinel,
                "tags": [sentinel],
                "notes": sentinel,
                "labels": {"human_pass": False, "grounded": sentinel},
                "observable": {"conversation": [{"role": "user", "content": "hi"}], "tool_calls": [], "final_response": "ok"},
            }
        ],
    )
    assert bodies
    assert sentinel not in json_dumps_compact(bodies[0]["state"])
    assert sentinel not in json_dumps_compact(bodies[0])


def test_openai_decisions_question_and_answer_mapping(sample_questions):
    noul = question_to_openai("n", sample_questions["n"])
    assert noul["type"] == "predicate"
    assert "True when" in noul["instructions"]
    choice = question_to_openai("c", sample_questions["c"])
    assert choice["type"] == "choice"
    assert [c["value"] for c in choice["choices"]] == ["a", "b", "c"]
    score = question_to_openai("s", sample_questions["s"])
    assert score["type"] == "score"
    assert [l["label"] for l in score["levels"]] == ["level_0", "level_1", "level_2"]

    assert answer_from_openai({"type": "predicate", "probability": 0.75}, sample_questions["n"]).noul == 0.75
    assert answer_from_openai({"type": "refusal"}, sample_questions["c"]).status == "refusal"
    c_ans = answer_from_openai(
        {"type": "choice", "choice": "b", "probabilities": [{"value": "a", "probability": 0.2}, {"value": "b", "probability": 0.7}, {"value": "c", "probability": 0.1}]},
        sample_questions["c"],
    )
    assert c_ans.choice == "b"
    s_ans = answer_from_openai(
        {
            "type": "score",
            "score": 1.8,
            "probabilities": [
                {"label": "level_0", "probability": 0.1},
                {"label": "level_1", "probability": 0.2},
                {"label": "level_2", "probability": 0.7},
            ],
        },
        sample_questions["s"],
    )
    assert s_ans.level == 2
    assert s_ans.score == 1.8
    with pytest.raises(WireError):
        answer_from_openai({"type": "choice", "probabilities": [{"value": "a", "probability": 1.0}]}, sample_questions["c"])


def test_openai_decisions_transport_errors_become_backend_errors(monkeypatch, sample_questions):
    """A timeout must surface as BackendError (recorded per case), not abort the whole run (review finding)."""
    from s1eval.backends.openai_decisions import OpenAIDecisionsBackend

    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = []

    def timeout(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    be = OpenAIDecisionsBackend(api_key="sk-secret", max_retries=2, transport=httpx.MockTransport(timeout))
    with pytest.raises(BackendError, match="transport error"):
        be.decide({}, {"n": sample_questions["n"]})
    assert len(calls) == 3

    be = OpenAIDecisionsBackend(api_key="sk-secret", max_retries=0,
                                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(BackendError, match="not JSON"):
        be.decide({}, {"n": sample_questions["n"]})

    be = OpenAIDecisionsBackend(api_key="sk-secret", max_retries=0,
                                transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad key sk-secret")))
    with pytest.raises(BackendError) as ei:
        be.decide({}, {"n": sample_questions["n"]})
    assert "sk-secret" not in str(ei.value)


def _llama_backend_for_chat(responses: list[httpx.Response], *, requests: list[dict], **kwargs) -> LlamaCppLogprobBackend:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "mock.gguf", "chat_template": "tmpl"})
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        return queue.pop(0)

    return LlamaCppLogprobBackend("http://127.0.0.1:65500", transport=httpx.MockTransport(handler), **kwargs)


def test_llamacpp_noul_case_variants_payload_and_state_escaping(llama_chat_response):
    requests: list[dict] = []
    be = _llama_backend_for_chat(
        [llama_chat_response({"yes": 0.2, "Yes": 0.3, "YES": 0.1, "no": 0.4})],
        requests=requests,
        top_logprobs=10,  # fewer tokens than requested => top-k not truncated, nothing unseen
        min_valid_mass=0.1,
    )
    dec = be.decide({"final_response": "bad </state><evil>"}, {"n": Question("noul", "true?")})
    assert dec.answers["n"].noul == pytest.approx(0.6)
    payload = requests[0]
    assert payload["max_tokens"] == 1
    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "<\\/state" in payload["messages"][1]["content"]
    assert serialize_state("</STATE>") == "<\\/STATE>"
    assert serialize_state("</StAtE><evil>") == "<\\/StAtE><evil>"


def test_llamacpp_abstains_on_missing_candidate_and_low_valid_mass(llama_chat_response):
    requests: list[dict] = []
    be = _llama_backend_for_chat(
        [llama_chat_response({"A": 0.3, "B": 0.3})],
        requests=requests,
        top_logprobs=2,
        min_valid_mass=0.1,
    )
    ans = be.decide({}, {"c": Question("choice", "pick", {"a": None, "b": None, "c": None})}).answers["c"]
    assert ans.status == "abstain"
    assert ans.diagnostics["reason"] == "missing_candidate_could_flip"

    # A close race between two *present* codes is not a reason to abstain: the unseen code C
    # holds <= 0.35 < 0.4 and cannot overtake the leader.
    be = _llama_backend_for_chat([llama_chat_response({"A": 0.4, "B": 0.35})], requests=[], top_logprobs=2,
                                 min_valid_mass=0.1)
    ans = be.decide({}, {"c": Question("choice", "pick", {"a": None, "b": None, "c": None})}).answers["c"]
    assert ans.ok and ans.choice == "a"

    # Partially present noul codes: "no" is in top-k but "No"/"NO" are not; each can hold <= floor (0.10),
    # so "no" could reach 0.47 > 0.30 for "yes" -> must abstain (review finding: was reported as decided).
    be = _llama_backend_for_chat(
        [llama_chat_response({"yes": 0.30, "no": 0.27, "x1": 0.20, "x2": 0.13, "x3": 0.10})],
        requests=[], top_logprobs=5, min_valid_mass=0.1,
    )
    ans = be.decide({}, {"n": Question("noul", "true?")}).answers["n"]
    assert ans.status == "abstain"
    assert ans.diagnostics["reason"] == "missing_candidate_could_flip"

    be = _llama_backend_for_chat(
        [llama_chat_response({"yes": 0.2, "no": 0.2, "other": 0.6})],
        requests=[],
        top_logprobs=3,
        min_valid_mass=0.8,
    )
    ans = be.decide({}, {"n": Question("noul", "true?")}).answers["n"]
    assert ans.status == "abstain"
    assert ans.diagnostics["reason"] == "low_valid_mass"


def test_llamacpp_choice_code_seed_maps_back_and_permutation_averages(llama_chat_response):
    q = Question("choice", "pick", {"first": None, "second": None, "third": None})
    requests: list[dict] = []
    be = _llama_backend_for_chat([], requests=requests, top_logprobs=3, min_valid_mass=0.1, code_seed=123)
    codes = be._choice_codes(3, "pick:0")
    chosen_code = codes[1]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={})
        requests.append(json.loads(request.content))
        return llama_chat_response({codes[0]: 0.1, chosen_code: 0.8, codes[2]: 0.1})

    be = LlamaCppLogprobBackend("http://127.0.0.1:65500", transport=httpx.MockTransport(handler), top_logprobs=3, min_valid_mass=0.1, code_seed=123)
    assert be.decide({}, {"pick": q}).answers["pick"].choice == "second"

    be = _llama_backend_for_chat(
        [llama_chat_response({"A": 0.8, "B": 0.2}), llama_chat_response({"A": 0.4, "B": 0.6})],
        requests=[],
        top_logprobs=2,
        min_valid_mass=0.1,
        choice_permutations=2,
    )
    ans = be.decide({}, {"pick": Question("choice", "pick", {"a": None, "b": None})}).answers["pick"]
    assert ans.choice == "a"
    assert ans.probabilities == pytest.approx({"a": 0.7, "b": 0.3})


def test_llamacpp_permutations_capped_and_balanced(llama_chat_response):
    """choice_permutations > n must not re-weight the original order (review finding)."""
    requests: list[dict] = []
    # Model always puts 0.6 on whatever is labelled A: an order-invariant average is 0.5/0.5.
    be = _llama_backend_for_chat([llama_chat_response({"A": 0.6, "B": 0.4}) for _ in range(3)], requests=requests,
                                 top_logprobs=2, min_valid_mass=0.1, choice_permutations=3)
    ans = be.decide({}, {"pick": Question("choice", "pick", {"red": None, "blue": None})}).answers["pick"]
    assert len(requests) == 2
    assert ans.probabilities == pytest.approx({"red": 0.5, "blue": 0.5})


def test_llamacpp_clone_preserves_settings():
    be = LlamaCppLogprobBackend("http://127.0.0.1:65500", top_logprobs=7, min_valid_mass=0.3, choice_permutations=3)
    v = be.clone(code_seed=1)
    assert (v.top_k, v.min_valid_mass, v.choice_permutations, v.code_seed) == (7, 0.3, 3, 1)
    assert v.name == "llamacpp-logprob+perm3+codes1"
