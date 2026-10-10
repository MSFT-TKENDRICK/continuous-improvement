import json
import math

import httpx
import pytest

from order_support.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    MicrosoftDecisionClient,
    MicrosoftDecisionError,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


def response_payload(*, answers=None, model="microsoft-decision-1", usage=None):
    return {
        "model": model,
        "answers": answers
        if answers is not None
        else {
            "gate": {"type": "noul", "noul": 0.9},
        },
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 12,
            "output_tokens": 0,
        },
    }


def make_client(handler, **kwargs):
    return MicrosoftDecisionClient(
        "https://decision-resource.services.ai.azure.com/",
        "support-decisions",
        api_key="api-secret",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_parses_all_answer_types_and_preserves_request_shape():
    captured = {}

    def handler(request):
        captured["request"] = request
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=response_payload(
                answers={
                    "gate": {"type": "noul", "noul": 0.8},
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "confidence": 0.72,
                        "probabilities": {"billing": 0.7, "support": 0.3},
                    },
                    "severity": {
                        "type": "score",
                        "score": 1.7,
                        "confidence": 0.81,
                        "legend": {"0": "low", "1": "medium", "2": "high"},
                        "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
                    },
                }
            ),
        )

    questions = {
        "gate": NoulQuestion(
            instructions="Is this actionable?",
            criteria={"true": "Actionable", "false": "Not actionable"},
        ),
        "route": ChoiceQuestion(
            instructions="Where should this go?",
            criteria={"billing": "Money", "support": "Everything else"},
        ),
        "severity": ScoreQuestion(
            instructions="How severe is it?",
            criteria=["low", "medium", "high"],
        ),
    }
    state = {"message": "I was charged twice.", "contacts": [1, 2]}

    with make_client(handler) as client:
        result = client.decide(state, questions)

    assert captured["url"] == (
        "https://decision-resource.services.ai.azure.com"
        "/providers/microsoft/v1/systemone"
    )
    assert captured["headers"]["api-key"] == "api-secret"
    assert "authorization" not in captured["headers"]
    assert captured["request"].extensions["timeout"] == {
        "connect": 30.0,
        "read": 30.0,
        "write": 30.0,
        "pool": 30.0,
    }
    assert captured["json"] == {
        "model": "support-decisions",
        "state": state,
        "questions": {
            "gate": {
                "type": "noul",
                "instructions": "Is this actionable?",
                "criteria": {"true": "Actionable", "false": "Not actionable"},
            },
            "route": {
                "type": "choice",
                "instructions": "Where should this go?",
                "criteria": {"billing": "Money", "support": "Everything else"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is it?",
                "criteria": ["low", "medium", "high"],
            },
        },
    }
    assert result.model == "microsoft-decision-1"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 0
    assert len(result.provenance) == 1
    assert result.provenance[0].deployment == "support-decisions"
    assert result.provenance[0].question_names == ("gate", "route", "severity")

    gate = result.answers["gate"]
    assert isinstance(gate, NoulAnswer)
    assert gate.probability == 0.8
    assert gate.verdict is True
    assert gate.confidence == 0.8

    route = result.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "billing"
    assert route.verdict == "billing"
    assert route.confidence == 0.72
    assert route.modal_choice == "billing"
    assert route.modal_probability == 0.7

    severity = result.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.score == 1.7
    assert severity.confidence == 0.81
    assert severity.modal_index == 2
    assert severity.modal_label == "high"
    assert severity.verdict == "high"


def test_bearer_auth_and_full_route_endpoint_are_normalized():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json=response_payload())

    client = MicrosoftDecisionClient(
        "https://example.test/providers/microsoft/v1/systemone/",
        "deployment",
        bearer_token="entra-secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.endpoint == "https://example.test"
        client.decide("hello", {"gate": NoulQuestion(instructions="Is it valid?")})
    finally:
        client.close()

    assert str(captured["request"].url) == (
        "https://example.test/providers/microsoft/v1/systemone"
    )
    assert captured["request"].headers["Authorization"] == "Bearer entra-secret"
    assert "api-key" not in captured["request"].headers


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"api_key": "key", "bearer_token": "token"},
        {"api_key": ""},
        {"bearer_token": " "},
    ],
)
def test_exactly_one_nonempty_credential_is_required(kwargs):
    with pytest.raises(ValueError, match="credential|exactly one"):
        MicrosoftDecisionClient("https://example.test", "deployment", **kwargs)


@pytest.mark.parametrize(
    "endpoint",
    [
        "example.test",
        "ftp://example.test",
        "https://user:password@example.test",
        "https://example.test?api-key=secret",
        "https://example.test#fragment",
    ],
)
def test_invalid_endpoints_are_rejected(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        MicrosoftDecisionClient(endpoint, "deployment", api_key="key")


def test_questions_and_state_are_validated_before_transport():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = make_client(handler)
    try:
        with pytest.raises(ValueError, match="question names"):
            client.decide("state", {"bad name": NoulQuestion(instructions="Question?")})
        with pytest.raises(ValueError, match="non-empty mapping"):
            client.decide("state", {})
        with pytest.raises(ValueError, match="must not be empty"):
            client.decide(" ", {"gate": NoulQuestion(instructions="Question?")})
        with pytest.raises(TypeError, match="object keys"):
            client.decide(
                {1: "not JSON"},
                {"gate": NoulQuestion(instructions="Question?")},
            )
        with pytest.raises(ValueError, match="non-finite"):
            client.decide(
                {"value": math.inf},
                {"gate": NoulQuestion(instructions="Question?")},
            )
    finally:
        client.close()
    assert calls == 0


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: NoulQuestion(), "require instructions"),
        (
            lambda: NoulQuestion(instructions="x", criteria={"maybe": "unclear"}),
            "true.*false",
        ),
        (lambda: ChoiceQuestion(criteria={}), "between 1 and 255"),
        (
            lambda: ChoiceQuestion(criteria={"bad option": "description"}),
            "question names",
        ),
        (lambda: ScoreQuestion(criteria=[]), "between 1 and 10"),
        (lambda: ScoreQuestion(criteria=["ok", ""]), "non-empty string"),
    ],
)
def test_question_definitions_are_validated(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_chunking_aggregates_usage_and_provenance():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        names = list(body["questions"])
        requests.append(names)
        answers = {
            name: {"type": "noul", "noul": index / 10}
            for index, name in enumerate(names, start=1)
        }
        return httpx.Response(
            200,
            json=response_payload(
                answers=answers,
                usage={"input_tokens": len(names) * 10, "output_tokens": len(names)},
            ),
        )

    questions = {
        f"q{index}": NoulQuestion(instructions=f"Question {index}?")
        for index in range(5)
    }
    with make_client(handler, max_questions_per_request=2) as client:
        result = client.decide("state", questions)

    assert requests == [["q0", "q1"], ["q2", "q3"], ["q4"]]
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 5
    assert [item.request_index for item in result.provenance] == [0, 1, 2]
    assert [item.usage.input_tokens for item in result.provenance] == [20, 20, 10]
    assert tuple(result.answers) == tuple(questions)


def test_chunking_rejects_inconsistent_underlying_models():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        name = next(iter(json.loads(request.content)["questions"]))
        return httpx.Response(
            200,
            json=response_payload(
                model=f"model-{calls}",
                answers={name: {"type": "noul", "noul": 0.5}},
            ),
        )

    questions = {
        "first": NoulQuestion(instructions="First?"),
        "second": NoulQuestion(instructions="Second?"),
    }
    with (
        make_client(handler, max_questions_per_request=1) as client,
        pytest.raises(MicrosoftDecisionError, match="model identities"),
    ):
        client.decide("state", questions)


def test_retries_retryable_status_and_honors_retry_after():
    calls = 0
    sleeps = []

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=response_payload())

    with make_client(
        handler,
        max_retries=1,
        max_retry_delay=2,
        sleep=sleeps.append,
    ) as client:
        result = client.decide(
            "state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    assert isinstance(result.answers["gate"], NoulAnswer)
    assert calls == 2
    assert sleeps == [2]
    assert result.provenance[0].attempts == 2


def test_retries_server_errors_with_exponential_backoff():
    calls = 0
    sleeps = []

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=response_payload())

    with make_client(
        handler,
        max_retries=1,
        retry_backoff=0.125,
        sleep=sleeps.append,
    ) as client:
        result = client.decide(
            "state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    assert calls == 2
    assert sleeps == [0.125]
    assert result.provenance[0].attempts == 2


def test_retries_transport_errors_with_exponential_backoff():
    calls = 0
    sleeps = []

    def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("state=private-state api-secret", request=request)
        return httpx.Response(200, json=response_payload())

    with make_client(
        handler,
        max_retries=2,
        retry_backoff=0.25,
        sleep=sleeps.append,
    ) as client:
        result = client.decide(
            "private-state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    assert calls == 3
    assert sleeps == [0.25, 0.5]
    assert result.provenance[0].attempts == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_nonretryable_http_failures_are_not_retried(status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            text="api-secret private-state should not appear in the exception",
        )

    with (
        make_client(handler, max_retries=3) as client,
        pytest.raises(MicrosoftDecisionError) as exc_info,
    ):
        client.decide(
            "private-state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    message = str(exc_info.value)
    assert message == f"Decision-1 returned HTTP {status}"
    assert "api-secret" not in message
    assert "private-state" not in message
    assert calls == 1


def test_exhausted_transport_failure_redacts_transport_message():
    def handler(request):
        raise httpx.ReadError(
            "api-secret and private-state were in the low-level error",
            request=request,
        )

    with (
        make_client(handler, max_retries=0) as client,
        pytest.raises(MicrosoftDecisionError) as exc_info,
    ):
        client.decide(
            "private-state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    message = str(exc_info.value)
    assert "api-secret" not in message
    assert "private-state" not in message
    assert message == "Decision-1 transport failed after 1 attempt(s)"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda body: body.pop("model"), "invalid model"),
        (lambda body: body.__setitem__("answers", []), "answers object"),
        (lambda body: body["answers"].clear(), "missing"),
        (
            lambda body: body["answers"]["gate"].__setitem__("type", "choice"),
            "expected 'noul'",
        ),
        (
            lambda body: body["answers"]["gate"].__setitem__("noul", 1.1),
            "between 0 and 1",
        ),
        (lambda body: body.pop("usage"), "invalid usage"),
        (
            lambda body: body["usage"].__setitem__("input_tokens", -1),
            "non-negative integer",
        ),
    ],
)
def test_malformed_noul_responses_are_rejected(mutate, match):
    def handler(request):
        body = response_payload()
        mutate(body)
        return httpx.Response(200, json=body)

    with (
        make_client(handler) as client,
        pytest.raises(MicrosoftDecisionError, match=match),
    ):
        client.decide(
            "state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )


@pytest.mark.parametrize(
    "answer,match",
    [
        (
            {
                "type": "choice",
                "choice": "unknown",
                "confidence": 0.9,
                "probabilities": {"billing": 0.5, "support": 0.5},
            },
            "unknown choice",
        ),
        (
            {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.7},
            },
            "contain exactly",
        ),
        (
            {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.7, "support": 0.2},
            },
            "sum to approximately 1",
        ),
        (
            {
                "type": "choice",
                "choice": "billing",
                "confidence": "NaN",
                "probabilities": {"billing": 0.7, "support": 0.3},
            },
            "finite number",
        ),
    ],
)
def test_invalid_choice_answers_are_rejected(answer, match):
    def handler(request):
        return httpx.Response(
            200,
            json=response_payload(answers={"route": answer}),
        )

    question = ChoiceQuestion(criteria={"billing": None, "support": None})
    with (
        make_client(handler) as client,
        pytest.raises(MicrosoftDecisionError, match=match),
    ):
        client.decide("state", {"route": question})


@pytest.mark.parametrize(
    "answer,match",
    [
        (
            {
                "type": "score",
                "score": 2.1,
                "confidence": 0.9,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.1, "1": 0.9},
            },
            "between 0 and 1",
        ),
        (
            {
                "type": "score",
                "score": 0.9,
                "confidence": 0.9,
                "legend": {"0": "minor", "1": "high"},
                "probabilities": {"0": 0.1, "1": 0.9},
            },
            "legend does not match",
        ),
        (
            {
                "type": "score",
                "score": 0.9,
                "confidence": 0.9,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": -0.1, "1": 1.1},
            },
            "between 0 and 1",
        ),
        (
            {
                "type": "score",
                "score": 0.9,
                "confidence": 1.1,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.1, "1": 0.9},
            },
            "between 0 and 1",
        ),
    ],
)
def test_invalid_score_answers_are_rejected(answer, match):
    def handler(request):
        return httpx.Response(
            200,
            json=response_payload(answers={"severity": answer}),
        )

    question = ScoreQuestion(criteria=["low", "high"])
    with (
        make_client(handler) as client,
        pytest.raises(MicrosoftDecisionError, match=match),
    ):
        client.decide("state", {"severity": question})


def test_invalid_json_response_is_rejected_without_body_leakage():
    def handler(request):
        return httpx.Response(
            200,
            content=b"api-secret private-state {not-json",
            headers={"Content-Type": "application/json"},
        )

    with (
        make_client(handler) as client,
        pytest.raises(MicrosoftDecisionError) as exc_info,
    ):
        client.decide(
            "private-state",
            {"gate": NoulQuestion(instructions="Gate?")},
        )

    assert str(exc_info.value) == "Decision-1 returned invalid JSON"


def test_closed_client_rejects_requests_and_reentry():
    client = make_client(lambda request: httpx.Response(200, json=response_payload()))
    client.close()
    client.close()

    with pytest.raises(MicrosoftDecisionError, match="closed"):
        client.decide("state", {"gate": NoulQuestion(instructions="Gate?")})
    with pytest.raises(MicrosoftDecisionError, match="closed"):
        client.__enter__()
