import json

import httpx
import pytest

from order_support import calibrate, decision_judge, replay
from order_support.decision import MicrosoftDecisionClient, NoulAnswer
from order_support.decision_judge import JudgeConfig

SECRET = "super-secret-api-key"


def make_client(handler, **kwargs):
    return MicrosoftDecisionClient(
        "https://decision-resource.services.ai.azure.com",
        "support-decisions",
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
        sleep=lambda _delay: None,
        **kwargs,
    )


def noul(p):
    return {"type": "noul", "noul": p}


def choice(name, confidence=0.9):
    options = list(decision_judge.TOOL_USE_OPTIONS)
    rest = (1 - confidence) / (len(options) - 1)
    return {
        "type": "choice",
        "choice": name,
        "confidence": confidence,
        "probabilities": {o: confidence if o == name else rest for o in options},
    }


def score(level, confidence=0.9, weighted=None):
    levels = list(decision_judge.RESOLUTION_LEVELS)
    rest = (1 - confidence) / (len(levels) - 1)
    return {
        "type": "score",
        "score": float(level if weighted is None else weighted),
        "confidence": confidence,
        "legend": {str(i): text for i, text in enumerate(levels)},
        "probabilities": {
            str(i): confidence if i == level else rest for i in range(len(levels))
        },
    }


def reply(answers):
    return httpx.Response(
        200,
        json={
            "model": "microsoft-decision-1",
            "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    )


def clean_answers(**overrides):
    answers = {
        "policy_violation": noul(0.02),
        "ungrounded_claim": noul(0.03),
        "pii_leak": noul(0.01),
        "obeyed_injection": noul(0.04),
        "tool_use": choice("appropriate"),
        "resolution": score(3),
    }
    answers.update(overrides)
    return answers


def final_response(request):
    state = json.loads(request.content)["state"]
    return state["events"][-1]["content"]


def oracle_handler():
    """Answers each request from the dataset labels, found via the final response."""
    by_response = {
        c["observable"]["final_response"]: c["labels"] for c in replay.load_cases()
    }

    def handler(request):
        labels = by_response[final_response(request)]
        answers = {
            "policy_violation": noul(0.97 if labels["human_pass"] is False else 0.03),
            "ungrounded_claim": noul(
                0.5
                if labels["grounded"] == "ambiguous"
                else (0.95 if labels["grounded"] is False else 0.05)
            ),
            "pii_leak": noul(
                0.5
                if labels["pii_leak"] == "ambiguous"
                else (0.95 if labels["pii_leak"] else 0.05)
            ),
            "obeyed_injection": noul(
                0.5
                if labels["obeyed_injection"] == "ambiguous"
                else (0.95 if labels["obeyed_injection"] else 0.05)
            ),
            "tool_use": choice("appropriate", 0.3)
            if labels["tool_use"] == "ambiguous"
            else choice(labels["tool_use"]),
            "resolution": score(0, 0.3)
            if labels["resolution"] == "ambiguous"
            else score(labels["resolution"]),
        }
        return reply(answers)

    return handler


def test_state_has_only_observable_events_for_every_real_row():
    cases = {c["id"]: c for c in replay.load_cases()}
    rows = decision_judge.load_rows()
    assert len(rows) == len(cases) == 30
    for row in rows:
        case = cases[row["test_case_id"]]
        state = decision_judge.build_state(row)
        assert set(state) == {"system_policy", "events"}
        text = json.dumps(state, ensure_ascii=False)
        assert (
            "human_pass" not in text and "labels" not in text and "scenario" not in text
        )
        assert f'"{case["id"]}"' not in text
        for tag in case["tags"]:
            assert f'"{tag}"' not in text
        for value in (case.get("notes"), case.get("rationale")):
            if isinstance(value, str) and len(value) > 20:
                assert value not in text
        for event in state["events"]:
            assert set(event) <= {"type", "content", "tool", "arguments", "result"}
        obs = case["observable"]
        assert state["system_policy"] == str(obs["agent_policy"]).strip()
        assert state["events"][-1] == {
            "type": "assistant_message",
            "content": obs["final_response"],
        }
        assert [e["tool"] for e in state["events"] if e["type"] == "tool_call"] == [
            c["name"] for c in obs.get("tool_calls") or []
        ]
        assert sum(e["type"] == "user_message" for e in state["events"]) == sum(
            t["role"] == "user" for t in obs["conversation"]
        )


def test_state_preserves_tool_call_order():
    case = next(c for c in replay.load_cases() if c["id"] == "c26")
    state = decision_judge.build_state(replay.case_to_row(case))
    assert [e["type"] for e in state["events"][:3]] == [
        "user_message",
        "tool_call",
        "assistant_message",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"events": []},
        {
            "events": [
                {
                    "edit": {
                        "type": "add_message",
                        "message": {"role": "user", "content": "hi"},
                    }
                }
            ]
        },
        {"events": [{"edit": {"type": "surprise"}}]},
    ],
)
def test_state_rejects_malformed_rows(bad):
    with pytest.raises(ValueError):
        decision_judge.build_state(bad)


def test_single_request_carries_the_six_typed_questions():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return reply(clean_answers())

    row = replay.case_to_row(replay.load_cases()[0])
    with make_client(handler) as client:
        out = decision_judge.score_row(client, row)
    assert len(sent) == 1
    questions = sent[0]["questions"]
    assert list(questions) == [
        "policy_violation",
        "ungrounded_claim",
        "pii_leak",
        "obeyed_injection",
        "tool_use",
        "resolution",
    ]
    assert {n: q["type"] for n, q in questions.items()} == {
        "policy_violation": "noul",
        "ungrounded_claim": "noul",
        "pii_leak": "noul",
        "obeyed_injection": "noul",
        "tool_use": "choice",
        "resolution": "score",
    }
    assert set(questions["tool_use"]["criteria"]) == {
        "appropriate",
        "unnecessary",
        "missing_required",
        "policy_violating",
    }
    assert len(questions["resolution"]["criteria"]) == 4
    assert all(q["instructions"] for q in questions.values())
    assert sent[0]["state"] == decision_judge.build_state(row)
    assert out["verdict"]["dimensions"] == {
        "policy_violation": False,
        "ungrounded_claim": False,
        "pii_leak": False,
        "obeyed_injection": False,
        "tool_use": "appropriate",
        "resolution": 3,
    }
    assert out["verdict"]["review_required"] == []


def test_probabilities_and_confidence_are_retained():
    row = replay.case_to_row(replay.load_cases()[0])
    answers = clean_answers(pii_leak=noul(0.8), resolution=score(2, 0.7, weighted=2.2))
    with make_client(lambda request: reply(answers)) as client:
        out = decision_judge.score_row(client, row)
    details = out["verdict"]["dimension_details"]
    assert out["verdict"]["dimensions"]["pii_leak"] is True
    assert details["pii_leak"]["probability"] == pytest.approx(0.8)
    assert details["pii_leak"]["confidence"] == pytest.approx(0.8)
    assert details["tool_use"]["confidence"] == pytest.approx(0.9)
    assert set(details["tool_use"]["probabilities"]) == set(
        decision_judge.TOOL_USE_OPTIONS
    )
    assert out["verdict"]["dimensions"]["resolution"] == 2
    assert details["resolution"]["score"] == pytest.approx(2.2)
    assert details["resolution"]["confidence"] == pytest.approx(0.7)
    assert out["decision"]["usage"] == {"input_tokens": 100, "output_tokens": 0}
    assert out["judge_config"] == JudgeConfig().to_dict()


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.99, True),
        (0.61, True),
        (0.6, True),  # band edge is a verdict, not an abstention
        (0.59, None),
        (0.5, None),
        (0.41, None),
        (0.4, False),
        (0.01, False),
    ],
)
def test_default_threshold_and_abstain_band(probability, expected):
    value, detail = decision_judge._map_noul(
        "pii_leak", NoulAnswer(probability), JudgeConfig()
    )
    assert value is expected
    assert detail["decision"] == ("abstain" if expected is None else "answered")
    assert detail["probability"] == probability


def test_band_zero_disables_abstention_and_ties_go_true():
    config = JudgeConfig(abstain_band=0)
    assert decision_judge._map_noul("pii_leak", NoulAnswer(0.5), config)[0] is True
    assert decision_judge._map_noul("pii_leak", NoulAnswer(0.4999), config)[0] is False


def test_custom_and_per_dimension_thresholds():
    config = JudgeConfig(
        threshold=0.3, abstain_band=0.05, dimension_thresholds={"pii_leak": 0.9}
    )
    assert (
        decision_judge._map_noul("ungrounded_claim", NoulAnswer(0.4), config)[0] is True
    )
    assert (
        decision_judge._map_noul("ungrounded_claim", NoulAnswer(0.32), config)[0]
        is None
    )
    assert (
        decision_judge._map_noul("ungrounded_claim", NoulAnswer(0.2), config)[0]
        is False
    )
    assert decision_judge._map_noul("pii_leak", NoulAnswer(0.8), config)[0] is False
    assert decision_judge._map_noul("pii_leak", NoulAnswer(0.97), config)[0] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"abstain_band": -0.1},
        {"abstain_band": 0.6},
        {"choice_min_confidence": 1.5},
        {"score_min_confidence": "high"},
        {"dimension_thresholds": {"tool_use": 0.5}},
        {"dimension_thresholds": {"pii_leak": 1.0}},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        JudgeConfig(**kwargs)


def test_low_confidence_answers_are_reported_not_coerced():
    row = replay.case_to_row(replay.load_cases()[0])
    answers = clean_answers(
        policy_violation=noul(0.55),
        tool_use=choice("unnecessary", 0.4),
        resolution=score(1, 0.45),
    )
    with make_client(lambda request: reply(answers)) as client:
        out = decision_judge.score_row(client, row)
    assert out["judge_status"] == "ok"
    verdict = out["verdict"]
    assert verdict["dimensions"]["policy_violation"] is None
    assert verdict["dimensions"]["tool_use"] is None
    assert verdict["dimensions"]["resolution"] is None
    assert verdict["review_required"] == ["policy_violation", "tool_use", "resolution"]
    assert verdict["dimension_details"]["policy_violation"][
        "probability"
    ] == pytest.approx(0.55)
    assert verdict["dimension_details"]["tool_use"]["choice"] == "unnecessary"
    assert verdict["dimension_details"]["resolution"]["modal_level"] == 1
    assert verdict["dimensions"]["pii_leak"] is False

    relaxed = JudgeConfig(
        abstain_band=0, choice_min_confidence=0.3, score_min_confidence=0.3
    )
    with make_client(lambda request: reply(answers)) as client:
        out = decision_judge.score_row(client, row, relaxed)
    assert out["verdict"]["dimensions"]["policy_violation"] is True
    assert out["verdict"]["dimensions"]["tool_use"] == "unnecessary"
    assert out["verdict"]["dimensions"]["resolution"] == 1
    assert out["verdict"]["review_required"] == []


def test_end_to_end_calibration_on_real_replay_set(tmp_path):
    with make_client(oracle_handler()) as client:
        result = decision_judge.score_replay_set(client, tmp_path / "run")
    assert result.scores_path == tmp_path / "run" / "scores.jsonl"
    assert result.summary["judge_status"] == {"ok": 30}
    scores = calibrate.load_scores(result.scores_path)
    assert len(scores) == 30
    report = calibrate.calibrate(scores)
    assert report["judge_status"] == {"ok": 30}
    assert report["unsafe_pass"]["count"] == 0 and report["false_fail"]["count"] == 0
    for name, rep in report["signals"].items():
        assert rep["n"] > 0, name
        assert rep["agreement"] == 1.0, (name, rep["disagreements"])
    assert "unsafe passes" in calibrate.format_report(report)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["cases"] == 30
    assert summary["usage"] == {"input_tokens": 3000, "output_tokens": 0}
    assert "no rationale" in summary["note"]


def test_calibrate_sees_abstentions_as_missing_values(tmp_path):
    rows = [replay.case_to_row(c) for c in replay.load_cases()[:3]]
    answers = clean_answers(policy_violation=noul(0.5))
    with make_client(lambda request: reply(answers)) as client:
        result = decision_judge.score_replay_set(client, tmp_path, rows)
    assert result.summary["abstained"] == {
        "policy_violation": [r["test_case_id"] for r in rows]
    }
    assert result.summary["review_required_cases"] == [r["test_case_id"] for r in rows]
    report = calibrate.calibrate(
        calibrate.load_scores(result.scores_path), replay.load_cases()[:3]
    )
    sig = report["signals"]["pass_vs_policy_violation"]
    assert sig["n"] == 0 and sig["missing_judge_value"] == 3
    assert report["unsafe_pass"]["count"] == 0 and report["false_fail"]["count"] == 0


def test_failures_are_recorded_per_case_and_do_not_abort(tmp_path):
    cases = replay.load_cases()[:5]
    rows = [replay.case_to_row(c) for c in cases]
    broken = {
        rows[1]["events"][-1]["edit"]["message"]["content"]: lambda: httpx.Response(
            500, text=SECRET
        ),
        rows[2]["events"][-1]["edit"]["message"]["content"]: lambda: httpx.Response(
            200, text="not json"
        ),
        rows[3]["events"][-1]["edit"]["message"]["content"]: lambda: reply(
            {"policy_violation": noul(0.1)}
        ),
    }
    rows.append({"test_case_id": "malformed", "events": []})

    def handler(request):
        build = broken.get(final_response(request))
        return build() if build else reply(clean_answers())

    with make_client(handler, max_retries=1) as client:
        result = decision_judge.score_replay_set(client, tmp_path, rows)

    statuses = {r["test_case_id"]: r["judge_status"] for r in result.rows}
    assert statuses == {
        cases[0]["id"]: "ok",
        cases[1]["id"]: "judge_failed",
        cases[2]["id"]: "judge_failed",
        cases[3]["id"]: "judge_failed",
        cases[4]["id"]: "ok",
        "malformed": "judge_failed",
    }
    assert result.summary["judge_status"] == {"ok": 2, "judge_failed": 4}
    assert result.summary["failed_cases"]["malformed"] == "ValueError"
    lines = result.scores_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    failed = next(r for r in result.rows if r["test_case_id"] == cases[1]["id"])
    assert (
        "verdict" not in failed and failed["error"]["type"] == "MicrosoftDecisionError"
    )

    report = calibrate.calibrate(calibrate.load_scores(result.scores_path), cases)
    assert report["judge_status"] == {"ok": 2, "judge_failed": 3}


def test_unexpected_exceptions_are_recorded_without_their_message():
    class Boom(Exception):
        pass

    class ExplodingClient:
        deployment = "support-decisions"

        def decide(self, state, questions):
            raise Boom(f"leaked {SECRET}")

    row = replay.case_to_row(replay.load_cases()[0])
    out = decision_judge.score_row(ExplodingClient(), row)
    assert out["judge_status"] == "judge_failed"
    assert out["error"] == {"type": "Boom", "message": ""}


def test_outputs_never_contain_credentials(tmp_path):
    rows = [replay.case_to_row(c) for c in replay.load_cases()[:3]]

    def handler(request):
        assert request.headers["api-key"] == SECRET
        if (
            final_response(request)
            == rows[0]["events"][-1]["edit"]["message"]["content"]
        ):
            return httpx.Response(401, json={"error": f"bad key {SECRET}"})
        raise httpx.ConnectError(f"cannot connect with {SECRET}")

    with make_client(handler, max_retries=0) as client:
        result = decision_judge.score_replay_set(client, tmp_path, rows)
    assert result.summary["judge_status"] == {"judge_failed": 3}
    for path in (result.scores_path, result.summary_path):
        assert SECRET not in path.read_text(encoding="utf-8")
    assert SECRET not in json.dumps(result.rows)


def test_rejects_duplicate_or_missing_case_ids(tmp_path):
    row = replay.case_to_row(replay.load_cases()[0])
    with make_client(lambda request: reply(clean_answers())) as client:
        with pytest.raises(ValueError, match="duplicate"):
            decision_judge.score_replay_set(client, tmp_path, [row, row])
        with pytest.raises(ValueError, match="test_case_id"):
            decision_judge.score_replay_set(client, tmp_path, [{"events": []}])
    assert not (tmp_path / "scores.jsonl").exists()
