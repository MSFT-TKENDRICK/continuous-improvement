import functools
import json

import httpx
import pytest
from test_decision_judge import clean_answers, noul, reply

from order_support import calibrate, cli, replay

ENV_NAMES = (
    cli.DECISION_ENDPOINT_ENV,
    cli.DECISION_DEPLOYMENT_ENV,
    cli.DECISION_API_KEY_ENV,
    cli.DECISION_BEARER_TOKEN_ENV,
)
SECRET = "do-not-leak-this-credential"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(
        cli.DECISION_ENDPOINT_ENV, "https://decision-resource.services.ai.azure.com"
    )
    monkeypatch.setenv(cli.DECISION_DEPLOYMENT_ENV, "support-decisions")
    monkeypatch.setenv(cli.DECISION_API_KEY_ENV, SECRET)


@pytest.fixture
def fake_transport(monkeypatch):
    """Route the CLI's client through a mock transport; any real network attempt fails the test."""
    requests = []

    def install(handler):
        def recording(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(
            cli.decision,
            "MicrosoftDecisionClient",
            functools.partial(
                cli.decision.MicrosoftDecisionClient,
                transport=httpx.MockTransport(recording),
                sleep=lambda _delay: None,
            ),
        )
        return requests

    return install


def test_help_works_without_env_and_without_network(capsys, fake_transport):
    requests = fake_transport(lambda request: pytest.fail("no request expected"))
    for argv in (["--help"], ["decide", "--help"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--abstain-band" in out and cli.DECISION_ENDPOINT_ENV in out
    assert requests == []


def test_arguments_are_parsed_with_judge_defaults():
    args = cli.build_parser().parse_args(["decide"])
    assert (args.threshold, args.abstain_band, args.timeout, args.max_retries) == (
        0.5,
        0.1,
        30.0,
        2,
    )
    assert args.output_dir == str(cli.DEFAULT_DECISION_OUTPUT)
    args = cli.build_parser().parse_args(
        [
            "decide",
            "--threshold",
            "0.7",
            "--abstain-band",
            "0",
            "--output-dir",
            "out",
            "--timeout",
            "5",
            "--max-retries",
            "0",
        ]
    )
    assert (args.threshold, args.abstain_band, args.timeout, args.max_retries) == (
        0.7,
        0.0,
        5.0,
        0,
    )
    assert args.output_dir == "out"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, [cli.DECISION_ENDPOINT_ENV, cli.DECISION_DEPLOYMENT_ENV]),
        (
            {cli.DECISION_ENDPOINT_ENV: "https://r.example"},
            [cli.DECISION_DEPLOYMENT_ENV],
        ),
        (
            {
                cli.DECISION_ENDPOINT_ENV: "https://r.example",
                cli.DECISION_DEPLOYMENT_ENV: "d",
            },
            [cli.DECISION_API_KEY_ENV, cli.DECISION_BEARER_TOKEN_ENV],
        ),
        (
            {
                cli.DECISION_ENDPOINT_ENV: "https://r.example",
                cli.DECISION_DEPLOYMENT_ENV: "d",
                cli.DECISION_API_KEY_ENV: "   ",
                cli.DECISION_BEARER_TOKEN_ENV: "",
            },
            [cli.DECISION_API_KEY_ENV, cli.DECISION_BEARER_TOKEN_ENV],
        ),
    ],
)
def test_missing_configuration_fails_fast(
    monkeypatch, capsys, fake_transport, tmp_path, env, expected
):
    requests = fake_transport(lambda request: pytest.fail("no request expected"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert cli.main(["decide", "--output-dir", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and all(name in err for name in expected)
    assert requests == [] and not (tmp_path / "out").exists()


def test_both_credentials_are_ambiguous_and_not_leaked(
    monkeypatch, capsys, configured, fake_transport, tmp_path
):
    fake_transport(lambda request: pytest.fail("no request expected"))
    monkeypatch.setenv(cli.DECISION_BEARER_TOKEN_ENV, SECRET + "-token")
    assert cli.main(["decide", "--output-dir", str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert "not both" in captured.err
    assert SECRET not in captured.out + captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["--threshold", "1.5"],
        ["--abstain-band", "0.9"],
        ["--timeout", "0"],
        ["--max-retries", "50"],
    ],
)
def test_invalid_numeric_options_fail_without_leaking(
    capsys, configured, fake_transport, tmp_path, argv
):
    requests = fake_transport(lambda request: pytest.fail("no request expected"))
    assert cli.main(["decide", "--output-dir", str(tmp_path / "out"), *argv]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert SECRET not in captured.out + captured.err
    assert requests == []


def test_bad_endpoint_is_rejected_without_echoing_credentials(
    monkeypatch, capsys, configured, tmp_path
):
    monkeypatch.setenv(
        cli.DECISION_ENDPOINT_ENV, f"https://user:{SECRET}@resource.example"
    )
    assert cli.main(["decide", "--output-dir", str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert "endpoint" in captured.err and SECRET not in captured.out + captured.err


def test_stale_inference_set_is_rejected(
    monkeypatch, capsys, configured, fake_transport, tmp_path
):
    requests = fake_transport(lambda request: pytest.fail("no request expected"))
    monkeypatch.setattr(replay, "is_current", lambda: False)
    assert cli.main(["decide", "--output-dir", str(tmp_path / "out")]) == 2
    assert "replay build" in capsys.readouterr().err
    assert requests == []


def test_run_writes_scores_that_calibrate_consumes(
    monkeypatch, capsys, configured, fake_transport, tmp_path
):
    requests = fake_transport(lambda request: reply(clean_answers()))
    out = tmp_path / "out"
    assert cli.main(["decide", "--output-dir", str(out), "--max-retries", "0"]) == 0
    stdout = capsys.readouterr().out
    cases = len(replay.load_cases())
    assert len(requests) == cases
    assert requests[0].headers["api-key"] == SECRET
    assert (out / "scores.jsonl").exists() and (out / "decision_run.json").exists()
    assert f"scored {cases} cases" in stdout and f"{cases} ok, 0 failed" in stdout
    assert "0 with abstained dimensions" in stdout
    assert f'order-support-evals calibrate --scores "{out / "scores.jsonl"}"' in stdout
    assert SECRET not in stdout

    assert cli.main(["calibrate", "--scores", str(out / "scores.jsonl")]) == 0
    result = calibrate.calibrate(calibrate.load_scores(out / "scores.jsonl"))
    assert result["signals"]
    assert "pass_vs_policy_violation" in capsys.readouterr().out


def test_bearer_token_and_options_reach_the_judge(
    monkeypatch, capsys, configured, fake_transport, tmp_path
):
    monkeypatch.delenv(cli.DECISION_API_KEY_ENV)
    monkeypatch.setenv(cli.DECISION_BEARER_TOKEN_ENV, SECRET)
    requests = fake_transport(lambda request: reply(clean_answers(pii_leak=noul(0.55))))
    out = tmp_path / "out"
    code = cli.main(
        [
            "decide",
            "--output-dir",
            str(out),
            "--abstain-band",
            "0.1",
            "--threshold",
            "0.5",
        ]
    )
    assert code == 0
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert "api-key" not in requests[0].headers
    stdout = capsys.readouterr().out
    assert "abstained pii_leak:" in stdout
    summary = json.loads((out / "decision_run.json").read_text(encoding="utf-8"))
    assert summary["config"]["abstain_band"] == 0.1
    assert len(summary["review_required_cases"]) == len(replay.load_cases())
    assert SECRET not in stdout + (out / "scores.jsonl").read_text(encoding="utf-8")


def test_failed_cases_are_reported_with_nonzero_exit(
    capsys, configured, fake_transport, tmp_path
):
    fake_transport(
        lambda request: httpx.Response(400, json={"error": {"message": "bad"}})
    )
    out = tmp_path / "out"
    assert cli.main(["decide", "--output-dir", str(out), "--max-retries", "0"]) == 1
    stdout = capsys.readouterr().out
    cases = len(replay.load_cases())
    assert f"0 ok, {cases} failed" in stdout
    assert (out / "scores.jsonl").exists()
    assert SECRET not in stdout
