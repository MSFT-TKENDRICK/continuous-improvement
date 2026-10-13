"""``ci-lab judge`` CLI wiring (offline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("litellm")

from ci_lab.judge import backends as B
from ci_lab.judge import cli
from ci_lab.judge import provider as P
from ci_lab.judge.s1types import Answer

ROOT = Path(__file__).resolve().parents[3]
REPLAY_CONFIG = ROOT / "evals" / "assert" / "judge_replay" / "eval_config.yaml"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ci-lab")
    cli.register(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(argv)


@pytest.fixture(autouse=True)
def _fresh_provider():
    P.unregister()
    yield
    P.unregister()


def test_subcommands_registered():
    assert _parse(["judge", "audit", "--labels", "l", "--judge", "a", "b"]).func is cli._cmd_audit
    args = _parse(["judge", "align", "--config", "c", "--labels", "l", "--transcripts", "t", "--out-dir", "o",
                   "--map", "grounded=!ungrounded_claim", "--dimension", "pii_leak"])
    assert args.func is cli._cmd_align and args.lm == "openai/local" and args.map == ["grounded=!ungrounded_claim"]
    assert _parse(["judge", "provider-check"]).model == "s1/scripted/default"


def test_audit_command(tmp_path, capsys):
    labels, scores = [], []
    for i in range(12):
        c = f"c{i}"
        labels.append({"case_id": c, "dimension": "pii_leak", "label": i % 2 == 0})
        labels.append({"case_id": c, "dimension": "resolution", "label": i % 3})
        scores.append({"test_case_id": c, "judge_status": "ok",
                       "verdict": {"dimensions": {"pii_leak": i % 2 == 0, "resolution": 0}}})
    lp, sp, out = tmp_path / "labels.jsonl", tmp_path / "scores.jsonl", tmp_path / "audit.json"
    lp.write_text("\n".join(json.dumps(r) for r in labels), encoding="utf-8")
    sp.write_text("\n".join(json.dumps(r) for r in scores), encoding="utf-8")
    base = ["judge", "audit", "--labels", str(lp), "--judge", str(sp), "--n-boot", "50", "--out", str(out)]
    args = _parse(base)
    assert args.func(args) == 0
    assert "pii_leak" in capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["primary"] == ["pii_leak"] and report["diagnostic"] == ["resolution"]
    args = _parse(base + ["--strict", "--json"])
    assert args.func(args) == 1
    assert json.loads(capsys.readouterr().out)["diagnostic"] == ["resolution"]


def test_provider_check_builtin_offline(capsys):
    args = _parse(["judge", "provider-check"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK s1/scripted/default path=s1")
    assert "polite" in out and "quality" in out


def test_provider_check_with_assert_config():
    result = cli.provider_check(config=str(REPLAY_CONFIG))
    assert result["ok"] and result["path"] == "s1" and result["stats"] == {"s1": 1, "fallback": 0}
    assert {"tool_use", "resolution", "policy_violation"} <= set(result["verdict"]["dimensions"])


def test_provider_check_fails_offline_when_s1_cannot_answer(capsys, monkeypatch):
    monkeypatch.delenv(P.FALLBACK_ENV, raising=False)
    B.register_script("abstain", lambda state, name, q: Answer.non_answer(q.type, "abstain", reason="test"))
    args = _parse(["judge", "provider-check", "--model", "s1/scripted/abstain"])
    assert args.func(args) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL s1/scripted/abstain path=unsupported") and "Abstained" in out
    import os

    assert P.FALLBACK_ENV not in os.environ


def test_provider_check_restores_handler():
    h = P.register()
    assert cli.provider_check()["ok"]
    assert "judge" not in h.__dict__ and P.register() is h
