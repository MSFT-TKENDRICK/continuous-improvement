"""HOOK(M16): sleep night -> ci_lab.lessons candidates as draft-PR proposals (offline, fake profile)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ci_lab.lessons.cluster import MineConfig
from ci_lab.sleep.backend import RolloutView
from ci_lab.sleep.fakes import (
    FakeOracle,
    FakeReflector,
    fake_run_target,
    make_fake_assert_eval,
)
from ci_lab.sleep.lessons_hook import (
    LESSONS_FORMAT,
    LESSONS_REL,
    LessonsReport,
    LessonsRequest,
    make_lessons_hook,
    parse_source,
    proposal_document,
    sanitize_candidate,
    trajectories_from_rollouts,
)
from ci_lab.sleep.night import SleepConfig, SleepDeps, run_night
from ci_lab.sleep.registry import HARNESS_EDITING
from ci_lab.sleep.runner import sequential_runner

OS = "harness-editing"
CASES = {f"c{i}": "inspect_before_edit" for i in range(6)}


def cfg_for(repo: Path, tmp_path: Path, **kw) -> SleepConfig:
    kw.setdefault("night_date", "20260921")
    kw.setdefault("base_sha", "a" * 40)
    kw.setdefault("n_boot", 300)
    kw.setdefault("targets", [HARNESS_EDITING])
    return SleepConfig(repo_root=repo, out_dir=tmp_path / "out" / "sleep-bundle", **kw)


def deps_for(**kw) -> SleepDeps:
    kw.setdefault("run_target", fake_run_target)
    kw.setdefault("oracle", FakeOracle())
    kw.setdefault("reflector", FakeReflector())
    kw.setdefault("assert_eval", make_fake_assert_eval(CASES))
    kw.setdefault("latest_delta", lambda: 0.05)
    kw.setdefault("run_workflow", sequential_runner)
    kw.setdefault("clock", lambda: datetime(2026, 9, 21, 7, 17, tzinfo=UTC))
    return SleepDeps(**kw)


def read_bundle(out: Path) -> tuple[dict, dict, dict, str]:
    def load(n: str) -> dict:
        return json.loads((out / n).read_text(encoding="utf-8"))

    return load("manifest.json"), load("experiment.json"), load("results.json"), \
        (out / "candidate.patch").read_text(encoding="utf-8")

RELAXED = MineConfig(min_support=2, min_slices=1, min_families=1, max_slice_share=1.0,
                     holdout_fraction=0.0, require_stability=False)


def _night(repo: Path, tmp: Path, date: str, hook, **kw):
    cfg = cfg_for(repo, tmp / date, night_date=date, lessons_hook=True, **kw)
    return cfg, run_night(cfg, deps_for(lessons=hook))


def test_hook_off_by_default_never_calls_hook(sleep_repo, tmp_path):
    calls: list = []
    cfg = cfg_for(sleep_repo, tmp_path)
    res = run_night(cfg, deps_for(lessons=lambda req: calls.append(req) or LessonsReport()))
    assert res.status == "accepted" and calls == []
    _, experiment, _, patch = read_bundle(cfg.out_dir)
    assert LESSONS_REL not in patch
    assert experiment["config"]["lessons_hook"] is False
    assert experiment["targets"][OS]["lessons"] is None


def test_hook_receives_typed_rollouts(sleep_repo, tmp_path):
    seen: list[LessonsRequest] = []

    def hook(req: LessonsRequest) -> LessonsReport:
        seen.append(req)
        return LessonsReport(counts={"trajectories_night": len(req.rollouts)})

    cfg, res = _night(sleep_repo, tmp_path, "20260921", hook)
    assert res.status == "accepted", res.error
    (req,) = seen
    assert req.target == OS and req.date == "20260921" and req.night_id == res.night_id
    assert req.rollouts and all(isinstance(r, RolloutView) for r in req.rollouts)
    assert {r.suite for r in req.rollouts} <= {"harness-proposal", "sleep"}
    _, experiment, _, patch = read_bundle(cfg.out_dir)
    assert experiment["config"]["lessons_hook"] is True
    assert experiment["targets"][OS]["lessons"] == {"trajectories_night": len(req.rollouts)}
    assert LESSONS_REL not in patch  # no candidates -> no proposal file


def test_hook_failure_is_soft(sleep_repo, tmp_path):
    def boom(req):
        raise RuntimeError("secret detail CASE-10001")

    cfg, res = _night(sleep_repo, tmp_path, "20260921", boom)
    assert res.status == "accepted", res.error
    _, experiment, _, patch = read_bundle(cfg.out_dir)
    assert experiment["targets"][OS]["lessons"] == {"error": "RuntimeError"}
    assert "CASE-10001" not in json.dumps(experiment) and LESSONS_REL not in patch


def test_real_hook_mines_candidates_across_nights_into_bundle(sleep_repo, tmp_path):
    store = tmp_path / "store"
    hook = make_lessons_hook(store, config=RELAXED)
    _, first = _night(sleep_repo, tmp_path, "20260921", hook)
    assert first.status == "accepted", first.error
    cfg, res = _night(sleep_repo, tmp_path, "20260922", hook)
    assert res.status == "accepted", res.error
    assert (store / OS / "trajectories.jsonl").is_file()
    _, experiment, _, patch = read_bundle(cfg.out_dir)
    counts = experiment["targets"][OS]["lessons"]
    assert counts["trajectories_total"] > counts["trajectories_night"] > 0
    assert counts["candidates"] >= 1, counts
    rel = f"{LESSONS_REL}/{res.night_id}.json"
    assert f"+++ b/{rel}" in patch
    hunk = patch.split(f"+++ b/{rel}", 1)[1].split("\ndiff --git ", 1)[0]
    added = "\n".join(line[1:] for line in hunk.splitlines() if line.startswith("+"))
    doc = json.loads(added)
    assert doc["format"] == LESSONS_FORMAT and doc["status"] == "proposed"
    cand = doc["targets"][OS]["candidates"][0]
    assert cand["status"] == "candidate" and cand["human_confirmed"] is False
    assert cand["fingerprint"]["pin"] == "sleep-fake"
    assert all(tok != "_" for g in cand["fingerprint"]["tool_ngrams"] for tok in g)
    # B8: only typed slots; no task text, resource ids, tool args or member ids
    for leak in ("CASE-1000", "Please help", "verified your resource", "members", "excerpt", "args"):
        assert leak not in added, leak


def test_trajectories_keep_rubric_ids_without_args():
    ro = RolloutView(case_id="t01", suite="harness", transcript=None, violations=(),  # type: ignore[arg-type]
                     rule_ids=("check.contains:verified", "rule:x"), passed=False)
    req = LessonsRequest(target=OS, night_id="sleep-20260921-1", date="20260921", profile="fake",
                         rollouts=(ro, ro))
    a, b = trajectories_from_rollouts(req)
    assert a.id != b.id
    assert a.split == "evolve" and a.slice == "20260921" and a.pin == "sleep-fake" and a.trusted
    assert list(a.outcome.rubric_fails) == ["check.contains"] and a.outcome.passed is False


def test_sanitize_candidate_drops_untyped_values():
    line = {"cluster": {"id": "c-1", "route": "R2", "status": "candidate", "members": ["a", "b", "c"],
                        "families": ["f1"], "slices": ["s1", "s2"],
                        "fingerprint": {"pin": "p", "oracle_rules": ["edit.limit"], "rubric_ids": [],
                                        "tool_ngrams": [["^", "read_file", "has space"]],
                                        "error_class": "free text here", "excerpt": "PII"}},
            "features": {"amount_gt": True, "n": 3, "who": "Jane Doe <j@x.com>", "bad key!": 1,
                         "tools": ["read_file"], "mixed": ["ok", "not ok"]},
            "trusted": True, "holdout_members": ["x"]}
    out = sanitize_candidate(line)
    assert out["support"] == 3 and out["families"] == 1 and out["slices"] == 2 and out["holdout"] == 1
    assert out["fingerprint"]["tool_ngrams"] == [["^", "read_file", "_"]]
    assert out["fingerprint"]["error_class"] is None
    assert out["features"] == {"amount_gt": True, "n": 3, "tools": ["read_file"]}
    assert "PII" not in json.dumps(out) and "Jane" not in json.dumps(out)


def test_proposal_document_none_without_candidates():
    assert proposal_document("n", {OS: LessonsReport(counts={"candidates": 0})}) is None
    doc = proposal_document("n", {OS: LessonsReport(candidates=[{"id": "c"}])})
    assert doc and doc["status"] == "proposed" and "never adopted" in doc["note"]


def test_parse_source():
    assert parse_source("usage:artifacts/u") == ("usage", Path("artifacts/u"))
    for bad in ("usage", "calibrate:x", "nope:x", "usage:"):
        with pytest.raises(ValueError):
            parse_source(bad)


def test_cli_lessons_flag_from_env(monkeypatch, tmp_path):
    import argparse

    from ci_lab.sleep import cli

    captured: dict = {}

    def fake_build(profile, cfg, agl, **kw):
        captured.update(cfg=cfg, **kw)
        raise ValueError("stop")

    monkeypatch.setattr("ci_lab.sleep.wiring.build_deps", fake_build)
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers(dest="cmd"))
    out = tmp_path / "out"
    for env, argv, want in (("", [], False), ("true", [], True), ("0", ["--lessons"], True)):
        monkeypatch.setenv("SLEEP_LESSONS", env)
        args = parser.parse_args(["sleep", "run", "--profile", "fake", "--out", str(out), "--date", "20260921",
                                  "--base-sha", "a" * 40, "--lessons-source", "usage:u", *argv])
        assert args.func(args) == 1
        assert captured["cfg"].lessons_hook is want
        assert captured["lessons_sources"] == ([("usage", Path("u"))] if want else [])
