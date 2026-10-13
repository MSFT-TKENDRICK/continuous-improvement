from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_DECISION,
    ATTR_NIGHT,
    SPAN_OPTIMIZER,
    SPAN_SLEEP_NIGHT,
    SPAN_STEP,
)
from ci_lab.sleep.fakes import (
    FakeOracle,
    FakeReflector,
    fake_run_target,
    make_fake_assert_eval,
)
from ci_lab.sleep.night import SleepConfig, SleepDeps, run_night
from ci_lab.sleep.registry import (
    HARNESS_EDITING,
    TRACE_TRIAGE,
    RegistryError,
    load_targets,
    parse_target,
)
from ci_lab.sleep.runner import sequential_runner

SHA = "b" * 40
CASES = {f"c{i}": "inspect_before_edit" for i in range(6)}
SECOND = TRACE_TRIAGE


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(obs, "tracer", lambda: tp.get_tracer("ci_lab"))
    return exp


def _deps(**kw) -> SleepDeps:
    kw.setdefault("run_target", fake_run_target)
    kw.setdefault("oracle", FakeOracle())
    kw.setdefault("reflector", FakeReflector())
    kw.setdefault("assert_eval", make_fake_assert_eval(CASES))
    kw.setdefault("latest_delta", lambda: 0.05)
    kw.setdefault("run_workflow", sequential_runner)
    kw.setdefault("clock", lambda: datetime(2026, 9, 22, 7, 17, tzinfo=UTC))
    return SleepDeps(**kw)


def _cfg(repo: Path, tmp_path: Path, **kw) -> SleepConfig:
    kw.setdefault("night_date", "20260922")
    kw.setdefault("base_sha", SHA)
    kw.setdefault("n_boot", 300)
    kw.setdefault("targets", [HARNESS_EDITING])
    return SleepConfig(repo_root=repo, out_dir=tmp_path / "out" / "sleep-bundle", **kw)


def _add_second(repo: Path, h) -> None:
    skill = repo / SECOND.skill_path
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(h.SKILL.replace("harness-editing", "trace-triage"), encoding="utf-8", newline="\n")
    h.write_tasks(repo / SECOND.tasks_file,
                  [h.task_row(i, id=f"r{i:02d}", project="trace-triage") for i in range(6)])


# ------------------------------------------------------------------ registry

def test_packaged_registry_is_harness_skills():
    targets = load_targets()
    assert targets == [HARNESS_EDITING, TRACE_TRIAGE]


@pytest.mark.parametrize("bad", [
    {"skill": "harness/skills/../../evil/SKILL.md"},
    {"skill": ".github/workflows/x/SKILL.md"},
    {"skill": "harness/skills/x/notes.md"},
    {"tasks": "experiments/sleep/tasks.pending.jsonl"},
    {"tasks": "experiments/other/tasks.jsonl"},
    {"name": "Bad Name"},
    {"surprise": 1},
    {"enabled": "yes"},
    {"owner_agent": ""},
])
def test_registry_rejects_bad_targets(bad):
    raw = {"name": "x", "skill": "harness/skills/x/SKILL.md", "owner_agent": "proposer",
           "eval_suite": "harness", **bad}
    with pytest.raises(RegistryError):
        parse_target(raw)


def test_registry_rejects_duplicates_and_disabled_only(tmp_path):
    p = tmp_path / "t.yaml"
    t = {"name": "a", "skill": "harness/skills/a/SKILL.md", "owner_agent": "o", "eval_suite": "s"}
    p.write_text(json.dumps({"format": "ci_lab.sleep.targets.v1", "targets": [t, t]}), encoding="utf-8")
    with pytest.raises(RegistryError, match="duplicate"):
        load_targets(p)
    p.write_text(json.dumps({"format": "ci_lab.sleep.targets.v1", "targets": [{**t, "enabled": False}]}),
                 encoding="utf-8")
    with pytest.raises(RegistryError, match="no enabled"):
        load_targets(p)
    assert load_targets(p, include_disabled=True)[0].enabled is False


def test_tasks_file_override_needs_single_target(sleep_repo, tmp_path):
    with pytest.raises(ValueError):
        _cfg(sleep_repo, tmp_path, targets=[HARNESS_EDITING, SECOND], tasks_file=tmp_path / "t.jsonl")


# ------------------------------------------------------------------ multi-target night

def test_two_targets_one_accepted_one_rejected(sleep_repo, tmp_path, h, exporter):
    _add_second(sleep_repo, h)
    strict = _deps(latest_delta=lambda: 1.0)
    deps = _deps(per_target=lambda t: strict if t.name == "trace-triage" else deps_os)
    deps_os = _deps()
    cfg = _cfg(sleep_repo, tmp_path, targets=[HARNESS_EDITING, SECOND])
    res = run_night(cfg, deps)
    assert res.status == "accepted", res.error
    assert res.targets == {"harness-editing": "accepted", "trace-triage": "rejected"}
    assert res.decisions["trace-triage"]["delta"] == 1.0
    results = json.loads((cfg.out_dir / "results.json").read_text(encoding="utf-8"))
    assert set(results["targets"]) == {"harness-editing", "trace-triage"}
    assert any(r.startswith("trace-triage: ") for r in results["reasons"])
    assert all(r.startswith(("trace-triage: ", "harness-editing: ")) for r in results["reasons"])
    patch = (cfg.out_dir / "candidate.patch").read_text(encoding="utf-8")
    assert f"diff --git a/{HARNESS_EDITING.skill_path}" in patch
    assert f"a/{SECOND.skill_path}" not in patch
    # state is the C10 ledger: per-target status + reviewed-task watermark for the usage gate
    state_new = (tmp_path / "state.json")
    h.git(sleep_repo, "apply", "--include=experiments/sleep/state.json", str(_write(tmp_path, patch)))
    state = json.loads((sleep_repo / "experiments/sleep/state.json").read_text(encoding="utf-8"))
    assert state["targets"]["trace-triage"] == {"last_status": "rejected", "accepted_total": 0}
    assert state["watermark"]["task_ids"]["trace-triage"] == [f"r{i:02d}" for i in range(6)]
    assert not state_new.exists()

    # tracing: one night trace; per-target child spans; optimizer spans per target
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == SPAN_SLEEP_NIGHT]
    assert len(roots) == 1 and roots[0].attributes[ATTR_NIGHT] == 1
    assert roots[0].attributes[ATTR_DECISION] == "accepted"
    tid = roots[0].context.trace_id
    assert all(s.context.trace_id == tid for s in spans), [(s.name, dict(s.attributes or {}), s.parent) for s in spans if s.context.trace_id != tid][:3]
    per_target = {s.attributes.get("sleep.target") for s in spans if s.name == SPAN_STEP}
    assert {"harness-editing", "trace-triage"} <= per_target
    assert len([s for s in spans if s.name == SPAN_OPTIMIZER]) == 2

    # live status marker (obs v2.3.1: per-writer status.d, read via read_status)
    status = obs.read_status(cfg.run_dir, res.night_id)
    assert status["phase"] == "done" and status["state"] == "accepted"
    assert status["target_status"] == res.targets and "night" in status.get("writers", ["night"])


def _write(tmp_path: Path, patch: str) -> Path:
    p = tmp_path / "multi.patch"
    p.write_text(patch, encoding="utf-8", newline="\n")
    return p


def test_agl_rows_route_by_target_and_unknown_split_still_checked(sleep_repo, tmp_path, h):
    _add_second(sleep_repo, h)
    rows = [{"target": "nope", "dataset_split": "heldout", "task": {"id": "x", "project": "p", "intent": "i"}}]
    res = run_night(
        _cfg(sleep_repo, tmp_path, targets=[HARNESS_EDITING, SECOND]),
        _deps(agl_records=lambda: rows),
    )
    assert res.status == "error" and "split" in res.error.lower()


def test_rerun_links_previous_night_trace(sleep_repo, tmp_path, exporter):
    cfg = _cfg(sleep_repo, tmp_path)
    run_night(cfg, _deps())
    first = next(s for s in exporter.get_finished_spans() if s.name == SPAN_SLEEP_NIGHT)
    exporter.clear()
    run_night(cfg, _deps())
    second = next(s for s in exporter.get_finished_spans() if s.name == SPAN_SLEEP_NIGHT)
    assert second.context.trace_id != first.context.trace_id
    assert [link.context.trace_id for link in second.links] == [first.context.trace_id]
