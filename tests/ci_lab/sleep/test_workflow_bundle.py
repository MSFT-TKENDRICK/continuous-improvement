from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from ci_lab.sleep.bundle import FileChange, make_patch, verify_bundle, write_bundle
from ci_lab.sleep.night import STEPS, WORKFLOW_PATH
from ci_lab.sleep.runner import WorkflowError, default_runner, load_actions, maf_runner, sequential_runner

# ------------------------------------------------------------------ sleep.yaml


def test_sleep_yaml_is_expression_free_and_ordered():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    actions = load_actions(text)
    assert [a["id"] for a in actions] == list(STEPS)
    assert [a["functionName"] for a in actions] == [f"sleep_{s}" for s in STEPS]
    doc = yaml.safe_load(text)

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from strings(v)

    assert not any(s.lstrip().startswith("=") for s in strings(doc))
    for bad in ("If", "ConditionGroup", "Foreach", "Goto"):
        assert f"kind: {bad}" not in text
    assert actions[0]["arguments"] == {"split": "evolve"}
    assert actions[1]["arguments"] == {"gate_mode": "on"}


@pytest.mark.parametrize("mutation", [
    lambda d: d["trigger"]["actions"][0]["arguments"].update(split="=Local.split"),
    lambda d: d["trigger"]["actions"].append({"kind": "If", "id": "x", "condition": "true"}),
    lambda d: d["trigger"]["actions"][0]["arguments"].update(split=["evolve"]),
    lambda d: d["trigger"]["actions"].append({"kind": "SendActivity", "id": "y", "activity": "hi"}),
])
def test_load_actions_rejects_expressions_and_control_flow(mutation):
    doc = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    mutation(doc)
    with pytest.raises(WorkflowError):
        load_actions(yaml.safe_dump(doc))


def _tools(log):
    def mk(name, param):
        def fn(**kw):
            assert list(kw) == [param]
            log.append((name, kw[param]))
            return {"ok": name}

        fn.__name__ = name
        fn.__annotations__ = {param: str, "return": dict}
        import inspect

        fn.__signature__ = inspect.Signature([inspect.Parameter(param, inspect.Parameter.KEYWORD_ONLY,
                                                                annotation=str)], return_annotation=dict)
        return fn

    params = {"sleep_harvest": "split", "sleep_consolidate": "gate_mode", "sleep_assert_gate": "suite_split",
              "sleep_record": "envelope_kind", "sleep_bundle": "layout"}
    return {n: mk(n, p) for n, p in params.items()}


def test_sequential_runner(tmp_path):
    log = []
    out = sequential_runner(WORKFLOW_PATH, _tools(log), tmp_path)
    assert list(out) == list(STEPS) and [n for n, _ in log] == [f"sleep_{s}" for s in STEPS]
    assert log[0] == ("sleep_harvest", "evolve")


def test_maf_runner_runs_declarative_workflow_with_checkpoints(tmp_path):
    pytest.importorskip("agent_framework_declarative")
    import asyncio

    log = []
    tools = _tools(log)

    # steps may call asyncio.run themselves (MAF agents) -> must not run on the loop thread
    def harvest(split: str) -> dict:
        asyncio.run(asyncio.sleep(0))
        log.append(("sleep_harvest", split))
        return {"n": 1}

    tools["sleep_harvest"] = harvest
    out = maf_runner(WORKFLOW_PATH, tools, tmp_path / "ckpt")
    assert [n for n, _ in log] == [f"sleep_{s}" for s in STEPS]
    assert set(out) == set(STEPS) and out["harvest"] == {"n": 1}
    assert log[1] == ("sleep_consolidate", "on")
    assert any((tmp_path / "ckpt").iterdir()), "MAF checkpoints should be written"


class Crash(BaseException):
    """Simulated process crash: declarative InvokeFunctionTool swallows Exception, not BaseException."""


def test_maf_runner_resumes_from_file_checkpoint(tmp_path):
    pytest.importorskip("agent_framework_declarative")
    log = []
    tools = _tools(log)
    inner = tools["sleep_consolidate"]
    crashed = []

    def consolidate(gate_mode: str) -> dict:
        if not crashed:
            crashed.append(gate_mode)
            raise Crash("killed mid-night")
        return inner(gate_mode=gate_mode)

    tools["sleep_consolidate"] = consolidate
    ckpt = tmp_path / "ckpt"
    with pytest.raises(Crash):
        maf_runner(WORKFLOW_PATH, tools, ckpt)
    assert [n for n, _ in log] == ["sleep_harvest"] and crashed == ["on"]
    assert (ckpt / "run.status").read_text(encoding="utf-8") == "running"
    assert list(ckpt.glob("*.json")), "the completed harvest superstep must be checkpointed"

    log.clear()
    out = maf_runner(WORKFLOW_PATH, tools, ckpt, resume=True)
    # resumed from the MAF checkpoint: harvest is not re-executed, the rest runs once each
    assert [n for n, _ in log] == [f"sleep_{s}" for s in STEPS[1:]]
    assert list(out) and set(out) == set(STEPS)
    assert out["harvest"] == {"ok": "sleep_harvest"}  # read back from the step journal
    assert (ckpt / "run.status").read_text(encoding="utf-8") == "done"

    # a finished run is not resumed again: resume=True on a done dir runs fresh
    log.clear()
    maf_runner(WORKFLOW_PATH, tools, ckpt, resume=True)
    assert [n for n, _ in log] == [f"sleep_{s}" for s in STEPS]


def test_maf_runner_without_resume_reruns_fresh(tmp_path):
    pytest.importorskip("agent_framework_declarative")
    log = []
    tools = _tools(log)
    inner = tools["sleep_record"]
    calls = []

    def record(envelope_kind: str) -> dict:
        calls.append(envelope_kind)
        if len(calls) == 1:
            raise Crash("killed")
        return inner(envelope_kind=envelope_kind)

    tools["sleep_record"] = record
    with pytest.raises(Crash):
        maf_runner(WORKFLOW_PATH, tools, tmp_path / "ckpt")
    log.clear()
    out = maf_runner(WORKFLOW_PATH, tools, tmp_path / "ckpt")
    assert [n for n, _ in log] == [f"sleep_{s}" for s in STEPS] and set(out) == set(STEPS)


# ------------------------------------------------------------------ default runner (fail-closed)


@pytest.fixture
def no_maf(monkeypatch):
    monkeypatch.delenv("SLEEP_RUNNER", raising=False)
    monkeypatch.setitem(sys.modules, "agent_framework_declarative", None)  # import -> ImportError


@pytest.mark.parametrize("profile", ["copilot", "offline", None])
def test_default_runner_raises_instead_of_falling_back(no_maf, profile):
    with pytest.raises(WorkflowError, match="refusing to fall back"):
        default_runner(profile)


def test_default_runner_sequential_only_when_explicit(no_maf, monkeypatch):
    assert default_runner("fake") is sequential_runner  # fake profile/tests only
    monkeypatch.setenv("SLEEP_RUNNER", "sequential")
    assert default_runner("copilot") is sequential_runner
    monkeypatch.setenv("SLEEP_RUNNER", "maf")
    with pytest.raises(WorkflowError, match="refusing"):
        default_runner("fake")
    monkeypatch.setenv("SLEEP_RUNNER", "bogus")
    with pytest.raises(WorkflowError, match="expected one of"):
        default_runner("copilot")


def test_default_runner_uses_maf_when_available(monkeypatch):
    pytest.importorskip("agent_framework_declarative")
    monkeypatch.delenv("SLEEP_RUNNER", raising=False)
    assert default_runner("copilot") is maf_runner
    assert default_runner("offline") is maf_runner
    assert default_runner("fake") is maf_runner


# ------------------------------------------------------------------ bundle


def test_patch_roundtrip_with_git_apply(tmp_path, h):
    repo = tmp_path / "r"
    skill = repo / "src/order_support/harness/skills/order-support/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(b"line1\r\nline2\r\n")
    (repo / "experiments/sleep").mkdir(parents=True)
    (repo / "experiments/sleep/state.json").write_bytes(b'{"night": 0}')  # no trailing newline
    h.init_repo(repo)
    patch = make_patch([
        FileChange("src/order_support/harness/skills/order-support/SKILL.md", "line1\r\nline2\r\n",
                   "line1\r\nline2\r\nline3\r\n"),
        FileChange("experiments/sleep/state.json", '{"night": 0}', '{"night": 1}\n'),
        FileChange("experiments/sleep/envelopes/n1.json", None, "{}\n"),
        FileChange("experiments/sleep/same.txt", "x", "x"),
    ])
    assert "same.txt" not in patch and "new file mode 100644" in patch
    p = tmp_path / "c.patch"
    p.write_bytes(patch.encode("utf-8"))
    h.git(repo, "apply", "--check", str(p))
    h.git(repo, "apply", str(p))
    assert skill.read_bytes() == b"line1\r\nline2\r\nline3\r\n"
    assert (repo / "experiments/sleep/state.json").read_bytes() == b'{"night": 1}\n'
    assert (repo / "experiments/sleep/envelopes/n1.json").read_bytes() == b"{}\n"


@pytest.mark.parametrize("path", ["../x", "experiments/sleep/../../etc/passwd", "/etc/passwd", "src/ci_lab/x.py",
                                  "experiments\\sleep\\x", ".github/workflows/x.yml", "experiments/sleep/.git/config",
                                  "C:/x"])
def test_make_patch_refuses_paths(path):
    with pytest.raises(ValueError):
        make_patch([FileChange(path, None, "x\n")])


def test_bundle_digests_and_tamper_detection(tmp_path):
    out = tmp_path / "b"
    m = write_bundle(out, patch="", experiment={"a": 1}, results={"b": 2}, base_sha="c" * 40,
                     night_id="sleep-20260101-1", date="20260101", accepted=False, ledger_update=False,
                     status="rejected")
    assert set(m["files"]) == {"candidate.patch", "experiment.json", "results.json"}
    assert verify_bundle(out)["base_sha"] == "c" * 40
    (out / "results.json").write_text(json.dumps({"b": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        verify_bundle(out)
    assert sorted(os.listdir(out)) == ["candidate.patch", "experiment.json", "manifest.json", "results.json"]
    assert Path(out / "manifest.json").read_text(encoding="utf-8").endswith("\n")
