from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ci_lab.bus.cli import make_voters_for, pools_from_env
from ci_lab.bus.voters.local import DeterministicCheckVoter
from ci_lab.bus.voters.remote import S1RubricVoter
from ci_lab.cli import main
from ci_lab.taskgraph.model import ContextRef, Deliverable, OutputSpec, StudentSpec
from ci_lab.taskgraph.students import AgentStudentFactory
from ci_lab.taskgraph.vault import RubricVault


def rubric(task: str, canary: str) -> dict[str, Any]:
    return {"id": f"{task}-rubric", "version": 1, "deliverable": task, "pass_score": 0.7, "canary": canary,
            "criteria": [
                {"id": "c-head", "description": "Heading first", "measure": "deterministic", "threshold": 1.0,
                 "required": True, "check": {"kind": "regex", "pattern": f"^# {task.upper()}$"}},
                {"id": "c-s1", "description": "States the topic", "measure": "s1", "threshold": 0.5,
                 "check": {"question": "Is the topic of the note stated?", "type": "noul", "options": []}}]}


def cli(capsys: pytest.CaptureFixture[str], *argv: Any) -> tuple[int, str]:
    rc = main([str(a) for a in argv])
    return rc, capsys.readouterr().out


@pytest.fixture
def graph(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path]:
    rubrics = tmp_path / "rubrics.json"
    rubrics.write_text(json.dumps({"rubrics": [rubric("a", "0123456789abcdef"), rubric("b", "fedcba9876543210")]}))
    rc, out = cli(capsys, "graph", "seal", rubrics, "--vault", tmp_path / "vault")
    assert rc == 0
    commit = {line.split("\t")[1]: line.split("\t")[2] for line in out.splitlines()}
    deliverables = [{"id": t, "title": f"Note {t}", "output": {"kind": "file", "path": f"out/{t}.md"},
                     "rubric_commitment": commit[t], "depends_on": deps, "budget": {"max_attempts": 2},
                     "instructions": f"Write the note.\n```\n# {t.upper()}\nbody\n```\n"}
                    for t, deps in (("a", []), ("b", ["a"]))]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"id": "g", "goal": "notes", "deliverables": deliverables}))
    return path, rubrics


def test_graph_and_bus_subcommands(tmp_path: Path, graph: tuple[Path, Path],
                                   capsys: pytest.CaptureFixture[str]) -> None:
    g, rubrics = graph
    assert cli(capsys, "graph", "validate", g, "--vault", tmp_path / "vault")[0] == 0
    rc, out = cli(capsys, "graph", "validate", g, "--vault", tmp_path / "empty")
    assert rc == 1 and "2 problem(s)" in out
    run = tmp_path / "run"
    rc, out = cli(capsys, "graph", "run", g, "--run-dir", run, "--run-id", "r1", "--rubric", rubrics,
                  "--student", "fake", "--challenger", "det", "--s1-model", "", "--no-telemetry", "--json")
    doc = json.loads((run / "graph_result.json").read_text())
    assert rc == 0 and json.loads(out) == doc and doc["ok"]
    assert {t: r["status"] for t, r in doc["tasks"].items()} == {"a": "committed", "b": "committed"}
    rc, out = cli(capsys, "graph", "show", run)
    assert rc == 0 and "b                committed" in out and "head r1/_run " in out
    assert cli(capsys, "bus", "heads", run / "bus", "r1")[1].count("\n") == 3
    rc, out = cli(capsys, "bus", "verify", run / "bus")
    assert rc == 0 and "3 topic(s), 0 corrupt" in out

    _, student = cli(capsys, "bus", "tail", run / "bus", "r1/a", "--role", "student")
    lines = student.splitlines()
    assert len(lines) == 2 and lines[0].split()[1] == "commit" and "not visible to student" in lines[1]
    _, orch = cli(capsys, "bus", "tail", run / "bus", "r1/a", "-n", "1000")
    assert " vote " in orch and "adversary:" in orch and "adversary:" not in student and " vote " not in student

    wal = next((run / "bus").rglob("*.jsonl"))
    wal.write_text(wal.read_text().replace('"commit"', '"commits"'))
    rc, out = cli(capsys, "bus", "verify", run / "bus")
    assert rc == 1 and "CORRUPT" in out


def test_run_maf_engine_unavailable(tmp_path: Path, graph: tuple[Path, Path], capsys: pytest.CaptureFixture[str],
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ci_lab.taskgraph.maf_engine", None)
    rc, out = cli(capsys, "graph", "run", graph[0], "--run-dir", tmp_path / "run", "--rubric", graph[1],
                  "--engine", "maf", "--student", "fake", "--no-telemetry")
    assert rc == 2 and "--engine maf unavailable" in out


def test_run_optimizer_gepa_wires_hardener(tmp_path: Path, graph: tuple[Path, Path],
                                           capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from ci_lab.adversary import harden
    from ci_lab.adversary.optim_adapters import GepaSoftQuestionAdapter
    from ci_lab.optim import lm

    made: list[Any] = []
    monkeypatch.setattr(lm, "make_lm", lambda profile, purpose: made.append((profile, purpose)) or "refl-lm")
    real = harden.Hardener
    monkeypatch.setattr(harden, "Hardener", lambda *a, **kw: made.append(real(*a, **kw)) or made[-1])
    base = ["graph", "run", graph[0], "--rubric", graph[1], "--student", "fake", "--s1-model", "", "--no-telemetry",
            "--optimizer", "gepa", "--profile", "offline"]
    rc, out = cli(capsys, *base, "--run-dir", tmp_path / "r0")
    assert rc == 2 and "--optimizer needs --challenger" in out and made == []
    rc, _ = cli(capsys, *base, "--run-dir", tmp_path / "r1", "--challenger", "det")
    assert rc == 0 and made[0] == ("offline", "optimizer")
    (opt,) = made[1].optimizers
    assert isinstance(opt.__self__, GepaSoftQuestionAdapter) and opt.__self__.reflection_lm == "refl-lm"
    assert opt.__self__.seed == made[1].seed and opt.__self__.scorer is None  # the hardener passes its scorer


def test_voters_for_and_pools(tmp_path: Path, graph: tuple[Path, Path]) -> None:
    assert inspect.signature(make_voters_for).parameters["domain_name"].default == "harness"
    vault = RubricVault(tmp_path / "vault")
    d = SimpleNamespace(rubric_commitment=vault.commitments()[0])
    on = make_voters_for(vault, run_dir=tmp_path, profile="offline", s1_model="s1/llamacpp/x")(d)
    assert [type(v) for v in on] == [DeterministicCheckVoter, S1RubricVoter] and on[1].model == "s1/llamacpp/x"
    assert [type(v) for v in make_voters_for(vault, run_dir=tmp_path, profile="offline", s1_model="")(d)] == [
        DeterministicCheckVoter]
    assert pools_from_env({"CI_POOL_S1": "3", "CI_POOL_GPU": "2"}).capacities | {"cpu": 1} == {
        "s1": 3, "llm": 4, "cpu": 1, "gpu": 2}


def test_agent_student_uses_student_spec_tools_in_workspace(tmp_path: Path) -> None:
    (tmp_path / "ctx").mkdir()
    (tmp_path / "ctx" / "log.txt").write_text("ORD-7")
    seen: dict[str, Any] = {}

    def builder(meta: Any, *, client: Any, bindings: Any, middleware: Any, **_: Any) -> Any:
        seen.update(tools=list(bindings), middleware=middleware)

        async def run(message: str, session: Any = None) -> None:
            assert bindings["submit_output"]("early").startswith("ERROR")
            bindings["write_file"]("out/a.md", "# A\n" + bindings["read_file"]("log.txt"))
            bindings["submit_output"]("wrote it")
        return SimpleNamespace(run=run)

    spec = StudentSpec.of(Deliverable("a", "A", "do it", OutputSpec("file", "out/a.md"), (ContextRef("file", "log.txt"),)))
    factory = AgentStudentFactory(context_root=tmp_path / "ctx", work_root=tmp_path / "ws", client=object(),
                                  builder=builder)
    out = asyncio.run(factory(spec, ["fw1", "fw2"]).run("msg"))
    assert out == "# A\nORD-7"
    assert seen["tools"] == ["list_files", "read_file", "write_file", "submit_output"]
    assert seen["middleware"][1:] == ["fw1", "fw2"]
