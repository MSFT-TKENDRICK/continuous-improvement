"""End-to-end offline run of ``examples/taskgraph`` through the CLI, and its telemetry export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ci_lab.cli import main
from ci_lab.telemetry.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[3]
EX = ROOT / "examples" / "taskgraph"
OFFLINE = ["--student", "fake", "--challenger", "det", "--s1-model", ""]


def run_args(run_dir: Path) -> list[str]:
    return ["graph", "run", str(EX / "graph.yaml"), "--run-dir", str(run_dir), "--run-id", "demo",
            "--rubric", str(EX / "rubrics.yaml"), *OFFLINE]


def test_example_validates_and_runs_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["graph", "seal", str(EX / "rubrics.yaml"), "--vault", str(tmp_path / "v")]) == 0
    assert main(["graph", "validate", str(EX / "graph.yaml"), "--vault", str(tmp_path / "v")]) == 0
    assert main([*run_args(tmp_path / "run"), "--no-telemetry"]) == 0
    doc = json.loads((tmp_path / "run" / "graph_result.json").read_text())
    assert {t: r["status"] for t, r in doc["tasks"].items()} == dict.fromkeys(
        ("brief", "facts", "risks", "summary"), "committed")
    assert doc["critical_path"][0] == "brief" and doc["critical_path"][-1] == "summary"
    capsys.readouterr()
    assert main(["bus", "verify", str(tmp_path / "run" / "bus")]) == 0
    assert "5 topic(s), 0 corrupt" in capsys.readouterr().out


def test_run_exports_attribute_only_bus_events_to_jsonl(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env |= {"CI_DASHBOARD_STATE": str(tmp_path / "no-dashboard.json"), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "ci_lab.cli", *run_args(tmp_path / "run")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stderr
    spans, bad = read_jsonl(sorted((tmp_path / "run" / "telemetry").glob("spans-*.jsonl")))
    assert bad == 0
    names = {s["name"] for s in spans}
    assert {"ci.taskgraph.cli", "ci.taskgraph.run", "ci.taskgraph.deliverable", "ci.taskgraph.attempt"} <= names
    events = [e for s in spans for e in s["events"] if e["name"] == "ci.bus.append"]
    assert {e["attributes"]["ci.bus.kind"] for e in events} >= {"manifest", "proposal", "vote", "commit"}
    assert all(set(e["attributes"]) == {"ci.bus.topic", "ci.bus.seq", "ci.bus.kind", "ci.bus.role"}
               for e in events)

def test_runs_sharing_a_run_dir_adopt_hardened_rubrics(tmp_path: Path) -> None:
    import asyncio
    from dataclasses import replace

    from ci_lab.bus import ids
    from ci_lab.bus.cli import _load_rubrics
    from ci_lab.bus.types import Author, RubricPatchBody
    from ci_lab.bus.wal import AgentBus
    from ci_lab.taskgraph.vault import RubricVault

    run_dir = tmp_path / "run"

    def run(run_id: str, *extra: str) -> set[str]:
        args = [*run_args(run_dir), "--no-telemetry", *extra]
        args[args.index("demo")] = run_id
        assert main(args) == 0
        st = AgentBus(run_dir / "bus").state(ids.task_topic(run_id, "brief"))
        return {e.body.rubric_version for e in st.proposals.values()}

    assert run("demo") == {"brief-rubric@v1"}
    brief = next(r for r in _load_rubrics([EX / "rubrics.yaml"]) if r.id == "brief-rubric")
    RubricVault.for_run(run_dir).seal(replace(brief, version=2))  # as if the hardener ran after the commit
    patch = RubricPatchBody(rubric_id="brief-rubric", from_version="brief-rubric@v1", to_version="brief-rubric@v2",
                            applies_from_attempt=None, applies_from_epoch=1, diff_sha256="0" * 64, metrics={},
                            accepted=True)
    asyncio.run(AgentBus(run_dir / "bus").append(ids.run_topic("demo"), "rubric_patch", Author("hardener", "h"), patch))
    assert run("next") == {"brief-rubric@v2"}
    assert run("pinned", "--no-adopt-hardened") == {"brief-rubric@v1"}
