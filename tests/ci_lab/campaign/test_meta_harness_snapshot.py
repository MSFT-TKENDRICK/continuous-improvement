"""Incumbent meta harness snapshots: ``MetaAgents.incumbent`` pins the evolvable specs per round."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ci_lab.campaign.wiring import META_HARNESS_DIR, META_HARNESS_RECORD, MetaAgents
from ci_lab.harness_tree import HarnessTreeError, repo_harness_dir, tree_digest
from ci_lab.meta.run import run_analyst, write_brief, write_failures
from ci_lab.meta.spec_loader import load_spec
from ci_lab.testing import Call, FakeChatClient

MARKER = "PINNED-INCUMBENT-MARKER"
DOMAIN = SimpleNamespace(surface_globs=("harness/**",), component_globs={"prompt": ("harness/prompts/**",)},
                         frozen_globs=())


def _agents(harness: Path, seen: list[str] | None = None) -> MetaAgents:
    def factory(*, profile, model, purpose, **_):
        if seen is not None:
            seen.append(purpose)
        return object()

    return MetaAgents(DOMAIN, factory, harness_dir=harness)


@pytest.fixture
def harness(tmp_path: Path) -> Path:
    dest = tmp_path / "src-harness"
    shutil.copytree(repo_harness_dir(), dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def test_incumbent_is_copied_recorded_and_reused(tmp_path: Path, harness: Path) -> None:
    agents, round_dir = _agents(harness), tmp_path / "round"
    round_dir.mkdir()
    snap = agents.incumbent(round_dir)
    assert snap.root == round_dir / META_HARNESS_DIR and snap.digest == tree_digest(harness)
    assert json.loads((round_dir / META_HARNESS_RECORD).read_text(encoding="utf-8"))["digest"] == snap.digest
    (harness / "prompts" / "analyst.md").write_text("changed after the round started\n", encoding="utf-8")
    again = agents.incumbent(round_dir)  # a resumed round keeps the pinned copy
    assert again == snap and again.digest != tree_digest(harness)


def test_tampered_incumbent_fails_closed(tmp_path: Path, harness: Path) -> None:
    agents, round_dir = _agents(harness), tmp_path / "round"
    round_dir.mkdir()
    snap = agents.incumbent(round_dir)
    (snap.root / "prompts" / "proposer.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(HarnessTreeError):
        agents.incumbent(round_dir)


def test_unrecorded_partial_copy_is_replaced(tmp_path: Path, harness: Path) -> None:
    round_dir = tmp_path / "round"
    (round_dir / META_HARNESS_DIR).mkdir(parents=True)
    (round_dir / META_HARNESS_DIR / "stray.txt").write_text("interrupted\n", encoding="utf-8")
    snap = _agents(harness).incumbent(round_dir)
    assert not (snap.root / "stray.txt").exists() and snap.digest == tree_digest(harness)


def test_client_model_comes_from_the_snapshot(tmp_path: Path, harness: Path) -> None:
    seen: list[str] = []
    agents = _agents(harness, seen)
    (tmp_path / "round").mkdir()
    snap = agents.incumbent(tmp_path / "round")
    agents.client("offline", "analyst", snap.root)
    assert seen == ["analyst"]
    assert load_spec("analyst", harness_dir=snap.root).harness_root == snap.root.resolve()


def test_run_analyst_uses_the_given_harness_dir(tmp_path: Path, harness: Path) -> None:
    prompt = harness / "prompts" / "analyst.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + f"\n{MARKER}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    write_brief(run_dir, "Round 1: analyze the failures.")
    write_failures(run_dir, [])
    client = FakeChatClient([[Call("submit_analysis", {"summary": "nothing", "patterns": [],
                                                       "suggested_components": []})]], default="UNEXPECTED")
    asyncio.run(run_analyst(run_dir, client, harness_dir=harness))
    msgs, options = client.requests[0]
    assert MARKER in " ".join(m.text or "" for m in msgs) + json.dumps(options, default=str)
