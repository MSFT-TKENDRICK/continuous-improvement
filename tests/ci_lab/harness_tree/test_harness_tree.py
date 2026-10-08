from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import yaml

from ci_lab.contracts import COMPONENT_OWNERS
from ci_lab.harness_tree import (
    FROZEN_MANIFEST,
    HarnessTree,
    HarnessTreeError,
    default_root,
    load_manifest,
    manifest_errors,
    materialize,
    repo_harness_dir,
    snapshot,
    tree_digest,
)


def _copy(tmp_path: Path) -> Path:
    dest = tmp_path / "harness"
    shutil.copytree(repo_harness_dir(), dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def test_repo_manifest_is_frozen_copy() -> None:
    root = repo_harness_dir()
    assert (root / "harness.yaml").read_bytes().replace(b"\r\n", b"\n") == \
        FROZEN_MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    m = load_manifest()
    assert m["owners"] == dict(COMPONENT_OWNERS)
    assert HarnessTree(root)._manifest_check() == []


def test_hash_mismatch_fails(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    p = root / "harness.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("max_nudges: 4", "max_nudges: 99"), encoding="utf-8")
    assert any("differs from the frozen manifest" in e for e in HarnessTree(root).validate())
    p.unlink()
    assert any("harness.yaml: missing" in e for e in HarnessTree(root).validate())


def test_manifest_errors_owners_and_unknown_component() -> None:
    m = yaml.safe_load(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert manifest_errors(m) == []
    bad = {**m, "owners": {**m["owners"], "prompt": "agl"}}
    assert any("owners" in e for e in manifest_errors(bad))
    bad = {**m, "components": {**m["components"], "weights": ["w/*"]}}
    assert any("unknown component" in e for e in manifest_errors(bad))
    bad = {**m, "components": {"prompt": ["../x/*.md"]}}
    assert any("relative to harness/" in e for e in manifest_errors(bad))
    bad = {**m, "caps": {**m["caps"], "eval": {"max_llm_calls": 1}}}
    assert any("caps.eval" in e for e in manifest_errors(bad))


def test_frozen_manifest_is_authority(tmp_path: Path) -> None:
    """A tree's own harness.yaml never widens the components (a mismatch only fails validation)."""
    root = _copy(tmp_path)
    (root / "harness.yaml").write_text("format: whatever\ncomponents: {prompt: ['**']}\n", encoding="utf-8")
    tree = HarnessTree(root)
    assert tree.component_of("weights/x.bin") is None
    assert tree.component_globs()["prompt"] == ("harness/prompts/**/*.md",)


def test_paths_are_contained(tmp_path: Path) -> None:
    tree = HarnessTree(_copy(tmp_path))
    assert tree.path("mcp/exposure.yaml").is_file()
    assert tree.agent_spec_path("analyst") == tree.root / "agents" / "analyst.yaml"
    assert tree.prompt_path("common.md") == tree.root / "prompts" / "common.md"
    assert tree.workflow_path("triage") == tree.root / "workflows" / "triage.yaml"
    for bad in ("../x", "/etc/passwd", "agents/../../x"):
        with pytest.raises(HarnessTreeError):
            tree.path(bad)
    with pytest.raises(HarnessTreeError):
        tree.agent_spec_path("../evil")


def test_orphan_file_and_component_of(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    tree = HarnessTree(root)
    assert tree.component_of("prompts/a/b.md") == "prompt"
    assert tree.component_of("skills/x/SKILL.md") == "skill"
    assert tree.component_of("mcp/exposure.yaml") == "mcp"
    (root / "notes.txt").write_text("x", encoding="utf-8")
    assert "notes.txt: not in any component" in tree.validate()


def test_symlink_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    try:
        os.symlink(root / "harness.yaml", root / "guards-link")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert any("symlinks are not allowed" in e for e in HarnessTree(root).validate())


def test_snapshot_digest_stable_and_sensitive(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    a = snapshot(root)
    assert a.digest == tree_digest(root) and a.digest.startswith("sha256:")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert snapshot(root).digest == a.digest
    (root / "guards").mkdir(exist_ok=True)
    (root / "guards" / "new.md").write_text("x", encoding="utf-8")
    b = snapshot(root)
    assert b.digest != a.digest
    with pytest.raises(HarnessTreeError):
        a.verify()
    assert b.verify() is b
    assert type(b).from_dict(b.as_dict()) == b


def test_materialize_pins_a_copy(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    snap = materialize(root, tmp_path / "round" / "harness")
    assert snap.digest == tree_digest(root) and snap.root != root.resolve()
    (root / "harness.yaml").write_text("changed", encoding="utf-8")
    assert snap.verify() is snap
    with pytest.raises(HarnessTreeError):
        materialize(root, snap.root)


def test_default_root_env_only_for_cli(tmp_path: Path) -> None:
    assert default_root({}) == repo_harness_dir()
    assert default_root({"CI_HARNESS_DIR": str(tmp_path)}) == tmp_path


def test_loops_clamped_and_validated(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    (root / "agents").mkdir(exist_ok=True)
    (root / "agents" / "analyst.yaml").touch()
    (root / "loops").mkdir(exist_ok=True)
    (root / "loops" / "loops.yaml").write_text(
        "format: ci_lab.harness.loops.v1\nagents:\n  analyst: {max_nudges: 9, max_turns: 3}\n"
        "code_mode: {max_runs: 99}\n", encoding="utf-8")
    tree = HarnessTree(root)
    caps = tree.manifest["caps"]
    assert tree.loops() == {"agents": {"analyst": {"max_nudges": caps["agents"]["max_nudges"], "max_turns": 3}},
                            "code_mode": {"max_runs": caps["code_mode"]["max_runs"]}}
    errs = tree.loop_errors()
    assert any("max_nudges=9 exceeds the frozen cap" in e for e in errs)
    assert any("max_runs=99 exceeds" in e for e in errs)
    (root / "loops" / "loops.yaml").write_text(
        "format: ci_lab.harness.loops.v1\nagents:\n  ghost: {max_retries: 1}\n", encoding="utf-8")
    errs = tree.loop_errors()
    assert any("no agents/ghost.yaml" in e for e in errs) and any("unknown knob" in e for e in errs)
    with pytest.raises(HarnessTreeError):
        tree.loops()
