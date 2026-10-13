from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ci_lab.cache import cow, env as cenv, venv as cvenv
from ci_lab.cache.evalcache import EvalCache, pin_hash
from ci_lab.contracts import EvaluatorPin, Profile, TaskScore, Violation

TREE = "a" * 40
PIN = EvaluatorPin("b" * 40, "gpt-x", "copilot")


# ---------------------------------------------------------------- cow


def test_cow_detect_returns_valid_mode(tmp_path):
    info = cow.detect(tmp_path)
    assert info.mode in cow.MODES
    assert cow.cow_mode(tmp_path / "not" / "yet" / "created") in cow.MODES
    assert info.reason


@pytest.mark.skipif(os.name != "nt", reason="Windows volume API")
def test_windows_volume_info_no_subprocess(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("subprocess used"))
    root, fs, flags = cow.windows_volume_info(tmp_path)
    assert root and fs and isinstance(flags, int)
    assert cow.detect(tmp_path).filesystem == fs


@pytest.mark.skipif(os.name != "nt", reason="Windows volume API")
@pytest.mark.parametrize("fs,flags,mode", [("ReFS", 0, "clone"), ("NTFS", cow.FILE_SUPPORTS_BLOCK_REFCOUNTING, "clone"),
                                           ("NTFS", cow.FILE_SUPPORTS_HARD_LINKS, "hardlink"),
                                           ("exFAT", 0, "copy"), ("FAT32", 0, "copy")])
def test_windows_mode_mapping(monkeypatch, tmp_path, fs, flags, mode):
    monkeypatch.setattr(cow, "windows_volume_info", lambda p: ("C:\\", fs, flags))
    assert cow.detect(tmp_path).mode == mode


def test_cross_volume_source_forces_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(cow, "same_volume", lambda a, b: False)
    monkeypatch.setattr(cow, "_detect_windows", lambda p: cow.CowInfo("clone", "ReFS", "x"))
    monkeypatch.setattr(cow, "_detect_posix", lambda p: cow.CowInfo("clone", "btrfs", "x"))
    assert cow.detect(tmp_path, source=tmp_path).mode == "copy"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux mountinfo")
def test_linux_filesystem_name(tmp_path):
    assert cow.linux_filesystem(tmp_path)


# ---------------------------------------------------------------- env


def test_shared_env(monkeypatch, tmp_path):
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "elsewhere")
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setattr(cenv, "cow_mode", lambda p, source=None: "hardlink")
    e = cenv.shared_env(tmp_path, root=tmp_path / "cache")
    assert e["UV_CACHE_DIR"] == str(tmp_path / "cache" / "uv")
    assert e["PYTHONPYCACHEPREFIX"] == str(tmp_path / "cache" / "pycache")
    assert e["UV_LINK_MODE"] == "hardlink"
    assert "UV_NO_SYNC" not in e
    assert cenv.shared_env(tmp_path, root=tmp_path, no_sync=True)["UV_NO_SYNC"] == "1"

    full = cenv.with_shared_env(tmp_path, root=tmp_path / "cache", no_sync=False)
    assert "UV_PROJECT_ENVIRONMENT" not in full and "UV_NO_SYNC" not in full
    assert full["PATH"] == os.environ["PATH"]
    assert cenv.with_shared_env(tmp_path, root=tmp_path / "cache")["UV_NO_SYNC"] == "1"

    monkeypatch.setattr(cenv, "cow_mode", lambda p, source=None: "clone")
    assert cenv.shared_env(tmp_path, root=tmp_path)["UV_LINK_MODE"] == "clone"
    monkeypatch.setattr(cenv, "cow_mode", lambda p, source=None: "copy")
    assert cenv.shared_env(tmp_path, root=tmp_path)["UV_LINK_MODE"] == "hardlink"


def test_shared_env_respects_explicit_uv_cache(monkeypatch, tmp_path):
    e = cenv.shared_env(tmp_path, root=tmp_path, base={"UV_CACHE_DIR": str(tmp_path / "mine")})
    assert e["UV_CACHE_DIR"] == str(tmp_path / "mine")


def test_cache_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_CACHE_DIR", str(tmp_path))
    assert cenv.cache_root() == tmp_path
    monkeypatch.delenv("CI_CACHE_DIR")
    monkeypatch.setenv("CI_WT_ROOT", str(tmp_path / "wt"))
    assert cenv.cache_root() == tmp_path / "wt" / ".cache"


# ---------------------------------------------------------------- venv


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))


def _golden(tmp_path: Path) -> Path:
    g = tmp_path / "golden" / ".venv"
    (g / "Lib").mkdir(parents=True)
    (g / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    (g / "Lib" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    return g


def test_provision_sync(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "uv.lock").write_text("", encoding="utf-8")
    monkeypatch.setenv("UV_NATIVE_TLS", "1")
    monkeypatch.setenv("CI_CACHE_DIR", str(tmp_path / "cache"))
    run = FakeRunner()
    res = cvenv.provision(wt, runner=run, offline=True)
    assert res.method == "sync" and res.venv == wt / ".venv"
    (cmd, kw), = run.calls
    assert cmd[:2] == ["uv", "sync"] and "--native-tls" in cmd and "--frozen" in cmd and "--offline" in cmd
    assert kw["cwd"] == str(wt) and kw["check"] is True
    assert kw["env"]["UV_CACHE_DIR"] == str(tmp_path / "cache" / "uv")
    assert "UV_NO_SYNC" not in kw["env"]

    monkeypatch.delenv("UV_NATIVE_TLS")
    (wt / "uv.lock").unlink()
    run = FakeRunner()
    cvenv.provision(wt, runner=run)
    assert run.calls[0][0] == ["uv", "sync"]


def test_provision_clones_golden_when_cow(monkeypatch, tmp_path):
    g = _golden(tmp_path)
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(cvenv, "cow_mode", lambda p, source=None: "clone")
    run = FakeRunner()
    res = cvenv.provision(wt, golden=g.parent, runner=run)
    assert res.method == "clone"
    assert (wt / ".venv" / "Lib" / "mod.py").read_text(encoding="utf-8") == "X = 1\n"
    assert len(run.calls) == 1  # uv sync still fixes up the delta


def test_provision_skips_clone_without_cow(monkeypatch, tmp_path):
    g = _golden(tmp_path)
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(cvenv, "cow_mode", lambda p, source=None: "hardlink")
    res = cvenv.provision(wt, golden=g, runner=FakeRunner())
    assert res.method == "sync" and not (wt / ".venv").exists()


# ---------------------------------------------------------------- evalcache


def _score(case="case-1", trial=0, score=0.75):
    return TaskScore(case, trial, "s1", score, (Violation("r.x", "major", "d"),), 10, 20, "m-1")


def test_evalcache_roundtrip(tmp_path):
    c = EvalCache(tmp_path)
    assert c.get(PIN, TREE, "evolve", "case-1", 0) is None
    p = c.put(PIN, TREE, "evolve", _score())
    assert p == c.path(PIN, TREE, "evolve", "case-1", 0)
    rel = p.relative_to(tmp_path).as_posix()
    assert rel == f"{pin_hash(PIN)}/{TREE}/evolve/case-1/0.json"
    assert c.get(PIN, TREE, "evolve", "case-1", 0) == _score()
    assert c.get(PIN, TREE, "evolve", "case-1", 0, suite="other") is None
    assert c.get(PIN, TREE, "heldout", "case-1", 0) is None
    assert c.get(PIN, "c" * 40, "evolve", "case-1", 0) is None
    assert c.get(EvaluatorPin("b" * 40, "gpt-y", "copilot"), TREE, "evolve", "case-1", 0) is None
    assert c.get(PIN, TREE, "evolve", "case-1", 1) is None


def test_evalcache_disabled_and_missing(tmp_path):
    off = EvalCache(tmp_path, enabled=False)
    assert off.put(PIN, TREE, "evolve", _score()) is None
    assert off.get(PIN, TREE, "evolve", "case-1", 0) is None
    assert not any(tmp_path.iterdir())
    assert EvalCache.for_profile(Profile.COPILOT, tmp_path).enabled is False
    assert EvalCache.for_profile("offline", tmp_path).enabled is True
    assert EvalCache(tmp_path).put(PIN, TREE, "evolve", _score(score=None)) is None


def test_evalcache_unsafe_case_ids_hashed(tmp_path):
    c = EvalCache(tmp_path)
    for case in ("../../evil", "con", "a/b", "x:y", "é"):
        p = c.put(PIN, TREE, "ood", _score(case=case))
        assert p.is_relative_to(Path(os.path.realpath(tmp_path)))
        assert p.parent.name.startswith("h-")
        assert c.get(PIN, TREE, "ood", case, 0) == _score(case=case)


def test_evalcache_validation_and_corruption(tmp_path):
    c = EvalCache(tmp_path)
    with pytest.raises(ValueError):
        c.path(PIN, "../x", "evolve", "c", 0)
    with pytest.raises(ValueError):
        c.path(PIN, TREE, "train", "c", 0)
    with pytest.raises(ValueError):
        c.path(PIN, TREE, "evolve", "c", -1)
    p = c.put(PIN, TREE, "evolve", _score())
    p.write_text("{not json", encoding="utf-8")
    assert c.get(PIN, TREE, "evolve", "case-1", 0) is None


def test_served_judge_models_checked(tmp_path):
    c = EvalCache(tmp_path)
    pin1 = EvaluatorPin("b" * 40, "gpt-x", "copilot", ("gpt-x-0601",))
    c.put(pin1, TREE, "evolve", _score())
    assert pin_hash(pin1) == pin_hash(PIN)
    assert c.get(pin1, TREE, "evolve", "case-1", 0) is not None
    assert c.get(PIN, TREE, "evolve", "case-1", 0) is not None  # unknown served set: accept
    assert c.get(EvaluatorPin("b" * 40, "gpt-x", "copilot", ("gpt-x-0901",)), TREE, "evolve", "case-1", 0) is None
