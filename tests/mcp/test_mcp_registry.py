import copy
import sys
from pathlib import Path

import pytest
import yaml

from ci_lab.mcp.registry import (
    FROZEN_SERVERS,
    McpConfigError,
    exposure_path,
    load_exposure,
    load_registry,
    parse_exposure,
    parse_registry,
    validate_exposure,
)

REPO = Path(__file__).resolve().parents[2]
RAW = yaml.safe_load(FROZEN_SERVERS.read_text(encoding="utf-8"))


def _reg(mutate):
    data = copy.deepcopy(RAW)
    mutate(data)
    return parse_registry(data)


def test_frozen_registry_loads():
    reg = load_registry()
    h = reg.servers["harness"]
    assert "read_component" in h.tools_allow and set(h.roots) == {"harness_root", "runs_root"}
    assert reg.code_mode.timeout_s_max > 0 and "json" in reg.code_mode.allowed_imports


def test_argv_and_roots_substitution(tmp_path):
    h = load_registry().servers["harness"]
    roots = h.resolve_roots({"harness_root": tmp_path}, base_dir=tmp_path / "repo")
    argv = h.argv(roots)
    assert argv[0] == sys.executable and argv[argv.index("--root") + 1] == str(tmp_path.resolve())
    assert argv[argv.index("--runs-root") + 1] == str((tmp_path / "repo" / "artifacts").resolve())
    with pytest.raises(McpConfigError):
        h.resolve_roots({"nope": tmp_path})


def test_env_is_allowlisted():
    h = load_registry().servers["harness"]
    assert h.env({"PYTHONPATH": "x", "GH_TOKEN": "secret"}) == {"PYTHONPATH": "x"}


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["servers"]["harness"].__setitem__("command", "python -m x"), "argv list"),
    (lambda d: d["servers"]["harness"].__setitem__("command", ["python", "a && b"]), "shell metacharacters"),
    (lambda d: d["servers"]["harness"]["command"].append("{nowhere}"), "unknown placeholder"),
    (lambda d: d["servers"]["harness"].__setitem__("shell", True), "unknown keys"),
    (lambda d: d["servers"]["harness"].__setitem__("tools_allow", []), "tools_allow"),
    (lambda d: d["code_mode"]["allowed_imports"].append("subprocess"), "forbidden modules"),
    (lambda d: d["code_mode"].__setitem__("timeout_s_max", -1), "timeout_s_max"),
    (lambda d: d.__setitem__("format", "v0"), "format"),
])
def test_registry_rejections(mutate, needle):
    with pytest.raises(McpConfigError) as ei:
        _reg(mutate)
    assert any(needle in e for e in ei.value.errors), ei.value.errors


def test_repo_exposure_is_valid_and_code_mode():
    exp = load_exposure(exposure_path(REPO / "harness"))
    assert exp.servers["harness"].mode == "code"
    caps = load_registry().code_mode
    assert exp.code_mode.timeout_s <= caps.timeout_s_max
    assert set(exp.code_mode.imports) <= set(caps.allowed_imports)


def test_exposure_defaults_and_tool_subset():
    reg = load_registry()
    exp = parse_exposure({"format": "ci_lab.mcp.exposure.v1", "servers": {"harness": {}}}, reg)
    assert set(exp.servers["harness"].tools) == set(reg.servers["harness"].tools_allow)
    assert exp.servers["harness"].mode == "direct"
    assert exp.code_mode.timeout_s == reg.code_mode.timeout_s_max
    exp = parse_exposure({"format": "ci_lab.mcp.exposure.v1",
                          "servers": {"harness": {"tools": {"list_components": "Short."}}}}, reg)
    assert exp.servers["harness"].tools == {"list_components": "Short."}


@pytest.mark.parametrize("data, needle", [
    ({"servers": {"evil": {}}}, "not in the frozen registry"),
    ({"servers": {"harness": {"tools": {"write_file": None}}}}, "tools_allow"),
    ({"servers": {"harness": {"mode": "shell"}}}, "mode"),
    ({"servers": {"harness": {"command": ["x"]}}}, "unknown keys"),
    ({"servers": {"harness": {"tools": {"list_components": "x" * 5000}}}}, "description"),
    ({"code_mode": {"timeout_s": 10_000}}, "timeout_s"),
    ({"code_mode": {"max_output_chars": 10**9}}, "max_output_chars"),
    ({"code_mode": {"imports": ["json", "os"]}}, "exceed"),
    ({"code_mode": {"allowed_imports": ["os"]}}, "unknown keys"),
    ({"env_allow": ["GH_TOKEN"]}, "unknown top-level"),
])
def test_exposure_rejects_anything_beyond_frozen_caps(data, needle):
    reg = load_registry()
    errs = validate_exposure({"format": "ci_lab.mcp.exposure.v1", **data}, reg)
    assert any(needle in e for e in errs), errs
    with pytest.raises(McpConfigError):
        parse_exposure({"format": "ci_lab.mcp.exposure.v1", **data}, reg)
