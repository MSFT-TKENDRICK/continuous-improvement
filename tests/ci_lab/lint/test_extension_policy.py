from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_extension_policy_node_tests():
    r = subprocess.run([NODE, "--test", str(Path(__file__).with_name("policy.test.mjs"))], cwd=REPO, check=False,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_extension_entrypoint_parses():
    ext = REPO / ".github" / "extensions" / "ci-guardrails" / "extension.mjs"
    r = subprocess.run([NODE, "--check", str(ext)], capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    src = ext.read_text(encoding="utf-8")
    assert 'from "@github/copilot-sdk/extension"' in src and "onPreToolUse" in src
    assert not (ext.parent / "package.json").exists()
