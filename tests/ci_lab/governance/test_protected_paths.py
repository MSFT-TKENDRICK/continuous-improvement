"""Protected write globs: tools/paths.PROTECTED_WRITE_GLOBS == the ACS manifests, enforced by ArmFS."""

from __future__ import annotations

import pytest
import yaml

from ci_lab.governance.policies import manifest_text
from ci_lab.tools.arm_fs import ArmFS
from ci_lab.tools.paths import PROTECTED_WRITE_GLOBS, PathRejected, check_writable


@pytest.mark.parametrize("name", ["meta_agents", "campaign"])
def test_manifest_protected_globs_match_the_tool_deny_list(name):
    doc = yaml.safe_load(manifest_text(name))
    found = [tuple(p["protected_globs"]) for p in doc["policies"].values() if "protected_globs" in p]
    assert found and all(g == PROTECTED_WRITE_GLOBS for g in found)


@pytest.mark.parametrize("rel", ["src/ci_lab/rules/x.yaml", "src/ci_lab/governance/policies/p.yaml",
                                 ".github/workflows/ci.yml", "evals/assert/x.yaml", "harness/.lkg/r.yaml",
                                 "harness/lessons/registry.yaml", "SRC/CI_LAB/GUARDS/g.md"])
def test_arm_fs_refuses_protected_writes_inside_the_surface(tmp_path, rel):
    fs = ArmFS(tmp_path, ("**",))
    with pytest.raises(PathRejected, match="protected"):
        fs.write(rel, "x")
    assert not any(tmp_path.rglob("*.*"))


def test_unprotected_surface_write_still_works(tmp_path):
    fs = ArmFS(tmp_path, ("harness/**",))
    assert fs.write("harness/prompts/system.md", "hi") == 2
    assert check_writable("harness/skills/changes/SKILL.md") == "harness/skills/changes/SKILL.md"
