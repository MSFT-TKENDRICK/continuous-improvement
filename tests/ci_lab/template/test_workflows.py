"""Workflow gates for the template: scheduled workflows are opt-in; template-init is hardened."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows"
SCHEDULED = {"campaign-scheduled.yml": "gate", "sleep-nightly.yml": "gate", "usage-harvest.yml": "harvest",
             "governance-native.yml": "native"}
ALWAYS_ON = ("tests.yml", "lint.yml")
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")


def load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def triggers(doc: dict) -> dict:
    on = doc.get("on", doc.get(True))
    return on if isinstance(on, dict) else {k: None for k in ([on] if isinstance(on, str) else on)}


def test_every_scheduled_workflow_is_listed():
    scheduled = {p.name for p in WF.glob("*.yml") if "schedule" in triggers(load(p.name))}
    assert scheduled == set(SCHEDULED)


@pytest.mark.parametrize(("name", "first"), sorted(SCHEDULED.items()))
def test_scheduled_workflows_are_opt_in(name: str, first: str):
    jobs = load(name)["jobs"]
    opt_in = jobs["opt_in"]
    assert opt_in["permissions"] == {} and "needs" not in opt_in and "if" not in opt_in
    assert opt_in["outputs"] == {"enabled": "${{ steps.opt_in.outputs.enabled }}"}
    (step,) = opt_in["steps"]
    assert step["env"] == {"CI_HARNESS_ENABLED": "${{ vars.CI_HARNESS_ENABLED }}"}
    assert '"$CI_HARNESS_ENABLED" = "true"' in step["run"] and "::notice" in step["run"]
    assert "docs/template.md" in step["run"] and "${{" not in step["run"]
    # the first real job waits for the gate; every other job depends on it (directly or transitively)
    assert jobs[first]["needs"] == "opt_in"
    assert jobs[first]["if"] == "needs.opt_in.outputs.enabled == 'true'"
    for job_name, job in jobs.items():
        if job_name not in ("opt_in", first):
            needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
            assert needs and "opt_in" not in needs, job_name
    # workflow_dispatch is gated too: there is no bypass input
    assert "CI_HARNESS_ENABLED" not in str(triggers(load(name)).get("workflow_dispatch") or "")


@pytest.mark.parametrize("name", ALWAYS_ON)
def test_tests_and_lint_are_not_gated(name: str):
    text = (WF / name).read_text(encoding="utf-8")
    assert "CI_HARNESS_ENABLED" not in text and "opt_in" not in load(name)["jobs"]


@pytest.mark.parametrize("name", sorted(p.name for p in WF.glob("*.yml")))
def test_no_hard_coded_repository(name: str):
    text = (WF / name).read_text(encoding="utf-8")
    assert "MSFT-TKENDRICK" not in text and "continuous-improvement" not in text


def test_template_init_is_manual_minimal_and_stdlib_only():
    doc = load("template-init.yml")
    on = triggers(doc)
    assert set(on) == {"workflow_dispatch"}
    assert on["workflow_dispatch"]["inputs"]["owners"]["required"] is True
    assert doc["permissions"] == {}
    (job,) = doc["jobs"].values()
    assert job["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert isinstance(job["timeout-minutes"], int) and job["timeout-minutes"] <= 15
    assert job["env"]["REPO"] == "${{ github.repository }}"
    assert job["env"]["OWNERS"] == "${{ inputs.owners }}"
    for step in job["steps"]:
        if "uses" in step:
            assert PINNED.match(step["uses"].split("#")[0].strip()), step["uses"]
            assert step["uses"].startswith("actions/checkout@")
            assert step["with"]["persist-credentials"] is False
        run = step.get("run", "")
        assert "${{" not in run and "inputs." not in run and "github.event." not in run
        assert "uv " not in run and "pip " not in run
        assert not any("secrets." in str(v) for v in (step.get("env") or {}).values())
    runs = "\n".join(s.get("run", "") for s in job["steps"])
    assert runs.count("python3 -I -B scripts/template_init.py") == 2 and "--apply" in runs
    assert '--owners "$OWNERS"' in runs
    assert "gh pr create" in runs and "gh pr merge" not in runs and "--force" not in runs


def test_no_workflow_uses_pull_request_target():
    for p in WF.glob("*.yml"):
        assert "pull_request_target" not in triggers(load(p.name)), p.name
