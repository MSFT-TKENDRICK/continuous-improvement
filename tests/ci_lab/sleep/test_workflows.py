"""Static lint of the sleep workflows (design C10, C22): least privilege, pinned actions, no uv in publish."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WF = ROOT / ".github" / "workflows"
FILES = ("sleep-nightly.yml", "usage-harvest.yml")
SHA_PIN = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def steps(job: dict) -> list[dict]:
    return job.get("steps") or []


def triggers(doc: dict) -> dict:
    return doc.get("on", doc.get(True))  # PyYAML parses bare `on` as True


@pytest.mark.parametrize("name", FILES)
def test_common_hardening(name):
    doc = load(name)
    raw = (WF / name).read_text(encoding="utf-8")
    assert doc["permissions"] == {}
    assert "pull_request_target" not in raw and "pull_request" not in triggers(doc)
    assert set(triggers(doc)) == {"schedule", "workflow_dispatch"}
    assert not (triggers(doc)["workflow_dispatch"] or {}).get("inputs")
    assert doc["concurrency"]["cancel-in-progress"] is False
    assert "secrets." not in raw
    for jname, job in doc["jobs"].items():
        assert isinstance(job.get("permissions"), dict), jname
        assert isinstance(job.get("timeout-minutes"), int) and job["timeout-minutes"] <= 120, jname
        for st in steps(job):
            if "uses" in st:
                assert SHA_PIN.match(st["uses"]), st["uses"]
                if st["uses"].startswith("actions/checkout@"):
                    assert st.get("with", {}).get("persist-credentials") is False, jname
                assert "cache" not in st["uses"], st["uses"]
            run = st.get("run", "")
            assert "github.event." not in run and "inputs." not in run, (jname, run)
            assert "${{" not in run, (jname, run)  # expressions only via env:


def test_nightly_jobs_and_permissions():
    jobs = load("sleep-nightly.yml")["jobs"]
    assert set(jobs) == {"gate", "evaluate", "publish"}
    assert triggers(load("sleep-nightly.yml"))["schedule"] == [{"cron": "17 7 * * *"}]
    assert jobs["gate"]["permissions"] == {"contents": "read"}
    ev = jobs["evaluate"]
    assert ev["permissions"] == {"contents": "read", "copilot-requests": "write"}
    assert ev["needs"] == "gate" and "needs.gate.outputs.run" in ev["if"]
    runs = "\n".join(s.get("run", "") for s in steps(ev))
    assert "ci-lab sleep run --profile copilot --out out/sleep-bundle" in runs
    assert "agentlightning.server" in runs and "redact-spans" in runs
    night = next(s for s in steps(ev) if s.get("id") == "night")
    assert night["env"]["COPILOT_GITHUB_TOKEN"] == "${{ github.token }}"
    uploads = [s["with"]["name"] for s in steps(ev) if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert uploads == ["sleep-bundle", "sleep-spans"]
    pub = jobs["publish"]
    assert pub["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert pub["environment"] == "sleep-publish" and pub["needs"] == "evaluate"
    assert "accepted == 'true'" in pub["if"] and "ledger_update == 'true'" in pub["if"]


@pytest.mark.parametrize("name,job,artifact", [("sleep-nightly.yml", "publish", "sleep-bundle"),
                                               ("usage-harvest.yml", "publish", "usage-bundle")])
def test_publish_jobs_run_only_the_stdlib_validator(name, job, artifact):
    pub = load(name)["jobs"][job]
    assert pub["environment"] == "sleep-publish"
    assert pub["permissions"] == {"contents": "write", "pull-requests": "write"}
    uses = [s["uses"].split("@")[0] for s in steps(pub) if "uses" in s]
    assert uses == ["actions/checkout", "actions/download-artifact"]
    checkout = steps(pub)[0]
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    dl = steps(pub)[1]["with"]
    assert dl["name"] == artifact and "runner.temp" in dl["path"]
    runs = [s["run"] for s in steps(pub) if "run" in s]
    assert runs == ['python3 -I scripts/sleep_publish.py --bundle "$BUNDLE_DIR"']
    text = repr(pub)
    assert "uv" not in re.findall(r"\buv\b", text) and "setup-uv" not in text and "pip" not in text
    assert "cache" not in text
    env = steps(pub)[-1]["env"]
    assert env["GH_TOKEN"] == "${{ github.token }}"
    assert not any("merge" in r for r in runs)


def test_usage_harvest_jobs():
    doc = load("usage-harvest.yml")
    jobs = doc["jobs"]
    assert set(jobs) == {"harvest", "publish"}
    assert jobs["harvest"]["permissions"] == {"contents": "read", "actions": "read"}
    runs = "\n".join(s.get("run", "") for s in steps(jobs["harvest"]))
    assert "ci-lab sleep harvest-usage" in runs and "--bundle out/usage-bundle" in runs
    assert "--open-pr" not in runs  # PRs only from the privileged publish job
    assert "ledger_update == 'true'" in jobs["publish"]["if"]
