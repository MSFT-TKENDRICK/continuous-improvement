"""``ci-lab template doctor``: offline readiness checks for a repository created from the template.

Every check is local (files, git config, the lint engine); nothing calls GitHub. Repository variables,
secrets and environments cannot be read offline, so the doctor checks that every one the workflows
reference is documented in docs/template.md and lists them for you to set. Exit 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ci_lab.template.codeowners import CODEOWNERS_REL, OWNER_RE, rules
from ci_lab.template.init import origin_repo
from ci_lab.template.marker import (
    MARKER_REL,
    MarkerError,
    read_marker,
    template_identity,
)

DOC_REL = "docs/template.md"
WORKFLOWS_REL = ".github/workflows"
EVALS_REL = "evals/assert"
HARNESS_MANIFEST_REL = "harness/harness.yaml"
FROZEN_MANIFEST_REL = "src/ci_lab/harness_tree/manifest.yaml"
HARNESS_POLICY_REL = "src/ci_lab/governance/policies/harness.acs.yaml"
HARNESS_DATASET_REL = "evals/datasets/harness.yaml"
SLEEP_TARGETS_REL = "src/ci_lab/sleep/targets.yaml"
OPT_IN_VAR = "CI_HARNESS_ENABLED"
_VARS_RE = re.compile(r"\bvars\.([A-Za-z_][A-Za-z0-9_]*)")
_SECRETS_RE = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)")
_ENV_RE = re.compile(r"^\s*environment:\s*['\"]?([A-Za-z0-9_.-]+)", re.MULTILINE)
_S1_ALIASES = ("local", "openai")


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    fix: str = ""


def _pass(name: str, detail: str) -> Check:
    return Check(name, "PASS", detail)


def _warn(name: str, detail: str, fix: str) -> Check:
    return Check(name, "WARN", detail, fix)


def _fail(name: str, detail: str, fix: str) -> Check:
    return Check(name, "FAIL", detail, fix)


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n")


# ---------------------------------------------------------------- checks


def check_marker(root: Path) -> list[Check]:
    init = "run `uv run ci-lab template init --owners \"@org/team\" --apply` and commit the result"
    try:
        marker = read_marker(root)
    except MarkerError as e:
        return [_fail("marker", str(e), f"fix the syntax of {MARKER_REL} or delete it and re-run init")]
    if marker is None:
        return [_fail("marker", f"{MARKER_REL} is missing", init)]
    if marker.get("role") != "derived" or marker.get("initialized") is not True:
        return [_fail("marker", f"{MARKER_REL} says role={marker.get('role')!r}, initialized="
                      f"{marker.get('initialized')!r}: this is (a copy of) the template, not initialized", init)]
    out = [_pass("marker", f"initialized {marker.get('repository')} from {marker.get('template_repository')} "
                           f"on {marker.get('initialized_on')}")]
    origin = origin_repo(root)
    if origin and str(marker.get("repository", "")).lower() != origin.lower():
        out.append(_warn("marker", f"marker repository {marker.get('repository')!r} differs from origin {origin!r}",
                         f"re-run `ci-lab template init --repo {origin} --apply` if the repository moved"))
    return out


def check_codeowners(root: Path) -> list[Check]:
    path = root / CODEOWNERS_REL
    if not path.is_file():
        return [_fail("codeowners", f"{CODEOWNERS_REL} is missing (protected harness paths have no reviewers)",
                      "restore it from the template and run `ci-lab template init --apply`")]
    try:
        marker = read_marker(root)
    except MarkerError:
        marker = None
    _, template_owners = template_identity(marker)
    mine = tuple(marker.get("owners") or ()) if marker and marker.get("role") == "derived" else ()
    rs = rules(path.read_text(encoding="utf-8"))
    out: list[Check] = []
    unowned = [p for p, o in rs if not o]
    bad = sorted({o for _, os_ in rs for o in os_ if not OWNER_RE.match(o)})
    template = [p for p, o in rs if set(o) & set(template_owners) and not set(o) <= set(mine)]
    off = [p for p, o in rs if mine and o and list(o) != list(mine) and p not in template]
    fix = 'run `ci-lab template init --owners "@org/team" --apply`'
    if not rs:
        out.append(_fail("codeowners", "no rules", fix))
    if unowned:
        out.append(_fail("codeowners", f"rules without owners: {', '.join(unowned)}", fix))
    if bad:
        out.append(_fail("codeowners", f"invalid owners: {', '.join(bad)}", "use @user, @org/team or an email"))
    if template:
        out.append(_fail("codeowners", f"{len(template)} rules still owned by the template owner "
                         f"{' '.join(template_owners)}", fix))
    if off:
        out.append(_warn("codeowners", f"{len(off)} rules differ from the marker owners {' '.join(mine)}: "
                         f"{', '.join(off)}", "make CODEOWNERS and the marker agree (tests check this)"))
    if mine and set(mine) & set(template_owners):
        out.append(_warn("codeowners", f"owners include the template owner {' '.join(template_owners)}",
                         "intended only if you own the template too"))
    if not out:
        out.append(_pass("codeowners", f"{len(rs)} rules owned by {' '.join(mine or rs[0][1])}; enable \"Require "
                                       "review from Code Owners\" branch protection so they block merges"))
    return out


def workflow_settings(root: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {"variables": set(), "secrets": set(), "environments": set()}
    for wf in sorted((root / WORKFLOWS_REL).glob("*.y*ml")):
        text = wf.read_text(encoding="utf-8")
        found["variables"] |= set(_VARS_RE.findall(text))
        found["secrets"] |= set(_SECRETS_RE.findall(text)) - {"GITHUB_TOKEN"}
        found["environments"] |= set(_ENV_RE.findall(text))
    return found


def check_settings(root: Path) -> list[Check]:
    found = workflow_settings(root)
    doc = root / DOC_REL
    text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    names = sorted(n for group in found.values() for n in group)
    missing = [n for n in names if f"`{n}`" not in text]
    out: list[Check] = []
    if not doc.is_file():
        out.append(_fail("settings", f"{DOC_REL} is missing", f"restore {DOC_REL} from the template"))
    elif missing:
        out.append(_fail("settings", f"workflow settings not documented in {DOC_REL}: {', '.join(missing)}",
                         f"add each one (in backticks) to the settings table in {DOC_REL}"))
    else:
        summary = "; ".join(f"{k}: {', '.join(sorted(v)) or '-'}" for k, v in found.items())
        out.append(_pass("settings", f"documented ({summary}); set them under Settings > Secrets and "
                                     "variables > Actions / Environments"))
    ungated = []
    for wf in sorted((root / WORKFLOWS_REL).glob("*.y*ml")):
        doc_ = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        on = doc_.get("on", doc_.get(True)) or {}
        if isinstance(on, dict) and "schedule" in on and f"vars.{OPT_IN_VAR}" not in wf.read_text(encoding="utf-8"):
            ungated.append(wf.name)
    if ungated:
        out.append(_fail("settings", f"scheduled workflows not gated on vars.{OPT_IN_VAR}: {', '.join(ungated)}",
                         f"add the `opt_in` job (see campaign-scheduled.yml) so they no-op until {OPT_IN_VAR}=true"))
    return out


def _suites(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for cfg in sorted((root / EVALS_REL).glob("harness_*/eval_config.yaml")):
        try:
            out.append((cfg, yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}))
        except yaml.YAMLError as e:
            out.append((cfg, {"__error__": str(e)}))
    return out


def _known_backends() -> tuple[str, ...]:
    try:
        from ci_lab.judge.backends import BACKENDS
    except ImportError:  # pragma: no cover - judge deps missing
        BACKENDS = ("llamacpp", "openai_decisions", "systemone", "scripted")
    return (*BACKENDS, *_S1_ALIASES)


def check_judge(root: Path) -> list[Check]:
    suites = _suites(root)
    if not suites:
        return [_fail("judge", f"no ASSERT suites under {EVALS_REL}/", "add at least one eval_config.yaml")]
    known = _known_backends()
    out: list[Check] = []
    llama = False
    for cfg, doc in suites:
        rel = cfg.relative_to(root).as_posix()
        judge = (doc.get("pipeline") or {}).get("judge") if isinstance(doc.get("pipeline"), dict) else None
        if "__error__" in doc:
            out.append(_fail("judge", f"{rel}: invalid YAML", doc["__error__"]))
            continue
        if not judge:
            continue
        model = str(((judge.get("model") or {}) if isinstance(judge.get("model"), dict) else {}).get("name") or "")
        if not model:
            out.append(_fail("judge", f"{rel}: pipeline.judge.model.name is empty",
                             "set it to s1/llamacpp/<model> (docs/judge.md)"))
        elif not model.startswith("s1/"):
            out.append(_warn("judge", f"{rel}: judge {model!r} bypasses the System-1 provider",
                             "use an s1/<backend>/<model> judge (docs/judge.md, C21/C25)"))
        else:
            backend = model.split("/")[1] if model.count("/") >= 1 else ""
            if backend not in known:
                out.append(_fail("judge", f"{rel}: unknown s1 backend {backend!r}",
                                 f"use one of {', '.join(known)} (docs/judge.md)"))
            llama = llama or backend in ("llamacpp", "local")
    if llama:
        url = os.environ.get("CI_S1_LLAMA_URL") or "http://127.0.0.1:8081 (default)"
        out.append(_pass("judge", f"s1 llama.cpp judge at {url}: start llama-server with logprobs before live "
                                  "evals (docs/judge.md); set CI_S1_LLAMA_URL to move it"))
    if not out and not llama:
        out.append(_pass("judge", "every suite's judge model is configured"))
    return out


def check_test_sets(root: Path) -> list[Check]:
    out: list[Check] = []
    generating = 0
    for cfg, doc in _suites(root):
        if "__error__" in doc:
            continue
        rel = cfg.parent.relative_to(root).as_posix()
        pipeline = doc.get("pipeline") if isinstance(doc.get("pipeline"), dict) else {}
        regen = (f"generate and freeze it with the configured ASSERT pipeline, then copy "
                 f"artifacts/results/<suite name>/test_set.jsonl to {rel}/test_set.jsonl")
        if "test_set" in pipeline:
            generating += 1
            ts = cfg.parent / "test_set.jsonl"
            if not ts.is_file():
                out.append(_fail("test-sets", f"{rel}: frozen test_set.jsonl is missing", regen))
                continue
            rows, bad = 0, 0
            for line in ts.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rows += isinstance(json.loads(line), dict)
                    except json.JSONDecodeError:
                        bad += 1
            if bad or not rows:
                out.append(_fail("test-sets", f"{rel}/test_set.jsonl: {rows} cases, {bad} invalid lines", regen))
        else:
            judge = pipeline.get("judge") if isinstance(pipeline.get("judge"), dict) else {}
            inf = judge.get("inference_set_path")
            if inf and not (cfg.parent / str(inf)).is_file():
                out.append(_fail("test-sets", f"{rel}: judge-only inference set {inf} is missing",
                                 "rebuild it with the configured ASSERT replay pipeline"))
    if not generating:
        out.append(_fail("test-sets", "no generating ASSERT suite (pipeline.test_set) found",
                         "add a suite for your agent (docs/template.md#swap-in-your-own-agent)"))
    elif not out:
        out.append(_pass("test-sets", f"{generating} generating suites have frozen test sets"))
    return out


def check_harness(root: Path) -> list[Check]:
    """Validate the repo-root harness control plane and its offline/scheduled defaults."""
    required = (
        HARNESS_MANIFEST_REL, FROZEN_MANIFEST_REL, HARNESS_POLICY_REL,
        HARNESS_DATASET_REL, SLEEP_TARGETS_REL,
        ".github/workflows/campaign-scheduled.yml", ".github/workflows/sleep-nightly.yml",
    )
    missing = [rel for rel in required if not (root / rel).is_file()]
    if missing:
        return [_fail("harness", f"missing required harness files: {', '.join(missing)}",
                      "restore the repo-root harness tree, frozen manifest, ACS policy and schedules")]
    errors: list[str] = []
    local = _norm((root / HARNESS_MANIFEST_REL).read_text(encoding="utf-8"))
    frozen = _norm((root / FROZEN_MANIFEST_REL).read_text(encoding="utf-8"))
    if local != frozen:
        errors.append(f"{HARNESS_MANIFEST_REL} differs from {FROZEN_MANIFEST_REL}")
    try:
        manifest = yaml.safe_load(local) or {}
        if manifest.get("format") != "ci_lab.harness.v1" or not manifest.get("required_agents"):
            errors.append(f"{HARNESS_MANIFEST_REL} is not a frozen ci_lab.harness.v1 manifest")
    except yaml.YAMLError:
        errors.append(f"{HARNESS_MANIFEST_REL} is invalid YAML")
    try:
        from ci_lab.governance.acs import load_manifest

        load_manifest((root / HARNESS_POLICY_REL).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - doctor reports a local policy validation failure
        errors.append(f"{HARNESS_POLICY_REL} is invalid ({type(exc).__name__})")
    try:
        dataset = yaml.safe_load((root / HARNESS_DATASET_REL).read_text(encoding="utf-8")) or {}
        ci = (dataset.get("tiers") or {}).get("ci") or {}
        if dataset.get("format") != "ci_lab.harness.dataset.v1":
            errors.append(f"{HARNESS_DATASET_REL} has the wrong format")
        if ci.get("profile") != "fake" or ci.get("judge") != "fake":
            errors.append(f"{HARNESS_DATASET_REL} does not provide an offline fake CI tier")
        suite_root = root / str(dataset.get("suite_root") or EVALS_REL)
        case_ids = {
            str(row.get("test_case_id"))
            for path in suite_root.glob("harness_*/test_set.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for row in [json.loads(line)]
            if isinstance(row, dict) and row.get("test_case_id")
        }
        wanted = {
            str(case_id)
            for split in (dataset.get("splits") or {}).values() if isinstance(split, list)
            for case_id in split
        }
        if not wanted or not wanted <= case_ids:
            errors.append(f"{HARNESS_DATASET_REL} does not resolve to frozen harness cases")
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        errors.append(f"{HARNESS_DATASET_REL} or its test sets are invalid")
    campaign = (root / ".github/workflows/campaign-scheduled.yml").read_text(encoding="utf-8")
    for token in ("--domain harness", 'rrsi_profile="harness"', "CI_LAB_TARGET_MODEL",
                  "CI_LAB_JUDGE_MODEL", "CI_S1_LLAMA_URL"):
        if token not in campaign:
            errors.append(f"campaign-scheduled.yml is missing {token}")
    sleep = (root / ".github/workflows/sleep-nightly.yml").read_text(encoding="utf-8")
    targets = (root / SLEEP_TARGETS_REL).read_text(encoding="utf-8")
    if "SkillOpt-Sleep v0.2.x" not in sleep or "harness/skills/" not in targets:
        errors.append("sleep-nightly.yml/targets.yaml do not default to harness SkillOpt-Sleep")
    if errors:
        return [_fail("harness", "; ".join(errors), "restore the frozen harness defaults from the template")]
    return [_pass("harness", "frozen manifest/policy valid; fake CI and scheduled campaign/sleep target harness")]


def check_lint(root: Path) -> list[Check]:
    from ci_lab.lint.engine import run
    from ci_lab.lint.spec import RuleLoadError

    try:
        res = run(root)
    except (RuleLoadError, RuntimeError) as e:
        return [_fail("lint", f"lint could not run: {e}", "fix lint/rules/*.yaml")]
    if res.exit_code:
        first = "; ".join(f"{f.path}:{f.line} {f.rule}" for f in res.findings[:5])
        return [_fail("lint", f"{len(res.findings)} findings ({first})", "run `uv run ci-lab lint` and fix them")]
    return [_pass("lint", f"0 errors ({len(res.findings)} warnings)")]


CHECKS: tuple[tuple[str, Callable[[Path], list[Check]]], ...] = (
    ("marker", check_marker), ("codeowners", check_codeowners), ("settings", check_settings),
    ("harness", check_harness), ("judge", check_judge), ("test-sets", check_test_sets),
    ("lint", check_lint),
)


def doctor(root: Path, *, skip: tuple[str, ...] = ()) -> list[Check]:
    out: list[Check] = []
    for name, fn in CHECKS:
        if name not in skip:
            out.extend(fn(root))
    return out


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{c.status}] {c.name}: {c.detail}")
        if c.fix and c.status != "PASS":
            lines.append(f"       fix: {c.fix}")
    fails = sum(c.status == "FAIL" for c in checks)
    warns = sum(c.status == "WARN" for c in checks)
    lines.append(f"doctor: {'NOT READY' if fails else 'ready'} ({fails} failed, {warns} warnings)")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from ci_lab.lint.engine import repo_root

    root = Path(args.root).resolve() if args.root else repo_root()
    checks = doctor(root, skip=("lint",) if args.skip_lint else ())
    if args.format == "json":
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(format_checks(checks))
    return 1 if any(c.status == "FAIL" for c in checks) else 0
