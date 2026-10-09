"""Profile wiring: builds :class:`~ci_lab.sleep.night.SleepDeps` for ``fake`` / ``offline`` /
``copilot``. Sibling fleet modules (providers, safety oracle, evaluation domain) are
discovered by import probing so this module lands independently; anything
REQUIRED for a real night that is missing raises :class:`WiringError` (no silent fallback to
fakes outside ``--profile fake``).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from ci_lab.contracts import (
    PROVIDER_MAPPING,
    EvalResult,
    Profile,
    Transcript,
    Violation,
)
from ci_lab.providers.offline import check_loopback, check_offline_endpoints
from ci_lab.sleep.fakes import (
    FakeOracle,
    FakeReflector,
    cases_from_tasks,
    fake_run_target,
    make_fake_assert_eval,
)
from ci_lab.sleep.harvest import load_reviewed_tasks, read_jsonl_rows
from ci_lab.sleep.night import SleepConfig, SleepDeps
from ci_lab.sleep.registry import HARNESS_EDITING, SkillTarget
from ci_lab.sleep.target import (
    make_harness_run_target,
    materialize_harness,
)

HARNESS_REL = Path("harness")
MODEL_ENV = {"target": "CI_LAB_SLEEP_TARGET_MODEL", "reflector": "CI_LAB_SLEEP_REFLECTOR_MODEL"}
DEFAULT_MODEL = {"copilot": "gpt-5-mini", "offline": "local"}


class WiringError(RuntimeError):
    pass


def model_uses(profile: Profile) -> list[Any]:
    """The Copilot models a ``copilot`` night uses (target + reflector), for the model preflight."""
    from ci_lab.providers.models import ModelUse

    if profile is not Profile.COPILOT:
        return []
    return [ModelUse(os.environ.get(env, "").strip() or DEFAULT_MODEL["copilot"], f"sleep {purpose}",
                     f"{env}=<id>") for purpose, env in MODEL_ENV.items()]


def _probe(candidates: Sequence[tuple[str, str]]) -> Any | None:
    for module, attr in candidates:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        if (obj := getattr(mod, attr, None)) is not None:
            return obj
    return None


# ------------------------------------------------------------------ clients

def client_factory(profile: Profile, purpose: str) -> Callable[[], Any]:
    model = os.environ.get(MODEL_ENV.get(purpose, ""), "") or DEFAULT_MODEL.get(profile.value, "")
    shared = _probe([("ci_lab.providers", "client_factory"), ("ci_lab.providers", "make_client"),
                     ("ci_lab.maf", "client_factory")])
    if shared is not None:
        return lambda: shared(profile=profile, model=model, purpose=purpose, rollout=None)
    if profile is Profile.COPILOT:
        cls = _probe([(PROVIDER_MAPPING["package"], PROVIDER_MAPPING["name"])])
        if cls is None:
            raise WiringError(f"{PROVIDER_MAPPING['package']}.{PROVIDER_MAPPING['name']} is not available")
        return lambda: cls(**{PROVIDER_MAPPING["model_field"]: model})
    if profile is Profile.OFFLINE:
        from agent_framework.openai import OpenAIChatClient

        base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "http://127.0.0.1:8080/v1"
        check_offline_endpoints()
        check_loopback("base_url", base)
        key = os.environ.get("OPENAI_API_KEY", "local")
        return lambda: OpenAIChatClient(model_id=model, base_url=base, api_key=key)
    raise WiringError(f"no client for profile {profile.value}")


class HarnessOracle:
    """Harness safety is enforced by ACS and the frozen evaluation suites, not case-specific rules."""

    def check(self, transcript: Transcript) -> list[Violation]:
        return []


# ------------------------------------------------------------------ ASSERT domain

def _candidate_rel(path: str, target: SkillTarget) -> Path:
    prefix = "harness/"
    if not path.startswith(prefix):
        raise WiringError(f"target {target.name!r}: path {path!r} is outside its harness")
    return Path(path.removeprefix(prefix))


def make_assert_eval(cfg: SleepConfig, target: SkillTarget = HARNESS_EDITING, *,
                     profile: Profile = Profile.FAKE,
                     k: int = 1) -> Callable[[str, str, str], EvalResult]:
    factory = _probe([("ci_lab.domain", "get_domain")])
    if factory is None:
        raise WiringError("ASSERT domain (ci_lab.domain) not available: the nightly gate cannot run")
    if target.eval_suite != "harness":
        raise WiringError(f"no ASSERT domain for eval suite {target.eval_suite!r}")
    domain = factory(target.eval_suite, repo_root=cfg.repo_root,
                     work_dir=Path(cfg.work_dir or cfg.out_dir) / "assert-domain",
                     profile=profile.value)
    assert cfg.work_dir is not None
    root = Path(cfg.work_dir) / "assert-harness" / target.name
    base = cfg.repo_root / HARNESS_REL
    skill_rel = _candidate_rel(target.skill_path, target)
    memory_rel = _candidate_rel(target.memory_path, target) if target.memory_path else None

    def assert_eval(skill: str, memory: str, variant: str) -> EvalResult:
        harness = materialize_harness(skill, memory, root, base, skill_rel=skill_rel, memory_rel=memory_rel)
        return asyncio.run(domain.evaluate(harness, "evolve", k, experiment_id="sleep", variant=variant))

    # the OES envelope of a night that never reaches the gate still records the gate's pin
    assert_eval.evaluator_pin = domain.pin  # type: ignore[attr-defined]
    return assert_eval


def latest_delta_from(repo: Path) -> Callable[[], float | None]:
    def latest() -> float | None:
        fn = _probe([("ci_lab.rrsi", "latest_delta"), ("ci_lab.ledger", "latest_delta")])
        if fn is not None:
            return fn(repo)
        files = sorted((repo / "experiments").glob("**/calibration*.json"), key=lambda p: p.as_posix())
        for p in reversed(files):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key in ("delta", "delta_aa"):
                if isinstance(doc.get(key), (int, float)):
                    return float(doc[key])
        return None

    return latest


# ------------------------------------------------------------------ deps

def _fake_target_deps(cfg: SleepConfig, target: SkillTarget) -> SleepDeps:
    path = cfg.tasks_path(target)
    tasks = load_reviewed_tasks(path) if path.exists() else []
    return SleepDeps(run_target=fake_run_target, oracle=FakeOracle(), reflector=FakeReflector(),
                     assert_eval=make_fake_assert_eval(cases_from_tasks(tasks)), latest_delta=lambda: 0.05)


def fake_deps(cfg: SleepConfig) -> SleepDeps:
    assert cfg.targets
    deps = _fake_target_deps(cfg, cfg.targets[0])
    if len(cfg.targets) > 1:
        deps.per_target = lambda t: _fake_target_deps(cfg, t)
    return deps


SUPPORTED_TARGETS = {
    ("proposer", "harness/skills/harness-editing/SKILL.md"),
    ("failure_analyst", "harness/skills/trace-triage/SKILL.md"),
}


def _real_target_deps(profile: Profile, cfg: SleepConfig, target: SkillTarget) -> SleepDeps:
    if (target.owner_agent, target.skill_path) not in SUPPORTED_TARGETS:
        raise WiringError(f"skill target {target.name!r}: no run_target harness for owner agent "
                          f"{target.owner_agent!r} with skill {target.skill_path!r}")
    assert cfg.work_dir is not None
    run_target = make_harness_run_target(
        client_factory(profile, "target"),
        harness_root=Path(cfg.work_dir) / "target-harness" / target.name,
        base_harness=cfg.repo_root / HARNESS_REL,
        owner_agent=target.owner_agent,
        skill_path=target.skill_path,
        memory_path=target.memory_path,
    )
    return SleepDeps(
        run_target=run_target,
        oracle=HarnessOracle(),
        reflector=_reflector(profile),
        assert_eval=make_assert_eval(cfg, target, profile=profile),
        latest_delta=latest_delta_from(cfg.repo_root),
    )


def build_deps(profile: Profile, cfg: SleepConfig, agl_exports: Iterable[Path] = (), *,
               lessons_dir: Path | None = None,
               lessons_sources: Iterable[tuple[str, Path]] = ()) -> SleepDeps:
    paths = list(agl_exports)
    if profile is Profile.FAKE:
        deps = fake_deps(cfg)
    else:
        assert cfg.targets
        if profile is Profile.OFFLINE:
            check_offline_endpoints()  # network-free: fail closed before any target is wired
        # fail fast for every target before the night starts
        per = {t.name: _real_target_deps(profile, cfg, t) for t in cfg.targets}
        deps = per[cfg.targets[0].name]
        if len(per) > 1:
            deps.per_target = lambda t: per[t.name]
    if paths:
        deps.agl_records = lambda: read_jsonl_rows(paths)
    if cfg.lessons_hook:
        from ci_lab.sleep.lessons_hook import make_lessons_hook

        store = Path(lessons_dir) if lessons_dir else Path(cfg.work_dir or cfg.out_dir) / "lessons"
        deps.lessons = make_lessons_hook(store, list(lessons_sources))
    return deps


def _reflector(profile: Profile) -> Any:
    from ci_lab.sleep.reflector import make_maf_reflector

    return make_maf_reflector(client_factory(profile, "reflector"))


__all__ = ["HarnessOracle", "WiringError", "build_deps", "client_factory", "fake_deps"]
