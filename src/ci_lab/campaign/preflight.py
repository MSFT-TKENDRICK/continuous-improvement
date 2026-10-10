"""Campaign model preflight (``copilot`` profile): fail before ``calibrate``/``run``/``confirm``
spend budget if a model the campaign will use is not available.

:func:`campaign_model_plan` lists every model the campaign can reach and who uses it:

* the meta agents (and subagents) from ``ci_lab.meta.specs`` (``CI_META_MODEL`` applied);
* the lesson synthesizer when the ``guard`` strategy is enabled;
* the optimizer LM (``CI_LAB_OPTIMIZER_MODEL``) for GEPA/SkillOpt/AGL. AGL uses the Copilot
  chat-client factory directly; GEPA/SkillOpt also check their copilot-serve endpoint;
* for the self-hosted harness, the pinned target model in ``CI_LAB_TARGET_MODEL``;
* for the legacy order-support domain, the order agent's model when
  ``ORDER_AGENT_PROFILE=copilot`` (``model.id`` of the incumbent harness ``agent.yaml``),
  or ``ORDER_AGENT_MODEL`` at ``OPENAI_API_BASE`` otherwise;
* the ASSERT tester (``CI_ASSERT_MODEL`` or each suite's ``default_model``) at ``OPENAI_API_BASE``.

Copilot-routed ids are checked against the Copilot SDK's ``list_models()``; OpenAI-compatible
endpoints against their ``GET /models``. The judge is the local s1 backend (not Copilot) and is
not checked here. Nothing is substituted: the error names the override to set.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ci_lab.contracts import Profile
from ci_lab.providers.models import (
    ListModels,
    ModelPreflightError,
    ModelUse,
    ServedModels,
    check_copilot_models,
    check_served_models,
)

__all__ = ["ModelPlan", "Preflight", "campaign_model_plan", "make_preflight"]

Preflight = Callable[[Mapping[str, Any]], Awaitable[None]]

META_HINT = "CI_META_MODEL=<id> (must be in the manifest allowed_models or CI_ALLOWED_MODELS)"
OPTIMIZER_HINT = "CI_LAB_OPTIMIZER_MODEL=<id>"
SERVE_HINT = ("point CI_COPILOT_SERVE_URL/CI_COPILOT_SERVE_KEY_FILE at a copilot-serve started with "
              "--model <id>")
ORDER_HINT = "the incumbent harness agent.yaml model.id (a committed harness change)"
ORDER_OFFLINE_HINT = "ORDER_AGENT_MODEL=openai/<served id>"
TESTER_HINT = "CI_ASSERT_MODEL=openai/<served id>"
TARGET_HINT = "CI_LAB_TARGET_MODEL=<copilot model id>"
OFFLINE_BASE_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL")


@dataclass
class ModelPlan:
    """``copilot``: ids for the Copilot SDK; ``served``: ``(base_url, api_key, uses)`` endpoint checks."""

    copilot: list[ModelUse] = field(default_factory=list)
    served: list[tuple[str, str, list[ModelUse]]] = field(default_factory=list)

    def serve(self, base: str, key: str, use: ModelUse) -> None:
        for b, k, uses in self.served:
            if (b, k) == (base, key):
                uses.append(use)
                return
        self.served.append((base, key, [use]))


def _strategies(hyper: Mapping[str, Any]) -> set[str]:
    raw = hyper.get("strategies") or ["agent"]
    return {s for s in (raw.split(",") if isinstance(raw, str) else raw) if s}


def _meta_uses(strategies: set[str], meta_harness_dir: Path | None = None) -> list[ModelUse]:
    from ci_lab.meta.spec_loader import AGENTS, SpecError, load_spec, subagent_specs

    uses: list[ModelUse] = []
    try:
        for key in AGENTS:
            spec = load_spec(key, harness_dir=meta_harness_dir)
            uses.append(ModelUse(spec.model, f"meta agent {key}", META_HINT))
            uses += [ModelUse(s.model, f"meta subagent {s.key}", META_HINT) for s in subagent_specs(spec)]
        if "guard" in strategies:
            from ci_lab.lessons_arm.agent import load_spec as load_synth

            spec, _ = load_synth()
            uses.append(ModelUse(str(spec["model"]["id"]), "lesson synthesizer (guard)", META_HINT))
    except (SpecError, ValueError) as exc:
        raise ModelPreflightError(f"meta-agent model selection is invalid: {exc}") from exc
    return uses


def _harness_model(harness_dir: Path) -> str | None:
    path = Path(harness_dir) / "agent.yaml"
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = doc.get("model") if isinstance(doc, Mapping) else None
    mid = model.get("id") if isinstance(model, Mapping) else model
    return str(mid) if mid else None


def _tester_models(evals_dir: Path) -> list[str]:
    out: list[str] = []
    for cfg_path in sorted(Path(evals_dir).glob("*/eval_config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        model = cfg.get("default_model") or {}
        name = model.get("name") if isinstance(model, Mapping) else model
        if name:
            out.append(str(name))
    return list(dict.fromkeys(out))


def _offline_endpoint(env: Mapping[str, str]) -> tuple[str, str] | None:
    base = next((env[v].strip() for v in OFFLINE_BASE_ENVS if env.get(v, "").strip()), "")
    return (base.rstrip("/"), env.get("OPENAI_API_KEY", "").strip() or "local") if base else None


def campaign_model_plan(hyper: Mapping[str, Any], *, env: Mapping[str, str] | None = None,
                        harness_dir: Path | None = None, evals_dir: Path | None = None,
                        tester_model: str | None = None, meta_harness_dir: Path | None = None,
                        domain_name: str = "harness") -> ModelPlan:
    """Every model a ``copilot``-profile campaign with ``hyper`` will use (see module doc)."""
    from ci_lab.optim import lm

    env = os.environ if env is None else env
    strategies = _strategies(hyper)
    plan = ModelPlan(copilot=_meta_uses(strategies, meta_harness_dir))
    optimizers = strategies & {"agl", "gepa", "skillopt"}
    served_optimizers = strategies & {"gepa", "skillopt"}
    if optimizers:
        model = lm.resolve_model(Profile.COPILOT, "optimizer", env=env)
        users = "optimizer LM (" + "/".join(sorted(optimizers)) + ")"
        plan.copilot.append(ModelUse(model, users, OPTIMIZER_HINT))
        if served_optimizers and not env.get(lm.AGL_BASE_URL_ENV, "").strip():
            try:
                base, key = lm.resolve_endpoint(Profile.COPILOT, env)
            except OSError as exc:
                raise ModelPreflightError(f"optimizer LM endpoint: cannot read the copilot-serve key ({exc}); "
                                          f"{SERVE_HINT}") from exc
            served_users = "optimizer LM (" + "/".join(sorted(served_optimizers)) + ")"
            plan.serve(base, key, ModelUse(model, served_users, f"{OPTIMIZER_HINT}, or {SERVE_HINT}"))
    if domain_name == "harness":
        from ci_lab.domain.harness import JUDGE_MODEL_ENV, S1_URL_ENV, TARGET_MODEL_ENV

        target = env.get(TARGET_MODEL_ENV, "").strip()
        judge = env.get(JUDGE_MODEL_ENV, "").strip()
        judge_url = env.get(S1_URL_ENV, "").strip()
        missing = [name for name, value in ((TARGET_MODEL_ENV, target), (JUDGE_MODEL_ENV, judge),
                                            (S1_URL_ENV, judge_url)) if not value]
        if missing:
            raise ModelPreflightError(
                "live harness campaigns require pinned target and System-1 judge settings: "
                + ", ".join(missing)
            )
        plan.copilot.append(ModelUse(target, "self-hosted harness target", TARGET_HINT))
        return plan
    from ci_lab.domain.order_support import ASSERT_MODEL_ENV, HARNESS_ROOT, REPO_ROOT

    offline = _offline_endpoint(env)
    if (env.get("ORDER_AGENT_PROFILE", "").strip().lower() or "offline") == "copilot":
        harness = Path(harness_dir) if harness_dir is not None else REPO_ROOT / HARNESS_ROOT
        if model := _harness_model(harness):
            plan.copilot.append(ModelUse(model, "order agent (ORDER_AGENT_PROFILE=copilot)", ORDER_HINT))
    elif offline is not None:
        model = env.get("ORDER_AGENT_MODEL", "").strip() or "openai/local"
        plan.serve(*offline, ModelUse(model, "order agent (ORDER_AGENT_PROFILE=offline)", ORDER_OFFLINE_HINT))
    if offline is not None:
        override = tester_model or env.get(ASSERT_MODEL_ENV, "").strip()
        testers = [override] if override else _tester_models(Path(evals_dir or REPO_ROOT / "evals" / "assert"))
        for model in testers:
            if model.startswith("openai/"):
                plan.serve(*offline, ModelUse(model, "ASSERT tester (eval_config default_model)", TESTER_HINT))
    return plan


def make_preflight(profile: Profile | str, *, harness_dir: Path | None = None, evals_dir: Path | None = None,
                   tester_model: str | None = None, env: Mapping[str, str] | None = None,
                   list_models: ListModels | None = None, fetch: ServedModels | None = None,
                   meta_harness_dir: Path | None = None, domain_name: str = "harness") -> Preflight | None:
    """The campaign preflight for ``profile``: ``None`` for ``fake``/``offline`` (no Copilot models).

    ``env`` is read when the preflight runs (default ``os.environ``); ``list_models``/``fetch``
    replace the Copilot SDK listing and the endpoint ``GET /models`` (tests)."""
    if Profile(profile) is not Profile.COPILOT:
        return None

    async def preflight(hyper: Mapping[str, Any]) -> None:
        plan = campaign_model_plan(hyper, env=env, harness_dir=harness_dir, evals_dir=evals_dir,
                                   tester_model=tester_model, meta_harness_dir=meta_harness_dir,
                                   domain_name=domain_name)
        await check_copilot_models(plan.copilot, list_models=list_models)
        for base, key, uses in plan.served:
            await asyncio.to_thread(check_served_models, base, key, uses, fetch=fetch)

    return preflight
