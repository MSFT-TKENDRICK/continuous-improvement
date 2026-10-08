"""``LessonSynthesizer``: MAF declarative agent for leftover clusters (design §13.3 step 4).

The agent (``specs/lesson_synthesizer.yaml``, expression-free) sees only a typed lesson view
(:func:`lesson_view`: ids, tool names, enum tokens — never trace text or tool args; C12/B3) and
acts only through ``submit_rule``. Its arguments (:class:`SubmitRuleArgs`) can only choose a
skeleton from :data:`SKELETONS`, the matching rung + template id, a target tool from the
vocabulary and typed slots; they are validated into a shadow :class:`~ci_lab.rulespec.RuleSpec`
by the same deterministic template synthesizers (``provenance.source = "synthesizer"``).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ci_lab.contracts import ChatClientFactory, Profile
from ci_lab.rulespec import LessonCluster, RuleSpec

from .features import FeatureKind, LessonFeatures, is_injection
from .synth import SYNTHESIZERS, SynthesisError
from .templates import (
    REQUIRED_TEMPLATES,
    SAFE_PATTERNS,
    TPL_AMOUNT_PRIOR,
    TPL_ARG_CONSTRAINT,
    TPL_PRIOR_CALL,
    TPL_REDACT_PATTERN,
    TPL_STATE_FLAG,
)

SPEC_PATH = Path(__file__).resolve().parent / "specs" / "lesson_synthesizer.yaml"
TERMINAL_TOOL = "submit_rule"
MAX_INVALID_SUBMISSIONS = 3

SKELETONS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    # skeleton: (rung, template id, slot names)
    "prior_call": ("R2", TPL_PRIOR_CALL, ("prior_tool", "subject_arg", "prior_subject_field",
                                          "prior_result_equals", "within")),
    "state_flag": ("R2", TPL_STATE_FLAG, ("flag", "subject_arg", "via_tool")),
    "arg_constraint": ("R1", TPL_ARG_CONSTRAINT, ("arg", "op", "values")),
    "amount_vs_prior": ("R2", TPL_AMOUNT_PRIOR, ("prior_tool", "subject_arg", "prior_subject_field",
                                                 "amount_arg", "prior_amount_field", "cmp_op", "within")),
    "response_pattern": ("R3", TPL_REDACT_PATTERN, ("flag", "pattern_classes")),
}
_TOOL_SLOTS = ("target_tool", "prior_tool", "via_tool")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class SynthesizerError(RuntimeError):
    """Base class for LessonSynthesizer failures."""


class NoRuleSubmitted(SynthesizerError):
    """The agent finished without a valid ``submit_rule`` call."""

    def __init__(self, cluster_id: str, errors: list[str]) -> None:
        super().__init__(f"{cluster_id}: agent never submitted a valid rule"
                         + (f" (last error: {errors[-1]})" if errors else ""))
        self.cluster_id = cluster_id
        self.errors = errors


class NoSkeletonFits(SynthesizerError):
    """The agent explicitly declined (``skeleton: none``): the lesson stays prose (R5/R6)."""


class SubmitRuleArgs(BaseModel):
    """The only arguments ``submit_rule`` accepts. Anything else is rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skeleton: FeatureKind | Literal["none"]
    rung: Literal["R1", "R2", "R3"] | None = None
    template_id: Literal[TPL_PRIOR_CALL, TPL_STATE_FLAG, TPL_ARG_CONSTRAINT,  # type: ignore[valid-type]
                         TPL_AMOUNT_PRIOR, TPL_REDACT_PATTERN] | None = None
    target_tool: str | None = None
    slots: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> SubmitRuleArgs:
        if self.skeleton == "none":
            return self
        rung, tpl, allowed = SKELETONS[self.skeleton]
        if self.rung is not None and self.rung != rung:
            raise ValueError(f"skeleton {self.skeleton} requires rung {rung}")
        if self.template_id is not None and self.template_id != tpl:
            raise ValueError(f"skeleton {self.skeleton} requires template_id {tpl}")
        if extra := sorted(set(self.slots) - set(allowed)):
            raise ValueError(f"slots {extra} not allowed for {self.skeleton}; allowed {list(allowed)}")
        return self

    def features(self, *, trusted: bool) -> LessonFeatures:
        if self.skeleton == "none":
            raise NoSkeletonFits("agent declined: no skeleton fits")
        slots = dict(self.slots)
        if isinstance(slots.get("values"), (str, int, float, bool)):
            slots["values"] = [slots["values"]]
        return LessonFeatures.model_validate({**slots, "kind": self.skeleton, "target_tool": self.target_tool,
                                              "trusted": trusted, "injection_suspect": False})


def _tokens(values: Iterable[Any]) -> list[str]:
    return sorted({str(v) for v in values if isinstance(v, str) and _SAFE_TOKEN.match(v)})


def tool_vocabulary(cluster: LessonCluster, features: LessonFeatures | None = None,
                    extra: Iterable[str] = ()) -> list[str]:
    """Tool names the agent may use: the cluster's tool n-grams, partial features, ``extra``."""
    names: list[Any] = [t for g in cluster.fingerprint.tool_ngrams for t in g]
    if features is not None:
        names += [getattr(features, s) for s in _TOOL_SLOTS]
    return [t for t in _tokens([*names, *extra]) if re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$", t)]


def lesson_view(cluster: LessonCluster, features: LessonFeatures | None, vocabulary: list[str]) -> dict[str, Any]:
    """The only input the agent sees: ids, enum tokens and counts (C12/B3)."""
    fp = cluster.fingerprint
    partial = None
    if features is not None:
        partial = {k: v for k, v in features.model_dump(mode="json", exclude_defaults=True).items()
                   if k not in ("trusted", "injection_suspect")}
    return {
        "lesson": {"cluster_id": cluster.id if _SAFE_TOKEN.match(cluster.id) else fp.digest(),
                   "route": cluster.route, "oracle_rules": _tokens(fp.oracle_rules),
                   "rubric_ids": _tokens(fp.rubric_ids), "error_class": (_tokens([fp.error_class]) or [None])[0],
                   "tool_ngrams": [list(g) for g in fp.tool_ngrams if all(_SAFE_TOKEN.match(t) for t in g)],
                   "n_members": len(cluster.members), "n_families": len(cluster.families),
                   "n_slices": len(cluster.slices)},
        "partial_features": partial,
        "tool_vocabulary": vocabulary,
        "skeletons": {k: {"rung": r, "template_id": t, "slots": list(s)} for k, (r, t, s) in SKELETONS.items()},
        "pattern_classes": sorted(SAFE_PATTERNS),
    }


def load_spec(path: Path = SPEC_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(declarative spec without x-ci, x-ci block)``; rejects PowerFx/``=`` expressions."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    xci = dict(spec.pop("x-ci", None) or {})

    def walk(node: Any, where: str) -> list[str]:
        if isinstance(node, str):
            return [where] if node.lstrip().startswith("=") else []
        if isinstance(node, Mapping):
            return [x for k, v in node.items() for x in walk(v, f"{where}.{k}")]
        if isinstance(node, list):
            return [x for i, v in enumerate(node) for x in walk(v, f"{where}[{i}]")]
        return []

    if bad := walk(spec, "spec") + walk(xci, "x-ci"):
        raise SynthesizerError(f"{path.name}: expressions are not allowed: {bad}")
    tools = [t.get("name") for t in spec.get("tools") or []]
    if tools != [TERMINAL_TOOL] or xci.get("terminal_tool") != TERMINAL_TOOL:
        raise SynthesizerError(f"{path.name}: must expose exactly one tool {TERMINAL_TOOL}")
    return spec, xci


AgentBuilder = Callable[[dict[str, Any], Any, Mapping[str, Callable[..., Any]]], Any]


def local_builder(spec: dict[str, Any], client: Any, bindings: Mapping[str, Callable[..., Any]]) -> Any:
    """Build via ``agent_framework_declarative.AgentFactory`` in safe mode with the injected client."""
    from agent_framework_declarative import AgentFactory

    doc = {k: v for k, v in spec.items() if k != "model"}  # injected client wins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factory = AgentFactory(client=client, bindings=dict(bindings), safe_mode=True, env_file_path=os.devnull)
        return factory.create_agent_from_dict(doc)


def default_builder() -> AgentBuilder:
    """``ci_lab.maf.loader.build_agent`` when importable (HOOK(M1)), else :func:`local_builder`."""
    try:
        from ci_lab.maf.loader import build_agent  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional parallel module
        return local_builder

    def build(spec: dict[str, Any], client: Any, bindings: Mapping[str, Callable[..., Any]]) -> Any:
        try:
            return build_agent(SPEC_PATH, client=client, bindings=dict(bindings))
        except Exception:  # noqa: BLE001 - loader contract still settling; building has no side effects
            return local_builder(spec, client, bindings)

    return build


class LessonSynthesizer:
    """Runs the declarative agent once per leftover cluster; returns a validated shadow rule."""

    def __init__(self, *, client: Any = None, client_factory: ChatClientFactory | None = None,
                 profile: Profile = Profile.COPILOT, model: str | None = None,
                 builder: AgentBuilder | None = None, spec_path: Path = SPEC_PATH) -> None:
        if client is None and client_factory is None:
            raise TypeError("LessonSynthesizer needs a client or a client_factory")
        self.spec, self.xci = load_spec(spec_path)
        self._client = client
        self._factory = client_factory
        self.profile = profile
        self.model = model or str((self.spec.get("model") or {}).get("id") or "")
        self.builder = builder or default_builder()

    def client(self) -> Any:
        if self._client is None:
            assert self._factory is not None
            self._client = self._factory(profile=self.profile, model=self.model,
                                         purpose=self.xci.get("purpose", "proposer"))
        return self._client

    async def synthesize(self, cluster: LessonCluster, features: LessonFeatures | None = None, *,
                         trusted: bool | None = None, vocabulary: Iterable[str] = ()) -> RuleSpec:
        if is_injection(cluster) or (features is not None and features.injection_suspect):
            raise SynthesisError("injection-suspect lessons are never synthesized (B3)")
        if trusted is None:
            trusted = features.trusted if features is not None else (cluster.human_confirmed
                                                                      or cluster.status == "confirmed")
        if not trusted and not cluster.human_confirmed:
            raise SynthesisError("untrusted (usage/PR) lesson needs a human-reviewed label first (B3)")
        vocab = tool_vocabulary(cluster, features, vocabulary)
        accepted: list[RuleSpec] = []
        errors: list[str] = []
        declined: list[bool] = []

        def submit_rule(skeleton: str, rung: str | None = None, template_id: str | None = None,
                        target_tool: str | None = None, slots: dict[str, Any] | None = None) -> str:
            kwargs = {k: v for k, v in {"skeleton": skeleton, "rung": rung, "template_id": template_id,
                                        "target_tool": target_tool, "slots": slots}.items() if v is not None}
            if accepted:
                return "ERROR: a rule was already accepted; reply done"
            if len(errors) >= MAX_INVALID_SUBMISSIONS:
                return "ERROR: too many invalid submissions; stop"
            try:
                args = SubmitRuleArgs.model_validate(kwargs)
                if args.skeleton == "none":
                    declined.append(True)
                    return "accepted"
                feats = args.features(trusted=True)
                bad = [getattr(feats, s) for s in _TOOL_SLOTS
                       if getattr(feats, s) is not None and getattr(feats, s) not in vocab]
                if bad:
                    raise ValueError(f"tools {bad} not in tool_vocabulary")
                rule = SYNTHESIZERS[args.skeleton](cluster, feats)
            except (ValidationError, ValueError, SynthesisError) as exc:
                msg = _short(exc)
                errors.append(msg)
                return f"ERROR: {msg}"
            accepted.append(rule.model_copy(update={
                "provenance": rule.provenance.model_copy(update={"source": "synthesizer"})}))
            return "accepted"

        agent = self.builder(self.spec, self.client(), {TERMINAL_TOOL: submit_rule})
        await agent.run(json.dumps(lesson_view(cluster, features, vocab), sort_keys=True))
        if declined and not accepted:
            raise NoSkeletonFits(f"{cluster.id}: agent declined; lesson stays prose (R5/R6)")
        if not accepted:
            raise NoRuleSubmitted(cluster.id, errors)
        return accepted[0]


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())[:400]
    return str(exc)[:400]


__all__ = ["MAX_INVALID_SUBMISSIONS", "REQUIRED_TEMPLATES", "SKELETONS", "SPEC_PATH", "LessonSynthesizer",
           "NoRuleSubmitted", "NoSkeletonFits", "SubmitRuleArgs", "SynthesizerError", "lesson_view", "load_spec",
           "tool_vocabulary"]
