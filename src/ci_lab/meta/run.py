"""Run the meta agents (design §5): build from the declarative spec through an injected
builder, run with a literal instruction, read the JSON written by the terminal ``submit_*``
tool. Dynamic inputs reach the agents only through their tools (bound to the run dir /
arm worktree here). A run that ends without a valid submission raises
:class:`MetaAgentError`.

All runners are idempotent under at-least-once execution (C5): with ``reuse=True`` an
existing valid submission is returned without calling the model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_COMPONENT,
    ATTR_EXPERIMENT,
    ATTR_PHASE,
    ATTR_PROFILE,
    ATTR_PURPOSE,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_STEP,
    TEXT_COMPONENTS,
    ArmContext,
    ChatClientFactory,
    CriticVerdict,
    Domain,
    Edit,
    FailureRecord,
)
from ci_lab.meta.spec_loader import (
    AgentBuilder,
    MetaAgentSpec,
    TerminalSubmitMiddleware,
    default_builder,
    load_spec,
)
from ci_lab.tools.arm_fs import DEFAULT_MAX_BYTES, make_arm_fs
from ci_lab.tools.briefs import make_brief_tools
from ci_lab.tools.commit import edits_since, git, make_commit_tool
from ci_lab.tools.critic_checks import (
    CriticConfig,
    LeakCorpus,
    SpecValidator,
    collect_diff,
    diff_text,
    run_checks,
)
from ci_lab.tools.paths import matches_any
from ci_lab.tools.submit import (
    SUBMISSION_FILES,
    AnalysisSubmission,
    ProposalSubmission,
    ReflectionSubmission,
    VerdictSubmission,
    make_submit_tools,
    read_submission,
    write_json_atomic,
)

__all__ = [
    "INSTRUCTION",
    "ArmSurface",
    "MetaAgentError",
    "ProposalResult",
    "ProposerStrategy",
    "default_arm_brief",
    "run_analyst",
    "run_critic",
    "run_meta_agent",
    "run_proposer",
    "run_reflector",
    "surface_text",
    "write_brief",
    "write_failures",
]

INSTRUCTION = "Read your brief with tools; finish by calling {tool}."
NUDGE = ("You have not called {tool} yet, so your work is not recorded. Finish now by calling {tool} "
         "(fix any ERROR it reported).")


class MetaAgentError(RuntimeError):
    """A meta agent finished without a valid terminal submission (or produced no edits)."""


# ---------------------------------------------------------------- run-dir helpers


def write_brief(run_dir: Path | str, brief: str | Mapping[str, Any]) -> Path:
    """Write ``brief.md`` (text) or ``brief.json`` (mapping) into the run dir."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(brief, str):
        path = root / "brief.md"
        path.write_text(brief, encoding="utf-8")
        (root / "brief.json").unlink(missing_ok=True)
    else:
        path = root / "brief.json"
        write_json_atomic(path, dict(brief))
        (root / "brief.md").unlink(missing_ok=True)
    return path


def write_failures(run_dir: Path | str, failures: Sequence[FailureRecord]) -> Path:
    path = Path(run_dir) / "failures.json"
    write_json_atomic(path, [asdict(f) if is_dataclass(f) else dict(f) for f in failures])  # type: ignore[arg-type]
    return path


def _read_valid(run_dir: Path, tool: str) -> BaseModel | None:
    try:
        return read_submission(run_dir, tool)
    except (ValidationError, ValueError, OSError):
        return None


# ---------------------------------------------------------------- generic runner


async def run_meta_agent(key: str, run_dir: Path | str, client: Any, bindings: Mapping[str, Callable[..., Any]], *,
                         builder: AgentBuilder | None = None, spec: MetaAgentSpec | None = None,
                         reuse: bool = True) -> BaseModel:
    """Build agent ``key`` with ``bindings`` (+ its terminal submit tool), run it, return the submission."""
    spec = spec or load_spec(key)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    terminal = spec.terminal_tool
    if reuse and (previous := _read_valid(root, terminal)) is not None:
        return previous
    (root / SUBMISSION_FILES[terminal][0]).unlink(missing_ok=True)

    all_bindings = {**bindings, terminal: make_submit_tools(root)[terminal]}
    missing = [t for t in spec.tools if t not in all_bindings]
    if missing:
        raise MetaAgentError(f"{spec.key}: no binding for tool(s) {', '.join(missing)}")

    def should_continue(**_: Any) -> bool:
        return _read_valid(root, terminal) is None

    def next_message(**_: Any) -> str:
        return NUDGE.format(tool=terminal)

    agent = (builder or default_builder())(
        spec, client=client, bindings={t: all_bindings[t] for t in spec.tools},
        middleware=[TerminalSubmitMiddleware(terminal)], loop_should_continue=should_continue,
        loop_next_message=next_message)
    session = agent.create_session() if hasattr(agent, "create_session") else None
    await agent.run(INSTRUCTION.format(tool=terminal), session=session)
    result = _read_valid(root, terminal)
    if result is None:
        raise MetaAgentError(f"{spec.key} finished without a valid {terminal} call")
    return result


async def run_analyst(run_dir: Path | str, client: Any, *, builder: AgentBuilder | None = None,
                      reuse: bool = True) -> AnalysisSubmission:
    """Analyst over ``failures.json`` (+ brief/history) in ``run_dir`` -> ``analysis.json``."""
    spec = load_spec("analyst")
    tools = make_brief_tools(run_dir, allowed=spec.documents or None)
    with obs.span(SPAN_STEP, {ATTR_PHASE: "analyze", ATTR_PURPOSE: "analyst"}):
        return await run_meta_agent("analyst", run_dir, client, tools, builder=builder, spec=spec,
                                    reuse=reuse)  # type: ignore[return-value]


async def run_reflector(run_dir: Path | str, client: Any, *, builder: AgentBuilder | None = None,
                        reuse: bool = True) -> ReflectionSubmission:
    """Reflector over ``results.json`` (+ brief/analysis/history) -> ``reflection.json``."""
    spec = load_spec("reflector")
    tools = make_brief_tools(run_dir, allowed=spec.documents or None)
    with obs.span(SPAN_STEP, {ATTR_PHASE: "reflect", ATTR_PURPOSE: "reflector"}):
        return await run_meta_agent("reflector", run_dir, client, tools, builder=builder, spec=spec,
                                    reuse=reuse)  # type: ignore[return-value]


# ---------------------------------------------------------------- arms


@dataclass(frozen=True)
class ArmSurface:
    """Domain-level edit scope shared by every arm (from :class:`~ci_lab.contracts.Domain`)."""

    surface_globs: Sequence[str]
    component_globs: Mapping[str, Sequence[str]]
    frozen_globs: Sequence[str] = ()
    max_file_bytes: int = DEFAULT_MAX_BYTES

    @classmethod
    def from_domain(cls, domain: Domain, *, max_file_bytes: int = DEFAULT_MAX_BYTES) -> ArmSurface:
        return cls(tuple(domain.surface_globs), {k: tuple(v) for k, v in domain.component_globs.items()},
                   tuple(domain.frozen_globs), max_file_bytes)

    def components(self, ctx: ArmContext) -> tuple[str, ...]:
        focus = tuple(ctx.directive.component_focus) or tuple(c for c in TEXT_COMPONENTS if c in self.component_globs)
        unknown = [c for c in focus if c not in self.component_globs]
        if unknown:
            raise ValueError(f"unknown component(s) in directive: {', '.join(unknown)}")
        return focus

    def writable_globs(self, components: Sequence[str]) -> tuple[str, ...]:
        return tuple(g for c in components for g in self.component_globs[c])


def default_arm_brief(ctx: ArmContext, components: Sequence[str]) -> str:
    d = ctx.directive
    lines = [
        f"# Arm {d.arm} of experiment {ctx.experiment_id}",
        "",
        f"- Components you may edit: {', '.join(components)}",
        f"- Edit budget: at most {d.edit_budget} commit_edit call(s)",
        f"- Mode: {'explore (try a lever not tried before)' if d.explore else 'exploit (refine what works)'}",
        f"- Failure records: {len(ctx.failures)} (read_brief name 'failures' if present, and 'analysis')",
    ]
    if ctx.budget_tokens:
        lines.append(f"- Token budget: {ctx.budget_tokens}")
    return "\n".join(lines) + "\n"


def _arm_attrs(ctx: ArmContext, phase: str, components: Sequence[str]) -> dict[str, Any]:
    return {ATTR_PHASE: phase, ATTR_EXPERIMENT: ctx.experiment_id, ATTR_VARIANT: ctx.directive.arm,
            ATTR_STRATEGY: ctx.directive.strategy, ATTR_COMPONENT: ",".join(components),
            ATTR_PROFILE: getattr(ctx.profile, "value", ctx.profile)}


@dataclass
class ProposalResult:
    submission: ProposalSubmission
    edits: list[Edit] = field(default_factory=list)


async def run_proposer(ctx: ArmContext, client: Any, *, surface: ArmSurface, builder: AgentBuilder | None = None,
                       reuse: bool = True) -> ProposalResult:
    """Proposer for one arm: edits the directive's components in ``ctx.worktree`` via ``commit_edit``.

    Writes a default ``brief.md`` / ``failures.json`` into ``ctx.run_dir`` when absent. The
    returned edits are read back from commit trailers on ``base_commit..HEAD``.
    """
    components = surface.components(ctx)
    with obs.span(SPAN_STEP, {**_arm_attrs(ctx, "propose", components), ATTR_PURPOSE: "proposer"}):
        run_dir = Path(ctx.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not any((run_dir / n).is_file() for n in ("brief.md", "brief.json")):
            write_brief(run_dir, default_arm_brief(ctx, components))
        if ctx.failures and not (run_dir / "failures.json").is_file():
            write_failures(run_dir, ctx.failures)
        spec = load_spec("proposer")
        tools: dict[str, Callable[..., Any]] = {**make_brief_tools(run_dir, allowed=spec.documents or None)}
        tools.update(make_arm_fs(ctx.worktree, surface.surface_globs, surface.frozen_globs, surface.max_file_bytes,
                                 writable_globs=surface.writable_globs(components)))
        tools["commit_edit"] = make_commit_tool(
            ctx.worktree, ctx.directive.edit_budget, component_globs=surface.component_globs,
            experiment_id=ctx.experiment_id, variant=ctx.directive.arm, surface_globs=surface.surface_globs,
            frozen_globs=surface.frozen_globs, allowed_components=components)
        submission = await run_meta_agent("proposer", run_dir, client, tools, builder=builder, spec=spec,
                                          reuse=reuse)
        edits = edits_since(ctx.worktree, ctx.base_commit)
        if not edits:
            raise MetaAgentError("proposer submitted without committing any edit")
        return ProposalResult(submission, edits)  # type: ignore[arg-type]


class ProposerStrategy:
    """``contracts.ArmStrategy`` "agent": the meta-agent proposer (for M10's AgentStrategy).

    Pass a ready ``client`` or a :class:`~ci_lab.contracts.ChatClientFactory`; the factory is
    called with the arm's profile, the proposer spec's model alias and purpose "proposer".
    """

    name = "agent"

    def __init__(self, surface: ArmSurface, *, client: Any = None, client_factory: ChatClientFactory | None = None,
                 builder: AgentBuilder | None = None, model: str | None = None, reuse: bool = True) -> None:
        if (client is None) == (client_factory is None):
            raise ValueError("pass exactly one of client= or client_factory=")
        self.surface = surface
        self.client = client
        self.client_factory = client_factory
        self.builder = builder
        self.model = model
        self.reuse = reuse

    def client_for(self, ctx: ArmContext) -> Any:
        if self.client is not None:
            return self.client
        model = self.model or load_spec("proposer").model
        return self.client_factory(profile=ctx.profile, model=model, purpose="proposer")  # type: ignore[misc]

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        result = await run_proposer(ctx, self.client_for(ctx), surface=self.surface, builder=self.builder,
                                    reuse=self.reuse)
        return result.edits


def surface_text(worktree: Path | str, rev: str, surface_globs: Sequence[str],
                 frozen_globs: Sequence[str] = ()) -> str:
    """Concatenated text of the surface at ``rev`` (pre-existing material is not a leak)."""
    wt = Path(worktree)
    names = [n for n in git(wt, "ls-tree", "-r", "-z", "--name-only", rev).split("\x00")
             if n and matches_any(n, surface_globs) and not matches_any(n, frozen_globs)]
    parts = []
    for n in names:
        try:
            parts.append(git(wt, "show", f"{rev}:{n}"))
        except RuntimeError:
            continue
    return "\n".join(parts)


def _critique(run_dir: Path, *, passed: bool, reasons: Sequence[str], source: str, base: str, head: str) -> None:
    write_json_atomic(run_dir / "critique.json", {"passed": passed, "reasons": list(reasons), "source": source,
                                                  "base": base, "head": head})


async def run_critic(ctx: ArmContext, client: Any, *, surface: ArmSurface, builder: AgentBuilder | None = None,
                     leak_corpus: LeakCorpus | None = None, spec_validator: SpecValidator | None = None,
                     config: CriticConfig | None = None, repairs: int = 0, reuse: bool = True) -> CriticVerdict:
    """Deterministic checks first (no model call when they fail), then the LLM critic.

    Writes ``critique.json`` (read by the proposer on repair) and ``diff.patch`` (read by the
    critic). The arm passes only on an explicit ``accept``.
    """
    components = surface.components(ctx)
    run_dir = Path(ctx.run_dir)
    with obs.span(SPAN_STEP, {**_arm_attrs(ctx, "critique", components), ATTR_PURPOSE: "critic"}) as span:
        run_dir.mkdir(parents=True, exist_ok=True)
        diff = collect_diff(ctx.worktree, ctx.base_commit)
        cfg = config or CriticConfig(
            surface_globs=surface.surface_globs, component_globs=surface.component_globs,
            frozen_globs=surface.frozen_globs, leak_corpus=leak_corpus, spec_validator=spec_validator,
            baseline_text=surface_text(ctx.worktree, diff.base, surface.surface_globs, surface.frozen_globs))
        reasons = run_checks(diff, cfg)
        reasons += [f"component: commit {c.sha[:12]} is tagged {c.component!r}; this arm may edit "
                    f"{', '.join(components)}" for c in diff.commits if c.component and c.component not in components]
        if len(diff.commits) > ctx.directive.edit_budget:
            reasons.append(f"size: {len(diff.commits)} commits exceed the edit budget {ctx.directive.edit_budget}")
        if reasons:
            _critique(run_dir, passed=False, reasons=reasons, source="checks", base=diff.base, head=diff.head)
            (run_dir / "verdict.json").unlink(missing_ok=True)
            span.set_attribute("ci.critic.passed", False)
            return CriticVerdict(passed=False, reasons=reasons, repairs=repairs)

        prior = _read_json(run_dir / "critique.json")
        same_diff = (prior.get("source") == "critic" and prior.get("head") == diff.head
                     and prior.get("base") == diff.base)
        (run_dir / "diff.patch").write_text(diff_text(ctx.worktree, diff.base, diff.head), encoding="utf-8")
        spec = load_spec("critic")
        tools: dict[str, Callable[..., Any]] = {**make_brief_tools(run_dir, allowed=spec.documents or None)}
        fs = make_arm_fs(ctx.worktree, surface.surface_globs, surface.frozen_globs, surface.max_file_bytes,
                         writable_globs=())
        tools.update(list_files=fs["list_files"], read_file=fs["read_file"])
        verdict: VerdictSubmission = await run_meta_agent(  # type: ignore[assignment]
            "critic", run_dir, client, tools, builder=builder, spec=spec, reuse=reuse and same_diff)
        passed = verdict.verdict == "accept"
        _critique(run_dir, passed=passed, reasons=list(verdict.reasons), source="critic", base=diff.base,
                  head=diff.head)
        span.set_attribute("ci.critic.passed", passed)
        return CriticVerdict(passed=passed, reasons=[] if passed else list(verdict.reasons), repairs=repairs)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
