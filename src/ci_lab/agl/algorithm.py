"""LLM-only structural harness optimization over Agent Lightning rollouts."""

from __future__ import annotations

import difflib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ci_lab import obs
from ci_lab.agl.credit import Credit, RolloutDigest, assign_credit, journal_credits
from ci_lab.agl.journal import RolloutRecord
from ci_lab.agl.metrics import rollout_metrics
from ci_lab.contracts import (
    COMPONENT_OWNERS,
    COMPONENTS,
    ArmContext,
    Edit,
    Profile,
    strategy_may_edit,
)
from ci_lab.harness_tree import HarnessTree
from ci_lab.strategies.base import (
    Committer,
    check_edit_budget,
    git_commit,
    optimizer_commit_message,
    optimizer_span,
    write_report,
)
from ci_lab.tools.paths import contained, matches_any, normalize_rel

MAX_FILE_CHARS = 20_000
MAX_CHANGED_LINES = 80
MAX_PROMPT_CHARS = 60_000
_COST_REASONS = frozenset({"cost.calls", "cost.tokens", "cost.wall"})


class ComponentOwnershipError(ValueError):
    """An AGL arm was directed at a component owned by another optimizer."""


class StructuralProposalError(ValueError):
    """The optimizer response was malformed, unsafe, or invalidated the harness tree."""


@dataclass(frozen=True)
class StructuralProposal:
    component: str
    path: str
    operation: str
    content: str | None
    hypothesis: str


def _latest_score(events: Sequence[Mapping[str, Any]]) -> tuple[float | None, Mapping[str, Any]]:
    for event in reversed(events):
        if event.get("event_type") not in ("ci.score", "reward"):
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        value = data.get("value")
        if value is None:
            return None, data
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value), data
    return None, {}


def digest_rollout(record: RolloutRecord) -> RolloutDigest | None:
    """Convert one materialized journal rollout to a bounded payload-free digest."""
    attempt = record.latest_attempt
    events = record.events_for(attempt_id=attempt) if attempt is not None else list(record.events)
    score, score_data = _latest_score(events)
    case = record.key.case_id if record.key is not None else str(record.input.get("case_id") or "")
    suite = str(score_data.get("suite") or record.input.get("suite") or "unknown")
    if not case:
        return None
    rules: list[str] = [str(r) for r in score_data.get("rule_ids") or () if r]
    for violation in score_data.get("violations") or ():
        if isinstance(violation, Mapping) and violation.get("rule_id"):
            rules.append(str(violation["rule_id"]))
    touches: dict[str, int] = {}
    initial = record.input.get("component_touches")
    if isinstance(initial, Mapping):
        for component, count in initial.items():
            if component in COMPONENTS and isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                touches[str(component)] = count
    for event in events:
        if event.get("event_type") not in ("ci.edit", "ci.touch"):
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        component = data.get("component")
        if component in COMPONENTS:
            count = data.get("count", 1)
            count = count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 1
            touches[str(component)] = touches.get(str(component), 0) + count
    return RolloutDigest(case, suite, score, tuple(rules), touches, rollout_metrics(events))


def _source_records(source: Any) -> list[RolloutRecord]:
    if source is None:
        return []
    if callable(getattr(source, "iter_rollouts", None)):
        return [r for r in source.iter_rollouts() if isinstance(r, RolloutRecord)]
    inner = getattr(source, "journal", None)
    if inner is not None and inner is not source:
        return _source_records(inner)
    rows = getattr(source, "rollouts", None)
    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes, Mapping)):
        return [r for r in rows if isinstance(r, RolloutRecord)]
    raise TypeError("AGL rollout source needs iter_rollouts() or RolloutRecord rows")


def _domain_journal(domain: Any) -> Any:
    current = domain
    for _ in range(3):
        if current is None:
            return None
        if getattr(current, "journal", None) is not None:
            return current.journal
        current = getattr(current, "inner", None)
    return None


def _schema(component: str, paths: Sequence[str]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": ["component", "path", "operation", "content", "hypothesis"],
            "properties": {
                "component": {"type": "string", "const": component},
                "path": {"type": "string", "enum": list(paths)},
                "operation": {"type": "string", "enum": ["replace", "delete"]},
                "content": {"type": ["string", "null"], "maxLength": MAX_FILE_CHARS},
                "hypothesis": {"type": "string", "minLength": 1, "maxLength": 300},
            }}


def _parse_proposal(text: str, component: str, paths: Sequence[str]) -> StructuralProposal:
    data = json.loads(text)
    required = {"component", "path", "operation", "content", "hypothesis"}
    if not isinstance(data, dict) or set(data) != required:
        raise StructuralProposalError("proposal has unknown or missing fields")
    if data["component"] != component or data["path"] not in paths:
        raise StructuralProposalError("proposal escaped the selected component")
    if data["operation"] not in ("replace", "delete") or not isinstance(data["hypothesis"], str) \
            or not 0 < len(data["hypothesis"]) <= 300:
        raise StructuralProposalError("bad proposal operation or hypothesis")
    content = data["content"]
    if data["operation"] == "replace" and (not isinstance(content, str) or not content or len(content) > MAX_FILE_CHARS):
        raise StructuralProposalError("replace needs bounded non-empty content")
    if data["operation"] == "delete" and content not in (None, ""):
        raise StructuralProposalError("delete content must be null")
    return StructuralProposal(component, str(data["path"]), str(data["operation"]), content, data["hypothesis"])


def _changed_lines(before: str, after: str) -> int:
    return sum(line[0] in "+-" for line in difflib.ndiff(before.splitlines(), after.splitlines()))


class LlmResourceAlgorithm:
    """AGL arm strategy: assign structural credit, then commit one contained small edit."""

    name = "agl"

    def __init__(self, *, journal: Any = None, store: Any = None, client: Any = None,
                 client_factory: Any = None, domain: Any = None, harness_dir: str | Path | None = None,
                 committer: Committer | None = None) -> None:
        if client is None and client_factory is None:
            raise TypeError("LlmResourceAlgorithm needs a client or ChatClientFactory")
        self.source = journal if journal is not None else store
        self.client = client
        self.client_factory = client_factory
        self.domain = domain
        self.harness_dir = str(harness_dir).replace("\\", "/").strip("/") if harness_dir is not None else None
        self.committer: Committer = committer or git_commit

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        from ci_lab.optim.lm import resolve_model

        return self.client_factory(profile=Profile.COPILOT,
                                   model=resolve_model(Profile.COPILOT, "optimizer"),
                                   purpose="optimizer")

    def _harness_rel(self) -> str:
        if self.harness_dir:
            return normalize_rel(self.harness_dir)
        if self.domain is not None:
            from ci_lab.domain.layout import harness_root

            return normalize_rel(harness_root(self.domain))
        return "harness"

    def _focus(self, ctx: ArmContext) -> tuple[str, ...]:
        focus = tuple(ctx.directive.component_focus)
        if not focus:
            return tuple(c for c in COMPONENTS if strategy_may_edit(self.name, c))
        for component in focus:
            if component not in COMPONENTS:
                raise ComponentOwnershipError(f"unknown harness component {component!r}")
            if not strategy_may_edit(self.name, component):
                owner = COMPONENT_OWNERS.get(component, "another strategy")
                raise ComponentOwnershipError(f"component {component!r} is owned by {owner}, not agl")
        return focus

    def _tree(self, ctx: ArmContext) -> tuple[str, HarnessTree]:
        rel = self._harness_rel()
        root = contained(Path(ctx.worktree), rel)
        tree = HarnessTree(root)
        if errors := tree.validate():
            raise StructuralProposalError("candidate harness is invalid before AGL: " + "; ".join(errors[:3]))
        return rel, tree

    def _files(self, rel: str, tree: HarnessTree, component: str) -> list[str]:
        globs = tree.component_globs().get(component, ())
        return [f"{rel}/{path}" for path in tree.files()
                if path != "harness.yaml" and matches_any(f"{rel}/{path}", globs)]

    def _digests(self, ctx: ArmContext) -> tuple[list[RolloutDigest], list[RolloutRecord]]:
        source = self.source if self.source is not None else _domain_journal(self.domain)
        records = _source_records(source) if source is not None else []
        cases = {failure.case_id for failure in ctx.failures}
        if cases:
            records = [r for r in records if r.key is not None and r.key.case_id in cases]
        digests = [d for record in records if (d := digest_rollout(record)) is not None]
        if not digests:
            digests = [RolloutDigest(f.case_id, f.suite, None, tuple(f.rule_ids), {}, {})
                       for f in ctx.failures]
        return digests, records

    def _journal(self, records: Sequence[RolloutRecord], credits: Sequence[Credit]) -> None:
        source = self.source if self.source is not None else _domain_journal(self.domain)
        journal = getattr(source, "journal", source)
        if journal is None or not callable(getattr(journal, "event", None)):
            return
        for record in records:
            if record.key is None:
                continue
            relevant = [c for c in credits if not c.evidence_ids
                        or any(e == record.key.case_id or e.startswith(record.key.case_id + ":")
                               for e in c.evidence_ids)]
            journal_credits(journal, record.key, relevant or credits)

    async def _proposal(self, client: Any, component: str, files: Sequence[str],
                        tree: HarnessTree, credits: Sequence[Credit]) -> StructuralProposal:
        snapshots: dict[str, str] = {}
        prefix = self._harness_rel() + "/"
        for path in files:
            text = tree.path(path.removeprefix(prefix)).read_text(encoding="utf-8")
            snapshots[path] = text[:MAX_FILE_CHARS]
        cost = any(c.component == component and c.reason_code in _COST_REASONS for c in credits)
        payload = {"component": component, "credits": [
            {"weight": c.weight, "reason_code": c.reason_code, "evidence_ids": list(c.evidence_ids)}
            for c in credits if c.component == component], "files": snapshots}
        instruction = ("Propose exactly one small structural harness edit. "
                       + ("Prefer deletion or tightening that reduces calls, steps, tools, or agents. " if cost else "")
                       + "Do not edit prompts, skills, guards, source, evals, governance, or the manifest. "
                       "Return only strict JSON.\n")
        prompt = instruction + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if len(prompt) > MAX_PROMPT_CHARS:
            raise StructuralProposalError("selected component snapshot is too large")
        schema = _schema(component, files)
        response = await client.get_response(
            prompt, options={"response_format": {"type": "json_schema",
                                                 "json_schema": {"name": "agl_edit", "strict": True,
                                                                 "schema": schema}}})
        return _parse_proposal(str(getattr(response, "text", response)), component, files)

    def _apply(self, ctx: ArmContext, rel: str, tree: HarnessTree,
               proposal: StructuralProposal) -> Edit:
        path = contained(Path(ctx.worktree), proposal.path)
        original = path.read_bytes()
        before = original.decode("utf-8")
        after = "" if proposal.operation == "delete" else str(proposal.content)
        if _changed_lines(before, after) > MAX_CHANGED_LINES:
            raise StructuralProposalError(f"edit exceeds {MAX_CHANGED_LINES} changed lines")
        try:
            if proposal.operation == "delete":
                path.unlink()
            else:
                newline = "\r\n" if b"\r\n" in original else "\n"
                normalized = after.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
                path.write_bytes(normalized.encode("utf-8"))
            if errors := tree.validate():
                raise StructuralProposalError("proposal invalidates harness: " + "; ".join(errors[:3]))
        except BaseException:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            raise
        message = optimizer_commit_message(
            ctx.directive.arm,
            self.name,
            proposal.component,
            proposal.hypothesis,
        )
        commit = self.committer(Path(ctx.worktree), [proposal.path], message)
        return Edit(proposal.component, proposal.hypothesis, (proposal.path,), commit)

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        with optimizer_span(self.name, ctx):
            if ctx.directive.edit_budget < 1:
                return []
            focus = self._focus(ctx)
            rel, tree = self._tree(ctx)
            available = {component: self._files(rel, tree, component) for component in focus}
            editable = tuple(component for component, files in available.items() if files)
            if not editable:
                raise StructuralProposalError("no files in the frozen manifest globs for AGL focus")
            digests, records = self._digests(ctx)
            client = self._client()
            credits = await assign_credit(digests, client=client, components=editable)
            self._journal(records, credits)
            component = next((c.component for c in credits if c.component in editable), editable[0])
            proposal = await self._proposal(client, component, available[component], tree, credits)
            edit = self._apply(ctx, rel, tree, proposal)
            obs.annotate({"ci.edits": 1, "rrsi.component": component})
            write_report(ctx, self.name, {
                "cost": {}, "component": component,
                "credits": [{"component": c.component, "weight": c.weight, "reason_code": c.reason_code,
                             "evidence_ids": list(c.evidence_ids)} for c in credits],
                "edits": [{"component": edit.component, "files": list(edit.files), "commit": edit.commit}],
                "acceptance": "diagnostic-only",
            })
            return check_edit_budget([edit], ctx)
