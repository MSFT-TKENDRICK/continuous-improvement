"""``guard`` arm strategy (design §11, §13.3 step 6; B2, B3, B4, N4).

``propose(ctx)`` turns lesson candidates into guard rule files in the arm worktree:

1. read ``<run_dir>/lessons/candidates.jsonl`` (nearest ancestor of ``ctx.run_dir``) — typed
   :class:`~ci_lab.lessons_arm.features.Candidate` lines only;
2. drop clusters that are injection-suspect, rejected/backlog, untrusted without a human label,
   already encoded, or touched by another arm (sibling ``proposal.json`` edits mapped via
   ``lessons/registry.yaml``, plus an exclusive per-lesson claim file — N4);
3. synthesize ≤ ``edit_budget`` rules: deterministic templates first, the
   :class:`~ci_lab.lessons_arm.agent.LessonSynthesizer` only for leftovers;
4. write ``<harness root>/guards/<lesson_id>.yaml`` — the guards dir the domain's agent loads
   (:func:`ci_lab.domain.layout.guards_rel`) and the only path this arm may write (never
   ``BUNDLE.lock`` or extractor files; enforced by a resolved-path check and a post-commit diff);
5. validate by loading the **full** bundle (``ci_lab.rules.load_bundle``), run the replay
   rejection filter (M16), and commit one git commit per :class:`~ci_lab.contracts.Edit`.

Rules are written ``mode: shadow`` by default (closed-loop scoring forces enforcement through
``CI_GUARDS=enforce``; see :mod:`ci_lab.lessons_arm.paired`).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ci_lab import obs
from ci_lab.contracts import (
    ATTR_EXPERIMENT,
    ATTR_PROFILE,
    ATTR_STRATEGY,
    ATTR_VARIANT,
    SPAN_OPTIMIZER,
    ArmContext,
    ChatClientFactory,
    Edit,
)
from ci_lab.rulespec import GUARD_BUNDLE_LOCK, GUARDS_DIR, LESSON_REGISTRY, LessonCluster, RuleSpec

from .bundle import BundleError, dump_rule_file, load_rules, rule_files, subset_bundle
from .features import Candidate, LessonFeatures, is_injection, read_candidates
from .seams import ReplayFn, default_replay, lessons_touching, load_registry
from .synth import SynthesisError, lesson_id_for, synthesize

STRATEGY = "guard"
COMMIT_TRAILER = "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
FALLBACK_IDENTITY = ("ci-lab arm", "ci-lab-arm@localhost")
CANDIDATES = Path("lessons") / "candidates.jsonl"
CLAIMS_DIR = "lesson_claims"
GUARD_COMPONENT = "guard"  # contracts.COMPONENTS (v2.4 integration)
_RUNG_ORDER = {"R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5}

Committer = Callable[[Path, Sequence[str], str], str]
"""``(worktree, files, message) -> commit sha`` (same shape as ``ci_lab.strategies``)."""


class GuardArmError(RuntimeError):
    pass


class GuardPathViolation(GuardArmError):
    """The arm tried to write outside ``<guards dir>/<lesson>.yaml``."""


class EditBudgetExceeded(GuardArmError, ValueError):
    pass


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=worktree, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def git_commit(worktree: Path, files: Sequence[str], message: str) -> str:
    """Stage exactly ``files`` and commit them; returns the new HEAD sha."""
    ident: list[str] = []
    try:
        _git(worktree, "config", "user.email")
    except subprocess.CalledProcessError:
        ident = ["-c", f"user.name={FALLBACK_IDENTITY[0]}", "-c", f"user.email={FALLBACK_IDENTITY[1]}"]
    _git(worktree, "add", "--", *files)
    _git(worktree, *ident, "commit", "-q", "-m", message, "--", *files)
    return _git(worktree, "rev-parse", "HEAD")


def guard_path(worktree: Path, lesson_id: str, guards_dir: str = GUARDS_DIR) -> tuple[Path, str]:
    """``(absolute, repo-relative posix)`` path for a lesson's rule file; raises
    :class:`GuardPathViolation` for anything but a plain ``<guards_dir>/<id>.yaml`` (``guards_dir``
    is the domain's ``<harness root>/guards``, :func:`ci_lab.domain.layout.guards_rel`)."""
    root = (Path(worktree) / guards_dir)
    rel = f"{guards_dir}/{lesson_id}.yaml"
    if (not lesson_id or "/" in lesson_id or "\\" in lesson_id or lesson_id.startswith(".")
            or "extractor" in lesson_id.casefold() or rel == f"{guards_dir}/{Path(GUARD_BUNDLE_LOCK).name}"):
        raise GuardPathViolation(f"refusing to write {rel!r}")
    cur = Path(worktree)
    for part in Path(guards_dir).parts:
        cur = cur / part
        if cur.is_symlink():
            raise GuardPathViolation(f"{cur} is a symlink")
    target = root / f"{lesson_id}.yaml"
    if target.is_symlink():
        raise GuardPathViolation(f"{target} is a symlink")
    if target.resolve().parent != root.resolve() or target.suffix != ".yaml":
        raise GuardPathViolation(f"refusing to write {target}")
    return target, rel


def find_candidates(run_dir: Path, levels: int = 3) -> Path | None:
    d = Path(run_dir)
    for _ in range(levels + 1):
        if (d / CANDIDATES).is_file():
            return d / CANDIDATES
        d = d.parent
    return None


@dataclass
class Proposal:
    lesson_id: str
    cluster: LessonCluster
    rule: RuleSpec
    source: str  # template | synthesizer


@dataclass
class GuardReport:
    candidates: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    edits: list[dict[str, Any]] = field(default_factory=list)
    replay: str = "unavailable"
    engine: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"candidates": self.candidates, "skipped": self.skipped, "rejected": self.rejected,
                "edits": self.edits, "replay": self.replay, "engine": self.engine}


class GuardStrategy:
    """:class:`~ci_lab.contracts.ArmStrategy` named ``guard``."""

    name = STRATEGY

    def __init__(self, *, synthesizer: Any = None, client_factory: ChatClientFactory | None = None,
                 committer: Committer | None = None, replay: ReplayFn | None = None,
                 trajectories: Any = None, dataset_texts: Sequence[str] = (),
                 candidates_path: Path | None = None, registry_path: Path | None = None,
                 extractor_paths: Sequence[Path] = (), vocabulary: Iterable[str] = (),
                 write_mode: Literal["shadow", "enforce"] = "shadow", require_replay: bool = False,
                 templates: Mapping[str, Any] | None = None, domain: Any = None,
                 guards_dir: str | None = None) -> None:
        from ci_lab.domain.layout import guard_extractors, guards_rel

        self.synthesizer = synthesizer
        self.client_factory = client_factory
        self.committer = committer or git_commit
        self.guards_rel = guards_dir or guards_rel(domain)
        extractor_paths = list(extractor_paths) or guard_extractors(domain)
        self.replay = replay if replay is not None else default_replay(trajectories, dataset_texts=dataset_texts,
                                                                        extractors=list(extractor_paths))
        self.candidates_path = candidates_path
        self.registry_path = registry_path
        self.extractor_paths = list(extractor_paths)
        self.vocabulary = list(vocabulary)
        self.write_mode = write_mode
        self.require_replay = require_replay
        self.templates = templates

    # ------------------------------------------------------------ selection

    def _registry(self, ctx: ArmContext) -> list[Any]:
        for p in (self.registry_path, Path(ctx.worktree) / LESSON_REGISTRY, Path(ctx.run_dir) / LESSON_REGISTRY):
            if p is not None and Path(p).is_file():
                return load_registry(Path(p))
        return []

    def _other_arms(self, ctx: ArmContext, registry: Sequence[Any]) -> set[str]:
        """Lessons touched by sibling arms' recorded proposals (N4)."""
        touched: set[str] = set()
        own = Path(ctx.run_dir).resolve()
        for prop in sorted(Path(ctx.run_dir).parent.glob("*/proposal.json")):
            if prop.parent.resolve() == own:
                continue
            try:
                data = json.loads(prop.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            files = [f for e in data.get("edits") or [] for f in e.get("files") or []]
            touched |= lessons_touching(files, registry, guards_dir=self.guards_rel)
        return touched

    def _claim(self, ctx: ArmContext, lesson_id: str) -> bool:
        """Exclusive per-round claim so two guard arms never encode the same lesson."""
        d = Path(ctx.run_dir).parent / CLAIMS_DIR
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{lesson_id}.json"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("arm") == ctx.directive.arm
            except (OSError, ValueError):
                return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"arm": ctx.directive.arm, "experiment_id": ctx.experiment_id}, fh)
        return True

    def _eligible(self, ctx: ArmContext, cands: Sequence[Candidate], report: GuardReport
                  ) -> list[tuple[str, LessonCluster, LessonFeatures | None]]:
        registry = self._registry(ctx)
        foreign = self._other_arms(ctx, registry)
        encoded = {e.lesson_id for e in registry if e.status in ("shadow", "enforced") and e.rule_ids}
        own_changed = self._changed_since_base(ctx)
        out: list[tuple[str, LessonCluster, LessonFeatures | None]] = []
        for c in cands:
            cl, feats = c.cluster, c.resolved()
            lid = lesson_id_for(cl)
            reason = None
            if is_injection(cl) or (feats is not None and feats.injection_suspect):
                reason = "injection_suspect"
            elif cl.status in ("rejected", "backlog"):
                reason = f"status_{cl.status}"
            elif cl.route not in ("R1", "R2", "R3"):
                reason = f"route_{cl.route}"
            elif ((feats is not None and not feats.trusted) or c.trusted is False) and not cl.human_confirmed:
                reason = "untrusted_unlabeled"
            elif lid in foreign:
                reason = "touched_by_other_arm"
            elif lid in encoded:
                reason = "already_encoded"
            elif (Path(ctx.worktree) / self.guards_rel / f"{lid}.yaml").exists() and \
                    f"{self.guards_rel}/{lid}.yaml" not in own_changed:
                reason = "guard_file_exists"
            if reason:
                report.skipped[lid] = reason
            else:
                out.append((lid, cl, feats))
        out.sort(key=lambda t: (_RUNG_ORDER.get(t[1].route, 9), t[2] is None, -len(t[1].families),
                                -len(t[1].members), t[0]))
        return out

    def _changed_since_base(self, ctx: ArmContext) -> set[str]:
        try:
            out = _git(Path(ctx.worktree), "diff", "--name-only", f"{ctx.base_commit}..HEAD", "--",
                       self.guards_rel)
        except (subprocess.CalledProcessError, OSError):
            return set()
        return {line.strip() for line in out.splitlines() if line.strip()}

    # ------------------------------------------------------------ synthesis

    def _llm(self, ctx: ArmContext) -> Any:
        if self.synthesizer is None and self.client_factory is not None:
            from .agent import LessonSynthesizer

            self.synthesizer = LessonSynthesizer(client_factory=self.client_factory, profile=ctx.profile)
        return self.synthesizer

    async def _synthesize(self, ctx: ArmContext, lid: str, cl: LessonCluster, feats: LessonFeatures | None,
                          report: GuardReport) -> Proposal | None:
        if feats is not None:
            try:
                return Proposal(lid, cl, synthesize(cl, feats), "template")
            except SynthesisError as exc:
                if "B3" in str(exc):
                    report.skipped[lid] = "untrusted_or_injection"
                    return None
        llm = self._llm(ctx)
        if llm is None:
            report.skipped[lid] = "leftover_no_synthesizer"
            return None
        try:
            rule = await llm.synthesize(cl, feats, vocabulary=self.vocabulary)
        except Exception as exc:  # noqa: BLE001 - typed agent errors are reported, never fatal
            report.skipped[lid] = f"synthesizer:{type(exc).__name__}"
            return None
        return Proposal(lid, cl, rule, "synthesizer")

    # ------------------------------------------------------------ write / validate / commit

    def _bundle_paths(self, worktree: Path) -> tuple[list[Path], list[Path]]:
        guards = Path(worktree) / self.guards_rel
        extractors = self.extractor_paths or sorted(guards.glob("*extractor*.yaml"))
        return rule_files(guards), list(extractors)

    def _apply(self, ctx: ArmContext, prop: Proposal, report: GuardReport) -> Edit | None:
        wt = Path(ctx.worktree)
        target, rel = guard_path(wt, prop.lesson_id, self.guards_rel)
        rule = prop.rule
        if self.write_mode == "enforce" and prop.source == "template":
            rule = rule.model_copy(update={"mode": "enforce"})
        prev = target.read_text(encoding="utf-8") if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dump_rule_file([rule]), encoding="utf-8", newline="\n")

        def undo(why: str, reasons: list[str]) -> None:
            if prev is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(prev, encoding="utf-8", newline="\n")
            report.rejected[prop.lesson_id] = [why, *reasons]

        rules_p, extr_p = self._bundle_paths(wt)
        try:
            loaded = load_rules(rules_p, extr_p, templates=self.templates)
        except BundleError as exc:
            undo("bundle", exc.errors)
            return None
        report.engine = loaded.engine
        replay_note = "unavailable"
        if self.replay is not None:
            work = Path(ctx.run_dir) / "replay" / prop.lesson_id
            work.mkdir(parents=True, exist_ok=True)
            ok, reasons = self.replay(subset_bundle(loaded, [rule], templates=self.templates), prop.cluster, work)
            if not ok:
                undo("replay", reasons)
                return None
            replay_note = "pass_to_closed_loop"
        elif self.require_replay:
            undo("replay", ["replay validator unavailable (HOOK M16)"])
            return None
        report.replay = replay_note if report.replay != "pass_to_closed_loop" else report.replay

        hypothesis = (f"guard: encode lesson {prop.lesson_id} as {rule.rung} rule {rule.id} "
                      f"(template {rule.template}, {prop.source}, mode {rule.mode}); replay {replay_note}")
        status = _git(wt, "status", "--porcelain", "--", rel)
        if not status:  # identical to HEAD: re-proposal on repair returns the existing commit
            sha = _git(wt, "log", "-1", "--format=%H", "--", rel)
        else:
            subject = f"guard: {rule.id} ({rule.rung}, {rule.mode}) for lesson {prop.lesson_id}"
            sha = self.committer(wt, [rel], f"{subject}\n\n{hypothesis}\n\n{COMMIT_TRAILER}")
            changed = _git(wt, "diff", "--name-only", f"{sha}~1", sha).splitlines()
            if changed != [rel]:
                raise GuardPathViolation(f"guard commit {sha[:12]} touched {changed}; only {rel} is allowed")
            if COMMIT_TRAILER not in _git(wt, "log", "-1", "--format=%B", sha):
                raise GuardArmError(f"guard commit {sha[:12]} lacks the required trailer")
        edit = Edit(component=GUARD_COMPONENT, hypothesis=hypothesis, files=(rel,), commit=sha)
        report.edits.append({"lesson_id": prop.lesson_id, "rule_id": rule.id, "source": prop.source,
                             "file": rel, "commit": sha, "digest": loaded.digest})
        return edit

    # ------------------------------------------------------------ ArmStrategy

    async def propose(self, ctx: ArmContext) -> list[Edit]:
        attrs = {ATTR_STRATEGY: self.name, ATTR_EXPERIMENT: ctx.experiment_id, ATTR_VARIANT: ctx.directive.arm,
                 ATTR_PROFILE: getattr(ctx.profile, "value", str(ctx.profile)),
                 "ci.edit_budget": ctx.directive.edit_budget}
        report = GuardReport()
        with obs.span(SPAN_OPTIMIZER, attrs):
            edits: list[Edit] = []
            if ctx.directive.edit_budget >= 1:
                path = self.candidates_path or find_candidates(Path(ctx.run_dir))
                cands = read_candidates(path) if path else []
                report.candidates = len(cands)
                for lid, cl, feats in self._eligible(ctx, cands, report):
                    if len(edits) >= ctx.directive.edit_budget:
                        break
                    if not self._claim(ctx, lid):
                        report.skipped[lid] = "claimed_by_other_arm"
                        continue
                    prop = await self._synthesize(ctx, lid, cl, feats, report)
                    if prop is not None and (edit := self._apply(ctx, prop, report)) is not None:
                        edits.append(edit)
            if len(edits) > ctx.directive.edit_budget:
                raise EditBudgetExceeded(f"{len(edits)} edits > edit_budget {ctx.directive.edit_budget}")
            obs.annotate({"ci.edits": len(edits), "ci.guard.skipped": len(report.skipped)})
        write_report(ctx, report)
        return edits


def write_report(ctx: ArmContext, report: GuardReport) -> Path:
    """``<run_dir>/optimizer/<arm>-guard.json`` (same location as ``ci_lab.strategies``)."""
    d = Path(ctx.run_dir) / "optimizer"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ctx.directive.arm}-{STRATEGY}.json"
    p.write_text(json.dumps({"experiment_id": ctx.experiment_id, "arm": ctx.directive.arm, "strategy": STRATEGY,
                             **report.as_dict()}, indent=2, sort_keys=True), encoding="utf-8")
    return p


def register() -> bool:
    """Register ``guard`` with M10's ``ci_lab.strategies`` registry.

    ``ci_lab.strategies.get_strategy("guard")`` calls this lazily (``EXTERNAL["guard"]``).
    """
    try:
        from ci_lab.strategies import (
            register_strategy,  # type: ignore[import-not-found]
        )
    except ImportError:
        return False
    register_strategy(STRATEGY, GuardStrategy)
    return True
