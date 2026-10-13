"""Real :class:`~ci_lab.campaign.deps.CampaignDeps` for the ``copilot`` and ``offline`` profiles.

========================  ===========================================================
field                     wiring
========================  ===========================================================
domain                    :class:`ci_lab.domain.order_support.OrderSupportDomain` behind
                          :class:`HarnessDomain` (slots are repo checkouts; ``evaluate`` gets
                          ``<slot>/<harness root>`` and results carry the git harness tree);
                          every scored case is an AGL rollout in :func:`campaign_journal`
                          (``run_root/agl``, mirrored to agl-server iff ``CI_LAB_AGL_URL``)
schedule / select         :mod:`ci_lab.campaign.rrsi_wiring` — RRSI Alg. 1 ``plan_round`` /
                          Alg. 2 ``select`` (cost rule, novelty, paired bootstrap CI)
calibrate_delta           ``rrsi_wiring.calibrate_delta`` — A/A bootstrap ``aa_delta``
confirm_test              ``rrsi_wiring.confirm_test`` — paired bootstrap + safety NI
build_envelope            ``rrsi_wiring.build_envelope`` — ``oes.build``, schema-validated
make_agent / critique     :class:`MetaAgents` — ``ci_lab.meta.run`` analyst/proposer/critic
                          over ``client_factory`` (``providers.factory.make_chat_client``)
                          from the round's pinned meta harness snapshot
provision_slot ...        :class:`GitOps` — ``ci_lab.gitops.slots.SlotPool`` per campaign;
                          ``resolve_incumbent`` = ``(HEAD commit, harness tree)``
ledger / outbox           :class:`~ci_lab.campaign.local.FileLedger` / ``FileOutbox``
publisher                 :class:`ci_lab.publish.github.GitHubPublisher`
strategy_kwargs           ``domain`` + ``client_factory`` (gepa/skillopt/guard take what they accept)
lane_voters               :mod:`.lane_wiring` — System-1 judge at ``$CI_S1_LLAMA_URL`` (else none)
adversary_complete        :mod:`.lane_wiring` — ``adversary`` spec model via ``client_factory``
                          (``copilot`` only)
========================  ===========================================================

``offline`` is network-free: every OpenAI-compatible endpoint in the environment and every s1
judge backend URL (``$CI_S1_LLAMA_URL``, ``$CI_S1_SYSTEMONE_URL``; see
:mod:`ci_lab.providers.offline`) must be loopback (llama-server / AGL proxy on this host),
publishing is always a dry run, and building the deps opens no connection (chat clients are
created per agent run).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ci_lab.campaign import lane_wiring, records, rrsi_wiring
from ci_lab.campaign.deps import CampaignDeps
from ci_lab.campaign.local import FileLedger, FileOutbox
from ci_lab.contracts import (
    ArmContext,
    CriticVerdict,
    Domain,
    Edit,
    EvalResult,
    FailureRecord,
    Profile,
)
from ci_lab.domain.layout import harness_root
from ci_lab.providers.offline import (  # noqa: F401 -- NetworkPolicyError is re-exported (campaign CLI)
    NetworkPolicyError,
    check_loopback,
    check_offline_endpoints,
)

if TYPE_CHECKING:
    from ci_lab.harness_tree import HarnessSnapshot
    from ci_lab.tools.critic_checks import LeakCorpus

log = logging.getLogger(__name__)

META_DIR = "meta"  # meta-agent run dir inside an arm/round dir (its proposal.json is not the arm's)
ANALYST_DIR = "analyst"
META_HARNESS_DIR = "meta-harness"  # per-round copy of the incumbent meta harness tree
META_HARNESS_RECORD = "meta-harness.json"  # its HarnessSnapshot (root + digest)
AGL_DIR = "agl"  # campaign rollout journal dir under run_root
AGL_URL_ENV = "CI_LAB_AGL_URL"
_EID_RE = re.compile(r"^(?P<cid>.+)-(?:r\d{2}|cal|confirm)$")  # round / calibration / confirm experiment ids

ClientFactory = Callable[..., Any]


class HarnessDomain:
    """The campaign's view of a :class:`~ci_lab.contracts.Domain` over repo-root slot worktrees.

    Slots (``GitOps.provision_slot``) and GEPA scratch copies are full repository checkouts,
    but ``Domain.evaluate`` takes the harness directory. ``evaluate(worktree, ...)`` runs the
    inner domain on ``worktree/<harness>`` and, for a clean checkout rooted at ``worktree``,
    labels the result with the git harness tree (``tree_of``, i.e. ``GitOps.harness_tree``) so
    ``EvalResult.harness_tree`` agrees with ``ArmResult.harness_tree``. Everything else is
    delegated unchanged.
    """

    def __init__(self, inner: Domain, harness: str, tree_of: Callable[[Path], str | None] | None = None) -> None:
        self.inner = inner
        self.harness = harness
        self.tree_of = tree_of
        self.name = inner.name
        self.surface_globs = inner.surface_globs
        self.frozen_globs = inner.frozen_globs
        self.component_globs = inner.component_globs
        self._inner_trees: dict[str, str] = {}  # git tree -> the inner domain's own tree label

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def harness_path(self, worktree: Path) -> Path:
        path = Path(worktree) / self.harness
        if not path.is_dir():
            raise FileNotFoundError(f"{worktree}: no harness directory {self.harness!r}")
        return path

    def splits(self) -> Mapping[str, Any]:
        return self.inner.splits()

    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult:
        worktree = Path(harness_dir)
        tree = self.tree_of(worktree) if self.tree_of is not None else None
        result = await self.inner.evaluate(self.harness_path(worktree), split, k, experiment_id=experiment_id,
                                           variant=variant)
        if tree and tree != result.harness_tree:
            self._inner_trees[tree] = result.harness_tree
            result = dataclasses.replace(result, harness_tree=tree)
        return result

    def failures(self, result: EvalResult) -> list[FailureRecord]:
        inner_tree = self._inner_trees.get(result.harness_tree)
        if inner_tree is not None:
            result = dataclasses.replace(result, harness_tree=inner_tree)
        return self.inner.failures(result)


class GitOps:
    """M6 gitops for the campaign: one :class:`~ci_lab.gitops.slots.SlotPool` per campaign id
    (arm slots are leased per ``eid/arm`` and stay checked out until the process exits, so the
    arm branch is intact for publishing)."""

    def __init__(self, repo_root: Path, harness: str, *, wt_root: Path | None = None,
                 incumbent_ref: str = "HEAD") -> None:
        from ci_lab.gitops import git

        self.repo = git.toplevel(repo_root)
        self.harness = harness
        self.wt_root = wt_root
        self.incumbent_ref = incumbent_ref
        self._pools: dict[str, Any] = {}

    def _pool(self, cid: str) -> Any:
        from ci_lab.gitops.slots import SlotPool

        if cid not in self._pools:
            self._pools[cid] = SlotPool(self.repo, cid, root=self.wt_root)
        return self._pools[cid]

    def provision_slot(self, eid: str, arm: str, base_commit: str) -> Path:
        from ci_lab.contracts import arm_branch

        cid = _EID_RE.match(eid)
        if cid is None:
            raise ValueError(f"not a campaign experiment id: {eid!r}")
        return self._pool(cid["cid"]).acquire(base_commit, arm_branch(eid, arm)).path

    def head_commit(self, worktree: Path) -> str:
        from ci_lab.gitops import git

        sha = git.rev_parse(worktree, "HEAD")
        if sha is None:
            raise RuntimeError(f"{worktree}: HEAD does not resolve")
        return sha

    def tree_of(self, repo: Path, commit: str) -> str:
        from ci_lab.gitops import git

        tree = git.tree_hash(repo, commit, self.harness)
        if tree is None:
            raise RuntimeError(f"{commit[:12]} has no harness tree {self.harness!r}")
        return tree

    def harness_tree(self, worktree: Path) -> str:
        return self.tree_of(worktree, self.head_commit(worktree))

    def clean_harness_tree(self, worktree: Path) -> str | None:
        """:meth:`harness_tree` when ``worktree`` is the root of a checkout whose harness has no
        uncommitted changes (so the tree names exactly what is on disk), else ``None``."""
        from ci_lab.gitops import git

        wt = Path(worktree)
        try:
            if not os.path.samefile(git.toplevel(wt), wt):
                return None
            if git.out(wt, "status", "--porcelain", "--untracked-files=all", "--", self.harness):
                return None
            return self.harness_tree(wt)
        except (git.GitError, OSError, RuntimeError):
            return None

    def resolve_incumbent(self) -> tuple[str, str]:
        from ci_lab.gitops import git

        commit = git.rev_parse(self.repo, self.incumbent_ref)
        if commit is None:
            raise RuntimeError(f"incumbent ref {self.incumbent_ref!r} does not resolve in {self.repo}")
        return commit, self.tree_of(self.repo, commit)


def _latest_rejection(arm_run: Any) -> list[str] | None:
    """Reasons of the most recent failed critique of ``arm_run`` (``None`` before any)."""
    attempt, last = 1, None
    while (verdict := arm_run.verdict(attempt)) is not None:
        last, attempt = verdict, attempt + 1
    return list(last.reasons) if last is not None and not last.passed else None


class _ProposerAgent:
    """``make_agent("proposer", ArmRun)``: :func:`ci_lab.meta.run.run_proposer` in the arm
    slot, then the arm's ``proposal.json`` in the shape the workflow gates on."""

    name = "Proposer"

    def __init__(self, agents: MetaAgents, arm_run: Any) -> None:
        self.agents, self.arm_run = agents, arm_run

    async def run(self, messages: Any = None, **_: Any) -> str:
        from ci_lab.meta.run import run_proposer

        rejected = _latest_rejection(self.arm_run)
        ctx = self.agents.arm_context(self.arm_run, rejected or ())
        snap = self.agents.incumbent(self.arm_run.round.dir)
        result = await run_proposer(ctx, self.agents.client(ctx.profile, "proposer", snap.root),
                                    surface=self.agents.surface, builder=self.agents.builder, reuse=rejected is None,
                                    harness_dir=snap.root)
        write_proposal(self.arm_run, result.edits)
        return result.submission.summary


class _AnalystAgent:
    """``make_agent("analyst", RoundContext)``: :func:`ci_lab.meta.run.run_analyst` over the
    round's typed failures, copied to the round's ``analysis.json``."""

    name = "Analyst"

    def __init__(self, agents: MetaAgents, round_ctx: Any) -> None:
        self.agents, self.round_ctx = agents, round_ctx

    async def run(self, messages: Any = None, **_: Any) -> str:
        from ci_lab.meta.run import run_analyst, write_brief, write_failures

        rctx = self.round_ctx
        run_dir = rctx.dir / ANALYST_DIR
        brief = {k: v for k, v in rctx.brief().items() if k not in ("failures", "analysis")}
        write_brief(run_dir, brief)
        write_failures(run_dir, rctx.brief()["failures"])
        snap = self.agents.incumbent(rctx.dir)
        submission = await run_analyst(run_dir, self.agents.client(rctx.env.profile, "analyst", snap.root),
                                       builder=self.agents.builder, harness_dir=snap.root)
        records.write_json(rctx.analysis_path, submission.model_dump(mode="json"))
        return submission.summary


def write_proposal(arm_run: Any, edits: list[Edit]) -> None:
    records.write_json(arm_run.proposal_path, {"strategy": arm_run.strategy, "edits": [
        {"component": e.component, "hypothesis": e.hypothesis, "files": list(e.files), "commit": e.commit}
        for e in edits]})


class MetaAgents:
    """``make_agent(role, ctx)`` and ``critique(arm_run, attempt)`` over ``ci_lab.meta``.

    ``client_factory(profile=, model=, purpose=)`` builds each agent's chat client from the
    meta spec's model alias; the spec builder validates against the manifest's
    ``allowed_models``.

    The evolvable specs come from the incumbent meta harness tree (``harness_dir``; ``None`` =
    the repo-root ``harness/``), pinned per round: the first agent of a round copies the tree
    to ``<round dir>/meta-harness`` and records its :class:`~ci_lab.harness_tree.HarnessSnapshot`
    in ``meta-harness.json``; the analyst, every proposal/repair and the critic of that round
    (and a resumed round) load from that copy, which must still match the recorded digest."""

    def __init__(self, domain: Domain, client_factory: ClientFactory, *, builder: Any = None,
                 leak_corpus: Any = None, harness_dir: Path | None = None) -> None:
        from ci_lab.meta.run import ArmSurface

        self.surface = ArmSurface.from_domain(domain)
        self.client_factory = client_factory
        self.builder = builder
        self.leak_corpus = leak_corpus
        self.harness_dir = harness_dir

    def client(self, profile: Profile, key: str, harness_dir: Path | None = None) -> Any:
        from ci_lab.meta.spec_loader import load_spec

        return self.client_factory(profile=profile, model=load_spec(key, harness_dir=harness_dir).model, purpose=key)

    def incumbent(self, round_dir: Path) -> HarnessSnapshot:
        """The round's pinned meta harness snapshot (recorded on first use, verified after)."""
        import shutil

        from ci_lab.harness_tree import HarnessSnapshot, materialize, repo_harness_dir

        record = Path(round_dir) / META_HARNESS_RECORD
        if (data := records.read_json(record)) is not None:
            return HarnessSnapshot.from_dict(data).verify()
        dest = Path(round_dir) / META_HARNESS_DIR
        shutil.rmtree(dest, ignore_errors=True)  # an unrecorded copy from an interrupted round start
        snap = materialize(self.harness_dir if self.harness_dir is not None else repo_harness_dir(), dest)
        records.write_json(record, snap.as_dict())
        return snap

    def arm_context(self, arm_run: Any, feedback: Any = ()) -> ArmContext:
        return dataclasses.replace(arm_run.strategy_context(list(feedback)), run_dir=arm_run.dir / META_DIR)

    def __call__(self, role: str, ctx: Any) -> Any:
        if role == "proposer":
            return _ProposerAgent(self, ctx)
        if role == "analyst":
            return _AnalystAgent(self, ctx)
        raise ValueError(f"no meta agent for role {role!r}")

    async def critique(self, arm_run: Any, attempt: int) -> CriticVerdict:
        from ci_lab.meta.run import run_critic

        ctx = self.arm_context(arm_run)
        snap = self.incumbent(arm_run.round.dir)
        return await run_critic(ctx, self.client(ctx.profile, "critic", snap.root), surface=self.surface,
                                builder=self.builder, leak_corpus=self.leak_corpus, repairs=int(attempt) - 1,
                                harness_dir=snap.root)


def default_client_factory(profile: Profile) -> ClientFactory:
    from ci_lab.providers.factory import make_chat_client

    def factory(*, profile: Profile = profile, model: str, purpose: str, **kw: Any) -> Any:
        if Profile(profile) is Profile.OFFLINE:
            check_offline_endpoints()
            check_loopback("base_url", kw.get("base_url"))
        return make_chat_client(profile=profile, model=model, purpose=purpose, **kw)  # type: ignore[arg-type]

    return factory


def campaign_journal(run_root: Path, *, offline: bool = False, environ: Mapping[str, str] | None = None) -> Any:
    """AGL rollout journal for campaign evaluations: a :class:`FileRolloutJournal` under
    ``run_root/agl``, mirrored to ``agl-server`` when ``CI_LAB_AGL_URL`` is set (bearer key
    ``CI_LAB_AGL_KEY``; the URL must be loopback for ``offline``)."""
    from ci_lab.agl.client import AglClient
    from ci_lab.agl.journal import FileRolloutJournal
    from ci_lab.agl.mirror import MirroringJournal
    from ci_lab.agl.server import KEY_ENV

    env = os.environ if environ is None else environ
    journal = FileRolloutJournal(Path(run_root) / AGL_DIR)
    url = env.get(AGL_URL_ENV)
    if not url:
        return journal
    if offline:
        check_loopback(f"${AGL_URL_ENV}", url)
    return MirroringJournal(journal, AglClient(url, env.get(KEY_ENV) or None))


FROZEN_TEST_SETS = "evals/assert/*/test_set.jsonl"


def campaign_leak_corpus(domain: Any, repo_root: Path) -> LeakCorpus | None:
    """The critic's leak corpus for a campaign: ``domain.leak_corpus()`` when the domain has
    one (``OrderSupportDomain``: frozen ASSERT test sets plus order identifiers), else the
    frozen ``evals/assert/*/test_set.jsonl`` files under ``repo_root``. ``None`` only when
    neither exists; the campaign then logs that its leak screen is off."""
    build = getattr(domain, "leak_corpus", None)
    if callable(build):
        return build()
    from ci_lab.tools.critic_checks import load_test_set_corpus

    paths = sorted(Path(repo_root).glob(FROZEN_TEST_SETS))
    if not paths:
        log.warning("no frozen test sets match %s under %s: the critic's leak screen is off",
                    FROZEN_TEST_SETS, repo_root)
        return None
    return load_test_set_corpus(paths)


def wired_deps(profile: Profile | str, *, run_root: Path, ledger_dir: Path | None = None,
               repo: str = "example/harness", dry_run_publish: bool = False, repo_root: Path | None = None,
               domain: Domain | None = None, client_factory: ClientFactory | None = None,
               wt_root: Path | None = None, incumbent_ref: str = "HEAD", leak_corpus: LeakCorpus | None = None,
               meta_harness_dir: Path | None = None, domain_name: str = "order_support",
               **overrides: Any) -> CampaignDeps:
    """:class:`CampaignDeps` for ``copilot``/``offline`` (``fake`` lives in :mod:`.fakes`).

    The meta agents load their evolvable specs from ``meta_harness_dir`` (default: the
    repo-root ``harness/``), snapshotted per round (:meth:`MetaAgents.incumbent`).

    The critic screens every candidate against ``leak_corpus`` (default:
    :func:`campaign_leak_corpus`, i.e. the frozen ASSERT test sets)."""
    from ci_lab.publish.github import GitHubPublisher

    profile = Profile(profile)
    if profile is Profile.FAKE:
        raise ValueError("use ci_lab.campaign.fakes.fake_deps for the fake profile")
    offline = profile is Profile.OFFLINE
    if offline:
        check_offline_endpoints()
    run_root = Path(run_root)
    if domain is None:
        from ci_lab.domain import get_domain
        from ci_lab.domain.order_support import REPO_ROOT

        repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        if domain_name == "harness":
            domain = get_domain(domain_name, repo_root=repo_root, work_dir=run_root / "domain",
                                profile=profile.value)
        else:
            domain = get_domain(domain_name, repo_root=repo_root, work_dir=run_root / "domain",
                                journal=campaign_journal(run_root, offline=offline))
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    gitops = GitOps(repo_root, harness_root(domain), wt_root=wt_root, incumbent_ref=incumbent_ref)
    if not isinstance(domain, HarnessDomain):
        domain = HarnessDomain(domain, gitops.harness, gitops.clean_harness_tree)
    client_factory = client_factory or default_client_factory(profile)
    agents = MetaAgents(domain, client_factory, harness_dir=meta_harness_dir,
                        leak_corpus=leak_corpus if leak_corpus is not None else campaign_leak_corpus(domain, gitops.repo))
    outbox = FileOutbox(run_root / "outbox.jsonl")
    publisher = GitHubPublisher(repo, outbox=outbox, dry_run=dry_run_publish or offline, git_cwd=gitops.repo,
                                journal=run_root / "publish-calls.jsonl")
    kwargs: dict[str, Any] = {
        "domain": domain, "make_agent": agents, "critique": agents.critique,
        "provision_slot": gitops.provision_slot, "head_commit": gitops.head_commit,
        "harness_tree": gitops.harness_tree, "resolve_incumbent": gitops.resolve_incumbent,
        "ledger": FileLedger(Path(ledger_dir) if ledger_dir is not None else gitops.repo / "experiments"),
        "publisher": publisher, "outbox": outbox,
        "strategy_kwargs": {"domain": domain, "client_factory": client_factory},
        "schedule": rrsi_wiring.schedule, "select": rrsi_wiring.select,
        "calibrate_delta": rrsi_wiring.calibrate_delta, "confirm_test": rrsi_wiring.confirm_test,
        "build_envelope": rrsi_wiring.build_envelope, "lane_voters": lane_wiring.lane_voters(),
        "adversary_complete": lane_wiring.adversary_complete(profile, client_factory)}
    if profile is Profile.COPILOT:
        from ci_lab.campaign.preflight import make_preflight

        inner = getattr(domain, "inner", domain)
        kwargs["preflight"] = make_preflight(
            profile, harness_dir=gitops.repo / harness_root(domain), evals_dir=getattr(inner, "evals_dir", None),
            tester_model=getattr(getattr(inner, "runner", None), "tester_model", None),
            meta_harness_dir=meta_harness_dir)
    kwargs.update(overrides)
    return CampaignDeps(**kwargs)
