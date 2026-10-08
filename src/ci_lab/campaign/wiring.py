"""Real :class:`~ci_lab.campaign.deps.CampaignDeps` for the ``copilot`` and ``offline`` profiles.

========================  ===========================================================
field                     wiring
========================  ===========================================================
domain                    :class:`ci_lab.domain.order_support.OrderSupportDomain`
make_agent / critique     :class:`MetaAgents` — ``ci_lab.meta.run`` analyst/proposer/critic
                          over ``client_factory`` (``providers.factory.make_chat_client``)
provision_slot ...        :class:`GitOps` — ``ci_lab.gitops.slots.SlotPool`` per campaign;
                          ``resolve_incumbent`` = ``(HEAD commit, harness tree)``
ledger / outbox           :class:`~ci_lab.campaign.local.FileLedger` / ``FileOutbox``
publisher                 :class:`ci_lab.publish.github.GitHubPublisher`
strategy_kwargs           ``domain`` + ``client_factory`` (gepa/skillopt/guard take what they accept)
========================  ===========================================================

``offline`` is network-free: every OpenAI-compatible endpoint in the environment must be
loopback (llama-server / AGL proxy on this host), publishing is always a dry run, and
building the deps opens no connection (chat clients are created per agent run).
"""
from __future__ import annotations

import dataclasses
import ipaddress
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ci_lab.campaign import records
from ci_lab.campaign.deps import CampaignDeps
from ci_lab.campaign.local import FileLedger, FileOutbox
from ci_lab.contracts import ArmContext, CriticVerdict, Domain, Edit, Profile

META_DIR = "meta"  # meta-agent run dir inside an arm/round dir (its proposal.json is not the arm's)
ANALYST_DIR = "analyst"
OFFLINE_ENDPOINT_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AGL_OPENAI_BASE_URL")
_GLOB_CHARS = re.compile(r"[*?\[]")
_EID_RE = re.compile(r"^(?P<cid>.+)-(?:r\d{2}|cal|confirm)$")  # round / calibration / confirm experiment ids

ClientFactory = Callable[..., Any]


class NetworkPolicyError(ValueError):
    """An ``offline`` endpoint is not on this host."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_offline_endpoints(env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`NetworkPolicyError` unless every configured endpoint is loopback."""
    env = os.environ if env is None else env
    for name in OFFLINE_ENDPOINT_ENVS:
        check_loopback(f"${name}", env.get(name))


def check_loopback(name: str, url: str | None) -> None:
    url = (url or "").strip()
    if not url:
        return
    host = urlsplit(url).hostname or ""
    if not _is_loopback(host):
        raise NetworkPolicyError(f"offline profile is network-free: {name} must point at a loopback host, "
                                 f"not {host or url!r}")


def harness_root(domain: Domain) -> str:
    """Directory prefix shared by the domain's surface globs (the harness tree, design §3)."""
    roots = set()
    for glob in domain.surface_globs:
        parts = []
        for part in glob.split("/"):
            if _GLOB_CHARS.search(part):
                break
            parts.append(part)
        roots.add("/".join(parts))
    if len(roots) != 1:
        raise ValueError(f"surface globs {list(domain.surface_globs)} do not share one harness root")
    return roots.pop()


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
        result = await run_proposer(ctx, self.agents.client(ctx.profile, "proposer"), surface=self.agents.surface,
                                    builder=self.agents.builder, reuse=rejected is None)
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
        submission = await run_analyst(run_dir, self.agents.client(rctx.env.profile, "analyst"),
                                       builder=self.agents.builder)
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
    ``allowed_models``."""

    def __init__(self, domain: Domain, client_factory: ClientFactory, *, builder: Any = None,
                 leak_corpus: Any = None) -> None:
        from ci_lab.meta.run import ArmSurface

        self.surface = ArmSurface.from_domain(domain)
        self.client_factory = client_factory
        self.builder = builder
        self.leak_corpus = leak_corpus

    def client(self, profile: Profile, key: str) -> Any:
        from ci_lab.meta.spec_loader import load_spec

        return self.client_factory(profile=profile, model=load_spec(key).model, purpose=key)

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
        return await run_critic(ctx, self.client(ctx.profile, "critic"), surface=self.surface, builder=self.builder,
                                leak_corpus=self.leak_corpus, repairs=int(attempt) - 1)


def default_client_factory(profile: Profile) -> ClientFactory:
    from ci_lab.providers.factory import make_chat_client

    def factory(*, profile: Profile = profile, model: str, purpose: str, **kw: Any) -> Any:
        if Profile(profile) is Profile.OFFLINE:
            check_offline_endpoints()
            check_loopback("base_url", kw.get("base_url"))
        return make_chat_client(profile=profile, model=model, purpose=purpose, **kw)  # type: ignore[arg-type]

    return factory


def wired_deps(profile: Profile | str, *, run_root: Path, ledger_dir: Path | None = None,
               repo: str = "example/harness", dry_run_publish: bool = False, repo_root: Path | None = None,
               domain: Domain | None = None, client_factory: ClientFactory | None = None,
               wt_root: Path | None = None, incumbent_ref: str = "HEAD", **overrides: Any) -> CampaignDeps:
    """:class:`CampaignDeps` for ``copilot``/``offline`` (``fake`` lives in :mod:`.fakes`)."""
    from ci_lab.publish.github import GitHubPublisher

    profile = Profile(profile)
    if profile is Profile.FAKE:
        raise ValueError("use ci_lab.campaign.fakes.fake_deps for the fake profile")
    offline = profile is Profile.OFFLINE
    if offline:
        check_offline_endpoints()
    run_root = Path(run_root)
    if domain is None:
        from ci_lab.domain.order_support import REPO_ROOT, OrderSupportDomain

        repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        domain = OrderSupportDomain(repo_root=repo_root, work_dir=run_root / "domain")
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    gitops = GitOps(repo_root, harness_root(domain), wt_root=wt_root, incumbent_ref=incumbent_ref)
    client_factory = client_factory or default_client_factory(profile)
    agents = MetaAgents(domain, client_factory)
    outbox = FileOutbox(run_root / "outbox.jsonl")
    publisher = GitHubPublisher(repo, outbox=outbox, dry_run=dry_run_publish or offline, git_cwd=gitops.repo,
                                journal=run_root / "publish-calls.jsonl")
    kwargs: dict[str, Any] = {
        "domain": domain, "make_agent": agents, "critique": agents.critique,
        "provision_slot": gitops.provision_slot, "head_commit": gitops.head_commit,
        "harness_tree": gitops.harness_tree, "resolve_incumbent": gitops.resolve_incumbent,
        "ledger": FileLedger(Path(ledger_dir) if ledger_dir is not None else gitops.repo / "experiments"),
        "publisher": publisher, "outbox": outbox,
        "strategy_kwargs": {"domain": domain, "client_factory": client_factory}}
    kwargs.update(overrides)
    return CampaignDeps(**kwargs)
