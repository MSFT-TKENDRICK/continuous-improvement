"""Layer 21b: real ``copilot``/``offline`` CampaignDeps (``ci_lab.campaign.wiring``), offline."""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ci_lab.campaign import records
from ci_lab.campaign.driver import Campaign
from ci_lab.campaign.wiring import (
    GitOps,
    HarnessDomain,
    MetaAgents,
    NetworkPolicyError,
    check_offline_endpoints,
    harness_root,
    wired_deps,
)
from ci_lab.cli import main
from ci_lab.contracts import EvalResult, EvaluatorPin, FailureRecord, Profile, TaskScore
from ci_lab.testing import Call, FakeChatClient

CID = "wire-camp"
PROMPT = "harness/prompts/system.md"
BASE_PROMPT = "You are a helpful order support agent.\n"
NEW_PROMPT = "You are a helpful order support agent.\nAlways verify identity first.\n"
FILES = {PROMPT: BASE_PROMPT, "harness/skills/refunds/SKILL.md": "# Refunds\nCheck the order first.\n",
         "README.md": "repo\n"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def make_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    git(root, "init", "-q")
    for key, value in (("user.name", "test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false"),
                       ("core.autocrlf", "false")):
        git(root, "config", key, value)
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root.resolve()


class GitDomain:
    """Scores 0.9 per case when the prompt says to verify identity, else 0.1.

    Like ``OrderSupportDomain``, ``evaluate`` takes the harness directory (not the repo-root slot)
    and labels results with its own content hash; ``HarnessDomain`` maps both."""

    name = "git-stub"
    surface_globs = ("harness/**",)
    frozen_globs = ("**/*.py",)
    component_globs: ClassVar[dict[str, tuple[str, ...]]] = {"prompt": ("harness/prompts/*.md",),
                                                             "skill": ("harness/skills/**",)}

    def __init__(self) -> None:
        self.harness_dirs: list[Path] = []

    def splits(self):
        return {"evolve": ("c1", "c2", "c3", "c4"), "heldout": ("h1", "h2")}

    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult:
        from ci_lab.domain.order_support import tree_hash

        self.harness_dirs.append(Path(harness_dir))
        text = (Path(harness_dir) / "prompts" / "system.md").read_text(encoding="utf-8")
        score = 0.9 if "verify identity" in text else 0.1
        return EvalResult(tree_hash(Path(harness_dir)), split,  # type: ignore[arg-type]
                          EvaluatorPin("stub-evaluator", "stub-judge", "fake"),
                          [TaskScore(c, t, "stub", score, tokens_in=5) for c in self.splits()[split]
                           for t in range(k)])

    def failures(self, result: EvalResult) -> list[FailureRecord]:
        assert result.harness_tree.startswith("sha256:"), "failures() must see the domain's own tree label"
        return [FailureRecord(s.case_id, s.suite, "low_score", (), {"score": s.score or 0.0})
                for s in result.scores if (s.score or 0.0) < 0.5]


class ScriptedClients:
    """``client_factory(profile=, model=, purpose=)`` with one fresh scripted client per run."""

    def __init__(self) -> None:
        self.calls: list[tuple[Profile, str, str]] = []

    def __call__(self, *, profile: Profile, model: str, purpose: str, **_: Any) -> FakeChatClient:
        self.calls.append((Profile(profile), model, purpose))
        if purpose == "analyst":
            return FakeChatClient([[Call("submit_analysis", {"summary": "identity gaps",
                                                             "suggested_components": ["prompt"]})]])
        if purpose == "proposer":
            return FakeChatClient([
                [Call("write_file", {"path": PROMPT, "content": NEW_PROMPT})],
                [Call("commit_edit", {"component": "prompt", "hypothesis": "Verify identity before refunds."})],
                [Call("submit_proposal_done", {"summary": "verify identity first"})],
            ])
        if purpose == "critic":
            return FakeChatClient([[Call("submit_verdict", {"verdict": "accept"})]])
        raise AssertionError(purpose)


def one_agent_arm(round_no, hyper, history):
    return [{"arm": "v1", "component": "prompt", "strategy": "agent", "budget": 1}]


@pytest.fixture
def no_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AGL_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def no_remote_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail any connection to a non-loopback address."""
    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            attempts.append(address)
            raise OSError(f"network access in an offline test: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def test_offline_endpoints_must_be_loopback() -> None:
    check_offline_endpoints({"OPENAI_API_BASE": "http://127.0.0.1:8081/v1", "OPENAI_BASE_URL": "",
                             "AGL_OPENAI_BASE_URL": "http://localhost:9000/v1"})
    check_offline_endpoints({"OPENAI_API_BASE": "http://[::1]:8081/v1"})
    with pytest.raises(NetworkPolicyError, match="AGL_OPENAI_BASE_URL"):
        check_offline_endpoints({"AGL_OPENAI_BASE_URL": "https://api.openai.com/v1"})


def test_harness_root_from_surface_globs() -> None:
    from ci_lab.domain.order_support import HARNESS_ROOT, OrderSupportDomain

    assert harness_root(GitDomain()) == "harness"
    assert harness_root(OrderSupportDomain) == HARNESS_ROOT  # type: ignore[arg-type]


def test_gitops_slots_trees_and_incumbent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    ops = GitOps(repo, "harness", wt_root=tmp_path / "wt")
    head = git(repo, "rev-parse", "HEAD").strip()
    assert ops.resolve_incumbent() == (head, git(repo, "rev-parse", "HEAD:harness").strip())
    slot = ops.provision_slot(f"{CID}-r01", "v1", head)
    assert slot == tmp_path / "wt" / CID / "s0"
    assert git(slot, "branch", "--show-current").strip() == f"exp/{CID}-r01/v1"
    assert ops.head_commit(slot) == head and ops.harness_tree(slot) == ops.resolve_incumbent()[1]
    assert ops.provision_slot(f"{CID}-cal", "h0", head) != slot  # one leased slot per eid/arm
    with pytest.raises(ValueError, match="experiment id"):
        ops.provision_slot("bogus", "v1", head)


def test_offline_round_with_meta_agents(tmp_path: Path, no_offline_env: None, no_remote_network: list[Any]) -> None:
    repo = make_repo(tmp_path / "repo")
    clients = ScriptedClients()
    domain = GitDomain()
    deps = wired_deps("offline", run_root=tmp_path / "runs", ledger_dir=tmp_path / "experiments", repo_root=repo,
                      domain=domain, client_factory=clients, wt_root=tmp_path / "wt", schedule=one_agent_arm)
    assert isinstance(deps.make_agent, MetaAgents) and deps.publisher.dry_run
    assert isinstance(deps.domain, HarnessDomain) and deps.domain.inner is domain
    assert deps.strategy_kwargs["domain"] is deps.domain
    camp = Campaign.new(CID, "offline", {"arms": 1, "aa_repeats": 2, "max_rounds": 1}, deps=deps,
                        run_root=tmp_path / "runs")
    asyncio.run(camp.calibrate())
    out = asyncio.run(camp.run(rounds=1))
    assert out["rounds"][0]["winner"] == "v1" and out["rounds"][0]["decision"] == "ship"

    run_dir = tmp_path / "runs" / f"{CID}-r01"
    assert records.read_json(run_dir / "analysis.json")["summary"] == "identity gaps"
    proposal = records.read_json(run_dir / "v1" / "proposal.json")
    [edit] = proposal["edits"]
    assert edit["files"] == [PROMPT] and edit["component"] == "prompt"
    assert records.read_json(run_dir / "v1" / "critique_1.json")["passed"]
    branch = f"exp/{CID}-r01/v1"
    assert git(repo, "rev-parse", branch).strip() == edit["commit"]
    assert git(repo, "show", f"{branch}:{PROMPT}") == NEW_PROMPT
    assert {p for _, _, p in clients.calls} == {"analyst", "proposer", "critic"}
    assert {pr for pr, _, _ in clients.calls} == {Profile.OFFLINE}
    assert no_remote_network == []
    assert domain.harness_dirs and all(d.name == "harness" and (d / "prompts").is_dir()
                                       for d in domain.harness_dirs)
    publish = [json.loads(line) for line in (tmp_path / "runs" / "publish-calls.jsonl").read_text().splitlines()]
    assert publish  # dry-run journal only


class RepoRootDomain(GitDomain):
    """``OrderSupportDomain``-shaped layout: the harness lives at ``src/order_support/harness``."""

    surface_globs = ("src/order_support/harness/**",)
    component_globs: ClassVar[dict[str, tuple[str, ...]]] = {
        "prompt": ("src/order_support/harness/prompts/*.md",)}

    async def evaluate(self, harness_dir: Path, split: str, k: int, *, experiment_id: str,
                       variant: str) -> EvalResult:
        assert (Path(harness_dir) / "agent.yaml").is_file(), f"{harness_dir} is not the harness directory"
        return await super().evaluate(harness_dir, split, k, experiment_id=experiment_id, variant=variant)


def test_repo_root_slot_evaluates_the_harness_subdir(tmp_path: Path) -> None:
    harness = "src/order_support/harness"
    files = {f"{harness}/agent.yaml": "name: order-support\n", f"{harness}/prompts/system.md": NEW_PROMPT,
             "README.md": "repo\n"}
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for key, value in (("user.name", "test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false"),
                       ("core.autocrlf", "false")):
        git(repo, "config", key, value)
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    repo = repo.resolve()

    inner = RepoRootDomain()
    deps = wired_deps("offline", run_root=tmp_path / "runs", ledger_dir=tmp_path / "experiments", repo_root=repo,
                      domain=inner, client_factory=ScriptedClients(), wt_root=tmp_path / "wt")
    head, base_tree = deps.resolve_incumbent()
    slot = deps.provision_slot(f"{CID}-r01", "v1", head)
    assert (slot / harness / "agent.yaml").is_file() and not (slot / "agent.yaml").exists()

    result = asyncio.run(deps.domain.evaluate(slot, "evolve", 1, experiment_id=f"{CID}-r01", variant="v1"))
    assert inner.harness_dirs == [slot / harness]
    assert result.harness_tree == deps.harness_tree(slot) == base_tree  # agrees with ArmResult.harness_tree
    assert [s.score for s in result.scores] == [0.9] * 4
    assert deps.domain.failures(result) == []  # failures() sees the inner domain's own label

    (slot / harness / "prompts" / "system.md").write_text(BASE_PROMPT, encoding="utf-8")  # dirty: no git label
    dirty = asyncio.run(deps.domain.evaluate(slot, "evolve", 1, experiment_id=f"{CID}-r01", variant="v1"))
    assert dirty.harness_tree.startswith("sha256:") and len(deps.domain.failures(dirty)) == 4

    with pytest.raises(FileNotFoundError, match="no harness directory"):
        asyncio.run(deps.domain.evaluate(tmp_path / "runs", "evolve", 1, experiment_id=f"{CID}-r01", variant="v1"))


def test_order_support_domain_gets_harness_dir_env(tmp_path: Path) -> None:
    import contextlib

    from ci_lab.domain.order_support import (
        HARNESS_ROOT,
        CaseOutcome,
        OrderSupportDomain,
        TestCase,
        stable_case_id,
    )

    seen: list[tuple[Path, str | None]] = []

    async def runner(case, *, harness_dir, key, env):  # type: ignore[no-untyped-def]
        seen.append((Path(harness_dir), env.get("ORDER_SUPPORT_HARNESS_DIR")))
        return CaseOutcome(verdict={"policy_violation": False})

    row = {"type": "prompt", "behavior": "refund", "seed": {"title": "t", "description": "d"}}
    case = TestCase(stable_case_id("order_support_refund_authorization", row), "order_support_refund_authorization",
                    "refund", "prompt", row, tmp_path / "eval_config.yaml")
    inner = OrderSupportDomain(cases=[case], runner=runner, work_dir=tmp_path / "work", use_default_oracle=False,
                               scope_factory=lambda key: contextlib.nullcontext(), heldout_fraction=0.0,
                               ood_fraction=0.0)
    slot = tmp_path / "slot"
    (slot / HARNESS_ROOT).mkdir(parents=True)
    (slot / HARNESS_ROOT / "agent.yaml").write_text("name: x\n", encoding="utf-8")
    wrapped = HarnessDomain(inner, harness_root(OrderSupportDomain))  # type: ignore[arg-type]
    split = next(name for name, ids in wrapped.splits().items() if ids)
    asyncio.run(wrapped.evaluate(slot, split, 1, experiment_id=f"{CID}-r01", variant="v1"))
    assert seen and all(d == slot / HARNESS_ROOT and env == str(slot / HARNESS_ROOT) for d, env in seen)


def test_load_deps_copilot_uses_copilot_chat_clients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ci_lab.campaign.cli import load_deps
    from ci_lab.providers import factory

    made: list[dict[str, Any]] = []
    monkeypatch.setattr(factory, "make_chat_client", lambda **kw: made.append(kw) or object())
    deps = load_deps(Profile.COPILOT, run_root=tmp_path, ledger_dir=tmp_path / "experiments",
                     repo="example/harness", dry_run_publish=False)
    assert not deps.publisher.dry_run and deps.domain.name == "order_support"
    deps.make_agent.client(Profile.COPILOT, "critic")
    deps.strategy_kwargs["client_factory"](model="m", purpose="guard")
    assert [(m["profile"], m["purpose"]) for m in made] == [(Profile.COPILOT, "critic"), (Profile.COPILOT, "guard")]


def test_cli_offline_rejects_remote_endpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             monkeypatch: pytest.MonkeyPatch, no_offline_env: None) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.example.com/v1")
    code = main(["campaign", "status", CID, "--profile", "offline", "--run-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and "loopback" in out["error"]


def test_cli_offline_new_is_network_free(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                         no_offline_env: None, no_remote_network: list[Any]) -> None:
    from ci_lab.domain.order_support import HARNESS_ROOT, REPO_ROOT

    code = main(["campaign", "new", CID, "--profile", "offline", "--run-dir", str(tmp_path / "runs"),
                 "--ledger-dir", str(tmp_path / "experiments")])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["profile"] == "offline"
    meta = json.loads((tmp_path / "experiments" / "campaigns" / CID / "campaign.json").read_text())
    head = git(REPO_ROOT, "rev-parse", "HEAD").strip()
    assert meta["base_commit"] == head and meta["domain"] == "order_support"
    assert meta["base_tree"] == git(REPO_ROOT, "rev-parse", f"HEAD:{HARNESS_ROOT}").strip()
    assert no_remote_network == []
