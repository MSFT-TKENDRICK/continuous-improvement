"""Campaign driver: RRSI rounds as OES experiments over declarative MAF workflows.

``Campaign.new`` -> ``calibrate`` (A/A, immutable δ) -> ``run`` (rounds until STOP
file / budget / max rounds) -> ``confirm`` (sealed held-out, one look) -> ``land``.
Each round builds ``round.yaml`` through ``deps.build_workflow`` and executes it
with ``deps.run_or_resume`` (checkpoint dir ``CI_RUN_DIR/<eid>/ckpt``); its
``run_arms`` step fans out one ``arm.yaml`` workflow per arm (C5). Re-running any
command after a crash resumes from checkpoints and durable markers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ci_lab.campaign import records
from ci_lab.campaign.defaults import DEFAULT_HYPER
from ci_lab.campaign.deps import CampaignDeps
from ci_lab.contracts import CAMPAIGN_RE, Profile, round_experiment_id
from ci_lab.workflows import CALIBRATE_YAML, CONFIRM_YAML, ROUND_YAML
from ci_lab.workflows.runtime import GatedAgent
from ci_lab.workflows.steps import (
    CalibrationContext,
    CampaignEnv,
    ConfirmContext,
    RoundContext,
    calibrate_tools,
    confirm_tools,
    round_tools,
)


def default_run_root() -> Path:
    return Path(os.environ.get("CI_RUN_DIR") or Path("artifacts") / "ci-runs")


def merge_hyper(hyperparams: Mapping[str, Any] | None) -> dict[str, Any]:
    hyper = dict(DEFAULT_HYPER)
    unknown = set(hyperparams or {}) - set(hyper)
    if unknown:
        raise ValueError(f"unknown hyperparameters: {sorted(unknown)}")
    hyper.update(hyperparams or {})
    for key in ("arms", "k", "max_rounds", "aa_repeats", "max_parallel_arms", "max_arm_attempts", "holdout_looks"):
        if int(hyper[key]) < 1:
            raise ValueError(f"{key} must be >= 1")
    if not 1 <= int(hyper["max_rounds"]) <= 99:
        raise ValueError("max_rounds must be in 1..99")
    return hyper


class Campaign:
    def __init__(self, env: CampaignEnv) -> None:
        self.env = env

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def new(cls, cid: str, profile: Profile | str, hyperparams: Mapping[str, Any] | None = None, *,
            deps: CampaignDeps, run_root: Path | None = None) -> Campaign:
        if not CAMPAIGN_RE.match(cid):
            raise ValueError(f"bad campaign id {cid!r}")
        profile = Profile(profile)
        hyper = merge_hyper(hyperparams)
        env = CampaignEnv(cid, profile, hyper, deps, Path(run_root or default_run_root()))
        meta_rel = env.rel("campaign.json")
        existing = deps.ledger.read_json(meta_rel)
        if existing is not None:
            if existing["profile"] != profile.value or existing["hyper"] != hyper:
                raise ValueError(f"campaign {cid} already exists with different settings")
            return cls(env)
        commit, tree = deps.resolve_incumbent()
        meta = {"campaignId": cid, "profile": profile.value, "hyper": hyper, "base_commit": commit,
                "base_tree": tree, "domain": getattr(deps.domain, "name", "unknown")}
        frontier = {"incumbent_commit": commit, "incumbent_tree": tree, "score": None, "round": 0, "eid": None}
        deps.ledger.cas_json(env.rel("frontier.json"), None, frontier)
        deps.ledger.write_json(meta_rel, meta)
        env.commit_ledger("new", f"Start campaign {cid}", [meta_rel, env.rel("frontier.json")])
        return cls(env)

    @classmethod
    def load(cls, cid: str, *, deps: CampaignDeps, run_root: Path | None = None) -> Campaign:
        if not CAMPAIGN_RE.match(cid):
            raise ValueError(f"bad campaign id {cid!r}")
        meta = deps.ledger.read_json("/".join(("campaigns", cid, "campaign.json")))
        if meta is None:
            raise FileNotFoundError(f"campaign {cid} not found (run `campaign new`)")
        return cls(CampaignEnv(cid, Profile(meta["profile"]), meta["hyper"], deps,
                               Path(run_root or default_run_root())))

    @property
    def cid(self) -> str:
        return self.env.cid

    @property
    def deps(self) -> CampaignDeps:
        return self.env.deps

    def _meta(self) -> dict[str, Any]:
        return self.deps.ledger.read_json(self.env.rel("campaign.json"))

    # ------------------------------------------------------------ calibration
    async def calibrate(self) -> float:
        cal = self.deps.ledger.read_json(self.env.rel("calibration.json"))
        if cal is not None:
            return float(cal["delta"])
        ctx = CalibrationContext(self.env)
        workflow = self.deps.build_workflow(CALIBRATE_YAML, {}, calibrate_tools(ctx), ctx.ckpt)
        await self.deps.run_or_resume(workflow, ctx.ckpt, "start")
        return self.env.delta()

    # ------------------------------------------------------------ rounds
    def _spent_tokens(self) -> int:
        cal = self.deps.ledger.read_json(self.env.rel("calibration.json")) or {}
        history = self.deps.ledger.read_jsonl(self.env.rel("history.jsonl"))
        return int(cal.get("tokens", 0)) + sum(int(h.get("tokens", 0)) for h in history)

    def _stop_reason(self, stop_file: Path | None) -> str | None:
        if stop_file is not None and Path(stop_file).exists():
            return "stop_file"
        budget = self.env.hyper.get("budget_tokens")
        if budget is not None and self._spent_tokens() >= int(budget):
            return "budget"
        return None

    async def run_round(self, round_no: int) -> dict[str, Any]:
        ctx = RoundContext(self.env, round_no)
        done = records.read_json(ctx.dir / "round.done")
        if done is not None:
            return done
        analyst = GatedAgent(self.deps.make_agent("analyst", ctx), ctx.analysis_path)
        workflow = self.deps.build_workflow(ROUND_YAML, {"Analyst": analyst}, round_tools(ctx), ctx.ckpt)
        await self.deps.run_or_resume(workflow, ctx.ckpt, "start")
        rec = records.read_json(ctx.dir / "record.done")
        pub = records.read_json(ctx.dir / "publish.done")
        if rec is None or pub is None:
            raise RuntimeError(f"{ctx.eid}: round workflow completed without record/publish markers")
        summary = {"eid": ctx.eid, "round": round_no, "decision": rec["decision"], "winner": rec["winner"],
                   "pr": (pub.get("result") or {}).get("pr")}
        records.write_json(ctx.dir / "round.done", summary)
        return summary

    async def run(self, rounds: int | None = None, stop_file: Path | None = None) -> dict[str, Any]:
        """Run rounds 1..``rounds`` (default ``max_rounds``); completed rounds are skipped.
        ``stop_file`` defaults to ``CI_RUN_DIR/<cid>/STOP`` (checked between rounds)."""
        self.env.delta()  # requires calibration
        stop_file = Path(stop_file) if stop_file is not None else self.env.campaign_dir / "STOP"
        last = min(int(rounds or self.env.hyper["max_rounds"]), int(self.env.hyper["max_rounds"]))
        summaries: list[dict[str, Any]] = []
        reason = "max_rounds"
        for round_no in range(1, last + 1):
            ctx = RoundContext(self.env, round_no)
            if (done := records.read_json(ctx.dir / "round.done")) is not None:
                summaries.append(done)
                continue
            if not ctx.dir.exists() and (stop := self._stop_reason(stop_file)) is not None:
                reason = stop
                break
            summaries.append(await self.run_round(round_no))
        return {"campaign": self.cid, "rounds": summaries, "stopped": reason}

    # ------------------------------------------------------------ inspection
    def status(self) -> dict[str, Any]:
        ledger = self.deps.ledger
        cal = ledger.read_json(self.env.rel("calibration.json"))
        history = ledger.read_jsonl(self.env.rel("history.jsonl"))
        in_flight = []
        for round_no in range(1, int(self.env.hyper["max_rounds"]) + 1):
            ctx = RoundContext(self.env, round_no)
            if ctx.dir.exists() and not (ctx.dir / "round.done").exists():
                in_flight.append(ctx.eid)
        return {"campaign": self.cid, "profile": self.env.profile.value, "hyper": dict(self.env.hyper),
                "calibrated": cal is not None, "delta": cal["delta"] if cal else None,
                "frontier": ledger.read_json(self.env.rel("frontier.json")),
                "rounds": [{k: h[k] for k in ("eid", "decision", "winner", "tokens")} for h in history],
                "in_flight": in_flight, "tokens_spent": self._spent_tokens(),
                "stack": ledger.read_json(self.env.rel("stack.json")),
                "confirm": ledger.read_json(self.env.rel("confirm.json")),
                "landed": ledger.read_json(self.env.rel("land.json"))}

    def readjudicate(self, eid: str) -> dict[str, Any]:
        """Recompute the selection from recorded evals + immutable δ; report drift."""
        rounds = {round_experiment_id(self.cid, r): r for r in range(0, 100)}
        if eid not in rounds:
            raise ValueError(f"{eid!r} is not a round of campaign {self.cid}")
        ledger = self.deps.ledger
        evals = ledger.read_json(self.env.rel("rounds", eid, "evals.json"))
        recorded = ledger.read_json(self.env.rel("rounds", eid, "decisions.json"))
        if evals is None or recorded is None:
            raise FileNotFoundError(f"no recorded evals/decision for {eid}")
        incumbent = records.arm_from_dict(evals["incumbent"])
        arms = {k: records.arm_from_dict(v) for k, v in evals["arms"].items()}
        fresh = dict(self.deps.select(incumbent.eval, arms, self.env.delta(), self.env.hyper))
        same = fresh.get("decision") == recorded.get("decision") and fresh.get("winner") == recorded.get("winner")
        return {"eid": eid, "recorded": {k: recorded.get(k) for k in ("decision", "winner")},
                "recomputed": {k: fresh.get(k) for k in ("decision", "winner")}, "consistent": same}

    # ------------------------------------------------------------ confirm / land
    async def confirm(self) -> dict[str, Any]:
        ctx = ConfirmContext(self.env)
        workflow = self.deps.build_workflow(CONFIRM_YAML, {}, confirm_tools(ctx), ctx.ckpt)
        await self.deps.run_or_resume(workflow, ctx.ckpt, "start")
        return self.deps.ledger.read_json(self.env.rel("confirm.json"))

    def land(self) -> dict[str, Any]:
        ledger = self.deps.ledger
        landed = ledger.read_json(self.env.rel("land.json"))
        if landed is not None:
            return landed
        confirm = ledger.read_json(self.env.rel("confirm.json"))
        if not confirm or confirm.get("decision") != "ship":
            raise RuntimeError("land requires a confirm decision of 'ship'")
        stack = ledger.read_json(self.env.rel("stack.json"))
        if not stack or not stack.get("layers"):
            raise RuntimeError("nothing to land: no accepted layers")
        result = self.deps.publisher.land(stack)  # each gh mutation is its own outbox op
        data = {"campaignId": self.cid, "result": result, "layers": [layer["pr"] for layer in stack["layers"]]}
        ledger.write_json(self.env.rel("land.json"), data)
        self.env.commit_ledger("land", f"Land campaign {self.cid}", [self.env.rel("land.json")])
        return data
