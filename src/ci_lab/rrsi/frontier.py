"""Incumbent frontier state and its pure transitions (frontier.json payload).

``trajectory[i]`` is the incumbent at the *start* of round i (``trajectory[0]`` = H0
baseline), which is exactly what ``schedule.stall_flag`` reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from ci_lab.contracts import ArmResult

from .codec import finite
from .selection import SelectionDecision

Status = Literal["active", "paused", "done"]


@dataclass(frozen=True)
class TrajectoryPoint:
    round: int
    score: float
    cost: float
    commit: str
    harness_tree: str
    arm: str | None = None          # winning arm that produced this incumbent (None = carried over)

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "score": finite(self.score), "cost": finite(self.cost), "commit": self.commit,
                "harness_tree": self.harness_tree, "arm": self.arm}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TrajectoryPoint:
        return cls(round=int(d["round"]), score=float(d["score"]), cost=float(d.get("cost") or 0.0),
                   commit=d["commit"], harness_tree=d["harness_tree"], arm=d.get("arm"))


@dataclass(frozen=True)
class Frontier:
    campaign_id: str
    round: int                      # next round to run
    s_star: float
    delta: float
    trajectory: tuple[TrajectoryPoint, ...]
    status: Status = "active"
    reruns: int = 0                 # consecutive reruns of the current round
    note: str = ""

    @property
    def incumbent(self) -> TrajectoryPoint:
        return self.trajectory[-1]

    @property
    def scores(self) -> list[float]:
        return [p.score for p in self.trajectory]

    def to_dict(self) -> dict[str, Any]:
        return {"campaign_id": self.campaign_id, "round": self.round, "s_star": finite(self.s_star),
                "delta": finite(self.delta), "trajectory": [p.to_dict() for p in self.trajectory],
                "status": self.status, "reruns": self.reruns, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Frontier:
        return cls(campaign_id=d["campaign_id"], round=int(d["round"]), s_star=float(d["s_star"]),
                   delta=float(d["delta"]), trajectory=tuple(TrajectoryPoint.from_dict(p) for p in d["trajectory"]),
                   status=d.get("status", "active"), reruns=int(d.get("reruns", 0)), note=d.get("note", ""))


def initial(campaign_id: str, *, commit: str, harness_tree: str, score: float, cost: float,
            delta: float) -> Frontier:
    """Frontier after baseline + A/A calibration: incumbent = H0, S* = S(H0)."""
    if delta < 0:
        raise ValueError("delta must be >= 0")
    return Frontier(campaign_id=campaign_id, round=0, s_star=score, delta=delta,
                    trajectory=(TrajectoryPoint(0, score, cost, commit, harness_tree),))


def advance(fr: Frontier, decision: SelectionDecision, arms: Sequence[ArmResult]) -> Frontier:
    """Apply a selection decision for round ``fr.round``.

    * ``ship``: winner becomes the incumbent (must be based on the incumbent commit).
    * ``do_not_ship``: incumbent kept, its contemporaneous score recorded.
    * ``rerun``: nothing changes except the rerun counter.
    S* only ever moves up.
    """
    if fr.status != "active":
        raise ValueError(f"frontier is {fr.status}; cannot advance")
    if decision.round != fr.round:
        raise ValueError(f"decision is for round {decision.round}, frontier expects round {fr.round}")
    if decision.delta != fr.delta:
        raise ValueError("decision delta differs from the campaign delta (delta is immutable)")
    if decision.decision == "rerun":
        return replace(fr, reruns=fr.reruns + 1, note="; ".join(decision.reasons))
    inc = fr.incumbent
    if decision.decision == "ship":
        arm = next((a for a in arms if a.arm == decision.winner), None)
        win = decision.winner_trace
        if arm is None or win is None:
            raise ValueError(f"winner {decision.winner!r} not among arms")
        if arm.base_commit != inc.commit:
            raise ValueError(f"winner {arm.arm} is based on {arm.base_commit}, not incumbent {inc.commit}")
        if not arm.head_commit or not arm.harness_tree:
            raise ValueError(f"winner {arm.arm} lacks head_commit/harness_tree")
        point = TrajectoryPoint(fr.round + 1, float(win.score), float(win.cost), arm.head_commit, arm.harness_tree,
                                arm.arm)
    else:
        score = decision.score_next if decision.score_next is not None else inc.score
        cost = decision.incumbent.get("cost", inc.cost)
        point = TrajectoryPoint(fr.round + 1, float(score), float(cost), inc.commit, inc.harness_tree, None)
    s_star = max(fr.s_star, decision.s_star_after, point.score)
    return replace(fr, round=fr.round + 1, s_star=s_star, trajectory=(*fr.trajectory, point), reruns=0, note="")


def pause(fr: Frontier, reason: str) -> Frontier:
    return replace(fr, status="paused", note=reason)


def resume(fr: Frontier) -> Frontier:
    if fr.status != "paused":
        raise ValueError("frontier is not paused")
    return replace(fr, status="active", note="")


def finish(fr: Frontier) -> Frontier:
    return replace(fr, status="done")


def rollback(fr: Frontier, to_round: int) -> Frontier:
    """Drop trajectory points after round ``to_round`` (needed before readjudicating an
    earlier round, since later arms were drafted from the old incumbent). S* is recomputed
    from the remaining trajectory."""
    if not 0 <= to_round <= fr.round:
        raise ValueError(f"cannot roll back to round {to_round}")
    traj = fr.trajectory[: to_round + 1]
    return replace(fr, round=to_round, trajectory=traj, s_star=max(p.score for p in traj), reruns=0, note="",
                   status="active")
