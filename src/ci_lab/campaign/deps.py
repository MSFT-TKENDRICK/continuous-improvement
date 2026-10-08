"""Narrow injection points for the campaign driver (wired by integration).

Every collaborator owned by another fleet module is a plain callable or a tiny
Protocol here, so the driver/steps code only against :mod:`ci_lab.contracts`:

========================  ==========================================  ======
field                     contract                                    owner
========================  ==========================================  ======
domain                    contracts.Domain                            M3/M8a
make_agent                (role, ctx) -> MAF agent (tools bound)      M8a
provision_slot            (eid, arm, base_commit) -> worktree Path    M6
head_commit/harness_tree  worktree -> sha / harness tree hash         M6
resolve_incumbent         () -> (commit, harness tree) of the base    M6
critique                  async (ArmRun, attempt) -> CriticVerdict    M8a
get_strategy              (name, **strategy_kwargs) -> ArmStrategy    M10
schedule                  (round_no, hyper, history) -> directives    M7
select                    (incumbent, arms, delta, hyper) -> verdict  M7
calibrate_delta           (aa_results, hyper) -> delta                M7
confirm_test              (h0, final, hyper) -> decision              M7
build_envelope            (kind, record) -> OES envelope dict         M4
ledger                    LedgerStore                                 M6
publisher                 Publisher (ci_lab.publish.github)           M8b
outbox                    contracts.Outbox                            M6
build_workflow            (yaml, agents, tools, ckpt_dir) -> workflow M1
run_or_resume             async (workflow, ckpt_dir, msg) -> result   M1
========================  ==========================================  ======
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ci_lab.contracts import ArmResult, ArmStrategy, CriticVerdict, Domain, EvalResult, Outbox
from ci_lab.workflows import runtime


class LedgerStore(Protocol):
    """Ledger files under ``experiments/`` (paths are ledger-relative, ``/``-separated)."""

    def read_json(self, rel: str) -> Any | None: ...
    def write_json(self, rel: str, obj: Any) -> None: ...
    def read_jsonl(self, rel: str) -> list[dict[str, Any]]: ...
    def append_jsonl(self, rel: str, obj: Mapping[str, Any], *, key: str) -> bool: ...
    def cas_json(self, rel: str, expected: Any | None, new: Any) -> bool: ...
    def commit(self, message: str, paths: Sequence[str]) -> str | None: ...


class Publisher(Protocol):
    """GitHub side effects for accepted arms (stack layers) and losers (archive tags)."""

    def publish_round(self, *, eid: str, winner: str | None, heads: Mapping[str, str],
                      stack: Mapping[str, Any], title: str, body: str) -> dict[str, Any]: ...
    def land(self, stack: Mapping[str, Any]) -> dict[str, Any]: ...


Schedule = Callable[[int, Mapping[str, Any], Sequence[Mapping[str, Any]]], list[dict[str, Any]]]
Select = Callable[[EvalResult, Mapping[str, ArmResult], float, Mapping[str, Any]], Mapping[str, Any]]
CalibrateDelta = Callable[[Sequence[EvalResult], Mapping[str, Any]], float]
ConfirmTest = Callable[[EvalResult, EvalResult, Mapping[str, Any]], Mapping[str, Any]]


def lazy_get_strategy(name: str, **kwargs: Any) -> ArmStrategy:
    """Resolve a non-agent arm strategy from M10 (imported lazily: dspy/skillopt are heavy, C26)."""
    from ci_lab.strategies import get_strategy

    return get_strategy(name, **kwargs)


def _defaults() -> Any:
    from ci_lab.campaign import defaults

    return defaults


@dataclass
class CampaignDeps:
    domain: Domain
    make_agent: Callable[[str, Any], Any]
    provision_slot: Callable[[str, str, str], Path]
    head_commit: Callable[[Path], str]
    harness_tree: Callable[[Path], str]
    resolve_incumbent: Callable[[], tuple[str, str]]
    critique: Callable[[Any, int], Awaitable[CriticVerdict]]
    ledger: LedgerStore
    publisher: Publisher
    outbox: Outbox
    schedule: Schedule = field(default_factory=lambda: _defaults().schedule)
    select: Select = field(default_factory=lambda: _defaults().select)
    calibrate_delta: CalibrateDelta = field(default_factory=lambda: _defaults().calibrate_delta)
    confirm_test: ConfirmTest = field(default_factory=lambda: _defaults().confirm_test)
    build_envelope: Callable[[str, Mapping[str, Any]], dict[str, Any]] = field(
        default_factory=lambda: _defaults().build_envelope)
    build_workflow: Callable[[Path, Mapping[str, Any], Mapping[str, Callable[..., Any]], Path], Any] = \
        runtime.build_workflow
    run_or_resume: Callable[[Any, Path, str], Awaitable[Any]] = runtime.run_or_resume
    get_strategy: Callable[..., ArmStrategy] = lazy_get_strategy
    strategy_kwargs: Mapping[str, Any] = field(default_factory=dict)
