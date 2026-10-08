"""RRSI core (M7): pure schedule / statistics / selection / frontier / attribution logic.

The only I/O lives in the JSONL/JSON helpers of ``history`` and ``readjudicate``.
See docs/rrsi.md.
"""

from .attribution import STRUCTURAL_COMPONENTS, ComponentStats, attribute, component_stats, novelty
from .frontier import Frontier, TrajectoryPoint, advance, initial
from .history import HistoryRecord, append_jsonl, read_jsonl, write_jsonl
from .params import PROFILES, TABLE5, Hyperparams, paper_reference, profile
from .readjudicate import Readjudication, load_round, save_round, verify
from .schedule import Directive, RoundSchedule, directives, edit_budget, plan_round, prune_set, stall_flag, untried
from .selection import ArmTrace, SelectionDecision, SelectionInputs, select
from .strategies import StrategyAllocation, StrategyStats, allocate_strategies, strategy_stats

__all__ = [
    "PROFILES", "STRUCTURAL_COMPONENTS", "TABLE5", "ArmTrace", "ComponentStats", "Directive", "Frontier",
    "HistoryRecord", "Hyperparams", "Readjudication", "RoundSchedule", "SelectionDecision", "SelectionInputs",
    "StrategyAllocation", "StrategyStats", "TrajectoryPoint", "advance", "allocate_strategies", "append_jsonl",
    "attribute", "component_stats", "directives", "edit_budget", "initial", "novelty", "paper_reference", "profile",
    "plan_round", "prune_set", "read_jsonl", "load_round", "save_round", "select", "stall_flag", "strategy_stats",
    "untried", "verify", "write_jsonl",
]