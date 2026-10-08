"""Lessons-as-structure rule engine (design §13, §13.6, §13.7).

Rules are data (``ci_lab.rulespec``); this package is the frozen interpreter the model cannot
bypass. Pure: no network, no LLM, no OTel global state; only the loaders touch the filesystem.
"""

from ci_lab.rules._bundle import (
    Bundle,
    Problem,
    RuleLoadError,
    build_bundle,
    compile_pattern,
    rule_order,
)
from ci_lab.rules.engine import (
    REDACTED,
    TRAJECTORY_END,
    Match,
    TrajectoryIndex,
    compute_flags,
    evaluate,
    evaluate_trajectory,
    redact,
)
from ci_lab.rules.loader import (
    LKG_DIRNAME,
    LOCK_NAME,
    default_templates,
    load_bundle,
    load_templates,
    load_with_lkg,
    write_lkg,
)

__all__ = [
    "LKG_DIRNAME",
    "LOCK_NAME",
    "REDACTED",
    "TRAJECTORY_END",
    "Bundle",
    "Match",
    "Problem",
    "RuleLoadError",
    "TrajectoryIndex",
    "build_bundle",
    "compile_pattern",
    "compute_flags",
    "default_templates",
    "evaluate",
    "evaluate_trajectory",
    "load_bundle",
    "load_templates",
    "load_with_lkg",
    "redact",
    "rule_order",
    "write_lkg",
]
