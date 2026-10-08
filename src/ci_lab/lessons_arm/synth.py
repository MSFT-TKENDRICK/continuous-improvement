"""Deterministic template synthesizers (design §13.3 step 4; B3). No LLM, no trace text.

Each synthesizer maps ``(LessonCluster, LessonFeatures)`` to :class:`~ci_lab.rulespec.RuleSpec`
objects with ``mode: shadow``, ``provenance.source: template``, a template id from the trusted
catalog and slots filled only with identifiers / enum tokens from the typed features.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ci_lab.rulespec import (
    AllPred,
    AnyPred,
    ArgPred,
    Cmp,
    LessonCluster,
    NotPred,
    PriorPred,
    Provenance,
    RuleSpec,
    StatePred,
    TextPred,
)

from .features import LessonFeatures
from .templates import (
    SAFE_PATTERNS,
    TPL_AMOUNT_PRIOR,
    TPL_ARG_CONSTRAINT,
    TPL_PRIOR_CALL,
    TPL_REDACT_PATTERN,
    TPL_STATE_FLAG,
)

_SLUG_BAD = re.compile(r"[^a-z0-9_.-]+")


class SynthesisError(ValueError):
    """A cluster's features are insufficient for the requested template."""


def lesson_id_for(cluster: LessonCluster) -> str:
    """Filesystem/rule-id safe lesson id derived from the cluster id (deterministic)."""
    slug = _SLUG_BAD.sub("-", cluster.id.casefold()).strip("-.")[:40] or cluster.fingerprint.digest()
    if not slug[0].isalpha():
        slug = "l" + slug
    return slug


def _rule_id(cluster: LessonCluster, suffix: str) -> str:
    return f"lsn.{lesson_id_for(cluster)}.{suffix}"[:64]


def _provenance(cluster: LessonCluster) -> Provenance:
    return Provenance(lesson_id=lesson_id_for(cluster), source="template",
                      evidence=[f"fingerprint:{cluster.fingerprint.digest()}"])


def _need(f: LessonFeatures, *names: str) -> None:
    missing = [n for n in names if getattr(f, n) in (None, [], {})]
    if missing:
        raise SynthesisError(f"{f.kind}: missing features {missing}")


def _join(f: LessonFeatures) -> list[tuple[str, str]]:
    return [(f"current.args.{f.subject_arg}", f"prior.args.{f.prior_subject_field or f.subject_arg}")]


def _rule(cluster: LessonCluster, suffix: str, **kw: object) -> RuleSpec:
    return RuleSpec(id=_rule_id(cluster, suffix), version=1, mode="shadow", provenance=_provenance(cluster),
                    **kw)  # type: ignore[arg-type]


def synth_prior_call(cluster: LessonCluster, f: LessonFeatures) -> RuleSpec:
    """Precondition-from-sequence: ``target`` requires an earlier ok ``prior_tool`` call on the
    same subject (``same`` join), optionally with typed ``prior.result`` equalities."""
    _need(f, "target_tool", "prior_tool", "subject_arg")
    conds = [ArgPred(kind="arg", path=f"prior.result.{k}", op="eq", value=v)
             for k, v in sorted(f.prior_result_equals.items())]
    where = None if not conds else conds[0] if len(conds) == 1 else AllPred(kind="all", of=conds)
    require = PriorPred(kind="prior", tool=f.prior_tool, status="ok", where=where, within=f.within, same=_join(f))
    return _rule(cluster, "prior", rung="R2", on="tool_call", target=f.target_tool, require=require,
                 action="block", severity="critical", template=TPL_PRIOR_CALL,
                 slots={"tool": f.target_tool, "prior_tool": f.prior_tool, "subject": f.subject_arg})


def synth_state_flag(cluster: LessonCluster, f: LessonFeatures) -> RuleSpec:
    """Precondition on a frozen-extractor flag bound to ``current.args.<subject_arg>``."""
    _need(f, "target_tool", "flag", "subject_arg", "via_tool")
    require = StatePred(kind="state", flag=f.flag, subject=f"current.args.{f.subject_arg}")
    return _rule(cluster, "state", rung="R2", on="tool_call", target=f.target_tool, require=require,
                 action="block", severity="critical", template=TPL_STATE_FLAG,
                 slots={"tool": f.target_tool, "flag": f.flag, "subject": f.subject_arg, "via_tool": f.via_tool})


def _constraint_slot(f: LessonFeatures) -> str:
    vals = "|".join(str(v).lower() if isinstance(v, bool) else str(v) for v in f.values)
    return f"{f.op} {vals}".strip()[:120]


def synth_arg_constraint(cluster: LessonCluster, f: LessonFeatures) -> RuleSpec:
    """R1 schema rung: one typed arg constraint on ``target_tool``."""
    _need(f, "target_tool", "arg", "op")
    if f.op in ("in", "nin"):
        if not f.values:
            raise SynthesisError(f"arg_constraint {f.op} needs values")
        value: object = list(f.values)
    elif f.op == "exists":
        value = None
    else:
        if len(f.values) != 1:
            raise SynthesisError(f"arg_constraint {f.op} needs exactly one value")
        value = f.values[0]
    require = ArgPred(kind="arg", path=f"current.args.{f.arg}", op=f.op, value=value)  # type: ignore[arg-type]
    return _rule(cluster, "arg", rung="R1", on="tool_call", target=f.target_tool, require=require,
                 action="block", template=TPL_ARG_CONSTRAINT,
                 slots={"tool": f.target_tool, "arg": f.arg, "constraint": _constraint_slot(f)})


def synth_amount_vs_prior(cluster: LessonCluster, f: LessonFeatures) -> RuleSpec:
    """R2 ``cmp``: ``current.args.<amount_arg> <cmp_op> prior.result.<prior_amount_field>`` on the
    same subject's earlier ok ``prior_tool`` call."""
    _need(f, "target_tool", "prior_tool", "subject_arg", "amount_arg", "prior_amount_field")
    require = PriorPred(kind="prior", tool=f.prior_tool, status="ok", within=f.within, same=_join(f),
                        cmp=[Cmp(current=f"current.args.{f.amount_arg}", op=f.cmp_op,
                                 prior=f"prior.result.{f.prior_amount_field}")])
    return _rule(cluster, "amount", rung="R2", on="tool_call", target=f.target_tool, require=require,
                 action="block", severity="critical", template=TPL_AMOUNT_PRIOR,
                 slots={"tool": f.target_tool, "arg": f.amount_arg, "prior_tool": f.prior_tool,
                        "prior_field": f.prior_amount_field})


def synth_response_pattern(cluster: LessonCluster, f: LessonFeatures) -> RuleSpec:
    """R3 redact: while ``flag`` is not established, response text must not match any pattern
    class from the fixed safe library (never trace literals)."""
    _need(f, "flag", "pattern_classes")
    texts = [TextPred(kind="text", matches=SAFE_PATTERNS[c]) for c in f.pattern_classes]
    matches = texts[0] if len(texts) == 1 else AnyPred(kind="any", of=texts)
    return _rule(cluster, "redact", rung="R3", on="response", target="*",
                 when=NotPred(kind="not", of=StatePred(kind="state", flag=f.flag)),
                 require=NotPred(kind="not", of=matches), action="redact", severity="critical",
                 template=TPL_REDACT_PATTERN,
                 slots={"pattern_class": ", ".join(f.pattern_classes), "flag": f.flag})


SYNTHESIZERS: dict[str, Callable[[LessonCluster, LessonFeatures], RuleSpec]] = {
    "prior_call": synth_prior_call,
    "state_flag": synth_state_flag,
    "arg_constraint": synth_arg_constraint,
    "amount_vs_prior": synth_amount_vs_prior,
    "response_pattern": synth_response_pattern,
}


def synthesize(cluster: LessonCluster, features: LessonFeatures) -> RuleSpec:
    """Run the template synthesizer selected by ``features.kind`` (B3: features must be trusted
    or the cluster human-confirmed; injection-suspect lessons are refused)."""
    if features.injection_suspect or any(r.startswith("injection.") for r in cluster.fingerprint.oracle_rules):
        raise SynthesisError("injection-suspect lessons are never synthesized (B3)")
    if not features.trusted and not cluster.human_confirmed:
        raise SynthesisError("untrusted (usage/PR) lesson needs a human-reviewed label first (B3)")
    return SYNTHESIZERS[features.kind](cluster, features)
