"""Validate OES 0.1.0 envelopes produced by ci_lab.

``validate_envelope(doc)`` returns a list of human-readable errors (empty = valid):

1. ``[schema]`` the vendored official OES 0.1.0 JSON Schema (draft 2020-12, formats checked);
2. ``[extension-schema]`` our extension schemas for
   ``extensions["com.microsoft.ci.rrsi"|"com.microsoft.ci.sleep"|"com.microsoft.ci.guard"]``
   (unknown extension keys are ignored, as OES requires);
3. ci_lab semantic rules (each error is prefixed with its rule id in brackets):
   ``schema-version``, ``content-hash``, ``result-hash``, ``baseline``, ``references``,
   ``decision``, ``non-compensatory``, ``rrsi``, ``holdout-looks``, ``confirm``, ``sleep``, ``guard``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from . import canonical
from .models import GUARD_EXT, NON_COMPENSATORY, OES_VERSION, RRSI_EXT, SLEEP_EXT

CORE_SCHEMA = "openexperiment-0.1.0.schema.json"
EXTENSION_SCHEMAS = {RRSI_EXT: "ext-com.microsoft.ci.rrsi.schema.json",
                     SLEEP_EXT: "ext-com.microsoft.ci.sleep.schema.json",
                     GUARD_EXT: "ext-com.microsoft.ci.guard.schema.json"}
OUR_OUTCOMES = ("ship", "do_not_ship", "rerun")
_ACTION = {"rollback": "roll_back"}


def schema_dir() -> Path:
    env = os.environ.get("CI_OES_SCHEMA_DIR")
    if env:
        return Path(env)
    bundled = Path(__file__).resolve().parent / "schemas"  # wheel: hatch force-include
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[3] / "schemas" / "oes"


@cache
def load_schema(name: str) -> dict[str, Any]:
    return json.loads((schema_dir() / name).read_text(encoding="utf-8"))


_formats = FormatChecker()


@_formats.checks("date-time", raises=ValueError)
def _is_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return True
    if "T" not in value.upper():
        raise ValueError("date-time needs a 'T' separator")
    parsed = datetime.fromisoformat(value.replace("z", "Z"))
    if parsed.tzinfo is None:
        raise ValueError("date-time needs a timezone offset")
    return True


@_formats.checks("uri", raises=ValueError)
def _is_uri(value: object) -> bool:
    if isinstance(value, str) and (":" not in value or any(c.isspace() for c in value)):
        raise ValueError("not an absolute URI")
    return True


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=_formats)


def _path(base: str, err: ValidationError) -> str:
    out = base
    for part in err.absolute_path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _schema_errors(name: str, doc: Any, base: str, rule: str) -> list[str]:
    errs = sorted(_validator(name).iter_errors(doc), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"[{rule}] {_path(base, e)}: {e.message}" for e in errs]


# ---------------------------------------------------------------- semantic rules

def _list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _baseline_ids(doc: Mapping[str, Any]) -> list[str]:
    return [v.get("id") for v in _list(doc.get("variants")) if v.get("role") in ("baseline", "control")]


def _shipped_variant(doc: Mapping[str, Any]) -> str | None:
    rrsi = _dict(_dict(doc.get("extensions")).get(RRSI_EXT))
    if rrsi.get("kind") == "round":
        return _dict(rrsi.get("selection")).get("winner")
    treatments = [v.get("id") for v in _list(doc.get("variants")) if v.get("role") == "treatment"]
    return treatments[0] if len(treatments) == 1 else None


def _rule_hashes(doc: Mapping[str, Any]) -> list[str]:
    errs = []
    decided = _dict(doc.get("decision")).get("status") == "decided"
    if canonical.HASH_FIELD not in doc:
        if decided:
            errs.append("[content-hash] decided envelopes must be hash-locked (missing contentHash)")
    elif not canonical.verify(doc):
        errs.append(f"[content-hash] contentHash mismatch: recorded {doc.get(canonical.HASH_FIELD)!r}, "
                    f"computed {canonical.content_hash(doc)!r}")
    rh = _dict(doc.get("provenance")).get("resultHash")
    if rh is not None and "results" in doc and rh != canonical.digest(doc["results"]):
        errs.append("[result-hash] provenance.resultHash does not match results")
    return errs


def _rule_variants(doc: Mapping[str, Any]) -> list[str]:
    variants = _list(doc.get("variants"))
    ids = [v.get("id") for v in variants]
    errs = []
    if len(set(ids)) != len(ids):
        errs.append(f"[baseline] duplicate variant ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    base = _baseline_ids(doc)
    if len(base) != 1:
        errs.append(f"[baseline] exactly one variant must have role baseline/control (found {len(base)})")
    return errs


def _rule_references(doc: Mapping[str, Any]) -> list[str]:
    metrics = {m.get("id") for m in _list(doc.get("metrics"))}
    variants = {v.get("id") for v in _list(doc.get("variants"))}
    base = _baseline_ids(doc)
    results = _dict(doc.get("results"))
    errs = []
    for i, r in enumerate(_list(results.get("metricResults"))):
        where = f"$.results.metricResults[{i}]"
        if r.get("metricId") not in metrics:
            errs.append(f"[references] {where}: metricId {r.get('metricId')!r} is not defined in metrics")
        cmp = _dict(r.get("comparison"))
        for key in ("baselineVariantId", "variantId"):
            if cmp.get(key) not in variants:
                errs.append(f"[references] {where}: {key} {cmp.get(key)!r} is not a variant id")
        if len(base) == 1 and cmp.get("baselineVariantId") != base[0]:
            errs.append(f"[references] {where}: baselineVariantId must be the baseline {base[0]!r}")
    for key in ("sampleSizes", "exposures"):
        extra = sorted(set(_dict(results.get(key))) - variants)
        if extra:
            errs.append(f"[references] results.{key} has unknown variants {extra}")
    return errs


def _rule_decision(doc: Mapping[str, Any]) -> list[str]:
    decision = _dict(doc.get("decision"))
    status, outcome = decision.get("status"), decision.get("outcome")
    exts = _dict(doc.get("extensions"))
    errs = []
    if _dict(doc.get("experiment")).get("status") == "decided" and status != "decided":
        errs.append("[decision] experiment.status is 'decided' but decision.status is not")
    if status == "decided" and outcome is None:
        errs.append("[decision] decision.status 'decided' requires decision.outcome")
    if outcome is not None and status not in ("decided", "superseded"):
        errs.append(f"[decision] decision.outcome {outcome!r} requires status decided/superseded")
    if outcome is not None and (RRSI_EXT in exts or SLEEP_EXT in exts) and outcome not in OUR_OUTCOMES:
        errs.append(f"[decision] ci_lab envelopes decide one of {OUR_OUTCOMES}, got {outcome!r}")
    action = _dict(doc.get("scorecard")).get("recommendedAction")
    if outcome is not None and action is not None and action != _ACTION.get(outcome, outcome):
        errs.append(f"[decision] scorecard.recommendedAction {action!r} contradicts decision.outcome {outcome!r}")
    if outcome == "ship":
        bad = [c.get("checkType") for c in _list(doc.get("qualityChecks"))
               if c.get("status") == "fail" and c.get("severity") in ("high", "critical")]
        if bad:
            errs.append(f"[decision] cannot ship with failing high/critical quality checks: {bad}")
        shipped = _shipped_variant(doc)
        roles = {v.get("id"): v.get("role") for v in _list(doc.get("variants"))}
        if shipped is None or roles.get(shipped) != "treatment":
            errs.append(f"[decision] ship requires a single shipped treatment variant (got {shipped!r})")
    return errs


def _rule_non_compensatory(doc: Mapping[str, Any]) -> list[str]:
    if _dict(doc.get("decision")).get("outcome") != "ship":
        return []
    shipped = _shipped_variant(doc)
    rrsi = _dict(_dict(doc.get("extensions")).get(RRSI_EXT))
    slack = _dict(rrsi.get("preRegistration")).get("nonInferiorityMargin", 0) if rrsi.get("kind") == "confirm" else 0
    results = [r for r in _list(_dict(doc.get("results")).get("metricResults"))
               if _dict(r.get("comparison")).get("variantId") == shipped]
    errs = [f"[non-compensatory] shipped variant {shipped!r} has a blocking result for {r.get('metricId')!r}"
            for r in results if r.get("decisionImpact") == "blocks_ship"]
    for m in _list(doc.get("metrics")):
        if m.get(NON_COMPENSATORY) is not True:
            continue
        mine = [r for r in results if r.get("metricId") == m.get("id")]
        if not mine:
            errs.append(f"[non-compensatory] guardrail {m.get('id')!r} has no result for shipped variant {shipped!r}")
            continue
        for r in mine:
            b, v = r.get("baselineValue"), r.get("variantValue")
            if not isinstance(b, (int, float)) or not isinstance(v, (int, float)):
                errs.append(f"[non-compensatory] guardrail {m.get('id')!r} result lacks baseline/variant values")
            elif m.get("direction") == "decrease_is_good" and v > b + slack:
                errs.append(f"[non-compensatory] guardrail {m.get('id')!r} worsened {b} -> {v} (allowed +{slack})")
            elif m.get("direction") == "increase_is_good" and v < b - slack:
                errs.append(f"[non-compensatory] guardrail {m.get('id')!r} worsened {b} -> {v} (allowed -{slack})")
    return errs


def _rule_rrsi(doc: Mapping[str, Any], look_counts: Mapping[str, int] | None) -> list[str]:
    ext = _dict(doc.get("extensions")).get(RRSI_EXT)
    if not isinstance(ext, dict):
        return []
    errs = []
    kind, cid, rnd = ext.get("kind"), ext.get("campaignId"), ext.get("round")
    exp_id = _dict(doc.get("experiment")).get("id")
    expected = {"calibration": f"{cid}-cal", "round": f"{cid}-r{rnd:02d}" if isinstance(rnd, int) else None,
                "confirm": f"{cid}-confirm"}.get(kind)
    if expected and exp_id != expected:
        errs.append(f"[rrsi] experiment.id {exp_id!r} should be {expected!r} for kind {kind!r}")
    variant_ids = {v.get("id") for v in _list(doc.get("variants"))}
    if "variants" in ext and set(ext["variants"]) != variant_ids:
        errs.append(f"[rrsi] extension variants {sorted(ext['variants'])} != OES variants {sorted(variant_ids)}")
    design = _dict(doc.get("design"))
    outcome = _dict(doc.get("decision")).get("outcome")
    if kind == "round":
        if design.get("type") != "abn":
            errs.append("[rrsi] round envelopes use design.type 'abn'")
        if design.get("multipleTestingPolicy") != "custom":
            errs.append("[rrsi] exploratory rounds must set design.multipleTestingPolicy 'custom'")
        sel = _dict(ext.get("selection"))
        winner = sel.get("winner")
        cands = {c.get("variantId"): c for c in _list(sel.get("candidates"))}
        if outcome == "ship" and winner is None:
            errs.append("[rrsi] decision ship requires selection.winner")
        if outcome == "do_not_ship" and winner is not None:
            errs.append(f"[rrsi] decision do_not_ship contradicts selection.winner {winner!r}")
        if winner is not None:
            if winner not in variant_ids or winner == (_baseline_ids(doc) or [None])[0]:
                errs.append(f"[rrsi] selection.winner {winner!r} is not a treatment variant")
            if not cands.get(winner, {}).get("admissible"):
                errs.append(f"[rrsi] selection.winner {winner!r} is not an admissible candidate")
            if outcome == "ship" and ext.get("ciLowerBound") != cands.get(winner, {}).get("ciLowerBound"):
                errs.append("[rrsi] ciLowerBound must equal the winner candidate's ciLowerBound")
    holdout = _dict(ext.get("holdout"))
    errs += _holdout_errors(holdout, look_counts, confirm=kind == "confirm")
    if kind == "confirm":
        pre = _dict(ext.get("preRegistration"))
        stats = _dict(pre.get("stats"))
        if design.get("alpha") != pre.get("alpha"):
            errs.append("[confirm] design.alpha must equal the pre-registered alpha")
        if design.get("peekingPolicy") != "fixed_horizon":
            errs.append("[confirm] confirmation must use peekingPolicy 'fixed_horizon'")
        if outcome == "ship":
            p, lo = stats.get("pValue"), stats.get("ciLower")
            if not isinstance(p, (int, float)) or p >= pre.get("alpha", 0):
                errs.append(f"[confirm] ship requires pValue < alpha ({p!r} vs {pre.get('alpha')!r})")
            if not isinstance(lo, (int, float)) or lo <= 0:
                errs.append(f"[confirm] ship requires a positive CI lower bound (got {lo!r})")
    return errs


def _holdout_errors(holdout: Mapping[str, Any], look_counts: Mapping[str, int] | None, *,
                    confirm: bool) -> list[str]:
    if not holdout:
        return []
    errs = []
    used, planned = holdout.get("looksUsed", 0), holdout.get("plannedLooks", 1)
    if used > planned:
        errs.append(f"[holdout-looks] held-out looks {used} exceed planned {planned}")
    if confirm and used < 1:
        errs.append("[holdout-looks] a confirmation is itself a look (looksUsed >= 1)")
    ledger = (look_counts or {}).get(holdout.get("datasetHash", ""))
    if ledger is not None and ledger > planned:
        errs.append(f"[holdout-looks] look ledger records {ledger} looks at {holdout.get('datasetHash')} "
                    f"(planned {planned})")
    return errs


def _rule_guard(doc: Mapping[str, Any], look_counts: Mapping[str, int] | None) -> list[str]:
    """v2.4 §13: a shipped guard bundle must have passed its paired ship rule (B1); guard
    evals off the evolve split are C15 held-out looks and share the global look budget."""
    ext = _dict(doc.get("extensions")).get(GUARD_EXT)
    if not isinstance(ext, dict):
        return []
    errs = _holdout_errors(_dict(ext.get("holdout")), look_counts, confirm=bool(ext.get("holdoutLook")))
    if _dict(doc.get("decision")).get("outcome") == "ship":
        if not ext.get("paired"):
            errs.append("[guard] ship requires a paired guard-off/on evaluation")
        ship = ext.get("ship")
        if isinstance(ship, dict) and not ship.get("ok"):
            errs.append(f"[guard] ship contradicts the guard ship rule: {'; '.join(ship.get('reasons') or ())}")
    return errs


def _rule_sleep(doc: Mapping[str, Any]) -> list[str]:
    ext = _dict(doc.get("extensions")).get(SLEEP_EXT)
    if not isinstance(ext, dict):
        return []
    errs = []
    night = str(ext.get("night", ""))
    if _dict(doc.get("experiment")).get("id") != f"sleep-{night.replace('-', '')}":
        errs.append(f"[sleep] experiment.id should be 'sleep-{night.replace('-', '')}'")
    leaked = {k: n for k, n in _dict(_dict(ext.get("tasks")).get("bySplit")).items() if k != "evolve" and n}
    if leaked:
        errs.append(f"[sleep] sleep tasks must come from the evolve split only (found {leaked})")
    tasks = _dict(ext.get("tasks"))
    if sum(_dict(tasks.get("byOrigin")).values()) != tasks.get("total"):
        errs.append("[sleep] tasks.total must equal the sum of tasks.byOrigin")
    outcome = _dict(doc.get("decision")).get("outcome")
    gate = _dict(ext.get("gate"))
    if outcome == "ship":
        failed = [g for g in ("skillopt", "assert") if not _dict(gate.get(g)).get("passed")]
        if failed:
            errs.append(f"[sleep] ship requires both gates to pass (failed: {failed})")
        if not ext.get("candidateDigest"):
            errs.append("[sleep] ship requires candidateDigest")
    elif ext.get("adoptionPr") is not None:
        errs.append(f"[sleep] adoptionPr must be null when outcome is {outcome!r}")
    return errs


def validate_envelope(doc: Any, *, look_counts: Mapping[str, int] | None = None) -> list[str]:
    """All schema + semantic errors for one envelope (``[]`` = valid).

    ``look_counts`` maps held-out ``datasetHash`` -> looks recorded in the global look ledger.
    """
    if not isinstance(doc, Mapping):
        return ["[schema] $: an OES envelope must be a JSON object"]
    doc = dict(doc)
    errors = _schema_errors(CORE_SCHEMA, doc, "$", "schema")
    exts = _dict(doc.get("extensions"))
    ext_ok = {}
    for key, name in EXTENSION_SCHEMAS.items():
        if key in exts:
            e = _schema_errors(name, exts[key], f'$.extensions["{key}"]', "extension-schema")
            errors += e
            ext_ok[key] = not e
    if doc.get("schemaVersion") != OES_VERSION:
        errors.append(f"[schema-version] schemaVersion {doc.get('schemaVersion')!r} is not {OES_VERSION}")
    errors += _rule_hashes(doc)
    if errors:
        return errors
    errors += _rule_variants(doc) + _rule_references(doc) + _rule_decision(doc) + _rule_non_compensatory(doc)
    if ext_ok.get(RRSI_EXT):
        errors += _rule_rrsi(doc, look_counts)
    if ext_ok.get(SLEEP_EXT):
        errors += _rule_sleep(doc)
    if ext_ok.get(GUARD_EXT):
        errors += _rule_guard(doc, look_counts)
    return errors


# ---------------------------------------------------------------- files

def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _no_constants(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_dupes,
                      parse_constant=_no_constants)


def validate_file(path: str | Path, *, look_counts: Mapping[str, int] | None = None) -> list[str]:
    try:
        doc = load_json(Path(path))
    except (OSError, ValueError) as exc:
        return [f"[json] $: cannot read JSON: {exc}"]
    return validate_envelope(doc, look_counts=look_counts)


def read_look_ledger(path: str | Path) -> dict[str, int]:
    """Count looks per ``datasetHash`` in a JSONL look ledger (one JSON object per line)."""
    counts: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            key = json.loads(line).get("datasetHash")
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def iter_rule_ids(errors: Iterable[str]) -> set[str]:
    return {e[1:e.index("]")] for e in errors if e.startswith("[") and "]" in e}
