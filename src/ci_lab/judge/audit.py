"""Judge-vs-human agreement audit (C21).

Joins human labels (JSONL rows ``{"case_id", "dimension", "label"}``) with ASSERT judge
results (``scores.jsonl`` rows ``{"test_case_id", "judge_status", "verdict": {"dimensions"}}``,
flat rows ``{"case_id", "dimensions"}`` or long rows ``{"case_id", "dimension", "value"}``) and
reports per dimension: exact agreement, within-one agreement (ordinal), Cohen's kappa,
quadratic-weighted kappa, Spearman's rho, seeded paired-bootstrap percentile CIs and the
confusion table. A dimension whose agreement metric (QWK for ordinal scales with three or
more levels, Cohen's kappa otherwise) is below ``floor`` - or that has fewer than ``min_n``
paired labels or an undefined kappa - is flagged ``diagnostic``: it may be reported but must
not gate promotion. The rest are ``primary``. Pure Python.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "ci-lab.judge-audit/1"
DEFAULT_FLOOR = 0.6
DEFAULT_MIN_N = 10
DEFAULT_BOOTSTRAP = 1000
SKIP_LABELS = {None, "", "ambiguous", "skip", "n/a", "na", "unknown"}

Value = Any


# ------------------------------------------------------------------ values

def norm_value(v: Any) -> Value:
    """Canonical comparable value: bools, ints and stripped strings ("true"/"3" coerced)."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        if s.lstrip("-").isdigit():
            return int(s)
        return s
    return v


def _scale_values(scale: Any) -> list[Value] | None:
    """Ordered values from an ASSERT ``dimension_scales`` entry (several shapes accepted)."""
    if isinstance(scale, Mapping):
        vals = scale.get("values", scale.get("levels"))
    else:
        vals = scale
    if isinstance(vals, Mapping):
        vals = list(vals)
    if not isinstance(vals, list) or not vals:
        return None
    out = []
    for v in vals:
        out.append(norm_value(v.get("value") if isinstance(v, Mapping) else v))
    return out


def infer_order(values: Iterable[Value], scale: list[Value] | None = None) -> list[Value] | None:
    """Ordinal order for a dimension, or None if nominal (unordered strings)."""
    vals = set(values)
    if scale:
        return None if vals - set(scale) else list(scale)
    if vals and all(isinstance(v, bool) for v in vals):
        return [False, True]
    if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return list(range(min(vals), max(vals) + 1))
    return None


# ------------------------------------------------------------------ metrics

def cohens_kappa(pairs: Sequence[tuple[Value, Value]]) -> float | None:
    """Unweighted kappa; None when undefined (no variance in either rater's labels)."""
    n = len(pairs)
    if n == 0:
        return None
    labels = {h for h, _ in pairs} | {j for _, j in pairs}
    po = sum(1 for h, j in pairs if h == j) / n
    hc = Counter(h for h, _ in pairs)
    jc = Counter(j for _, j in pairs)
    pe = sum(hc[lab] * jc[lab] for lab in labels) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def weighted_kappa(pairs: Sequence[tuple[Value, Value]], order: Sequence[Value]) -> float | None:
    """Quadratic-weighted kappa over ``order``; None when undefined."""
    n = len(pairs)
    k = len(order)
    if n == 0 or k < 2:
        return None
    idx = {v: i for i, v in enumerate(order)}
    hc = Counter(idx[h] for h, _ in pairs)
    jc = Counter(idx[j] for _, j in pairs)
    w = lambda a, b: ((a - b) / (k - 1)) ** 2  # noqa: E731
    observed = sum(w(idx[h], idx[j]) for h, j in pairs) / n
    expected = sum(hc[a] * jc[b] * w(a, b) for a in hc for b in jc) / (n * n)
    if expected == 0:
        return None
    return 1 - observed / expected


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rho with average ranks for ties; None when either side is constant."""
    if len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _metrics(pairs: Sequence[tuple[Value, Value]], order: Sequence[Value] | None) -> dict[str, float | None]:
    n = len(pairs)
    out: dict[str, float | None] = {
        "exact": (sum(1 for h, j in pairs if h == j) / n) if n else None,
        "kappa": cohens_kappa(pairs),
        "within1": None, "qwk": None, "spearman": None,
    }
    if order and n:
        idx = {v: i for i, v in enumerate(order)}
        if len(order) >= 3:
            out["within1"] = sum(1 for h, j in pairs if abs(idx[h] - idx[j]) <= 1) / n
        out["qwk"] = weighted_kappa(pairs, order)
        out["spearman"] = spearman([idx[h] for h, _ in pairs], [idx[j] for _, j in pairs])
    return out


def bootstrap_ci(pairs: Sequence[tuple[Value, Value]], stat: Callable[[Sequence[tuple[Value, Value]]], float | None],
                 *, n_boot: int = DEFAULT_BOOTSTRAP, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float] | None:
    """Paired percentile bootstrap; resamples where ``stat`` is undefined are dropped."""
    if len(pairs) < 2 or n_boot <= 0:
        return None
    rng = rng or random.Random(0)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        v = stat([pairs[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        return None
    vals.sort()
    lo = vals[max(0, int(math.floor(alpha / 2 * len(vals))))]
    hi = vals[min(len(vals) - 1, int(math.ceil((1 - alpha / 2) * len(vals))) - 1)]
    return lo, hi


# ------------------------------------------------------------------ loading

def load_labels(path: str | Path) -> dict[tuple[str, str], Value]:
    """``{(case_id, dimension): label}``; skipped/ambiguous labels are kept as None."""
    out: dict[tuple[str, str], Value] = {}
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            key = (str(row["case_id"]), str(row["dimension"]))
            label = row["label"]
        except KeyError as e:
            raise ValueError(f"{path}:{i}: label rows need case_id, dimension, label") from e
        out[key] = None if (isinstance(label, str) and label.strip().lower() in SKIP_LABELS) else norm_value(label)
    return out


def write_labels(path: str | Path, labels: Mapping[tuple[str, str], Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for (case, dim), label in sorted(labels.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            f.write(json.dumps({"case_id": case, "dimension": dim, "label": label}) + "\n")


@dataclass
class JudgeResults:
    rows: dict[str, list[dict[str, Value]]] = field(default_factory=dict)  # case -> [dims per trial]
    scales: dict[str, list[Value]] = field(default_factory=dict)
    status: Counter = field(default_factory=Counter)

    def add(self, case: str, dims: Mapping[str, Any]) -> None:
        self.rows.setdefault(case, []).append({k: norm_value(v) for k, v in dims.items()})

    def value(self, case: str, dim: str) -> Value:
        """Modal judge value across trials (ties broken by first occurrence); None if absent."""
        vals = [r[dim] for r in self.rows.get(case, []) if r.get(dim) is not None]
        if not vals:
            return None
        counts = Counter(vals)
        best = max(counts.values())
        return next(v for v in vals if counts[v] == best)


def _judge_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.rglob("scores.jsonl")))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError("no judge result files found")
    return files


def load_judge(paths: Iterable[str | Path]) -> JudgeResults:
    res = JudgeResults()
    long: dict[str, dict[str, Any]] = {}
    for f in _judge_files(paths):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case = str(row.get("case_id", row.get("test_case_id")))
            for name, sc in (row.get("dimension_scales") or {}).items():
                vals = _scale_values(sc)
                if vals:
                    res.scales.setdefault(name, vals)
            if "dimension" in row and "value" in row:
                long.setdefault(case, {})[str(row["dimension"])] = row["value"]
                continue
            if "verdict" in row or "judge_status" in row:
                status = row.get("judge_status", "ok")
                res.status[status] += 1
                if status != "ok":
                    res.rows.setdefault(case, [])
                    continue
                res.add(case, (row.get("verdict") or {}).get("dimensions") or {})
            elif isinstance(row.get("dimensions"), Mapping):
                res.status["ok"] += 1
                res.add(case, row["dimensions"])
    for case, dims in long.items():
        res.status["ok"] += 1
        res.add(case, dims)
    return res


def parse_dimension_map(items: Iterable[str]) -> dict[str, str]:
    """``["grounded=!ungrounded_claim", "pii_leak"]`` -> ``{"grounded": "!ungrounded_claim", ...}``."""
    out = {}
    for it in items:
        human, sep, judge = it.partition("=")
        out[human.strip()] = judge.strip() if sep else human.strip()
    return out


# ------------------------------------------------------------------ audit

def _judge_value(judge: JudgeResults, case: str, spec: str) -> Value:
    invert = spec.startswith("!")
    v = judge.value(case, spec.lstrip("!"))
    if invert:
        return (not v) if isinstance(v, bool) else None
    return v


def audit_dimension(name: str, pairs: list[tuple[str, Value, Value]], order: list[Value] | None, *,
                    floor: float, min_n: int, n_boot: int, seed: int, missing: int = 0,
                    skipped: int = 0) -> dict[str, Any]:
    vals = [(h, j) for _, h, j in pairs]
    m = _metrics(vals, order)
    ordinal = bool(order) and len(order) >= 3
    metric = "qwk" if ordinal else "kappa"
    rng = random.Random(f"{seed}:{name}")
    cis: dict[str, list[float] | None] = {}
    for key in ("exact", "kappa", "within1", "qwk", "spearman"):
        if m[key] is None:
            cis[key] = None
            continue
        stat = (lambda ps, k=key: _metrics(ps, order)[k])
        ci = bootstrap_ci(vals, stat, n_boot=n_boot, rng=rng)
        cis[key] = [round(ci[0], 4), round(ci[1], 4)] if ci else None
    reasons = []
    if len(vals) < min_n:
        reasons.append(f"n={len(vals)} < min_n={min_n}")
    if m[metric] is None:
        reasons.append(f"{metric} undefined (no label variance)")
    elif m[metric] < floor:
        reasons.append(f"{metric}={m[metric]:.3f} < floor={floor}")
    confusion = Counter((h, j) for h, j in vals)
    return {
        "n": len(vals),
        "scale": "ordinal" if ordinal else ("binary" if order and len(order) == 2 else "nominal"),
        "order": order,
        **{k: (round(v, 4) if v is not None else None) for k, v in m.items()},
        "ci95": cis,
        "agreement_metric": metric,
        "status": "diagnostic" if reasons else "primary",
        "reasons": reasons,
        "missing_judge_value": missing,
        "skipped_labels": skipped,
        "confusion": {f"human={h} judge={j}": c for (h, j), c in sorted(confusion.items(), key=str)},
        "disagreements": [c for c, h, j in pairs if h != j],
    }


def audit(labels: Mapping[tuple[str, str], Value], judge: JudgeResults, *,
          dimension_map: Mapping[str, str] | None = None, orders: Mapping[str, list[Value]] | None = None,
          floor: float = DEFAULT_FLOOR, min_n: int = DEFAULT_MIN_N, n_boot: int = DEFAULT_BOOTSTRAP,
          seed: int = 0) -> dict[str, Any]:
    """Compute the per-dimension audit report (a JSON-serialisable dict)."""
    dims = sorted({d for _, d in labels})
    dmap = {d: d for d in dims}
    dmap.update(dimension_map or {})
    report: dict[str, Any] = {}
    for dim in dims:
        spec = dmap[dim]
        pairs: list[tuple[str, Value, Value]] = []
        missing = skipped = 0
        for (case, d), label in sorted(labels.items()):
            if d != dim:
                continue
            if label is None:
                skipped += 1
                continue
            jv = _judge_value(judge, case, spec)
            if jv is None:
                missing += 1
                continue
            pairs.append((case, label, jv))
        scale = None
        if orders and dim in orders:
            scale = [norm_value(v) for v in orders[dim]]
        elif not spec.startswith("!"):
            scale = judge.scales.get(spec)
        order = infer_order([h for _, h, _ in pairs] + [j for _, _, j in pairs], scale)
        r = audit_dimension(dim, pairs, order, floor=floor, min_n=min_n, n_boot=n_boot, seed=seed,
                            missing=missing, skipped=skipped)
        r["judge_dimension"] = spec
        report[dim] = r
    return {
        "schema": SCHEMA,
        "floor": floor, "min_n": min_n, "n_boot": n_boot, "seed": seed,
        "judge_status": dict(judge.status),
        "dimensions": report,
        "primary": [d for d, r in report.items() if r["status"] == "primary"],
        "diagnostic": [d for d, r in report.items() if r["status"] == "diagnostic"],
    }


def run_audit(labels_path: str | Path, judge_paths: Sequence[str | Path], *, out: str | Path | None = None,
              experiment_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Load, audit, optionally write ``out`` (JSON) - inside a ``ci.evaluator`` span."""
    from ci_lab import obs
    from ci_lab.contracts import ATTR_EXPERIMENT, ATTR_PURPOSE, SPAN_EVALUATOR_EXPERIMENT

    attrs = {ATTR_PURPOSE: "judge_audit"}
    if experiment_id:
        attrs[ATTR_EXPERIMENT] = experiment_id
    with obs.span(SPAN_EVALUATOR_EXPERIMENT, attrs):
        result = audit(load_labels(labels_path), load_judge(judge_paths), **kwargs)
        obs.annotate({"ci.judge.primary": len(result["primary"]), "ci.judge.diagnostic": len(result["diagnostic"])})
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def format_report(result: Mapping[str, Any]) -> str:
    def f(x: Any) -> str:
        return "  -  " if x is None else f"{x:.3f}"

    def ci(x: Any) -> str:
        return "" if not x else f" [{x[0]:.2f},{x[1]:.2f}]"

    lines = [f"floor={result['floor']} min_n={result['min_n']} judge_status={result['judge_status']}",
             f"{'dimension':<22}{'n':>4} {'exact':>6} {'±1':>6} {'kappa':>6} {'qwk':>6} {'rho':>6}  status"]
    for d, r in result["dimensions"].items():
        lines.append(f"{d:<22}{r['n']:>4} {f(r['exact']):>6} {f(r['within1']):>6} {f(r['kappa']):>6} "
                     f"{f(r['qwk']):>6} {f(r['spearman']):>6}  {r['status']}"
                     f"{ci(r['ci95'].get(r['agreement_metric']))}"
                     + (f"  ({'; '.join(r['reasons'])})" if r["reasons"] else ""))
    return "\n".join(lines)
