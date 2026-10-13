"""Pure rule evaluator (design §13.2, §13.6 B5/B7/N6). No I/O, no network, no LLM, no OTel state.

See ``docs/rules.md`` for the exact semantics implemented here.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from ci_lab.rules._bundle import Bundle
from ci_lab.rulespec import (
    TEXT_MAX_BYTES,
    AllPred,
    AnyPred,
    ArgPred,
    CountPred,
    ExtractorSpec,
    GuardView,
    NotPred,
    PriorPred,
    RuleSpec,
    StatePred,
    TextPred,
    TrajectoryStep,
    normalize_subject,
)

On = Literal["tool_call", "response", "trajectory"]
REDACTED = "[redacted]"
TRAJECTORY_END = -1  # Match.step_index for whole-trajectory (target "*") R4 evaluations


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


@dataclass(frozen=True)
class Match:
    """A rule that fired (all-match telemetry, N6). ``message``/``fix`` come from trusted templates."""

    rule: RuleSpec
    message: str
    fix: str
    see: str
    step_index: int


# ---------------------------------------------------------------- typed comparisons


def _cls(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str"
    if v is None:
        return "null"
    return "other"


def strict_eq(a: Any, b: Any) -> bool:
    """Equality without coercion: bool is not a number, str never equals a number."""
    c = _cls(a)
    return c != "other" and c == _cls(b) and a == b


def _order(op: str, a: Any, b: Any) -> bool:
    c = _cls(a)
    if c not in ("num", "str") or c != _cls(b):
        return False
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    if op == "lt":
        return a < b
    return a <= b  # le


def compare(op: str, a: Any, b: Any) -> bool:
    """Typed comparison; missing/mismatched types => False (also for ``ne``)."""
    if a is MISSING or b is MISSING:
        return False
    if op == "eq":
        return strict_eq(a, b)
    if op == "ne":
        c = _cls(a)
        return c != "other" and c == _cls(b) and a != b
    return _order(op, a, b)


def _same(a: Any, b: Any) -> bool:
    """``prior.same`` join: strings via normalize_subject on both sides, numbers exact."""
    if a is MISSING or b is MISSING:
        return False
    ca, cb = _cls(a), _cls(b)
    if ca == cb == "str":
        return normalize_subject(a) == normalize_subject(b)
    return ca == cb == "num" and a == b


def cap_text(s: str) -> str:
    """Cap to TEXT_MAX_BYTES of UTF-8 (B7), never splitting a code point."""
    if len(s) * 4 <= TEXT_MAX_BYTES:
        return s
    b = s.encode("utf-8", "surrogatepass")
    if len(b) <= TEXT_MAX_BYTES:
        return s
    return b[:TEXT_MAX_BYTES].decode("utf-8", "ignore")


# ---------------------------------------------------------------- paths


@lru_cache(maxsize=4096)
def parse_path(path: str) -> tuple[str, tuple[str | int, ...]]:
    """``current.args.a[0].b`` -> ("current.args", ("a", 0, "b")). Raises ValueError if malformed."""
    for ns in ("current.args", "prior.args", "prior.result", "args", "result"):
        if path == ns or path.startswith((ns + ".", ns + "[")):
            rest = path[len(ns):]
            break
    else:
        raise ValueError(f"bad path {path!r}")
    segs: list[str | int] = []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == ".":
            j = i + 1
            while j < len(rest) and (rest[j].isascii() and (rest[j].isalnum() or rest[j] == "_")):
                j += 1
            name = rest[i + 1:j]
            if not name or name[0].isdigit():
                raise ValueError(f"bad path {path!r}")
            segs.append(name)
            i = j
        elif ch == "[":
            j = rest.find("]", i)
            idx = rest[i + 1:j] if j > 0 else ""
            if not idx.isascii() or not idx.isdigit():
                raise ValueError(f"bad path {path!r}")
            segs.append(int(idx))
            i = j + 1
        else:
            raise ValueError(f"bad path {path!r}")
    return ns, tuple(segs)


def walk(root: Any, segs: Sequence[str | int]) -> Any:
    cur = root
    for s in segs:
        if isinstance(s, int):
            if isinstance(cur, (list, tuple)) and 0 <= s < len(cur):
                cur = cur[s]
            else:
                return MISSING
        elif isinstance(cur, Mapping) and s in cur:
            cur = cur[s]
        else:
            return MISSING
    return cur


# ---------------------------------------------------------------- trajectory index


class TrajectoryIndex:
    """Call/result pairing, statuses and extractor events over a step sequence.

    Built once; queries take ``upto`` (number of visible steps) so replay can evaluate every
    prefix without rebuilding. Results at positions >= ``upto`` are invisible.
    """

    def __init__(self, steps: Sequence[TrajectoryStep], extractors: Sequence[ExtractorSpec] = ()) -> None:
        self.steps = tuple(steps)
        self.call_pos: list[int] = []
        self.by_tool: dict[str, list[int]] = defaultdict(list)
        self.result_of: dict[int, int] = {}
        open_by_id: dict[str, int] = {}
        open_by_tool: dict[str, list[int]] = defaultdict(list)
        for k, s in enumerate(self.steps):
            if s.kind == "tool_call":
                self.call_pos.append(k)
                self.by_tool[s.tool or ""].append(k)
                open_by_tool[s.tool or ""].append(k)
                if s.call_id:
                    open_by_id[s.call_id] = k
            elif s.kind == "tool_result":
                c = None
                if s.call_id and s.call_id in open_by_id:
                    c = open_by_id[s.call_id]
                else:
                    cands = [p for p in open_by_tool.get(s.tool or "", ())
                             if not (s.call_id and self.steps[p].call_id)]
                    if cands:
                        blocked = [p for p in cands if self.steps[p].status == "blocked"]
                        if s.status == "blocked":
                            c = (blocked or cands)[0]
                        else:
                            live = [p for p in cands if p not in blocked]
                            c = live[0] if live else None
                if c is not None:
                    self.result_of[c] = k
                    open_by_tool[self.steps[c].tool or ""].remove(c)
                    cid = self.steps[c].call_id
                    if cid and open_by_id.get(cid) == c:
                        del open_by_id[cid]
        self._events = self._extract(extractors)
        self._flags_cache: dict[int, dict[str, set[str]]] = {}

    # -- calls
    def calls_before(self, upto: int) -> int:
        return bisect_left(self.call_pos, upto)

    def status(self, pos: int, upto: int) -> str | None:
        """Status of the tool_call at ``pos`` as visible before ``upto``."""
        call = self.steps[pos]
        if call.status == "blocked":
            return "blocked"
        r = self.result_of.get(pos)
        if r is None or r >= upto:
            return None
        return self.steps[r].status

    def result(self, pos: int, upto: int) -> Any:
        r = self.result_of.get(pos)
        if r is None or r >= upto or self.steps[r].result is None:
            return MISSING
        return self.steps[r].result

    # -- flags (B5)
    def _extract(self, extractors: Sequence[ExtractorSpec]) -> list[tuple[int, str, str, bool, int, int | None]]:
        by_tool: dict[str, list[ExtractorSpec]] = defaultdict(list)
        for e in extractors:
            by_tool[e.tool].append(e)
        events = []
        for c, r in sorted(self.result_of.items(), key=lambda x: x[1]):
            call, res = self.steps[c], self.steps[r]
            if call.status == "blocked" or res.status != "ok" or not isinstance(res.result, Mapping):
                continue
            for e in by_tool.get(call.tool or "", ()):
                ns, segs = parse_path(e.subject)
                subj = walk(call.args if ns == "args" else res.result, segs)
                if _cls(subj) not in ("str", "num"):
                    continue
                _, vsegs = parse_path(e.result_path)
                on = strict_eq(walk(res.result, vsegs), e.equals)
                events.append((r, e.flag, normalize_subject(subj), on, self.calls_before(r), e.ttl_steps))
        return events

    def flags_at(self, upto: int) -> dict[str, set[str]]:
        got = self._flags_cache.get(upto)
        if got is not None:
            return got
        latest: dict[tuple[str, str], tuple[bool, int, int | None]] = {}
        for r, flag, subj, on, ordinal, ttl in self._events:
            if r >= upto:
                break
            latest[(flag, subj)] = (on, ordinal, ttl)
        n = self.calls_before(upto)
        out: dict[str, set[str]] = {}
        for (flag, subj), (on, ordinal, ttl) in latest.items():
            if on and (ttl is None or n - ordinal < ttl):
                out.setdefault(flag, set()).add(subj)
        self._flags_cache[upto] = out
        return out


# ---------------------------------------------------------------- predicate evaluation


class _Ctx:
    __slots__ = ("args", "bundle", "idx", "prior_args", "prior_result", "text", "upto")

    def __init__(self, bundle: Bundle, idx: TrajectoryIndex, upto: int, pending: TrajectoryStep | None) -> None:
        self.bundle = bundle
        self.idx = idx
        self.upto = upto
        self.args: Any = pending.args if pending is not None else {}
        self.text: str | None = cap_text(pending.text) if pending is not None and pending.text is not None else None
        self.prior_args: Any = MISSING
        self.prior_result: Any = MISSING

    def resolve(self, path: str) -> Any:
        ns, segs = parse_path(path)
        root = {"current.args": self.args, "prior.args": self.prior_args, "prior.result": self.prior_result}.get(ns)
        if root is None or root is MISSING:
            return MISSING
        return walk(root, segs)


def _arg(p: ArgPred, ctx: _Ctx) -> bool:
    v = ctx.resolve(p.path)
    if p.op == "exists":
        return (v is not MISSING) == (p.value is not False)
    if v is MISSING:
        return False
    if p.op == "in":
        return any(strict_eq(v, x) for x in p.value)
    if p.op == "nin":
        c = _cls(v)
        if c == "other":
            return False
        if p.value and not any(_cls(x) == c for x in p.value):
            return False
        return not any(strict_eq(v, x) for x in p.value)
    if p.op == "matches":
        return isinstance(v, str) and ctx.bundle.pattern(p.value).search(cap_text(v)) is not None
    return compare(p.op, v, p.value)


def _prior(p: PriorPred, ctx: _Ctx) -> bool:
    idx, upto = ctx.idx, ctx.upto
    lo = 0
    if p.within is not None:
        n = idx.calls_before(upto)
        lo = idx.call_pos[n - p.within] if n > p.within else 0
    cands = idx.call_pos if p.tool == "*" else idx.by_tool.get(p.tool, ())
    end = bisect_left(cands, upto)
    start = bisect_left(cands, lo)
    for k in range(end - 1, start - 1, -1):
        pos = cands[k]
        st = idx.status(pos, upto)
        if p.status == "any":
            if st == "blocked":
                continue
        elif st != p.status:
            continue
        ctx.prior_args = idx.steps[pos].args
        ctx.prior_result = idx.result(pos, upto)
        try:
            if not all(_same(ctx.resolve(c), ctx.resolve(q)) for c, q in p.same):
                continue
            if not all(compare(c.op, ctx.resolve(c.current), ctx.resolve(c.prior)) for c in p.cmp):
                continue
            if p.where is not None and not _eval(p.where, ctx):
                continue
            return True
        finally:
            ctx.prior_args = MISSING
            ctx.prior_result = MISSING
    return False


def _count(p: CountPred, ctx: _Ctx) -> bool:
    idx, upto = ctx.idx, ctx.upto
    cands = idx.call_pos if p.tool == "*" else idx.by_tool.get(p.tool, ())
    end = bisect_left(cands, upto)
    if p.status == "any":
        n = end
    else:
        n = sum(1 for k in range(end) if idx.status(cands[k], upto) == p.status)
    return _order(p.op, n, p.n) if p.op != "eq" else n == p.n


def _state(p: StatePred, ctx: _Ctx) -> bool:
    subjects = ctx.idx.flags_at(ctx.upto).get(p.flag, set())
    if p.subject is None:
        has = bool(subjects)
    else:
        v = ctx.resolve(p.subject)
        if _cls(v) not in ("str", "num"):
            return False
        has = normalize_subject(v) in subjects
    return has == p.value


def _eval(p: Any, ctx: _Ctx) -> bool:
    if isinstance(p, AllPred):
        return all(_eval(c, ctx) for c in p.of)
    if isinstance(p, AnyPred):
        return any(_eval(c, ctx) for c in p.of)
    if isinstance(p, NotPred):
        return not _eval(p.of, ctx)
    if isinstance(p, ArgPred):
        return _arg(p, ctx)
    if isinstance(p, PriorPred):
        return _prior(p, ctx)
    if isinstance(p, CountPred):
        return _count(p, ctx)
    if isinstance(p, StatePred):
        return _state(p, ctx)
    if isinstance(p, TextPred):
        return ctx.text is not None and ctx.bundle.pattern(p.matches).search(ctx.text) is not None
    raise TypeError(f"unknown predicate {type(p).__name__}")  # closed union: unreachable


def fires(rule: RuleSpec, ctx: _Ctx) -> bool:
    """Fires = ``when`` (default true) AND NOT ``require``."""
    if rule.when is not None and not _eval(rule.when, ctx):
        return False
    return not _eval(rule.require, ctx)


# ---------------------------------------------------------------- public evaluation


def _targets(rule: RuleSpec, on: str, tool: str | None) -> bool:
    if on == "trajectory":
        return rule.target == "*" if tool is None else rule.target == tool
    return rule.target == "*" or rule.target == tool


def _run(bundle: Bundle, idx: TrajectoryIndex, upto: int, pending: TrajectoryStep | None,
         on: str, step_index: int) -> list[Match]:
    tool = pending.tool if pending is not None else None
    rules = [r for r in bundle.rules_for(on) if _targets(r, on, tool)]
    if not rules:
        return []
    ctx = _Ctx(bundle, idx, upto, pending)
    out = []
    for r in rules:
        if fires(r, ctx):
            msg, fix = bundle.render(r)
            out.append(Match(rule=r, message=msg, fix=fix, see=r.see, step_index=step_index))
    return out


_PENDING_KIND = {"tool_call": "tool_call", "response": "response"}


def evaluate(bundle: Bundle, view: GuardView, *, on: On) -> list[Match]:
    """All rules with ``rule.on == on`` that fire for ``view.pending``, ordered (rung, id).

    - ``tool_call``: pending must be a tool_call step; targets ``*`` and ``pending.tool``.
    - ``response``: pending must be a response step; targets ``*`` and ``pending.tool``.
    - ``trajectory`` (R4): pending ``None`` => whole-trajectory rules (target ``*``) over
      ``view.steps``, ``step_index = -1``; pending tool_call => per-step R4 rules targeting it.
    """
    pending = view.pending
    if on in _PENDING_KIND:
        if pending is None or pending.kind != _PENDING_KIND[on]:
            raise ValueError(f"evaluate(on={on!r}) needs a pending {_PENDING_KIND[on]} step")
    elif on == "trajectory":
        if pending is not None and pending.kind != "tool_call":
            raise ValueError("evaluate(on='trajectory') takes pending None or a tool_call step")
    else:
        raise ValueError(f"unknown on {on!r}")
    idx = TrajectoryIndex(view.steps, bundle.extractors)
    step_index = pending.i if pending is not None else TRAJECTORY_END
    return _run(bundle, idx, len(view.steps), pending, on, step_index)


def evaluate_trajectory(bundle: Bundle, steps: Sequence[TrajectoryStep]) -> list[Match]:
    """Replay a completed trajectory: each tool_call step is evaluated as pending (``tool_call``
    rules, then per-step ``trajectory`` rules) with the earlier steps as context; each response
    step against ``response`` rules; finally whole-trajectory rules (target ``*``) once over all
    steps. Ordered by step position, then (rung, id); final R4 matches last."""
    steps = tuple(steps)
    idx = TrajectoryIndex(steps, bundle.extractors)
    out: list[Match] = []
    for k, s in enumerate(steps):
        if s.kind == "tool_call":
            got = _run(bundle, idx, k, s, "tool_call", s.i) + _run(bundle, idx, k, s, "trajectory", s.i)
            out.extend(sorted(got, key=lambda m: _key(m.rule)))
        elif s.kind == "response":
            out.extend(_run(bundle, idx, k, s, "response", s.i))
    out.extend(_run(bundle, idx, len(steps), None, "trajectory", TRAJECTORY_END))
    return out


def _key(r: RuleSpec) -> tuple[int, str]:
    from ci_lab.rules._bundle import rule_order

    return rule_order(r)


def compute_flags(bundle: Bundle, steps: Sequence[TrajectoryStep]) -> dict[str, set[str]]:
    """Derived state flags after ``steps``: flag -> normalized subjects (TTL applied)."""
    return {k: set(v) for k, v in TrajectoryIndex(steps, bundle.extractors).flags_at(len(steps)).items()}


def redact(text: str, matches: Sequence[Match], bundle: Bundle) -> str:
    """Mask spans matched by the ``require`` text predicates of matched ``redact`` rules.

    All spans are computed on the ORIGINAL text (later rules never see earlier masks), merged,
    and replaced by ``[redacted]``. Existing ``[redacted]`` tokens are opaque, so the operation
    is idempotent. Scans the full text (RE2 is linear-time) so nothing past the evaluation cap leaks.
    """
    pats: list[str] = []
    for m in matches:
        if m.rule.action == "redact":
            for p in bundle.redact_patterns(m.rule):
                if p not in pats:
                    pats.append(p)
    if not pats or not text:
        return text
    tokens = []
    start = text.find(REDACTED)
    while start != -1:
        tokens.append((start, start + len(REDACTED)))
        start = text.find(REDACTED, start + len(REDACTED))
    spans: list[tuple[int, int]] = []
    for p in pats:
        for mo in bundle.pattern(p).finditer(text):
            a, b = mo.span()
            if a == b or any(ta <= a and b <= tb for ta, tb in tokens):
                continue
            for ta, tb in tokens:
                if ta < b and a < tb:
                    a, b = min(a, ta), max(b, tb)
            spans.append((a, b))
    if not spans:
        return text
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts, last = [], 0
    for a, b in merged:
        parts.append(text[last:a])
        parts.append(REDACTED)
        last = b
    parts.append(text[last:])
    return "".join(parts)
