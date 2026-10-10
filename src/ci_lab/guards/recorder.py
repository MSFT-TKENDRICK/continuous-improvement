"""Per-conversation trajectory recording and the closed :class:`GuardView` (B2).

The recorder is append-only and authoritative for what guards see: tool calls (typed args), tool
results (structured JSON only; anything else is ``result=None``), blocked calls, and response
digests. User text never enters guards (B3): user steps carry only a digest. No suite / split /
case / env / evaluator metadata is representable here — the view is built solely from steps.

When bound to an ``AgentSession.state`` mapping, every step is mirrored (JSON-compatible) under
:data:`STATE_KEY`, so MAF session serialization (``to_dict``/``from_dict``) resumes guard state.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ci_lab.rulespec import GuardDecision, GuardView, TrajectoryStep, canonical_json

STATE_KEY = "ci_lab.guards"
STATE_VERSION = 1

DecisionSink = Callable[[GuardDecision], None]


def text_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_args(arguments: Any) -> dict[str, Any]:
    """Typed tool args as plain JSON values (Mapping or pydantic model; anything else → {})."""
    if isinstance(arguments, BaseModel):
        arguments = arguments.model_dump(mode="json")
    if not isinstance(arguments, Mapping):
        return {}
    try:
        value = json.loads(canonical_json(dict(arguments)))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def structured_result(raw: Any) -> dict[str, Any] | None:
    """A tool result as a dict when it is structured JSON, else None (never parsed from prose)."""
    if isinstance(raw, Mapping):
        try:
            value = json.loads(canonical_json(dict(raw)))
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    text: str | None
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif getattr(item, "type", None) == "text" and isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
            else:
                return None
        text = "".join(parts)
    else:
        text = getattr(raw, "text", None) if getattr(raw, "type", None) == "text" else None
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class TrajectoryRecorder:
    """Ordered :class:`TrajectoryStep` list for one conversation (session/thread)."""

    def __init__(self, key: str, *, state: MutableMapping[str, Any] | None = None) -> None:
        self.key = key
        self._steps: list[TrajectoryStep] = []
        self._state: MutableMapping[str, Any] | None = None
        self.blocks_total = 0
        if state is not None:
            self.bind_state(state)

    # ------------------------------------------------------------ persistence
    def bind_state(self, state: MutableMapping[str, Any]) -> None:
        """Mirror into ``state`` (an ``AgentSession.state``). Existing mirrored steps win (resume)."""
        self._state = state
        raw = state.get(STATE_KEY)
        if isinstance(raw, Mapping) and raw.get("version") == STATE_VERSION:
            try:
                steps = [TrajectoryStep.model_validate(s) for s in raw.get("steps") or []]
            except ValidationError:
                steps = []  # unreadable mirror: start empty (no flags => more restrictive, fail-safe)
            if [s.i for s in steps] == list(range(len(steps))):
                self._steps = steps
                self.blocks_total = int(raw.get("blocks_total") or 0)
        self._sync()

    def _sync(self) -> None:
        if self._state is not None:
            self._state[STATE_KEY] = {
                "version": STATE_VERSION,
                "steps": [s.model_dump(mode="json", exclude_defaults=True) for s in self._steps],
                "blocks_total": self.blocks_total,
            }

    # ------------------------------------------------------------ reading
    @property
    def steps(self) -> tuple[TrajectoryStep, ...]:
        return tuple(self._steps)

    @property
    def next_index(self) -> int:
        return len(self._steps)

    def view(self, pending: TrajectoryStep | None = None,
             extra: Iterable[TrajectoryStep] = ()) -> GuardView:
        """Closed guard view: recorded steps (+ ``extra`` batch-preflight calls) and ``pending``."""
        return GuardView(steps=(*self._steps, *extra), pending=pending)

    # ------------------------------------------------------------ steps
    def pending_call(self, tool: str, call_id: str | None, args: Any, *, offset: int = 0) -> TrajectoryStep:
        return TrajectoryStep(i=self.next_index + offset, kind="tool_call", tool=tool, call_id=call_id,
                              args=json_args(args))

    def pending_response(self, text: str) -> TrajectoryStep:
        return TrajectoryStep(i=self.next_index, kind="response", text=text, text_digest=text_digest(text))

    def _append(self, **fields: Any) -> TrajectoryStep:
        step = TrajectoryStep(i=self.next_index, **fields)
        self._steps.append(step)
        self._sync()
        return step

    def record_user(self, text: str) -> TrajectoryStep:
        return self._append(kind="user", text_digest=text_digest(text))

    def record_call(self, tool: str, call_id: str | None, args: Any, *, blocked: bool = False) -> TrajectoryStep:
        if blocked:
            self.blocks_total += 1
        return self._append(kind="tool_call", tool=tool, call_id=call_id, args=json_args(args),
                            status="blocked" if blocked else None)

    def record_result(self, tool: str, call_id: str | None, raw: Any, *, error: bool = False) -> TrajectoryStep:
        """``status`` is "error" when the tool raised or returned a structured ``{"error": ...}``
        result, so ``prior{status: ok}`` never matches failures."""
        result = None if error else structured_result(raw)
        failed = error or (result is not None and "error" in result)
        return self._append(kind="tool_result", tool=tool, call_id=call_id, result=result,
                            status="error" if failed else "ok")

    def record_response(self, text: str) -> TrajectoryStep:
        """Responses keep only a digest: session state must not persist response text (PII)."""
        return self._append(kind="response", text_digest=text_digest(text))


class JsonlDecisionSink:
    """Appends each :class:`GuardDecision` as one canonical JSON line (e.g.
    ``<run_dir>/guards/decisions.jsonl``) for paired guard-off/on metrics (B1)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, decision: GuardDecision) -> None:
        line = canonical_json(decision.model_dump(mode="json")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def read_decisions(path: Path | str) -> list[GuardDecision]:
    p = Path(path)
    if not p.exists():
        return []
    return [GuardDecision.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def as_sink(sink: DecisionSink | Path | str | None) -> DecisionSink | None:
    if sink is None or callable(sink):
        return sink
    return JsonlDecisionSink(sink)
