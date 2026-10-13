"""MAF 1.19 guard middleware: pre-tool guards (R2), batch preflight (B5), output lint (R3).

* :class:`GuardFunctionMiddleware` evaluates every tool call before it runs. Matches are recorded
  first (B1); an enforced block short-circuits the tool with canonical guard JSON (B6); after
  ``max_blocks_per_turn`` blocks the next block is terminal and stops the loop (B4).
* :class:`GuardPreflightMiddleware` (chat middleware: runs per model call inside MAF's function
  loop, after the model answered and before any call executes) evaluates all side-effecting calls
  of the batch against one snapshot so their verdicts don't depend on scheduling (B5).
* :class:`GuardAgentMiddleware` scopes the conversation/turn, lints the final text (R3) and
  buffers streams (:mod:`ci_lab.guards.stream`).

Failure policy (N1): evaluation errors on side-effecting tools raise ``MiddlewareFailure`` (the
run aborts); on read-only tools and responses guards degrade to warn-only.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agent_framework import (
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    ChatContext,
    ChatMiddleware,
    Content,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    MiddlewareFailure,
    MiddlewareTermination,
    ResponseStream,
)

from ci_lab import obs
from ci_lab.contracts import ATTR_GUARD_ACTION, ATTR_GUARD_DEGRADED
from ci_lab.guards.runtime import (
    RESPONSE_BLOCKED_TEMPLATE,
    TERMINAL_TEMPLATE,
    ConvState,
    GuardRuntime,
    Verdict,
    render,
)
from ci_lab.guards.stream import GuardStreamingError, guarded_stream
from ci_lab.rulespec import guard_result_json

Next = Callable[[], Awaitable[None]]


def _role(message: Any) -> str:
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role))


def _call_args(content: Any) -> dict[str, Any]:
    try:
        parsed = content.parse_arguments()
    except Exception:  # noqa: BLE001 - malformed model JSON: evaluate with no args (fails requires)
        return {}
    return dict(parsed) if parsed else {}


class GuardFunctionMiddleware(FunctionMiddleware):
    """Pre-tool guard (R2) around every tool invocation."""

    def __init__(self, runtime: GuardRuntime) -> None:
        self.runtime = runtime

    async def process(self, context: FunctionInvocationContext, call_next: Next) -> None:
        rt = self.runtime
        conv = rt.conversation(context.session)
        tool = context.function.name
        side = rt.is_side_effect(tool)
        if side:
            async with conv.side_effect_lock:  # B5: serialized per conversation
                await self._guarded(context, call_next, conv, tool, side=True)
        else:
            await self._guarded(context, call_next, conv, tool, side=False)

    async def _guarded(self, context: FunctionInvocationContext, call_next: Next, conv: ConvState,
                       tool: str, *, side: bool) -> None:
        rt = self.runtime
        call_id = context.metadata.get("call_id")
        if conv.turn.terminal:
            self._terminal(context, conv, tool, call_id, conv.turn.terminal_rule, "")
            return
        verdict = conv.turn.preflight.pop(call_id, None) if side and call_id else None
        if verdict is not None:
            block = verdict.block  # decided (and recorded) against the batch snapshot
        else:
            pending = conv.recorder.pending_call(tool, call_id, context.arguments)
            try:
                matches = rt.evaluate(conv.recorder.view(pending), on="tool_call")
            except Exception as exc:
                rt.record_degraded(conv, pending)
                if side:
                    raise MiddlewareFailure(
                        f"guard evaluation failed for side-effecting tool {tool!r} ({type(exc).__name__})") from exc
                matches = []
            decisions = rt.decide(conv, matches, pending)
            block = rt.first_enforced(matches, decisions, "block")
        if block is not None:
            self._block(context, conv, tool, call_id, block)
            return
        conv.recorder.record_call(tool, call_id, context.arguments)
        try:
            await call_next()
        except (MiddlewareTermination, MiddlewareFailure):
            raise
        except Exception:
            conv.recorder.record_result(tool, call_id, None, error=True)
            raise
        conv.recorder.record_result(tool, call_id, context.result)

    def _block(self, context: FunctionInvocationContext, conv: ConvState, tool: str,
               call_id: str | None, match: Any) -> None:
        rt = self.runtime
        turn = conv.turn
        terminal = (turn.blocks >= rt.max_blocks_per_turn
                    or conv.recorder.blocks_total >= rt.max_blocks_per_conversation)
        turn.blocks += 1
        obs.annotate({ATTR_GUARD_ACTION: "block"})
        if terminal:
            turn.terminal = True
            turn.terminal_rule = match.rule.id
            self._terminal(context, conv, tool, call_id, match.rule.id, match.see)
            raise MiddlewareTermination("guard: block limit reached; terminal response", result=context.result)
        conv.recorder.record_call(tool, call_id, context.arguments, blocked=True)
        context.result = guard_result_json(match.rule.id, match.message, match.fix, match.see)

    def _terminal(self, context: FunctionInvocationContext, conv: ConvState, tool: str,
                  call_id: str | None, rule_id: str, see: str) -> None:
        conv.recorder.record_call(tool, call_id, context.arguments, blocked=True)
        message, fix = render(self.runtime.template(TERMINAL_TEMPLATE), {"rule": rule_id})
        context.result = guard_result_json(rule_id, message, fix, see, terminal=True)


class GuardPreflightMiddleware(ChatMiddleware):
    """Batch preflight (B5): verdicts for all side-effecting calls of one model response, taken
    against the same pre-batch snapshot (plus earlier allowed calls of the batch, in order)."""

    def __init__(self, runtime: GuardRuntime) -> None:
        self.runtime = runtime

    async def process(self, context: ChatContext, call_next: Next) -> None:
        await call_next()
        conv = self.runtime.conversation(getattr(context, "session", None))
        result = context.result
        if isinstance(result, ResponseStream):
            def hook(response: Any) -> None:
                self.preflight(conv, response)
            result.with_result_hook(hook)
        elif result is not None:
            self.preflight(conv, result)

    def preflight(self, conv: ConvState, response: Any) -> None:
        rt = self.runtime
        calls = [c for m in getattr(response, "messages", None) or [] for c in m.contents
                 if c.type == "function_call" and not getattr(c, "informational_only", False)
                 and c.name and rt.is_side_effect(c.name)]
        extra: list[Any] = []
        for c in calls:
            pending = conv.recorder.pending_call(c.name, c.call_id, _call_args(c), offset=len(extra))
            try:
                matches = rt.evaluate(conv.recorder.view(pending, extra), on="tool_call")
            except Exception as exc:
                rt.record_degraded(conv, pending)
                raise MiddlewareFailure(
                    f"guard preflight failed for side-effecting tool {c.name!r} ({type(exc).__name__})") from exc
            decisions = rt.decide(conv, matches, pending)
            block = rt.first_enforced(matches, decisions, "block")
            if c.call_id:
                conv.turn.preflight[c.call_id] = Verdict(pending=pending, block=block)
            if block is None:
                extra.append(pending)


class GuardAgentMiddleware(AgentMiddleware):
    """Run scope: binds the conversation, resets the turn, lints the delivered response (R3)."""

    def __init__(self, runtime: GuardRuntime) -> None:
        self.runtime = runtime

    async def process(self, context: AgentContext, call_next: Next) -> None:
        rt = self.runtime
        if context.stream and not rt.buffer_streams:
            raise GuardStreamingError("guards require buffered streaming (install_guards(buffer_streams=True)) "
                                      "or a non-streaming run: R3 output lint must see the full response")
        conv = rt.start_run(context.session)
        if rt.degraded:
            obs.annotate({ATTR_GUARD_DEGRADED: True})
        for m in context.messages or []:
            if _role(m) == "user":
                conv.recorder.record_user(m.text or "")
        token = rt.bind(conv)
        try:
            await call_next()
        finally:
            rt.unbind(token)
        result = context.result
        if isinstance(result, ResponseStream):
            context.result = guarded_stream(result, bind=lambda: rt.bind(conv), unbind=rt.unbind,
                                            finish=lambda r: self.finish(conv, r))
        elif isinstance(result, AgentResponse):
            context.result = self.finish(conv, result)

    def finish(self, conv: ConvState, response: AgentResponse) -> AgentResponse:
        rt = self.runtime
        if conv.turn.terminal:
            message, _ = render(rt.template(TERMINAL_TEMPLATE), {"rule": conv.turn.terminal_rule})
            response.messages.append(Message(role="assistant", contents=[Content.from_text(message)]))
            conv.recorder.record_response(message)
            return response
        text = "\n".join(t for t in (_texts(m) for m in response.messages if _role(m) == "assistant") if t)
        if not text:
            return response
        pending = conv.recorder.pending_response(text)
        try:
            matches = rt.evaluate(conv.recorder.view(pending), on="response")
        except Exception:  # noqa: BLE001 - responses degrade to warn-only (N1)
            rt.record_degraded(conv, pending)
            matches = []
        decisions = rt.decide(conv, matches, pending)
        block = rt.first_enforced(matches, decisions, "block")
        if block is not None:
            message, _ = render(rt.template(RESPONSE_BLOCKED_TEMPLATE), {"rule": block.rule.id})
            _rewrite(response, lambda _t: "")
            response.messages.append(Message(role="assistant", contents=[Content.from_text(message)]))
            obs.annotate({ATTR_GUARD_ACTION: "block"})
        else:
            redacts = [m for m, d in zip(matches, decisions, strict=True) if d.enforced and d.action == "redact"]
            if redacts:
                try:
                    # N6: every redact rule masks the ORIGINAL text in one idempotent pass.
                    _rewrite(response, lambda t: rt.redact(t, redacts))
                    obs.annotate({ATTR_GUARD_ACTION: "redact"})
                except Exception:  # noqa: BLE001
                    rt.record_degraded(conv, pending)
        conv.recorder.record_response(text)
        return response


def _texts(message: Any) -> str:
    return "".join(c.text or "" for c in message.contents if c.type == "text")


def _rewrite(response: AgentResponse, fn: Callable[[str], str]) -> None:
    """Rewrite assistant text contents in place; drop assistant messages left empty."""
    kept: list[Any] = []
    for m in response.messages:
        if _role(m) == "assistant":
            contents: Sequence[Any] = m.contents
            new = []
            for c in contents:
                if c.type == "text":
                    t = fn(c.text or "")
                    if t:
                        new.append(Content.from_text(t))
                else:
                    new.append(c)
            if not new:
                continue
            m.contents = new
        kept.append(m)
    response.messages[:] = kept
