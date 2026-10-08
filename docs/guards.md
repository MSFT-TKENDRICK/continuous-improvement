# Guards: runtime trajectory correction (M15)

Guards turn lessons into structure the model cannot bypass (design §13, §13.6, §13.7). Rules are
data (`ci_lab.rulespec.RuleSpec`, YAML), evaluated by the frozen engine `ci_lab.rules` (M14), and
enforced by `ci_lab.guards` inside Microsoft Agent Framework (MAF 1.19) middleware.

## Architecture

| Piece | MAF hook | Responsibility |
|---|---|---|
| `TrajectoryRecorder` | (state) | Per-conversation ordered `TrajectoryStep`s (user, tool calls + args, tool results parsed from JSON text, responses). Builds the closed `GuardView`, with no suite/split/case/env (B2). Mirrored into `AgentSession.state["ci_lab.guards"]` so a checkpointed session resumes with its trajectory. |
| `GuardAgentMiddleware` | `AgentMiddleware` | Opens the conversation, records the user turn, lints the final response (R3 `on: response`), appends the terminal message, and buffers streams. |
| `GuardPreflightMiddleware` | `ChatMiddleware` (after `call_next`) | **Batch preflight (B5).** Sees the model's `function_call` contents before the function loop runs any of them. Evaluates every side-effecting call against one snapshot (plus earlier allowed calls in the same batch) and pins verdicts. |
| `GuardFunctionMiddleware` | `FunctionMiddleware` | Evaluates `on: tool_call` and records each match before acting (B1). On an enforced block, sets `context.result` to `guard_result_json(...)` and skips the tool (B6). Serializes side-effecting tools per conversation with an `asyncio.Lock`. |
| `guarded_stream` | `ResponseStream` | Buffers the entire agent stream and runs R3 lint before releasing the first update. Unbuffered streaming raises `GuardStreamingError`. |
| `GuardRuntime` | n/a | Mode resolution, the decision log/sink, OTel span events, block counters, and the N1 failure policy. |

Evaluation order matches the engine's `(rung, id)` order. Every match is recorded and later rules see
the original text (N6). Redaction is applied once with all redact matches (`rules.redact`).

## Public API (`ci_lab.guards`)

```python
from ci_lab.guards import install_guards, load_guard_bundle, read_decisions
from ci_lab.guards.domains.order_support import (TOOL_POLICIES, load_order_support_bundle,
                                                 verify_identity_tool)

bundle, degraded = load_order_support_bundle()          # rules.load_with_lkg + frozen extractors
middleware = install_guards(bundle=bundle, tool_policies=TOOL_POLICIES, degraded=degraded,
                            sink=run_dir / "guards" / "decisions.jsonl", mode_override=None)
agent = Agent(client=client, tools=[..., verify_identity_tool()], middleware=list(middleware))
response = await agent.run(msg, session=session)        # one AgentSession per conversation
middleware.runtime.decisions                             # list[GuardDecision]
```

`install_guards` takes these keyword arguments:

- `bundle`, `tool_policies`, `sink`, `mode_override`, `degraded`
- `engine` (a test seam)
- `max_blocks_per_turn` (default `MAX_GUARD_BLOCKS_PER_TURN`)
- `max_blocks_per_conversation` (6)
- `default_side_effect` (True)
- `buffer_streams` (True)

`tool_policies` can be `{tool: bool}` or tool_spec mappings that carry `side_effect`. Tools that are not
listed count as side-effecting, which is the conservative default. `wrap_tools` is not needed because
everything happens in middleware.

## Modes (B1)

Mode is resolved with this precedence:

1. The `mode_override` argument.
2. The `CI_GUARDS` env var (`off|shadow|enforce`). It is read only in `install.resolve_mode`.
3. Each rule's `mode`. All seed rules are `mode: shadow`.

| Mode | Effect |
|---|---|
| `shadow` | Records the decision and the span event. Behavior is unchanged. |
| `enforce` + `block` | Returns the guard JSON instead of running the tool, and records the step as `status: "blocked"`. |
| `warn` | The tool runs and its result is untouched. Telemetry only. |
| `off` | Still evaluates and records every attempt, with shadow semantics, so the guard-off arms of paired runs can measure attempted violations. It is stored as `mode: "shadow"`, because `GuardDecision.mode` has no `"off"` (see the change requests). |

## Termination (B4)

`max_blocks_per_turn` blocks are allowed per turn. The next one returns
`guard_result_json(..., terminal=True)` with template `guard.terminal` and raises
`MiddlewareTermination(result=...)`. The MAF function loop then stops gracefully, and the agent
middleware ends the turn with the terminal message. There are no remediation loops.

## Failure policy (N1)

| Where the engine or evaluation fails | Result |
|---|---|
| Side-effecting tool, or preflight | `MiddlewareFailure`: fail closed and abort the run. Sibling tools that are already running are not cancelled, but the lock and preflight mean that none of them are side-effecting. |
| Read-only tool or response | Degraded warn-only: the call proceeds, and a `GuardDecision(rule_id="guards.degraded", degraded=True)` is recorded with `ATTR_GUARD_DEGRADED`. |

If loading the bundle fails, `load_guard_bundle` falls back to LKG (`rules.load_with_lkg`) and sets
`degraded=True`. If no LKG exists, it returns `(None, True)` and logs every
`RuleLoadError.details` problem as `[GUARDS][ERROR] <file>: <violation> Fix: <fix>`.

Engine semantics are in `docs/rules.md` (M14) and are binding. The ones that matter for guards:

- Flags come only from extractors, so always pass `extractor_paths`. `load_order_support_bundle()`
  does this.
- A later `verified: false` revokes the flag. TTL counts `tool_call` steps, including blocked ones.
- `prior status: any` ignores blocked calls.
- Guards evaluate only `tool_call` and `response` online. R4 `trajectory` rules (`target: "*"` →
  `step_index = -1`) are offline-only (`rules.evaluate_trajectory`).

To validate seed rules, run:

```text
python -m ci_lab.rules.cli rules check src/order_support/harness/guards/order_support.yaml --extractors src/ci_lab/guards/domains/order_support/extractors.yaml
```

## Telemetry and metrics

- Every match adds a `contracts.SPAN_GUARD` event on the current span. Its attributes are
  `ATTR_GUARD_RULE/VERSION/MODE/ACTION/ENFORCED/BUNDLE`, plus `DEGRADED` when it applies. The code uses
  the OTel API only and never sets a provider.
- Enforced blocks call `obs.annotate(ATTR_GUARD_ACTION=...)`.
- The sink writes one canonical `GuardDecision` per JSONL line, for example to
  `<run_dir>/guards/decisions.jsonl`. Each decision carries `attempt_digest` and `step_index` for paired
  metrics. Read the file back with `read_decisions(path)`.

## Order-support domain pack (§13.7)

- **`verify_identity(order_id, full_name, email_or_phone)`** is frozen and deterministic, and matches
  against `order_support.data`:
  - Names are compared ignoring case and whitespace.
  - Emails match exactly after casefolding. Phones match on all digits or the last 4.
  - It returns only `{"verified", "order_id"}` and never echoes PII.
- **`extractors.yaml`** sets `identity_verified` from `verify_identity` when `result.verified == true`.
  The subject is `args.order_id` and `ttl_steps` is 50.
- **Seed rules** live in `src/order_support/harness/guards/order_support.yaml`, all `mode: shadow`:

| Rule | What it checks |
|---|---|
| `refund.requires_verified_identity` | A refund needs a verified identity for the same order. |
| `verify.before_lookup` | Anti-gaming: blocks `verify_identity` after a successful `lookup_order` on the same order, so claims can't be copied from tool output. |
| `refund.order_eligible` | The order must be eligible for a refund. |
| `refund.amount_within_total` | `cmp` check that the refund amount is within the order total. |
| `pii.before_verification` | R3 redact of emails, phones, and street addresses (RE2 patterns). |

## Known risks

- R3 redaction rewrites the returned response, not MAF's stored session history or the PII-bearing
  tool results kept in `session.state`. The `prior` predicates need those results.
- The R3 PII state is subjectless, so verifying any order disables redaction for the conversation.
- Response blocks can't be reached through validated rules (`target: "*"` + `block` is rejected). The
  middleware supports them, and this is covered by a test with an injected match.
- Checkpoint resume works through `AgentSession.to_dict/from_dict` (the state mirror). Without a
  session, every `agent.run` is a fresh conversation, so cross-turn verification is lost.

## Installation in order-support (HOOK(M3), wired)

The order-support agent wires these hooks in `src/order_support/guarding.py`, which is called from
`agent._run_turn`:

- **Install once.** Guards are installed once per process with `install_guards(bundle=load_order_support_bundle(...), tool_policies=agent.tool_policies())`.
  - The cache key is the guards dir plus `CI_GUARDS`, `CI_GUARD_DECISIONS` and `CI_RUN_DIR`.
  - The bundle comes from `<harness>/guards` when that directory holds `*.yaml`. Otherwise it comes from the packaged seeds, because a `None` bundle fails closed.
  - The middleware order is `[_TurnGuard, OpenInferenceChatMiddleware, *guards]`.
- **Default is shadow.** With `CI_GUARDS` unset, the seed rules are shadow, so they record decisions but never block.
- **One session per conversation.** Each conversation gets one `AgentSession(session_id="os-<uuid>")`.
  - The key is the bound ASSERT case plus the first user message. A first turn always starts a new session.
  - Earlier turns are fed by a non-storing `TranscriptProvider`, so the model input is byte-identical to the old text-seeded history, while guard state (`identity_verified`, prior lookups) carries across turns.
  - Sessions are in-process only. Persisting `session.to_dict()` across processes is not wired.
- **`verify_identity`.** The frozen `verify_identity` is bound through `tools.TOOLS`, so it gets a TOOL span. It is declared in `harness/agent.yaml`, and `harness/prompts/identity.md` tells the model to call it with the values the customer stated.
- **`side_effect` flags.**
  - `agent.yaml` marks `side_effect: true` on `issue_refund` and `escalate_to_human`.
  - `agent.tool_policies()` ORs these flags with `TOOL_POLICIES`, so a spec can only add side effects.
  - The key is stripped before the spec reaches MAF's `AgentFactory`.
- **Decision sinks.**
  - When `$CI_GUARD_DECISIONS` is set, every decision, `{"kind":"call",...}` record and `{"kind":"opportunity","n":1,"on":"tool_call"|"response",...}` record goes to `<root>/<case_id>/<trial>.jsonl`.
  - Each line is tagged with `case_id`/`trial`, which is the layout `ci_lab.lessons_arm.paired.read_run` reads. Case ids are sanitized to `[A-Za-z0-9._-]` in the path only.
  - The case comes from `CI_CASE_ID`/`CI_TRIAL`. If those are unset, it comes from the ASSERT case bound by `assert_wrapper`. Lines with no known case go to `_unattributed.jsonl`.
  - Without the variable, `$CI_RUN_DIR/guards/decisions.jsonl` gets the decisions only. With neither set, nothing is written.
- **Seeds.** The chat `seed` option is `ORDER_AGENT_SEED`, or a hash of `CI_CASE_ID|CI_TRIAL`. It never depends on `CI_VARIANT`/`CI_GUARDS`, so the guard-off and guard-on arms see the same randomness.

Known limits:
- Two concurrent repeats of the same case with the same first message, inside one ASSERT case run, share a session key.
- `runtime.decisions` and the per-conversation runtime state live for the whole process.
- The Copilot provider ignores `seed`.

The original hook notes are kept below for reference.

```text
# HOOK(M3) src/order_support/agent.py::_run_turn
#   Build guards once per process (not per turn); keep the existing middleware first:
#     bundle, degraded = load_order_support_bundle()
#     guards = install_guards(bundle=bundle, degraded=degraded, tool_policies=<tool_specs or TOOL_POLICIES>,
#                             sink=run_dir / "guards" / "decisions.jsonl")
#     middleware = [_TurnGuard(...), otel.OpenInferenceChatMiddleware(...), *guards]
#   Pass a persistent AgentSession(session_id=<conversation id>) per conversation:
#     agent.run(..., session=session). Today history is text-seeded with no session, so guard state
#     (identity_verified, prior lookups) would reset every turn. Persist session.to_dict() with the
#     conversation if turns cross processes.
# HOOK(M3) src/order_support/maf_tools.py::bindings + harness/agent.yaml
#   Bind verify_identity_tool() (or wrap ci_lab.guards.domains.order_support.verify_identity) as
#   "verify_identity" and declare it in agent.yaml tools. The instructions should tell the model to
#   verify identity with values the CUSTOMER stated, before lookup_order and refunds.
# HOOK(M3) tool_specs: add `side_effect: true` to issue_refund and escalate_to_human (false elsewhere),
#   then pass those specs as tool_policies (the same shape as TOOL_POLICIES).
# HOOK(M3) _configure_client sets allow_concurrent_invocation=False. Preflight still applies, and the
#   per-conversation lock is redundant there but harmless.
# HOOK(M3) streaming: keep buffer_streams=True (the default). Unbuffered streaming raises.
```
