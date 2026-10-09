# Guards: runtime trajectory correction

`ci_lab.guards` evaluates frozen `RuleSpec` YAML with the frozen `ci_lab.rules` engine inside
Microsoft Agent Framework middleware. Rules see a closed `GuardView`; suite, split, case, evaluator,
and environment data are not exposed to the target agent.

## Architecture

| Piece | MAF hook | Responsibility |
|---|---|---|
| `TrajectoryRecorder` | state | Records ordered user, tool, result, and response steps and mirrors them into `AgentSession.state`. |
| `GuardAgentMiddleware` | `AgentMiddleware` | Opens the conversation, records user and terminal response steps, and applies response rules. |
| `GuardPreflightMiddleware` | `ChatMiddleware` | Evaluates a batch of proposed side-effecting calls before any call in the batch runs. |
| `GuardFunctionMiddleware` | `FunctionMiddleware` | Evaluates tool-call rules, records decisions, and skips an enforced blocked call. |
| `guarded_stream` | `ResponseStream` | Buffers the stream so response rules run before output is released. |
| `GuardRuntime` | runtime | Resolves mode, records decisions and span events, and enforces block limits. |

Evaluation order follows `(rung, id)`. Every match is recorded. Redaction is applied once after all
matching redact rules are collected.

## Modes

Mode precedence is: explicit `mode_override`, then `CI_GUARDS=off|shadow|enforce`, then the rule's
own mode.

| Mode | Effect |
|---|---|
| `shadow` | Record the decision and span event without changing behavior. |
| `enforce` | Apply block, warn, or redact actions. |
| `off` | Evaluate and record attempts with non-enforcing semantics for paired experiments. |

Repeated enforced blocks terminate the turn with a bounded guard result. Side-effecting tool or
preflight failures fail closed. Read-only tool and response failures are recorded as degraded and
continue with warn-only behavior.

## Bundles and the harness

The repo-root `harness/guards/` directory is guard-owned, frozen from candidate edits, and excluded from
surface metrics. Campaign candidates cannot alter guard policy. New rule bundles are reviewed
through CODEOWNERS and the frozen-path guardrail.

`install_guards` accepts the loaded bundle, tool side-effect policy, optional decision sink, mode,
and block caps. Unlisted tools are treated as side-effecting. The engine receives extractor paths
explicitly; target output cannot create trusted measurements or flags.

## Telemetry

Every match emits a `contracts.SPAN_GUARD` event with bounded scalar attributes for rule, version,
mode, action, enforcement, bundle, and degraded status. Decision sinks use canonical JSONL records
with attempt digest and step index; raw tool output is not required by the guard metrics.

## Validation

Rule semantics and validation are documented in [rules.md](rules.md). Structural protection of
rules and guard assets is documented in [lint.md](lint.md). The harness injection suite exercises
instruction containment, while runtime guard tests cover preflight, session resume, block limits,
redaction, and degraded behavior.

## Known limits

- Guards are application-layer middleware, not an OS sandbox.
- Buffered streaming is required for response enforcement.
- Session-backed trajectory state persists only as far as the configured `AgentSession` storage.
- Human review remains required for changes to guard policy and frozen rule assets.
