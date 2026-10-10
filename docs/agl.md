# `ci_lab.agl` — Agent Lightning journal and structural optimizer

`ci_lab.agl` uses Agent Lightning 1.0.2 only as a rollout journal/store substrate. The
implementation is CPU-only and LLM-only:

- it deliberately never imports or uses `verl`;
- it does not implement PPO, RL training, or a trainer;
- it needs no GPU;
- its structural proposal model is created through the existing
  `ChatClientFactory` with `Profile.COPILOT`, purpose `optimizer`;
- DSPy/GEPA exclusively owns prompt optimization;
- SkillOpt exclusively owns skill optimization.

AGL owns the structural components `agent`, `loop`, `workflow`, `mcp`, `client_tool`,
`config`, `context_mgmt`, and `memory`. It refuses `prompt`, `skill`, and `guard`.

```text
scored run
  └─ RolloutScope
       ├─ FileRolloutJournal ── best-effort mirror ── AGL 1.0.2 server/store
       ├─ deterministic ci.metric
       └─ typed rollout digest
            ├─ LLM credit assignment ── deterministic heuristic fallback
            └─ LlmResourceAlgorithm ── one contained structural Edit
```

## Modules

| Module | Responsibility |
|---|---|
| `journal.py` | Append-only JSONL source of truth with event-id deduplication |
| `scope.py` | Rollout lifecycle, model/tool events, scores, reward, and final metric event |
| `metrics.py` | Pure `rollout_metrics(events)` aggregation |
| `credit.py` | Bounded `RolloutDigest`, `Credit`, strict LLM credit assignment, fallback, journaling |
| `algorithm.py` | `LlmResourceAlgorithm`, structural ownership and frozen-manifest containment |
| `client.py` | Thin synchronous AGL 1.0.2 REST client |
| `server.py` | Loopback `agl-server` process manager |
| `mirror.py` | Journal-first best-effort server mirroring and model-request normalization |
| `export.py` | `TaskScore`, `EvalResult`, SkillOpt records, and OES metrics |
| `tracing.py` | Optional telemetry processor attachment |

## Journal and idempotency

Each rollout has one `root/<rollout_id>.jsonl` file. Records are `start`, `event`, or
`finish`. Writes hold a cross-process lock and tolerate a truncated crash tail.

Every event has a deterministic caller-supplied id. `RolloutScope` derives ids with
`contracts.op_id(rollout_id, attempt_id, event_type, logical_name)`. Replaying the same
attempt therefore keeps the first logical event and does not duplicate it. The first
terminal finish also wins.

The important event types are:

| Event | Typed data |
|---|---|
| `model_request` | Model metadata, latency, status, and token usage |
| `ci.tool_call` | Tool name and optional runtime delta; never arguments or output |
| `ci.score` | Named evaluator score and typed rule/violation metadata |
| `ci.metric` | `wall_ms`, `llm_calls`, `tool_calls`, `tokens_in`, `tokens_out` |
| `ci.credit` | Component, bounded weight, reason code, evidence ids |
| `reward` | AGL-compatible telemetry reward; ASSERT remains authoritative |

### Runtime metrics

`rollout_metrics(events)` is pure and deterministic. It:

- counts and sums `model_request` latency and token usage;
- aggregates typed `ci.runtime`, `ci.usage`, `ci.llm_call`, and `ci.tool_call` deltas;
- treats `ci.score` and `ci.metric` as summaries whose valid fields override earlier
  values rather than being added.

At a normal or failed finish, `RolloutScope` measures elapsed body time, aggregates the
journal events, and emits one `ci.metric` named `finish`. Its event id is deterministic,
so duplicate finish/replay cannot create a second logical metric event. Interrupted
`BaseException` exits remain running and emit no finish metric.

AGL 1.0.2 server configuration has no rollout-hooks loading seam. There is no
`CiRolloutHooks` and `AglServer` does not attempt hooks wiring; metrics are journal-side.

## Typed credit assignment

`RolloutDigest` contains only:

- case and suite;
- quality score;
- violation rule ids;
- per-component touch counts;
- runtime/token metrics.

It cannot contain raw tool output. String lengths, rule counts, component names, metric
names, scores, and counts are bounded and validated.

`assign_credit(digests, *, client, components)` sends one batch request with a strict JSON
schema. It retries malformed output once. If both attempts fail, a frozen deterministic
fallback:

1. maps explicit rule-id prefixes to components;
2. attributes call-count outliers to `loop`;
3. attributes token outliers to `agent`;
4. uses typed component touches for otherwise unexplained low scores.

`journal_credits` writes typed `ci.credit` events with deterministic ids.

## Structural algorithm

`LlmResourceAlgorithm` implements the ordinary `ArmStrategy.propose` API.

1. Read terminal rollouts from the supplied journal/store and convert them to bounded
   digests.
2. Assign credit with the optimizer chat client.
3. Restrict candidates with `strategy_may_edit("agl", component)`.
4. Read component files using `HarnessTree.component_globs()` from the frozen manifest.
5. Ask for exactly one `replace` or `delete` operation on one existing allowed file.
6. Prefer deletion/tightening when cost credit points at calls, steps, tools, or agents.
7. Enforce path containment, response schema, content and changed-line bounds.
8. Re-run `HarnessTree.validate()` and restore the original bytes if validation fails.
9. Commit exactly that file and return one typed `contracts.Edit`.

The strategy writes only inside the supplied arm worktree and explicit harness directory.
The frozen `harness/harness.yaml`, `src/**`, `evals/**`, governance, and every path outside
the selected component glob are unavailable to the proposal.

The frozen arm workflow is `src/ci_lab/workflows/arm_agl.yaml`; candidate target workflows
remain under `harness/workflows/`.

## Server and mirroring

`AglServer` launches `python -m agentlightning.server` on loopback with a random API key.
The key is passed through the child environment, never the command line. The server is a
best-effort store/proxy; the local journal remains authoritative.

`MirroringJournal` writes locally first. On a server failure it marks the rollout dirty,
goes offline, and lets the scored run continue. `sync()` later recreates missing rollouts,
posts missing events using their deterministic ids, and reconciles terminal state.

## Exports

`task_scores()` reads the latest scored attempt. Runtime fields come from `ci.metric`,
then `ci.score` attributes, then model-request latency/count fallback. Token usage and the
served model are preserved. Missing trials remain `score=None`.

`oes_metric_values()` includes score/safety metrics plus:

- `cost_tokens_per_task`;
- `wall_ms_per_task`;
- `llm_calls_per_task`;
- `tool_calls_per_task`.

`skillopt_task_records()` remains evolve-only and consumes typed fields. This export does
not give AGL ownership of skills: SkillOpt remains the only skill optimizer.

## Limitations

- The AGL store is in-memory; rebuild it with `sync(all_rollouts=True)`.
- The server event API exposes the last attempt, so sync temporarily selects each attempt.
- Sequence ids require deterministic event order; concurrent producers should pass names.
- Copilot sampling has no seed/temperature guarantee. Determinism comes from typed
  validation, the journal, heuristic fallback, and paired evaluation.
