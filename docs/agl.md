# `ci_lab.agl` — Agent Lightning 1.0.2 data plane (M5)

Journal-first rollout recording (design §4, C2). The local append-only journal is the source
of truth; `agl-server` is a best-effort mirror. Built on the AGL **v1** REST API only (no
LitAgent / Trainer / APO — those were removed in 1.0).

```
RolloutScope ──► MirroringJournal ──► FileRolloutJournal (root/<rollout_id>.jsonl)   ← truth
                        │
                        └─(best effort, retried by sync())─► AglClient ──► agl-server (loopback)
export.py ◄── journal: TaskScore / EvalResult (RRSI), SkillOpt TaskRecord dicts, OES metrics
```

## Modules

| Module | Main API |
|---|---|
| `journal.py` | `FileRolloutJournal(root, *, fsync=True)` → `start/event/finish/events` (the `contracts.RolloutJournal` protocol), `append_*` (→ bool), `load(rollout_id)`, `iter_rollouts()`, `rollout_ids()` |
| `client.py` | `AglClient(base_url, key)`: `healthz`, `register_models`, `create_rollout`, `get_rollout`, `patch_state`, `patch_status`, `post_event`, `get_events`, `proxy_base_url`; `AglError`, `AglConflict` |
| `server.py` | `AglServer(model_name, endpoints, *, port=None, startup_timeout=30, log_path=None)` context manager → `.client`, `.base_url`, `.proxy_base_url(...)` |
| `mirror.py` | `MirroringJournal(journal, client=None)` → `sync()`, `import_server_events(key)`; `model_request_data(raw)`, `model_request_recorder(scope=None)` |
| `scope.py` | `RolloutScope(journal, key, input)` (sync + async CM) → `.reward()`, `.score()`, `.record_model_request()`, `.fail()`; `current_rollout` contextvar |
| `export.py` | `expected_keys`, `task_scores`, `eval_result`, `skillopt_task_records`, `oes_metric_values`, `HoldoutViolation` |
| `tracing.py` | `attach_telemetry(provider, processors=None)`: the C28 seam |

## Journal format

One JSONL file per rollout. Each line is
`{"v":1,"kind":"start|event|finish","rollout_id","attempt_id","ts",...}`.

- **start**: carries `key` and `input`, and is written once per attempt.
- **event**: carries `event_id`, `event_type` and `data`.
- **finish**: carries `status`, and is written once per rollout. The first terminal state wins; a conflicting second finish is ignored with a warning.

Writes hold a cross-process lock on `root/.journal.lock` (`msvcrt`/`fcntl`). Events are deduped by the caller-supplied `event_id` across all attempts.

Crash safety:
- An unterminated last line is ignored if it is not valid JSON, or kept if it is.
- The next append starts on a fresh line.

## Event ids and idempotency

`RolloutScope` derives every id as `contracts.op_id(rollout_id, attempt, event_type, name)`.
`name` is a logical name (`reward` → `"reward"`, `score` → the score name, or `name=` for
`record_model_request`) or a per-event-type sequence (`"#0"`, `"#1"`, …). Re-running a scope after a
checkpoint resume therefore journals nothing twice, provided the body is deterministic in its order of events.

How a scope exits:

| Body outcome | Result |
|---|---|
| Completes normally | rollout `succeeded` |
| Raises an `Exception` | `ci.error` `{type}` event, then rollout `failed` |
| `fail()` was called | rollout `failed` |
| Raises a `BaseException` (cancel, Ctrl-C) | rollout left `running`, so a resume can finish it |

## Event types

| `event_type` | `data` |
|---|---|
| `model_request` | AGL proxy field set: `model, model_version, request, response, server{model,endpoint,version}, latency_ms, http_status, status, retry_count, usage, finish_reason`; extras under `ci` (e.g. `ci.served_model`) |
| `reward` | AGL `RewardData`: `{value, message, source, reason}` |
| `ci.score` | `{name, value, **attrs}`. Attrs read by export: `suite, category, rule_ids, violations[{rule_id,severity,detail}], excerpt` |
| `ci.error` | `{type}` (exception class only, no message) |

When mirrored, each event's data also gets `ci_event_id`, which `sync()` uses to detect events the server already has.

### Copilot adapter

`model_request_recorder()` returns a callable for `CopilotChatClient(on_model_request=...)`.

Each dict it receives is normalised by `model_request_data`:
- Proxy field names are used as-is.
- Aliases are also accepted: `served_model`, `endpoint`/`provider`, `error`, `input_tokens`/`output_tokens`, `messages`, `content`.
- An optional `name` key gives the event a stable id.

The event is recorded on the active `current_rollout` scope. Calls made outside any scope are dropped.

## AGL server

`AglServer` runs `sys.executable -m agentlightning.server` with these Hydra overrides:
- `host=127.0.0.1` and a free `port`
- `default_proxy.model_name`, `default_proxy.include_log_probs=False`, and the train/val temperatures
- options that disable Hydra's output and log files

The random key is passed as `key=${oc.env:CI_LAB_AGL_KEY}` through the child's environment, so it is never on a command line or in a log. `__repr__` hides it too.

Lifecycle:
- **Startup**: the server polls `/healthz` until it responds (default limit 30 s), then registers `endpoints` for `model_name`. Startup takes about 6 s on Windows.
- **Port retries**: it retries on a new port only when the process exits early.
- **Stop**: it kills the process tree. On Windows this uses `taskkill /T`, because the venv `python.exe` is a launcher. Elsewhere it uses `killpg`.

REST shapes used (agentlightning 1.0.2):

| Call | Notes |
|---|---|
| `POST /api/models` `[{model, endpoint, version}]` | Upsert |
| `POST /api/rollouts` `[{rollout_id, input, is_train, metadata}]` | Returns the existing rollout unchanged for a known id (409 is also handled) |
| `PATCH /api/rollouts/{id}` `{"status": {state, last_attempt_id}}` | Valid transitions: `queuing→running|failed`, `running→succeeded|failed`. `patch_state` reconciles a 409 by re-reading: it accepts an existing terminal state and advances `queuing→running→succeeded` |
| `POST /api/rollouts/{id}/attempt/{aid}/events` `{event_type, data}` | — |
| `GET /api/rollouts/{id}/events` | Returns only the events of `status.last_attempt_id` |
| Proxy (OpenAI base URL) `/proxy/rollout/{id}/attempt/{aid}/mode/{train\|val}/openai/v1` | — |

## Mirroring and sync

`MirroringJournal` writes to the journal first and then mirrors to the server.

On a mirror failure it:
- logs a warning (never the key);
- marks the rollout dirty;
- goes offline, so later writes don't wait on a dead server.

`sync()` reconciles each dirty rollout in order:
1. Create the rollout on the server if it is missing.
2. Move it to `running`.
3. For each attempt, set `last_attempt_id` and post the events the server doesn't have.
4. Restore the latest attempt and patch the terminal state.

`sync(all_rollouts=True)` replays the whole journal into a fresh server. AGL's store is in-memory, so this is needed after a restart.

`import_server_events(key)` journals `model_request` events that the AGL proxy wrote directly. Their ids are deterministic and `routed_experts` is dropped.

## Exports

**Task scores** (`task_scores(journal, expected_keys(exp, variant, case_ids, k), score_name=None)`):
- Score source: the last `reward` event, or with `score_name` set, the `ci.score` event of that name. The latest attempt that has a match wins.
- Missing rollouts give `score=None`, and inherit their case's suite.
- Also returned: violations taken from the `ci.score` attrs, tokens summed from `usage`, and `served_model`.
- `eval_result(...)` wraps the scores into a `contracts.EvalResult`.

**SkillOpt records** (`skillopt_task_records(journal, keys, split="evolve")`):
- Returns SkillOpt-Sleep `TaskRecord` dicts with id `agl:<case_id>`.
- Raises `HoldoutViolation` for any split other than `evolve`, or for a rollout whose input declares a non-evolve split (C15).
- Uses only typed fields (C12):
  - From the rollout input: `intent`, `context_excerpt`, `reference_kind`, `reference`, `judge`.
  - From `ci.score` attrs: `category`, `rule_ids`, `excerpt`. `excerpt` is dropped for injection suites.
- Model and tool payloads are never read. Cases without an `intent` are skipped.

**OES metrics** (`oes_metric_values(scores)`):
- Returns: `evolve_score` (a missing trial counts as 0), `safety_score`, `critical_unsafe_pass`, `critical_violations`, `cost_tokens_per_task`, `missing_trial_rate`, `n_tasks`, and one `suite.<name>` per suite.
- `safety_score` covers the suites `indirect_prompt_injection`, `refund_authorization` and `identity_verification`.

## Telemetry (design §12.3, C28)

Each `RolloutScope` runs inside one `ci.case` span. This uses `ci_lab.obs`, which is a no-op when no tracer provider is installed.

- **Span attributes**: `agl.rollout_id`, `agl.attempt_id`, `ci.case_id`, `ci.trial`, `oes.experiment_id`, `oes.variant` and `ci.split`. A `reward` named `"reward"` also sets `ci.score`.
- **Status**: the span ends with status ERROR if the rollout fails.
- **Existing span**: if the caller (for example the ASSERT runner) has already opened a `ci.case` span, the scope adds its attributes to that span instead of nesting a second one.
- **Context**: the span is the current OTel context inside the scope, so async tasks and MAF GenAI spans become its children.

This module never calls `set_tracer_provider`. `ci_lab.telemetry.setup` (M12) owns the global provider.

agentlightning 1.0.2 has no tracer provider of its own. If an AGL component ever gets one, call `attach_telemetry(provider)` on it. It adds `ci_lab.telemetry.span_processors()` to that provider once per provider, and does nothing until M12 provides that function.

`AglServer` is a long-lived service, so it removes `TRACEPARENT`/`TRACESTATE` from its child process's environment. `AglClient` sends `obs.carrier()` W3C headers on every request instead (design §12.5).

Callers that send OpenAI requests through `proxy_base_url(...)` should also attach `obs.carrier()` headers to each request.

When the scope reuses an existing `ci.case` span, it sets its attributes with `obs.annotate`. Span attributes are bounded scalars, and only the exception type is recorded.

## Limitations

- The AGL store is in-memory. Use `sync(all_rollouts=True)` to rebuild it from the journal.
- `GET /events` only returns the events of the last attempt. `sync()` switches `last_attempt_id` per attempt to work around this.
- Sequence-numbered event ids assume a deterministic event order within a scope. Use explicit `name=`s when events are recorded concurrently.
- `RolloutHooks` are not used (C2). Scoring is explicit via `RolloutScope`.
