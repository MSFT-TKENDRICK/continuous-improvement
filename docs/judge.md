# `ci_lab.judge` — System-1 judge provider, agreement audit, rubric alignment

`ci_lab.judge` makes ASSERT's LLM judge auditable and improvable without changing ASSERT:

| Part | Module | Design |
|---|---|---|
| `s1/<backend>/<model>` LiteLLM provider that answers ASSERT's judge call with constrained categorical decisions | `provider.py`, `assert_contract.py`, `backends.py`, `s1types.py` | C21, C25 |
| Judge-vs-human agreement audit; low-agreement dimensions become `diagnostic` | `audit.py` | C21 |
| DSPy rubric alignment that writes an evaluator-experiment *proposal* and never adopts it | `align.py` | C17, C19, C26 |
| `ci-lab judge audit \| align \| provider-check` | `cli.py` | — |

`audit` and `align` run inside `obs.span(SPAN_EVALUATOR_EXPERIMENT, {ci.purpose: judge_audit|judge_align, ...})`.
They use only `obs.span` and `obs.annotate`, with scalar attributes (obs v2.3.1).

## 1. The `s1` provider

### Integration hook (ASSERT wrapper)

ASSERT calls the judge through LiteLLM inside the wrapper process, so registering the provider there is enough.
`ci_lab.domain.harness_assert_wrapper.install()` registers `ci_lab.judge.provider` before ASSERT starts and accepts only `harness_*` suite directories.

Every suite under `evals/assert/` pins `pipeline.judge.model.name: s1/llamacpp/qwen3.5-4b`, so ASSERT judges through the System-1 provider by default (a test in `tests/integration/test_reallib_assert.py` enforces this). To pick another System-1 backend for one run, without editing the suite:

```powershell
uv run python -m ci_lab.domain.harness_assert_wrapper run evals/assert/harness_triage/eval_config.yaml --override pipeline.judge.model.name=s1/scripted/default
```

`register()` is idempotent and thread-safe. It adds one `{"provider": "s1", ...}` entry to `litellm.custom_provider_map` and re-runs `litellm.utils.custom_llm_setup()`. `register(force=True)` replaces the handler, and `unregister()` removes it.

The CLI is `ci-lab judge` (registered in `COMMAND_MODULES`).

### Model strings and backends

These are ported from `s1eval` on `origin/main`.

| Model string | Backend | Confidence |
|---|---|---|
| `s1/llamacpp/<name>` (alias `local`) | llama-server `/v1/chat/completions` with `max_tokens=1`, `top_logprobs=40` and thinking off. Each decision is a single code token, and its distribution is renormalised over the valid codes. The decision abstains if the valid probability mass is below 0.5. The URL is `$CI_S1_LLAMA_URL` (default `http://127.0.0.1:8081`). `$CI_S1_CHOICE_PERMUTATIONS` averages over N option orders. | yes (logprobs) |
| `s1/openai_decisions/<model>` (alias `openai`) | OpenAI Decisions API (`POST /v1/decisions`). Needs `$OPENAI_API_KEY`. | reported only (`confidence` field) |
| `s1/systemone/<model>`, `s1/<preset>/<model>` | TypeSafe System One wire format (`POST /v1/systemone`). The URL is `$CI_S1_SYSTEMONE_URL`, else the preset's. Presets: `typesafe` (`https://api.typesafe.ai`, model `jev-latest`, key `$TYPESAFE_API_KEY`) and `openrouter` (`https://openrouter.ai/api`, model `~typesafe/jev-latest`, key `$OPENROUTER_API_KEY`); plain `systemone` uses `$CI_S1_API_KEY`. | yes (distribution) |
| `s1/scripted/<name>` | Offline and deterministic: `default`, `uniform`, or anything registered with `backends.register_script()`. | scripted |

### Host-wide admission queue (local llama.cpp)

Parallel harness ASSERT suites, campaign arms, and `ci-lab judge` all judge through one llama-server. `ci_lab.judge.admission` makes them queue for it host-wide instead of piling requests onto the server:

- **Slot leases.** A caller holds one of `capacity` leases while it talks to the server. Leases are OS file locks (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX) on `<lock root>/s1-<sha256(url)[:12]>/slot-<i>.lock`, keyed by the normalised server URL, so the OS releases them if the holder dies.
- **Where the wait happens.** the harness ASSERT wrapper takes the lease around ASSERT's `_single_judge_call`, *before* ASSERT's per-call timeout starts, and holds it across ASSERT's retries of that call. `LlamaCppLogprobBackend.decide` takes a lease itself only when its caller doesn't already hold one (the lease is a context variable, so nested acquisition is reentrant and `asyncio.to_thread` shares it). This covers campaigns, `provider-check` and other non-ASSERT callers.
- **Fairness.** Waiters drop a locked ticket file in `queue/` and only the `capacity` oldest live tickets compete for a free slot, so service is FIFO within windows of `capacity`. Tickets of dead waiters are reaped. Waiters poll with jittered backoff (50 ms to 1 s).
- **Observability.** Waits are logged at INFO every 30 s. The lease's wait, slot and capacity are added to the current OTel span as `ci.s1.queue_wait_s`, `ci.s1.slot` and `ci.s1.capacity`.

| Variable | Effect |
|---|---|
| `CI_S1_MAX_INFLIGHT` | Leases per server. `0` disables admission. Default: `min(/props total_slots, 1)`, or 1 if `/props` is unreachable. Every process sharing a server must resolve the same value. |
| `CI_S1_LOCK_DIR` | Lock root. Default `%LOCALAPPDATA%\ci-lab\locks` on Windows, `${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-~/.cache}}/ci-lab/locks` on POSIX. Tests point it at a temporary directory. |
| `CI_S1_PIN_SLOT` | `1` sends llama.cpp `id_slot=<lease index>` with each request. Off by default (see below). |
| `CI_S1_ADMISSION_LOG` | Directory. Each process appends one JSON line per lease (`wait_s` = queue time, `held_s` = service time, slot, capacity) to `admission-<pid>.jsonl`. `run-suites` sets it per suite. |
| `ORDER_EVALS_JUDGE_TIMEOUT_S` / `--judge-timeout` | ASSERT's per-call budget for local s1 judges, measured from lease acquisition. Default max(1200 s, ASSERT's 300 s or `--model-timeout`). |
| `CI_ASSERT_CASE_TIMEOUT_S` | Wall clock for one campaign case subprocess, which includes queue wait. Default 1800 s; `0` disables it. Raise it when campaigns share the judge with long `run-suites` runs. |

**Why one call in flight.** These figures come from four real ASSERT judge calls, replayed against the live llama-server (Qwen3.5-4B Q4_K_M, `-np 4 -t 11`, CPU only). Each call was nine ~1.6K-token decisions.

| In flight | Calls/min | Per-call service time | Prefill tok/s |
|---|---|---|---|
| 1 (pinned) | 0.61 | 75–153 s | 56 |
| 2 (pinned) | 0.41 | 288–297 s | 38 |
| 4 (pinned) | 0.53 | 437–451 s | 49 |
| 4 (unpinned) | 0.65 | 355–368 s | 60 |

The work is prefill-bound on the CPU, so throughput stays flat (other jobs on the box add noise) while each call's latency grows with the number in flight. At 4 in flight a single call exceeds ASSERT's default 300 s timeout even with no queue at all. Keep `-np` at 2–4 if other clients need their own slots and prompt caches, but leave `CI_S1_MAX_INFLIGHT` at 1. A GPU server can use more.

**`id_slot`.** llama-server accepts `id_slot` on `/v1/chat/completions`: a pinned request reuses that slot's cache, while another slot starts cold. Its own similarity-based slot choice, however, already hit the prompt cache on every unpinned request measured (36/36). With one lease, pinning would send every suite to slot 0, where interleaved suites would evict each other's cached prefix. Pinning is therefore opt-in.

### What it returns

The provider reads ASSERT's request from LiteLLM: the system prompt with the taxonomy JSON and dimension rubrics, the user transcript, and the strict `response_format` json_schema. It reproduces the response schema exactly. The output passes `assert_ai.core.judge.has_successful_judge_verdict` and `normalize_transcript_judge_verdict`, and the tests run it through ASSERT's real `_single_judge_call`.

Following C21, every judgement is a categorical decision:

- **Boolean dimensions** become a `noul` question, which returns P(true). The value is `P(true) >= 0.5`, and confidence is `|2p − 1|`.
- **Ordinal and nominal scales** become a `choice` question with one option per scale point, using the scale labels as option text. Nullable dimensions get an extra `not_applicable` option, which sets `dimension_applicability`.
- **Taxonomy behaviours** become a `choice` question with the options `not_relevant`, `satisfied` and `violated`. `node_judgments` keeps every relevant behaviour, with `high`/`medium`/`low` confidence taken from P(violated | relevant).
- **The built-in `policy_violation` and `overrefusal`** are derived from the node judgments in the same way ASSERT's normaliser enforces them.

Probabilities and confidence appear in three places:

- the justification strings
- `response._hidden_params["s1"]`
- the optional JSONL sidecar at `$CI_S1_JUDGE_LOG`

`$CI_S1_RUBRICS` points to a YAML or JSON file with a `{rubrics: {dimension: text}}` mapping, for example align's `candidate_rubrics.yaml`. The rubrics in that file override the request's rubrics, which lets you run an evaluator experiment without editing the suite.

Cost is one backend call per custom dimension plus one per taxonomy behavior in the selected harness suite.

### Fallback (C25)

In these cases the handler re-issues the original messages and params with `litellm.(a)completion(model=$CI_S1_FALLBACK_MODEL or "openai/local")`:

- the request is not a transcript-judge call (no strict schema, or an unknown schema shape)
- a decision abstains
- the backend errors
- the assembled verdict would violate the schema

When a fallback happens, `_hidden_params["s1_fallback"]` records the reason, and `handler.stats` counts `s1` and `fallback` calls. An `s1/` fallback model is refused, so the handler can't loop back on itself.

### Self-test

```powershell
uv run ci-lab judge provider-check                     # built-in mini contract, scripted backend
uv run ci-lab judge provider-check --config evals/assert/harness_triage/eval_config.yaml --model s1/llamacpp/qwen3.5-4b
```

The self-test makes one ASSERT judge call. By default the fallback is disabled, so the check stays offline and fails fast with its reason. Use `--allow-fallback` to permit the network fallback.

## 2. Audit — `ci-lab judge audit`

```powershell
uv run ci-lab judge audit --labels labels.jsonl --judge runs/<run>/judge `
    --map grounded=!ungrounded_claim --floor 0.6 --min-n 10 --out audit.json
```

- **Labels** are JSONL rows of `{case_id, dimension, label}`. A `skip`, `ambiguous` or empty label is excluded.
- **Judge inputs** are ASSERT `scores.jsonl` files or directories, which are searched recursively. Rows can be ASSERT rows (`test_case_id`, `verdict.dimensions`, `dimension_scales`), flat `{case_id, dimensions}` rows, or long `{case_id, dimension, value}` rows. When there are multiple trials, the modal value is used.
- **`--map human=judge`** renames a dimension. `!judge` inverts a boolean.

The report gives, for each dimension:

- n, missing and skipped counts
- exact agreement
- ±1 agreement (ordinal only)
- Cohen's κ
- quadratic-weighted κ (ordinal scales with 3 or more points)
- Spearman ρ
- seeded paired-bootstrap 95% CIs

The headline metric is QWK for ordinal scales with 3 or more points, and κ otherwise. A dimension is `primary` only if n ≥ `--min-n` and the metric is defined and ≥ `--floor`. Otherwise it is `diagnostic` (C21): report it, but don't let it gate decisions. `--strict` exits 1 if any dimension is diagnostic. The implementation is pure Python.

## 3. Align — `ci-lab judge align`

```powershell
uv run ci-lab judge align --config evals/assert/harness_triage/eval_config.yaml `
    --labels labels.jsonl --transcripts transcripts.jsonl --out-dir runs/align1 `
    --lm openai/local --api-base http://127.0.0.1:8081/v1 --map grounded=!ungrounded_claim
```

DSPy is imported lazily (C26), and both its disk and memory caches are disabled (C19). For each labelled dimension, align runs these steps:

1. **Split** the labelled cases into a seeded held-out set (`--heldout`, default 0.3) and a training set divided into `--k-folds` folds.
2. **Propose** `--candidates` rewritten rubrics from the baseline's training-set disagreements.
3. **Select** a candidate only if its mean cross-validated exact agreement beats the baseline by at least `--min-gain` *and* it wins at least half the folds.
4. **Confirm** the selected candidate on the held-out cases, using audit metrics with bootstrap CIs.
5. **Run metamorphic probes**, which check that labels stay stable when they shouldn't change:
   - reversed option order
   - an injected "grader note" in the transcript
   - a paraphrased rubric

   A candidate fails if its flip rate exceeds the baseline's by more than `--probe-tolerance`.

Align writes four outputs to `--out-dir`:

- `candidate_rubrics.yaml`: schema `ci-lab.judge-rubrics/1`. `rubrics` holds only the changed dimensions, and `all_rubrics` holds every dimension. It can be used directly as `$CI_S1_RUBRICS`.
- `candidate_eval_config.yaml`: the suite config with the candidate rubrics applied and relative paths made absolute.
- `align_report.json`: per-dimension CV, held-out and probe evidence.
- `evaluator_proposal.json`: `kind: evaluator_experiment`, with these fields:
  - `adopt: false` and `decision: pending`
  - `requires_new_epoch: true`
  - `baseline_pin` and `candidate_pin`, which are `contracts.EvaluatorPin` values. The candidate tree is `<base>+rubrics-sha256:<hash>`.
  - `recommendation`: `run_experiment` or `no_change`
  - the evidence

**Align never adopts.** Per C17, a new evaluator pin means a new campaign epoch, so a human or the campaign layer has to accept the proposal and start that epoch.

## Caveats

- The provider depends on LiteLLM's `CustomLLM` and `custom_provider_map` internals, and on ASSERT's prompt layout. The private `assert_ai` helpers (`_build_judge_request`, `_single_judge_call`, `_transcript_from_dict`) are used only in tests and in `provider-check`.
- Citations are best-effort: they quote the opening of the final assistant message. `narrative` is a fixed statement.
- s1eval's llamacpp `conformance()` pre-flight check was not ported.
- Align quality depends on the label count. Below about 30 labels per dimension, expect `no_change` or wide CIs.
