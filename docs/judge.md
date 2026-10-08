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
In `src/order_support/cli.py`, `cmd_run`, add this line right after `_load_dotenv()`:

```python
from ci_lab.judge.provider import register as _s1_register; _s1_register()
```

Then select the judge model per run, without editing the suite:

```powershell
uv run order-support-evals run evals/assert/judge_replay/eval_config.yaml --override default_model.name=s1/llamacpp/qwen3.5-4b
```

`register()` is idempotent and thread-safe. It adds one `{"provider": "s1", ...}` entry to `litellm.custom_provider_map` and re-runs `litellm.utils.custom_llm_setup()`. `register(force=True)` replaces the handler, and `unregister()` removes it.

To expose the CLI, the integrator adds `"judge"` to `COMMAND_MODULES` in `src/ci_lab/cli.py`.

### Model strings and backends

These are ported from `s1eval` on `origin/main`.

| Model string | Backend | Confidence |
|---|---|---|
| `s1/llamacpp/<name>` (alias `local`) | llama-server `/v1/chat/completions` with `max_tokens=1`, `top_logprobs=40` and thinking off. Each decision is a single code token, and its distribution is renormalised over the valid codes. The decision abstains if the valid probability mass is below 0.5. The URL is `$CI_S1_LLAMA_URL` (default `http://127.0.0.1:8081`). `$CI_S1_CHOICE_PERMUTATIONS` averages over N option orders. | yes (logprobs) |
| `s1/openai_decisions/<model>` (alias `openai`) | OpenAI Responses API with a strict decisions schema. Needs `$OPENAI_API_KEY`. | reported only (`confidence` field) |
| `s1/systemone/<model>`, `s1/<preset>/<model>` | System One `/v1/decide` wire format. Uses `$CI_S1_SYSTEMONE_URL` and `$CI_S1_API_KEY`, or the preset's key env. | yes (distribution) |
| `s1/scripted/<name>` | Offline and deterministic: `default`, `uniform`, or anything registered with `backends.register_script()`. | scripted |

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

Cost: one backend call per custom dimension plus one per taxonomy behaviour. That is about 15 calls per case on `judge_replay`.

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
uv run ci-lab judge provider-check --config evals/assert/judge_replay/eval_config.yaml --model s1/llamacpp/qwen3.5-4b
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
uv run ci-lab judge align --config evals/assert/judge_replay/eval_config.yaml `
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
