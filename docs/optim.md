# `ci_lab.optim` — DSPy LM, GEPA and SkillOpt over harness text

`import ci_lab.optim` (and all submodules) never imports `dspy`, `gepa`, `skillopt_sleep`
or `litellm`; they load on first use.

DSPy is the programmable-prompt layer. ax-llm (the DSPy-style TypeScript library) is not used,
because the harness is Python only.

## Ownership

The authoritative table is `ci_lab.contracts.COMPONENT_OWNERS`:

| Component | Dedicated owner |
|---|---|
| `prompt` | `gepa` |
| `skill` | `skillopt` |
| `guard` | `guard` |
| `agent`, `loop`, `workflow`, `mcp`, `client_tool`, `config`, `context_mgmt`, `memory` | `agl` |

`strategy_may_edit("agent", component)` also permits the general MAF proposer to edit any text
component except `guard`. That broad proposal permission does not transfer dedicated optimization
ownership. In particular, AGL refuses prompts and skills.

## `ci_lab.optim.lm`

`make_lm(profile, purpose="optimizer", *, model=None, cache=None, api_base=None, api_key=None,
fake_answers=None, env=None, **lm_kwargs) -> dspy.LM`

- Routes LiteLLM `openai/<model>` (`engine="litellm"`) to the first endpoint found:
  1. `AGL_OPENAI_BASE_URL` (AGL proxy, C24), for any profile;
  2. `copilot`: `CI_COPILOT_SERVE_URL` (default `http://127.0.0.1:8765/v1`), with the key
     from `CI_COPILOT_SERVE_KEY` or the file named by `CI_COPILOT_SERVE_KEY_FILE`;
  3. `offline`: `OPENAI_API_BASE` / `OPENAI_BASE_URL` (default `http://127.0.0.1:8081/v1`,
     llama-server), with the key from `OPENAI_API_KEY` (default `local`).
- Model: the `model` argument, then `CI_LAB_<PURPOSE>_MODEL`, then the profile default.
- Cache (C19): `cache_enabled(profile, purpose, cache)` is **off** for `copilot`, for
  `fake`, and for `purpose="judge"`, whatever `cache` says. Only offline non-judge calls
  default to on. `disable_dspy_cache()` turns off DSPy's disk and memory caches globally.
- `fake` profile: a deterministic `DummyLM` that returns `fake_answers` verbatim through
  `_RawTextAdapter`, so no network is used. Tests rely on it.
- `lm_usage_tokens(lm)` sums the usage recorded in `lm.history`.

## `ci_lab.optim.scoring`

The evolve-only guard sits between optimizers and evaluators:

- `EvolveScorer` is `(candidate: {target_id: text}, case_ids) -> [CaseOutcome]`. It may be
  sync or async.
- `DomainEvolveScorer(domain, worktree, scratch, ...)` wraps `contracts.Domain`:
  1. copies the worktree to scratch and writes the candidate;
  2. calls `domain.evaluate(..., "evolve", ...)`;
  3. maps the result through `domain.failures` into per-case outcomes.

  Results are cached per candidate hash. Asking for any non-evolve case raises
  `HeldOutAccessError`.
- `EvolveGuard` enforces the evolve allow-list and `MetricBudget`, a hard cap on per-case
  metric calls (`BudgetExhausted`).
- `render_failure(FailureRecord)` produces the typed text used as reflection feedback
  (C20: reflection datasets are built **only** from FailureRecords).

## `ci_lab.optim.targets`

`TextTarget(path, key=None)`: a whole file, or a YAML string at `path#dotted.key`.

`resolve_targets(worktree, focus, ...)` expands each `component_focus` entry:

- component names (`prompt`, `skill`, `memory`) expand through the domain's
  `component_globs` merged over `DEFAULT_COMPONENT_GLOBS`;
- entries containing `/` or `.` are explicit target ids;
- non-text components are skipped.

Targets outside `surface_globs`, or inside `frozen_globs`, raise `TargetError`.

## `ci_lab.optim.gepa`

`optimize_texts(seed, scorer, evolve_cases, *, reflection_lm, budget_tokens=None,
config=GepaConfig(), incumbent_failures=()) -> TextOptimization`

- Runs `gepa.optimize` with a custom `ComponentAdapter(GEPAAdapter)` whose candidate is
  `{target_id: text}`.
- Budget (C18):
  - `max_metric_calls = min(config.max_metric_calls, budget_tokens // tokens_per_metric_call)`;
  - the run is skipped when the cap cannot fit one iteration;
  - a stop callback and `MetricBudget` guarantee the scorer is never called beyond the cap.
- Data: only `evolve_cases`, optionally reduced to a stable sub-split (`max_cases`), are
  split into train/val. Held-out splits are never touched.
- The reflection LM goes through `DspyReflectionLM`. For HTTP (LiteLLM) engines it sends
  `obs.carrier()` as `extra_headers` on each call.
- `TextOptimization` holds `seed`, `best`, `changed`, `seed_score`, `best_score`, `note`
  and `diagnostics`, plus `cost: OptimizerCost` (metric calls and budget, refused calls,
  scorer and reflection tokens, candidates, wall time).
- GEPA's own val acceptance is **diagnostic only**. Promotion is decided by the round's
  gates.

## `ci_lab.optim.skillopt`

`optimize_skill((skill_id, text), scorer, evolve_cases, *, reflection_lm, memory=None,
edit_budget=1, budget_tokens=None, config=SkillOptConfig(), incumbent_failures=())`

- Runs one SkillOpt-Sleep `dream_consolidate(gate_mode="on")` epoch. The dependency and nightly
  workflow are restricted to SkillOpt 0.2.x (`skillopt>=0.2.0,<0.3`).
- Uses a `CliBackend` subclass:
  - `attempt` and `judge` go to the guarded evolve scorer;
  - reflection goes to our LM;
  - TaskRecords carry only evolve case ids and suite/category.
- `fit_cases` picks the largest stable evolve sub-split whose
  `calls_needed(train, val, memory)` fits the metric cap.
- `edit_budget` bounds SkillOpt's edit records per document.
- The SkillOpt gate decides whether `best` differs from `seed`. That gate is still only
  diagnostic for the round.

## Deviations

- The design names `dspy.GEPA`. We use `gepa.optimize` with a custom adapter (§11.2,
  "custom GEPAAdapter") because `dspy.GEPA`'s `DspyAdapter` needs LM-run predictors,
  while our metric is an external harness evaluation. DSPy provides the LM.
- `Domain.evaluate` has no case filter. Each candidate is therefore evaluated on the whole
  evolve split once (cached), and budget is charged per requested case as a proxy.
