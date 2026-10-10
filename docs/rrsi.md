# RRSI core (`ci_lab.rrsi`)

This package is a pure implementation of Algorithm 1 (the schedule) and Algorithm 2 (selection) from RRSI, extended with the design v2 §9 safeguards C8, C11 and C16.

Nothing in it calls models, git or a clock. The only I/O is the JSON/JSONL helpers in `history.py` and `readjudicate.py`. The callers (the round runner and the CLI) own evaluation, drafting, and git. The campaign driver reaches `plan_round`, `select`, `stats.aa_delta` and `stats.confirm_test` through the adapters in `ci_lab.campaign.rrsi_wiring` (copilot and offline profiles; see [campaign.md](campaign.md)).

## Modules

| Module | Public API |
|---|---|
| `params` | `Hyperparams` (frozen; `with_`, `to_dict`, `from_dict`), `PROFILES` (`smoke`, `local`, `paper`, `harness`), `HARNESS_STRATEGIES`, `profile(name, **overrides)`, `TABLE5`, `paper_reference(domain)` |
| `schedule` | `edit_budget(t, T, b_min, b_max)`, `stall_flag(traj, t, w, delta)`, `untried(history)`, `component_yield`, `prune_set(history, t, n_prune)`, `exploration_slots`, `plan_round(t, hp, history, trajectory, delta, arms=None, *, experiment_id=None)` → `RoundSchedule` (`.directives` rich, `.arm_directives` contract), `directives(...)` → `tuple[contracts.ArmDirective, ...]` |
| `strategies` | `strategy_stats(history)`, `starved(stats, t, K)`, `allocate_strategies(t, n_arms, history, hp)` → `StrategyAllocation` (strategy + reason + Thompson draws per slot) |
| `history` | `HistoryRecord` (round, arm, edits, score, cost, delta_s, delta_c, accepted = *a*, novelty, admissible, reasons), `read_jsonl`, `append_jsonl` (rejects a duplicate (round, arm)), `write_jsonl` (atomic), `accepted_counts`, `tried_components`, `before`, `replace_round` |
| `stats` | `paired_bootstrap`, `bootstrap_means`, `mean_missing_zero`, `task_score`, `missing_rate`, `mean_cost`, `mean_calls`, `median_wall_ms`, `runtime_observed`, `aa_delta`, `aa_repeats_for_precision`, `confirm_test`, `non_inferiority`, `derive_seed` |
| `selection` | `SelectionInputs`, `select(inputs, guard=None)` → `SelectionDecision` (`decision` ∈ ship / do_not_ship / rerun, a per-arm `ArmTrace` rule trace, `to_dict()` that is JSON-safe) |
| `frontier` | `Frontier`, `initial`, `advance(frontier, decision, arms)`, `pause`, `resume`, `finish`, `rollback` |
| `attribution` | `novelty(edits, accepted)`, `attribute(decision, arms)` → history records (*a* = 1 only for the winner's edits), `component_stats(history)` |
| `readjudicate` | `save_round`, `load_round`, `round_record`, `readjudicate(source, hp=None)` → `Readjudication`, `verify(source)` |

## Profiles

All profiles share the paper's coding-domain weights: w_s = 0, w_c = 15, w_n = 0.5, β0 = 0.10, β1 = 44.5.

| | T | k | N | b_min..b_max | w | m_draft | n_prune | M | n_bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 3 | 1 | 2 | 1..2 | 1 | 1 | 2 | 12 | 2 000 |
| local | 8 | 2 | 2 | 1..3 | 2 | 1 | 3 | 40 | 10 000 |
| paper | Table 5 (coding) | | | | | | | | 10 000 |

- In every profile, δ is `None` because it is calibrated by A/A. `paper_reference(domain)` carries the paper's fixed δ for comparison only.
- The smoke and local window and budget values are our own choices and do not come from the paper.
- `harness` (self-hosted harness target) copies `local` and turns on resource regularization: w_x = 0.5, `x_cap` = 0.10, `calls_cap` = 0.15, `wall_cap` = None, `require_resource_metrics` = True (see [Resource regularization](#resource-regularization)). Its strategies are `HARNESS_STRATEGIES`, computed at import as `agent`, `gepa`, `skillopt`, `agl` filtered by `contracts.STRATEGIES`, so `agl` joins automatically once it is registered.
- `smoke`, `local` and `paper` keep every resource knob off (w_x = 0, no caps), so their decisions are paper-faithful. `DEFAULT_STRATEGIES` excludes the opt-in `guard` and `agl` strategies.

## Semantics

### Scores and cost

- **Score.** Ŝ is the mean of per-case means, with a missing trial counted as 0.
  - When `cases` is supplied, the trial universe is fixed at `cases × k`; otherwise it is the union of observed (case, trial) keys.
- **Cost.** C is the mean of tokens_in + tokens_out over completed trials. ΔC = (C′ − C_t) / C_t.
  - If C_t = 0, the arm is inadmissible (`cost_baseline_zero`).

### Selection (Algorithm 2)

Each arm is compared with the *contemporaneous* incumbent evaluation. Pairs are matched by case and trial, and nothing is cached across rounds.

- **ΔS > δ: cost rule.** The arm must satisfy ΔC ≤ β0 + β1·ΔS. With `require_ci_lower` on (C16), it must also satisfy CI_lower(ΔS) > 0.
- **ΔS ≤ δ: weighted rule.** The arm must satisfy w_s·ΔS − w_c·ΔC + w_n·ν > 0 (strict, so a tie at 0 is rejected).
  - ν is the number of structural components (`client_tool`, `skill`, `memory`) the arm touches that have never been *accepted* before.
- **Guards.** These are non-compensatory: no score gain can offset them. Each one makes the arm inadmissible:
  - more critical safety violations than the incumbent (C11);
  - a critic rejection;
  - `base_commit` ≠ the incumbent commit;
  - a failed optional domain guard.
- **Floor.** S′ ≥ S* − δ.
- **Round-level failures.** Each of these sets the decision to `rerun`, and nothing changes:
  - the incumbent's missing-trial rate is > `missing_invalid_frac` (10%, strict);
  - the evaluator pin or split does not match;
  - there are no trials.
- **Winner.** The winner is the argmax of S′ over admissible arms. Ties go to lower cost, then (only when the profile is resource-aware, i.e. w_x > 0, a cap is set or `require_resource_metrics` is on) to lower surface complexity, then to the higher CI lower bound, then to the lexicographically smallest arm name.
- **S\* update.** S* ← max(S*, S_{t+1}), so S* never decreases.
  - On `do_not_ship`, S_{t+1} is the re-measured incumbent score.
- **Seeds.** Each arm's bootstrap seed is `derive_seed(hp.seed, round, arm)`, so a decision is a pure function of the stored inputs.

### Resource regularization

S stays quality-only (A13): runtime and surface size never enter S, ΔS or the bootstrap. Instead, each arm trace records three resource deltas against the contemporaneous incumbent. Each is `metrics.simplicity.relative_change(new, base)` = (new − base) / max(base, 1), and is `None` when either side lacks the data:

- **ΔX**: relative change of the evaluated surface `complexity` (`EvalResult.surface["complexity"]`).
- **ΔCalls**: relative change of the mean `llm_calls + tool_calls` per completed trial (`stats.mean_calls`). Runtime counts as observed when some completed trial has wall ms or calls (`stats.runtime_observed`).
- **ΔWall**: relative change of the median `wall_ms` over completed trials (`stats.median_wall_ms`).

The trace also records `complexity`, `calls_per_task`, `wall_ms_p50`, `simplicity_score` and a `resources` mapping (tree validity, missing metrics, per-cap detail, credit eligibility, pass/fail). The `params` block records the five knobs. `HistoryRecord` is unchanged.

The knobs (all off in `smoke`, `local` and `paper`):

- **Caps** (`x_cap`, `calls_cap`, `wall_cap`). A set cap is a non-compensatory gate in both branches: Δ > cap makes the arm inadmissible (reason `x_cap: dX … > …`, and so on). A cap is skipped when its delta is unavailable.
- **`require_resource_metrics`.** Missing surface complexity or runtime on the arm or the incumbent makes the arm inadmissible (reason `resource_metrics_missing`).
- **w_x.** The weighted rule becomes w_s·ΔS − w_c·ΔC + w_n·ν − w_x·max(ΔX, 0) + credit > 0, with the new terms shown in `rule.terms`. The cost rule (Eq. 7) is unchanged apart from the caps.
- **Simplicity credit (A16).** credit = w_x·max(−ΔX, 0), but only when the arm is otherwise admissible on quality and safety: ΔS ≥ 0, no increase in critical safety violations, and a valid tree. `simplicity_score` (`metrics.simplicity.simplicity_score` against the incumbent surface) is recorded only for such arms. A pure simplification (ΔS = 0, ΔC = 0) therefore ships under `harness` but not under the paper profiles.
- **Tree validity.** `surface["tree_valid"] < 1` (set by the harness-tree domain) makes the arm inadmissible under any profile (reason `tree_invalid`). Surfaces without the key count as valid.

`tests/ci_lab/rrsi/test_default_profile_regression.py` checks that the default profiles decide exactly as the pre-resource code did, using a golden recorded before the change.

### C16 and the paper's rule

The paper accepts an arm on the point estimate alone. The C16 gate only *narrows* that acceptance region:

- An arm that clears the cost rule but whose bootstrap lower bound (two-sided percentile CI at `ci_level`) is ≤ the threshold is **rejected**. It does **not** fall back to the weighted rule.
- Within-band acceptances do not claim a score gain, so the gate does not apply to them. They still face the floor and every guard.
- The threshold defaults to 0, as the task specifies. Design v2 §9 says "> δ", which is available as `ci_lower_threshold="delta"`.
- Setting `require_ci_lower=False` reproduces the paper's rule exactly.

### Statistics (C8)

- **`aa_delta`.** Pools the |bootstrap mean| of paired A/A differences over all repeat pairs, then sets δ = max(q-quantile, 1/M).
- **`aa_repeats_for_precision(h, pilot)`.** Returns R = ⌈(z·sd/h)²⌉, with a minimum of 5.
- **`confirm_test`.** A one-sided test at α: it passes when the α-quantile of the bootstrap ΔS is > 0. It is meant for held-out confirmation.
- **`non_inferiority`.** Passes when the bound is > −margin. It is meant for safety metrics.

### Schedule (Algorithm 1)

- **Budget.** b_t = ⌈b_min + ½(b_max − b_min)(1 + cos(πt/T))⌉, so b_0 = b_max and b_T = b_min.
- **Stall.** σ_t = 1 when t ≥ w and S_t − S_{t−w} ≤ δ.
- **Untried components (U).** U holds components with no measured edits.
- **Prune set (B).** B holds components whose best ΔS over the last `n_prune` rounds is ≤ 0. A component that was not tried in that window counts as −∞.
- **Exploration.** When stalled, the first m = min(m_draft, |U|, N) arms explore one untried component each.
- **Exploitation.** The other arms focus on non-pruned components, ranked by success rate, then mean ΔS, then vocabulary order.
- **Output.** `directives(...)` emits one `contracts.ArmDirective(arm, strategy, component_focus, edit_budget, explore)` per arm.
- **Telemetry.** `plan_round` runs inside `obs.span(SPAN_STEP, {ci.phase: "plan", rrsi.round, oes.experiment_id})` and also records `ci.strategy` (comma-joined, in arm order), `rrsi.budget` and `rrsi.stalled`. Without a telemetry provider the span is a no-op.

### Arm strategies (design §11.2, C23)

The strategy assignment is orthogonal to the component schedule: slot *i* receives both a component focus and a strategy.

- **Strategy set.** `hp.strategies` defaults to `params.DEFAULT_STRATEGIES`
  (`agent`, `gepa`, `skillopt`). The `harness` profile and
  `campaign.defaults.DEFAULT_HYPER["strategies"]` use `HARNESS_STRATEGIES`:
  `agent`, `gepa`, `skillopt`, and `agl`. `guard` is accepted but remains opt-in; include it
  explicitly when a campaign should evaluate a lessons/guard arm.

- **Evidence.** Each strategy has its own statistics, kept separate from the per-component statistics.
  - The unit is the measured arm: one arm counts as one trial for its `HistoryRecord.strategy`, however many edits or components it carries.
  - Success means the arm was elected (*a* = 1).
  - The posterior is Beta(a0 + successes, b0 + failures), with `hp.strategy_prior` defaulting to (1, 1).
- **Floor.** A strategy with no measured arm in the last K = `hp.strategy_floor_every` rounds (default 3), or never, is forced first, oldest first.
  - This also rotates untried strategies into the early rounds.
  - `Hyperparams` rejects configurations where |strategies| > N·K, because the floor would be infeasible.
- **Thompson sampling.** The remaining slots each take a fresh draw θ_s ~ Beta per strategy and pick the argmax, with ties going to vocabulary order.
  - `hp.strategy_cap` optionally caps how many arms one strategy can get in a round.
  - The draws are seeded by `derive_seed(hp.seed, t, "strategy")`, so a plan is a pure function of (t, hp, history < t).
- **Strategy-blind selection.** Selection never reads `ArmResult.strategy`. The strategy is carried only into attribution (`HistoryRecord.strategy`) and the stored round record. A test proves that relabelling strategies leaves the decision byte-identical.
- **Limitation.** An arm that fails before evaluation leaves no history record, so its strategy stays "starved" and is retried in the next round.

### Re-adjudication

- `save_round` stores the full `SelectionInputs` (evals, arms, history, hp, δ) together with the decision.
- `readjudicate` re-runs `select` from that record. δ always comes from storage, so an `hp` override cannot change δ but can change β and w.
- `verify` is true iff the stored decision is reproduced bit-for-bit.
- Before re-adjudicating a past round, call `frontier.rollback` first.
