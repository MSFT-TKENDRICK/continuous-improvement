# RRSI core (`ci_lab.rrsi`)

This package is a pure implementation of Algorithm 1 (the schedule) and Algorithm 2 (selection) from RRSI, extended with the design v2 §9 safeguards C8, C11 and C16.

Nothing in it calls models, git or a clock. The only I/O is the JSON/JSONL helpers in `history.py` and `readjudicate.py`. The callers (the round runner and the CLI) own evaluation, drafting, and git.

## Modules

| Module | Public API |
|---|---|
| `params` | `Hyperparams` (frozen; `with_`, `to_dict`, `from_dict`), `PROFILES` (`smoke`, `local`, `paper`), `profile(name, **overrides)`, `TABLE5`, `paper_reference(domain)` |
| `schedule` | `edit_budget(t, T, b_min, b_max)`, `stall_flag(traj, t, w, delta)`, `untried(history)`, `component_yield`, `prune_set(history, t, n_prune)`, `exploration_slots`, `plan_round(t, hp, history, trajectory, delta, arms=None)` → `RoundSchedule`, `directives(...)` → `tuple[Directive, ...]` |
| `history` | `HistoryRecord` (round, arm, edits, score, cost, delta_s, delta_c, accepted = *a*, novelty, admissible, reasons), `read_jsonl`, `append_jsonl` (rejects a duplicate (round, arm)), `write_jsonl` (atomic), `accepted_counts`, `tried_components`, `before`, `replace_round` |
| `stats` | `paired_bootstrap`, `bootstrap_means`, `mean_missing_zero`, `task_score`, `missing_rate`, `mean_cost`, `aa_delta`, `aa_repeats_for_precision`, `confirm_test`, `non_inferiority`, `derive_seed` |
| `selection` | `SelectionInputs`, `select(inputs, guard=None)` → `SelectionDecision` (`decision` ∈ ship / do_not_ship / rerun, a per-arm `ArmTrace` rule trace, `to_dict()` that is JSON-safe) |
| `frontier` | `Frontier`, `initial`, `advance(frontier, decision, arms)`, `pause`, `resume`, `finish`, `rollback` |
| `attribution` | `novelty(edits, accepted)`, `attribute(decision, arms)` → history records (*a* = 1 only for the winner's edits), `component_stats(history)` |
| `readjudicate` | `save_round`, `load_round`, `round_record`, `readjudicate(source, hp=None)` → `Readjudication`, `verify(source)` |

## Profiles

All three profiles share the paper's coding-domain weights: w_s = 0, w_c = 15, w_n = 0.5, β0 = 0.10, β1 = 44.5.

| | T | k | N | b_min..b_max | w | m_draft | n_prune | M | n_bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 3 | 1 | 2 | 1..2 | 1 | 1 | 2 | 12 | 2 000 |
| local | 8 | 2 | 2 | 1..3 | 2 | 1 | 3 | 40 | 10 000 |
| paper | Table 5 (coding) | | | | | | | | 10 000 |

- In every profile, δ is `None` because it is calibrated by A/A. `paper_reference(domain)` carries the paper's fixed δ for comparison only.
- The smoke and local window and budget values are our own choices and do not come from the paper.

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
- **Winner.** The winner is the argmax of S′ over admissible arms. Ties go to lower cost, then to the higher CI lower bound, then to the lexicographically smallest arm name.
- **S\* update.** S* ← max(S*, S_{t+1}), so S* never decreases.
  - On `do_not_ship`, S_{t+1} is the re-measured incumbent score.
- **Seeds.** Each arm's bootstrap seed is `derive_seed(hp.seed, round, arm)`, so a decision is a pure function of the stored inputs.

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

### Re-adjudication

- `save_round` stores the full `SelectionInputs` (evals, arms, history, hp, δ) together with the decision.
- `readjudicate` re-runs `select` from that record. δ always comes from storage, so an `hp` override cannot change δ but can change β and w.
- `verify` is true iff the stored decision is reproduced bit-for-bit.
- Before re-adjudicating a past round, call `frontier.rollback` first.
