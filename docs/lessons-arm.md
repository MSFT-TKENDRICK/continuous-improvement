# `ci_lab.lessons_arm`: lessons become guard rules as an experiment arm

This package implements design §13.3 steps 4–8, the binding §13.6 items (B1–B4, N3–N6) and the guard
arm in §11. The lifecycle runs:

**candidate cluster → `RuleSpec` (shadow) → guard arm commit → paired guard-off/on eval → OES ext →
N3 promotion → N5 prose deletion → N6 retirement.**

Every new rule starts in `mode: shadow`. Rule text comes only from the trusted M14 template catalog
(B3). Trace text never enters a rule; slot values are tool, argument, flag or enum names only.

## Public API

| Module | Contents |
|---|---|
| `features` | `LessonFeatures` (typed slots only), `derive_features`, `Candidate` (`{"cluster","features","trusted",…}` from M16's `candidates.jsonl`; unknown keys are ignored; `features: {}` falls back to derivation), `read_candidates` |
| `synth` | Deterministic templates, no LLM: `synth_prior_call` (R2 `prior` with a `same` subject join), `synth_state_flag`, `synth_arg_constraint` (R1), `synth_amount_vs_prior` (R2 `cmp`), `synth_response_pattern` (R3 redact; RE2 patterns come from the fixed safe library), `synthesize`, `lesson_id_for` |
| `agent` | `LessonSynthesizer`: a MAF declarative agent (`specs/lesson_synthesizer.yaml`, expression-free) with one `submit_rule` tool (`SubmitRuleArgs`: rung, target, skeleton, template, slots). Its purpose is `"proposer"`, served via `ChatClientFactory`. It raises `NoRuleSubmitted` / `NoSkeletonFits`. |
| `bundle` | `load_rules(rules, extractors)` → `LoadedRules` (through `ci_lab.rules.load_bundle`; the extractor paths are always passed), `subset_bundle`, `rule_files`, `read_rule_file`, `dump_rule_file`, `BundleError` |
| `strategy` | `GuardStrategy` (`contracts.ArmStrategy`, name `"guard"`) and `register()`. See "Guard arm" below. |
| `paired` | `paired_eval(domain, harness_dir, split, k, *, trials, experiment_id, variant, decisions_dir) -> GuardMetrics`, `guard_ship_ok(m, incumbent, margin, *, epsilon)` (B1), `agent_safety_rate` (the attempted rate; guards are never credited), `guard_paired_eval_step` (the workflow step) |
| `envelope` | `guard_extension(metrics, *, split, bundle_digest, …) -> {"com.microsoft.ci.guard": {...}}`, `holdout_look_required(split)` (anything other than `evolve` or `aa` counts as a C15 look) |
| `promote` | `promote(rule_id, decisions, labels, *, repo, branch=None) -> PromotionResult` (N3) |
| `prose` | `propose_deletions(entry, rules, *, root) -> ProseProposal`, `apply_deletion(prop, worktree) -> Edit` (N5) |
| `retire` | `exposure`, `retirement_candidates` (dormant/rare after `opportunities ≥ 200`, never counted in nights), `apply_ablation(c, worktree) -> Edit` (N6) |
| `workflows/arm_guard.yaml` | An expression-free MAF workflow with the same shape as campaign's `arm_skillopt.yaml`, plus a `guard_paired_eval` step |

## Guard arm (`GuardStrategy.propose`)

1. It reads `<run_dir>/lessons/candidates.jsonl` (searching up to 3 parent levels) and `ctx.failures`.
2. It skips clusters that are untrusted and unlabeled. It also skips clusters already touched by another arm, using registry conflicts plus `lesson_claims/` (N4).
3. It synthesizes at most `edit_budget` rules, trying templates first and the agent second.
4. It writes **only** `harness/guards/<lesson_id>.yaml`. A resolved-path check and a post-commit diff check both enforce this; `BUNDLE.lock` and extractor files are never touched.
5. It loads the full bundle (with extractors) through `ci_lab.rules`.
6. It runs the M16 replay filter (`ci_lab.lessons.replay.validate_ok`).
7. It makes one commit per Edit, with the trailer, and returns `Edit(component="guard", …)`.
8. It writes `<run_dir>/optimizer/<arm>-guard.json`.

## Paired evaluation (B1/B4)

`Domain.evaluate` runs twice on identical cases, trials and seeds, once with `CI_GUARDS=off` and once with `CI_GUARDS=enforce`. Stochastic providers get at least 3 trials.

Each run's GuardDecision JSONL goes to `$CI_GUARD_DECISIONS/<variant>/<case_id>/<trial>.jsonl`. Optional lines carry `{"kind":"opportunity","n"}` or `{"kind":"call",…}` records.

The metrics are:

| Metric | Definition |
|---|---|
| attempted violations | guard-off oracle violations ∪ recorded block/redact attempts |
| delivered violations | guard-on oracle |
| false denial | enforced blocks on pairs whose guard-off run was clean |
| substitutions | another side-effecting call after a block |

The ship rule is: delivered violations go down, AND completion is non-inferior within the C5 margin, AND the false-denial UCB is at most ε.

## CLI: `ci-lab lessons-arm`

| Command | Behavior | Exit codes |
|---|---|---|
| `promote --rule ID --decisions D.jsonl [--labels L.jsonl] [--repo .] [--out x.patch] [--branch name]` | N3 gate via `ci_lab.lessons.stats.promotion_ok`, using `promotion_ok_stratified` when any `intent` is present. Only decisions for the rule's current version count. At `SHADOW_MAX_NIGHTS` or more the verdict is `expired`. It writes a `git apply`-able patch and, optionally, a local branch built with git plumbing, without a checkout. It never merges or pushes. | 0 promote, 1 hold/expired, 2 error |
| `prose --lesson ID [--repo .] [--registry …] [--out x.patch]` | N5 proposal. It deletes only sentences whose normalized tokens are a subset of the enforced rules' rendered template message, fix and slots. Sentences carrying policy-intent or recovery cues are kept, and so are code fences and at least one sentence per section. | |
| `retire --decisions D.jsonl… [--min-opportunities 200] [--max-fire-ucb 0.005]` | Prints ablation-arm candidates. | |

Labels are lines of the form `{"attempt_digest", "label": "tp"|"fp", "intent"?}`. Opportunity lines take the form `{"kind":"opportunity","n","rule_id"?|"target"?,"intent"?,"night"?}`.

**N5 requirement (documented, enforced by the campaign).** A prose-deletion arm needs all of the following before it can ship:

- it passes on OOD cases;
- it passes on **at least 2 model pins** (C4);
- it has an automatic rollback canary.

The lesson stays in the registry.

## HOOK notes


- Done (21b) `# HOOK(M8b)`: `ARM_YAMLS["guard"]` is `ci_lab/lessons_arm/workflows/arm_guard.yaml`, and the `guard_paired_eval` step is bound to `ci_lab.lessons_arm.paired.guard_paired_eval_step`. The step is idempotent through `run_dir/guard_eval.json` and skips when `eval.json` is skipped.
- Done (21b) `# HOOK(M4)`: the `com.microsoft.ci.guard` ext schema is `schemas/oes/ext-com.microsoft.ci.guard.schema.json`. When `holdout_look_required(split)` is true, the campaign appends one look per round to the C15 holdout ledger through `ledger.looks.record_look`. The payload carries `holdout.{datasetHash, plannedLooks, looksUsed}`.
- `# HOOK(M1)`: `agent.default_builder()` uses `ci_lab.maf.loader.build_agent` when it can be imported, and otherwise falls back to `local_builder`, which runs the real MAF tool loop. It does not yet pass `allowed_models`, so the loader call fails and falls back.
- Done (21a/21b) `# HOOK(M3)`: `order_support.guarding` writes each case's guard sink to `$CI_GUARD_DECISIONS/<case_id>/<trial>.jsonl`, tags each line with `case_id`/`trial`, and emits call and opportunity records. The seed comes from `CI_CASE_ID|CI_TRIAL`, so it does not depend on `variant`. `Domain.evaluate` sets these per case.
- `# HOOK(M15)`: `CI_GUARDS` is read only in `ci_lab.guards.install.resolve_mode`. Under `off`, attempts are still recorded, with `mode="off"`.
- `# HOOK(M16)`: the replay filter (`lessons.replay.validate_ok`), `candidates.jsonl`, and `Registry` / `conflicts`.
- `# HOOK(M14)`: `default_templates()` must contain `precondition.prior_call`, `precondition.state_flag`, `arg.constraint`, `amount.not_exceed_prior` and `response.redact_pattern` (see `templates.catalog_gaps`).
- `# HOOK(integration)`:
  - done (20): `"lessons_arm"` is in `ci_lab.cli.COMMAND_MODULES`, which adds the CLI group `lessons-arm`;
  - done (21b): `ci_lab.strategies.EXTERNAL["guard"]` is `ci_lab.lessons_arm.strategy`; `get_strategy("guard")` imports it lazily and calls `register()`.

## Contract change requests

- Done (integration v2.4.1): `contracts.COMPONENTS` has a dedicated `"guard"` entry, so guard edits get their own credit assignment. Text strategies use `contracts.TEXT_COMPONENTS`, which excludes it.
- Done (integration v2.4.1): `GuardDecision` has `case_id`/`trial` and `mode="off"`, which off-mode attempts now record. Opportunity records are wired in the order-support layer.
- State-flag lessons depend on domain extractors, because the arm may not write extractor files.
