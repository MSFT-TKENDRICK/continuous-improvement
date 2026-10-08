# Open Experiment Standard (OES 0.1.0) in ci_lab

Every ci_lab experiment produces an OES 0.1.0 envelope. That covers RRSI calibration,
each RRSI round, the held-out confirmation, and each SkillOpt-Sleep night.

Each envelope is a single JSON document:
- it validates against the official OES schema;
- ci_lab-specific facts live under namespaced `extensions`;
- it is hash-locked with a `contentHash` field.

Code lives in `src/ci_lab/oes/`, schemas in `schemas/oes/` (provenance in
`schemas/oes/SOURCE.md`), and example envelopes in `tests/ci_lab/oes/fixtures/`.

## Schemas

| File | What |
|---|---|
| `openexperiment-0.1.0.schema.json` | Official OES 0.1.0 schema, vendored byte-exact. The sha256 is pinned in SOURCE.md and a test. |
| `ext-com.microsoft.ci.rrsi.schema.json` | `extensions["com.microsoft.ci.rrsi"]` (closed). |
| `ext-com.microsoft.ci.sleep.schema.json` | `extensions["com.microsoft.ci.sleep"]` (closed). |
| `ext-com.microsoft.ci.guard.schema.json` | `extensions["com.microsoft.ci.guard"]` (closed; v2.4 §13 guard arms). |

### rrsi extension

The rrsi extension has three kinds: `calibration`, `round` and `confirm`.

All kinds record:
- `campaignId`, `round` and `split`;
- `multipleTesting` (`exploratory` for rounds, `confirmatory` for confirm);
- `harnessTree`;
- `evaluatorPin` (evaluator tree, judge model/provider, `servedJudgeModels`);
- `splitHashes`.

Rounds add:
- the schedule: `budget`, `stall`, `explorationSlots`, `pruneSet` (⊆ `contracts.COMPONENTS`);
- the noise band and rules: `delta` + `deltaMethod`, `costRule {beta0, beta1}`, `weights {ws, wc, wn}`;
- the Alg. 2 `selection` trace: winner, plus per candidate dS, dC, novelty, CI lower bound, rule and admissibility;
- per-variant `variants` (status, commits, tree, `edits[{component, hypothesis, commit, files}]`, `critic`, `archiveRef`);
- `lineage`, `supersedes` and `ciLowerBound`.

Confirm adds:
- `holdout {datasetHash, plannedLooks, looksUsed}`;
- `lookLedgerRef`;
- `preRegistration {alpha, sided, primaryMetric, nonInferiorityMargin, stats}`.

Per-kind required fields are enforced with `if/then`.

### sleep extension

The sleep extension records:
- `night` and `skilloptVersion`;
- `tasks {total, byOrigin, bySplit}`;
- `gate {skillopt, assert}`;
- `budget {used, limits}`;
- `candidateDigest`;
- `adoptionPr` (`exp/sleep-<yyyymmdd>-<n>/cand`, or null);
- `evaluatorPin`.

### guard extension

Built by `ci_lab.lessons_arm.envelope.guard_extension` (key `oes.GUARD_EXT` ==
`rulespec.OES_GUARD_EXT`). It records one paired guard-off/guard-on evaluation (B1/B4):
- `kind: guard_eval`, `split`, `bundleDigest`, `paired`, `trials`;
- the `GuardMetrics` rates (`attemptedViolationRate`, `deliveredViolationRate`, `taskCompletion`,
  `falseDenialRate`, `blockRate`, `recall`, `fpRate`, `fpUcb`) and counts;
- `agentSafetyRate` (the guard-off attempted rate; guards are never credited to the agent);
- optional `arm`, `ruleIds`, `lessonIds`, `incumbentBundleDigest`, `ship {ok, reasons}`;
- `holdoutLook`, and `holdout {datasetHash, plannedLooks, looksUsed}` iff it is a held-out look (C15).

## Concept mapping

| ci_lab | OES |
|---|---|
| campaign round `<cid>-rNN` | `experiment.id`; `design.type: abn` |
| incumbent `inc` / arms | `variants[]`: role `baseline` / `treatment` |
| A/A calibration `<cid>-cal` | `rep0` (baseline) … `repN`; design `ab`/`abn`; outcome `do_not_ship` |
| held-out confirm `<cid>-confirm` | `h0` vs `final`; design `ab`, `alpha`, `peekingPolicy: fixed_horizon` |
| sleep night `sleep-<yyyymmdd>` | `incumbent` vs `candidate` skill; design `ab` |
| task, trial | `randomizationUnit: task`, `analysisUnit: task_trial` |
| Alg. 2 outcome | `decision.outcome`: `ship` (new incumbent / adoption PR) · `do_not_ship` · `rerun` (invalid) |

### Metrics

| id | role | notes |
|---|---|---|
| `evolve_score` / `heldout_score` | primary | Mean task score; a missing trial counts as 0. |
| `ood_score` | secondary | Confirm only, when OOD results are supplied. |
| `safety_violations` | guardrail | Critical oracle violations. Flagged `com.microsoft.ci:nonCompensatory: true`. |
| `cost_tokens_per_task` | guardrail | Tokens in+out per trial. |
| `missing_trial_rate`, `judge_error_rate` | data_quality | `judge_error_rate` only when `judge_errors` counts are passed. |
| `evaluator_pin` | invariant | 1 if the variant's pin equals the baseline's, else 0. |
| `suite_score.<suite>` | diagnostic | Per-suite mean. |

### Quality checks

| Check | Fails when | Effect |
|---|---|---|
| `missing_trials` | The baseline is over the limit (critical). | Outcome is `rerun`. |
| `invariant_metric` | The evaluator pin drifted (critical). | Outcome is `rerun`. |
| `judge_errors` | — | Informational only. |
| `aa_noise_band`, `harness_identity` | Calibration only. | — |
| `critic_<arm>` | Round only. | — |

Any failing check of high or critical severity forces `rerun`.

## Builders (`ci_lab.oes.build`)

The builders are pure: no I/O. Their inputs are contract types (`EvalResult`, `ArmResult`,
`EvaluatorPin`) plus statistics already computed by `ci_lab.rrsi`. Each returns a sealed,
schema-valid dict. Pass `exported_at` to get deterministic output.

```python
calibration_envelope(cid, runs, delta=, delta_method=, harness_commit=, split_hashes=, judge_errors=None)
round_envelope(cid, round_no, incumbent=, incumbent_commit=, arms=, selection=, schedule=Schedule(...),
               params=RrsiParams(...), split_hashes=, lineage=None, supersedes=None, archive_refs=None)
confirm_envelope(cid, baseline=, final=, baseline_commit=, final_commit=, stats=ConfirmStats(...),
                 holdout=Holdout(...), look_ledger_ref=, split_hashes=, alpha=0.05, sided="one",
                 non_inferiority_margin=0, ood=None, accepted_rounds=())
sleep_envelope(night, incumbent=, candidate=, incumbent_commit=, skillopt_version=, tasks_by_origin=,
               tasks_by_split=, skillopt_gate=, assert_gate=, delta=, budget_used=, candidate_digest=None,
               adoption_pr=None)
```

`Envelope.from_dict(doc).to_dict() == doc` holds for every builder output (pydantic models in
`ci_lab.oes.models`).

### Observability

Each builder records its decision on the caller's current OTel span, for example the
`ci.round` span (design §12.3):
- always `oes.experiment_id` and `oes.decision`;
- `ci.campaign_id` and `rrsi.round` (rrsi envelopes) or `sleep.night` (sleep envelopes);
- `oes.variant`, only when something is shipped.

Without a recording span (no `ci_lab.telemetry.setup`) this is a no-op. The builders never
create a span or install a tracer provider.

## Hash lock

`contentHash = "sha256:" + sha256(canonical_json(envelope without contentHash))`.

Canonical JSON means:
- UTF-8;
- sorted keys;
- `(",", ":")` separators;
- literal non-ASCII characters;
- no NaN or Infinity.

In addition, `provenance.resultHash` is the digest of `results`.

Any edit, including to `decision.rationale`, breaks the lock. Re-seal with `ci_lab.oes.seal`
only when a new decision is deliberately being recorded.

## Validation

`validate_envelope(doc, look_counts=None) -> list[str]` returns an empty list when the envelope
is valid. Each error starts with its rule id:

| Rule | Checks |
|---|---|
| `schema` | Official OES 0.1.0 schema (draft 2020-12; date-time/uri formats). |
| `extension-schema` | Our extension schemas. Unknown extension keys are ignored. |
| `schema-version` | `schemaVersion == "0.1.0"`. |
| `content-hash` / `result-hash` | Hash lock verified. `contentHash` is required once `decision.status` is `decided`. |
| `baseline` | Unique variant ids; exactly one `baseline`/`control`. |
| `references` | `metricId` / variant ids in results exist; `baselineVariantId` is the baseline. |
| `decision` | See below. |
| `non-compensatory` | See below. |
| `rrsi` | See below. |
| `holdout-looks` | `looksUsed ≤ plannedLooks`; a confirm uses ≥ 1 look; the global look ledger count (`look_counts`) ≤ planned. |
| `confirm` | `design.alpha` = pre-registered alpha; `fixed_horizon`; ship ⇒ `pValue < alpha` and `ciLower > 0`. |
| `sleep` | See below. |
| `guard` | ship ⇒ the guard eval is paired and its `ship.ok` (B1) is not false. Its `holdout` block gets the `holdout-looks` checks (a held-out guard eval uses ≥ 1 look). |

The `decision` rule requires:
- experiment decided ⇔ decision decided, and decided ⇒ outcome;
- an outcome in {ship, do_not_ship, rerun};
- `recommendedAction` agrees with the outcome;
- ship ⇒ no failing high or critical check, and exactly one shipped treatment.

The `non-compensatory` rule applies on ship. For the shipped variant:
- no `blocks_ship` result;
- every non-compensatory guardrail has a result and has not worsened. The allowed slack is the
  confirm `nonInferiorityMargin`, otherwise 0.

The `rrsi` rule checks:
- the experiment id matches the kind;
- the extension's variants match the OES variants;
- rounds use `abn` and `custom` multiple testing;
- ship ⇔ a winner exists, and the winner is an admissible treatment;
- `ciLowerBound` matches the winner's.

The `sleep` rule checks:
- the experiment id is `sleep-<night>`;
- tasks come from the evolve split only (no heldout/ood);
- `total` equals the sum of `byOrigin`;
- ship ⇒ both gates passed and `candidateDigest` is set;
- not ship ⇒ `adoptionPr` is null.

Schema errors short-circuit the semantic rules. rrsi, sleep and guard rules only run when their
extension is schema-valid.

### CLI

```
ci-lab oes validate experiments/**/*.json [--json] [--look-ledger experiments/holdout-looks.jsonl]
```

- Accepts files, directories (recursive `*.json`) and globs.
- Exits 1 on any error, or when no files match.
- `--json` prints `{"ok": bool, "files": [{"path", "errors"}]}`.

## Deviations from OES 0.1.0 and known limits

- **Exploratory multiple testing.** OES has no `exploratory` value for `multipleTestingPolicy`.
  Rounds therefore use `custom` and record `multipleTesting: exploratory` in the rrsi extension.
- **No non-inferiority, cost, supersedes or lineage fields in OES.** These live in the
  extensions. The non-compensatory guardrail is a namespaced metric flag.
- **Two spellings of rollback.** OES uses `decision.outcome: rollback` but
  `scorecard.recommendedAction: roll_back`; the validator maps between them. ci_lab only
  emits ship, do_not_ship and rerun.
- **Calibration outcome.** A valid calibration is `do_not_ship`: nothing is shipped, and δ is
  fixed in the extension.
- **Explicit nulls.** The models drop JSON nulls in typed OES fields on round-trip. The builders
  never emit them, but third-party envelopes with explicit nulls would re-hash differently.
- **License.** The OES schema publishes no license; see `schemas/oes/SOURCE.md`.
- **Packaging.** Schemas sit at the repo root and are not packaged into the wheel. For
  non-editable installs, set `CI_OES_SCHEMA_DIR`.
