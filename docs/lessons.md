# Lessons: mining traces into structural rules

`ci_lab.lessons` turns real failures into **lessons** (design §13.3) and routes each lesson to the
strongest enforcement rung that fits it. Pipeline:

```
harvest (adapters) → reduce (B3, untrusted only) → family holdout split → fingerprint → cluster
  → route (ladder R1..R6) → candidates.jsonl → [M17 synthesis] → replay (rejection filter) → closed loop
```

The replay validator is a **rejection filter only** (B4): it can `reject` or `pass_to_closed_loop`,
never accept. Acceptance is the guard-on/guard-off closed loop + OES gates.

## CLI (`ci-lab lessons …`)

| command | what | exit codes |
|---|---|---|
| `harvest --source assert\|spans\|agl\|usage\|calibrate --in <path> --out <dir> [--split evolve] [--pin P] [--slice S] [--labels L.jsonl]` | adapt traces → `<dir>/trajectories.jsonl` (merged, de-duplicated by id) + `harvest_report.json` | 0; **3** if any sealed/unknown-split record was refused (others still written) |
| `mine --in <dir> --run <run_dir> [--registry registry.yaml] [--min-support 3] [--min-slices 2]` | split/cluster/route → `<run>/lessons/candidates.jsonl` + `mine_report.json` | 0 |
| `validate --rules <yaml> --in <dir> [--run R --cluster ID] [--extractors X.yaml]… [--dataset eval.yaml]… [--epsilon E] [--out report.json]` | replay rejection filter → `ReplayReport` JSON | 0 pass_to_closed_loop; 2 reject |
| `confirm <cluster_id> --run R --by NAME [--reject] [--note …] [--yes]` | human confirmation | 0; 2 unknown cluster; **4** refused (CI/agent env or non-TTY without `--yes`) |

Each phase runs in `obs.span(contracts.SPAN_LESSONS, {ATTR_PHASE: harvest|cluster|route|replay})`
(replay also sets `ATTR_LESSON=<cluster id>`). `lessons/cli.py` exposes `register(subparsers)` and
`main(argv)`.

## Python API

* `ci_lab.lessons.stats`: `clopper_pearson(k, n, confidence=.95) -> (lo, hi)`, `cp_upper` /
  `cp_lower(k, n, confidence, one_sided=False)` (exact Clopper-Pearson by bisection on the log-space
  binomial CDF, pure Python), `promotion_ok(opportunities, fires, adjudicated_positives,
  false_positives) -> (bool, reasons)` using `rulespec.PROMOTE_*` (N3), `promotion_ok_stratified`.
* `ci_lab.lessons.harvest`: `harvest(source, path, opts, stats)`, `from_transcript(transcript,
  violations, split=…)`, `make_trajectory(...)`, `HarvestOptions`, `HarvestStats`.
* `ci_lab.lessons.reduce`: `reduce_trajectory(t)` (B3).
* `ci_lab.lessons.fingerprint`: `fingerprint(t) -> rulespec.Fingerprint`, `cluster_id(fp)`,
  `canonical_calls`, `tool_ngrams`, `is_failure`, `is_good`.
* `ci_lab.lessons.cluster`: `split_by_family`, `is_holdout(family)`, `mine(trajs, MineConfig,
  decisions) -> MineResult`, `append_confirmation`, `load_confirmations`.
* `ci_lab.lessons.route`: `route_all(clusters, members, corpus, registry) -> [Routing]`.
* `ci_lab.lessons.replay`: `validate(rules, trajectories, *, cluster=None, dataset_texts=(),
  config=ReplayConfig(), engine=None, work_dir=None, extractors=()) -> ReplayReport`; `validate_ok(...) -> (ok,
  reasons)`; `leak_screen`, `safety_check`, `paraphrases`. `rules` may be a `RuleSpec`, a list, a
  `RuleFile`, or an already-loaded `ci_lab.rules.Bundle`. Rules using `state` flags need the
  extractor YAML paths (`extractors=` / `--extractors`): flags come only from extractors, and a flag
  rule without one fails to load (→ reject).
* `ci_lab.lessons.registry`: `Registry.load/save/upsert/by_cluster/prose_fix_count/touched_lessons`,
  `conflicts(lesson_ids_touched_by_arms, interaction_evaluated=()) -> {lesson_id: [arms]}` (N4).
* `ci_lab.lessons.pipeline`: `run_harvest`, `run_mine`, `run_validate` (what the CLI calls).

## Data policy

* **Splits (B2, C15).** Only `evolve` (trusted eval/ASSERT/AGL/calibrate) and `usage` are accepted.
  `heldout`, `ood`, `aa`, `confirm`, `sealed`, `test`, `holdout`, or a missing/unknown split raise
  `SealedSplitError`; adapters log `REFUSED <digest>` at ERROR, count them in `sealed_refused`, and the
  CLI exits 3. Sealed data never reaches mining, replay, or any output file.
* **Untrusted reduction (B3, C12).** `usage` traces and injection-suspect trajectories are reduced:
  user text dropped; response text → `text_digest` only; args/results → typed shapes (bools, bounded
  numbers, closed-vocabulary enums, ids → `#id:<keyed hash>` of the normalized id so same-subject joins
  survive, other strings → `#txt:<keyed hash>`); family/slice/pin hashed unless already identifiers.
  Set `CI_LESSONS_SALT` to key the hashes. Usage trajectories are `trusted=False` until a human label
  exists (`--labels`, JSONL `{id, label: good|bad}`). Untrusted-only clusters get status `backlog`.
* **Injection.** Trajectories with an `injection*` oracle rule or from an injection suite/label are
  `injection_suspect=True`, always reduced, routed to R6 with no features (excluded from synthesis).

## Mining

* **Fingerprint (N2)** = pin + sorted oracle rule ids + rubric fail ids + canonical tool 3-grams ending
  at the failure anchor + error class. Canonicalization dedupes immediate retries (keeps the last) and
  collapses repeated lookups (`get_*`, `lookup_*`, `search_*`, …). Anchor = first errored/blocked call,
  else the first call whose tool shares a ≥4-char stem with an oracle rule id, else end of trajectory.
  Cluster id = `lc-<16 hex of fingerprint digest>`; a new pin means new ids.
* **Holdout by family before clustering (N2).** `sha256(salt|holdout|family)` sends ~50% of families
  to holdout; all paraphrases/trials of a family land on the same side. Holdout members are never mined;
  they are the replay recall set.
* **Clustering.** min support 3 trusted members, ≥2 slices, ≥2 families, no slice >80%, and stability:
  present in both the early and late half of the corpus' time slices. Dropped patterns are reported in
  `mine_report.json` with reasons. Status `candidate`; `human_confirmed` only via `ci-lab lessons
  confirm` (decisions in `confirmations.jsonl` survive re-mining).
* **Router (ladder).** injection → R6; `pii.*` → R3 `{pattern_classes}`; failing calls share an arg
  range/enum violation vs passing calls → R1 `{target_tool, arg, op, values}`; failing call lacks a
  same-subject prior tool that passing calls have → R2 `{target_tool, subject_arg, prior_tool,
  prior_subject_field}` (or mismatched prior result field `{arg: "result.k", op, values}` / amount
  comparison `{amount_arg, prior_amount_field, op}`); other oracle-rule clusters → R4; rubric-only → R6.
  A fingerprint already "fixed" by prose ≥2 times in the registry is forced to R4 (structural review).

## Replay rejection filter (B4)

Rejects on any of: invalid spec; regex not RE2-compilable (checked before the engine sees it); bundle
load failure (`ci_lab.rules.load_bundle`; unknown template ids etc.); holdout recall < 0.6 or no
holdout members; < 30 known-good negatives; Clopper-Pearson 95% UCB of the FP rate on passing /
human-good evolve trajectories > ε (`PROMOTE_FP_UCB`); per-rule block rate > 10% or aggregate > 20% of
evolve trajectories; leak screen hit (word 3-gram overlap with eval dataset texts, memorized case ids,
≥16-char verbatim substrings, or regexes that fail ≥50% of deterministic metamorphic paraphrase probes).
Leak findings carry digests only, never literals. Only `evolve` trajectories are replayed.

## File formats

* `<out>/trajectories.jsonl` — one `rulespec.Trajectory` per line (canonical JSON).
* `<out>/harvest_report.json` — `{source, harvested, untrusted, injection_suspect, invalid,
  sealed_refused, refused: [digests], total_in_dir}`.
* `<run>/lessons/candidates.jsonl` — `{"cluster": LessonCluster, "features": {...}, "route_reasons":
  [...], "forced_structural": bool, "trusted": bool, "holdout_members": [trajectory ids]}`.
  `features` is `{}` when no typed separator was found (M17 derives from the fingerprint).
* `<run>/lessons/mine_report.json` — counts, dropped clusters + reasons, routes.
* `<run>/lessons/confirmations.jsonl` — `{cluster_id, decision: confirmed|rejected, by, note, ts}`.
* `lessons/registry.yaml` — `{schema_version: 1, lessons: [LessonEntry…]}` (reads also accept a list
  or `{lesson_id: entry}`).

## Integration hooks

* `# HOOK(M3)` ASSERT/oracle: `--source assert` reads JSONL `{case_id, suite, split, trial?, family?,
  pin?, slice?|ts?, passed?|score?, transcript: {messages, tool_calls[{call_id, name, arguments,
  result, turn}]}, violations: [{rule_id, severity, detail}], rubric_fails?}` — i.e.
  `dataclasses.asdict(contracts.Transcript)` + `Violation`s. In-process callers use
  `harvest.from_transcript(transcript, violations, split="evolve", suite=…, pin=…)`. Violation
  `detail` text is never stored.
* `# HOOK(M12)` span records: `--source spans` reads the telemetry span JSONL (`schemaVersion`,
  `traceId`, `spanId`, `parentSpanId`, `attributes`); one trajectory per `ci.case` span (attrs
  `ci.case_id`, `ci.split`, `ci.trial`, `ci.oracle_rules`), tool spans via `tool.name`/`input.value`/
  `output.value` (OpenInference) or `gen_ai.tool.*` (OTel GenAI), agent output via `output.value`.
* `# HOOK(M5)` AGL: `--source agl` reads a `FileRolloutJournal` directory (`<rollout_id>.jsonl` with
  `kind: start|event|finish` records); latest attempt per rollout; OpenAI-style messages → steps.
* `# HOOK(M9)` / `HOOK(M16)` sleep nightly: **wired** in `ci_lab.sleep.lessons_hook` and enabled
  with `ci-lab sleep run --lessons` or `SLEEP_LESSONS=1`; it is off by default. The night's judged
  rollouts are converted with `make_trajectory(source="assert", split="evolve", slice=<night date>)`
  into a local store. Optional local `--lessons-source` inputs go through `run_harvest`. The hook
  then calls `run_mine`. Sanitized candidates (B8) are proposed in the sleep draft-PR bundle as
  `experiments/sleep/lessons/<night>.json` and are never adopted automatically. See
  [sleep.md](sleep.md#pipeline).
* `# HOOK(M8b)` campaign: before adopting several arms, call `registry.conflicts({arm: lesson_ids})`
  (lesson ids via `Registry.touched_lessons(rule_ids, prose_anchors)`); refuse non-empty results unless
  the arms' interaction was evaluated.
* `# HOOK(M17)` synthesis/guard arm consumes `candidates.jsonl` and calls `replay.validate`.
* `# HOOK(cli)` `ci_lab/cli.py` `COMMAND_MODULES` does not list `"lessons"` yet (contract change
  request); until then use `python -m ci_lab.lessons.cli lessons …`.
