# RRSI campaigns (M8b)

A campaign runs RRSI rounds over the harness. Each round is recorded as an OES
experiment and driven by declarative, expression-free MAF workflows. It covers
design §5, §8 and §9 (C5, C6, C8, C14), and design-oes-rrsi-v1 §3 and §5.

```
ci-lab campaign new          <cid> --profile copilot|offline|fake [--hyper k=v ...]
ci-lab campaign calibrate    <cid>     # A/A runs -> immutable delta
ci-lab campaign run          <cid> [--rounds N] [--stop-file PATH] [--defer-publish PATH]
ci-lab campaign publish      <cid> --requests PATH   # replay deferred publish requests
ci-lab campaign status       <cid>
ci-lab campaign readjudicate <cid> <eid>   # recompute selection from the ledger; exit 1 on drift
ci-lab campaign confirm      <cid>     # sealed held-out, one global look
ci-lab campaign land         <cid>     # requires confirm decision "ship"
```

Common flags (accepted by every subcommand):

| Flag | Meaning |
| --- | --- |
| `--profile` | `copilot` (default), `offline` or `fake`. |
| `--run-dir` | Run root. Defaults to `$CI_RUN_DIR`, then `artifacts/ci-runs`. |
| `--ledger-dir` | Ledger root. |
| `--repo owner/name` | GitHub repository for publishing. |
| `--dry-run-publish` | Record the intended `gh`/`git` calls instead of running them. |

The `fake` profile is fully offline: stub domain, fake slots and FakeChatClient agents. Publishing is always dry-run, and all state lives under `<run-dir>/_fake`.

The `copilot` and `offline` profiles are wired by `ci_lab.campaign.wiring.wired_deps`:

- **Domain:** `OrderSupportDomain(repo_root=REPO_ROOT, work_dir=<run-dir>/domain, journal=campaign_journal(<run-dir>))`. Its cases are the frozen ASSERT test sets (`evals/assert/<suite>/test_set.jsonl`, else `artifacts/results/<suite>/test_set.jsonl`); each case's category is its generated `dimensions.behavior`, which keeps the evolve, held-out and OOD splits non-empty. `splits()` raises if no test set is found. Per case/trial runs get `CI_CASE_ID`/`CI_TRIAL`, and `CI_TELEMETRY`, `CI_RUN_DIR` and `CI_GUARD_DECISIONS` are passed through.
- **Agent Lightning rollouts:** every scored case/trial (arms, incumbent, A/A and held-out evals) is an AGL rollout (`start` → `ci.score` event → `finish`) in a `FileRolloutJournal` under `<run-dir>/agl`. When `CI_LAB_AGL_URL` is set (bearer key `CI_LAB_AGL_KEY`), the journal is a `MirroringJournal` that best-effort mirrors to `agl-server` (journal first; mirror failures never stall the campaign). For `offline`, `CI_LAB_AGL_URL` must be loopback.
- **RRSI / OES (M7, M4):** `ci_lab.campaign.rrsi_wiring` (see [Selection, calibration and envelopes](#selection-calibration-and-envelopes-m7-m4)).
- **Git (M6):** `GitOps` keeps one `gitops.slots.SlotPool` per campaign under `$CI_WT_ROOT/<cid>`. `provision_slot` checks out `exp/<eid>/<arm>` from the incumbent. `resolve_incumbent` returns `(commit, harness tree)`, where the tree is the `tree_hash` of the harness root (the common prefix of `domain.surface_globs`). Slot leases are not released within a process.
- **Agents (M8a):** `MetaAgents` runs `meta.run.run_proposer`/`run_analyst`/`run_critic` with MAF specs validated against the manifest `allowed_models`. Proposer runs write under `<arm>/meta/`, because `submit_proposal` would otherwise overwrite the arm's `proposal.json`. A repair re-runs the proposer with the failing critique's reasons as feedback. The default meta model is `claude-sonnet-5`; `CI_META_MODEL` selects another allowlisted one (see [models.md](models.md)).
- **Model preflight (copilot only):** `CampaignDeps.preflight` (`ci_lab.campaign.preflight.make_preflight`) checks every model the campaign will use (meta agents, synthesizer, optimizer LM, order agent, ASSERT tester) before `calibrate`, `run` or `confirm` spends budget. A missing model exits with code 2 and names the override to set ([models.md](models.md)).
- **Strategies (M10):** `strategy_kwargs = {domain, client_factory}`.
- **Ledger and publishing:** `FileLedger` at `--ledger-dir` (default `<repo>/experiments`); `FileOutbox` at `<run-dir>/outbox.jsonl`; `GitHubPublisher` with journal `<run-dir>/publish-calls.jsonl`.
- **Chat clients:** `providers.factory.make_chat_client(profile, ...)`. The `copilot` profile uses the Copilot SDK client.

The `offline` profile is network-free:

- Publishing is always dry-run.
- `OPENAI_API_BASE`, `OPENAI_BASE_URL`, `AGL_OPENAI_BASE_URL`, `CI_LAB_AGL_URL`, the s1 judge backends `CI_S1_LLAMA_URL` and `CI_S1_SYSTEMONE_URL` (plus any other `CI_S1_*_URL`) and each chat client's `base_url` must be loopback (`localhost`/127.x/::1). Otherwise `NetworkPolicyError` exits with code 2 ([providers.md](providers.md#offline-network-policy-offlinepy)).

Missing optional integrations (an `ImportError`) also exit with code 2 ("integration pending").

## Workflows (`src/ci_lab/workflows/*.yaml`)

The workflows contain only `InvokeFunctionTool` and `InvokeAzureAgent` actions with literal arguments. They use no `=` expressions and no `If`, `ConditionGroup`, `Foreach` or `Goto`. `ci_lab.workflows.assert_expression_free` enforces this.

| file | actions |
| --- | --- |
| `round.yaml` | `begin_round` → Analyst → `run_arms` → `select` → `record` → `publish` |
| `arm_agent.yaml` | `provision_slot` → Proposer → `critique_1` → `repair_1` → `critique_2` → `repair_2` → `critique_final` → `evaluate` → `finalize_arm` |
| `arm_gepa.yaml`, `arm_skillopt.yaml` | the same, with the Proposer replaced by `propose` (`arguments: {strategy: gepa\|skillopt}`) |
| `lessons_arm/workflows/arm_guard.yaml` | guard arms: `ARM_YAMLS["guard"]`, which adds `guard_paired_eval` after `evaluate` |
| `calibrate.yaml` | `aa_runs` → `delta` → `record` |
| `confirm.yaml` | `reserve_look` → `evaluate_heldout` → `decide` → `record` |

Dynamic context never appears in the YAML. The run dir, eid and arm are bound into the registered tool closures (`ci_lab.workflows.steps`), and the YAML passes only `step:` names, `attempt:` numbers and `split:` names.

Repairs are unrolled. `repair_N` is a no-op when `critique_N` passed. Otherwise it re-invokes the proposer through an injected callable.

### Arm strategies (design §11.2)

Each directive carries `strategy` (`contracts.STRATEGIES`; default `agent`) and the driver runs `arm_<strategy>.yaml`:

- `agent`: the MAF Proposer agent (M8a), whose `submit_proposal` writes `proposal.json`.
- `gepa` / `skillopt`: `CampaignDeps.get_strategy(name, **strategy_kwargs)` resolves an `ArmStrategy` (default: lazy `ci_lab.strategies.get_strategy`, M10). The `propose` step calls `await strategy.propose(contracts.ArmContext)` once (marker `proposal.json`); a failed critique re-runs it with the critic reasons appended as `critic_rejected` `FailureRecord`s.
- `guard` (§13): runs `lessons_arm/workflows/arm_guard.yaml`. `guard_paired_eval` is bound to `ci_lab.lessons_arm.paired.guard_paired_eval_step`, which runs guard-off/on paired evals and gates the arm. Hyper keys:
  - `guard_trials`: paired repetitions. `None` gives ≥ 3 trials per case when the run is stochastic (B4).
  - `guard_stochastic`: `None` means stochastic iff the profile is Copilot.
  - `guard_margin`: B1 non-inferiority margin.
  The shipped guard arm's round envelope carries the `com.microsoft.ci.guard` OES extension. Guard evals on a held-out split reserve one C15 look per round in `holdout-looks.jsonl`.
- `ArmContext.evolve_case_ids` is set from `domain.splits()["evolve"]`. Text strategies score only these cases.
- `ArmResult.strategy` is filled; the incumbent pseudo-arm reports `incumbent`. `ArmResult.cost` carries the strategy's optimizer cost block (`<arm>/optimizer/<arm>-<strategy>.json`).
- **Edit scope (B2/N5).** Before evaluation, the arm's declared edit files plus its actual `base..HEAD` diff are checked with `strategies.base.edit_scope_violations`:
  - text strategies may never write `**/harness/guards/**`;
  - `guard` may write only guard rule files (never `BUNDLE.lock`).
  A violating arm is skipped (`edit_scope: ...`).

For `copilot`/`offline`, RRSI (`rrsi.schedule.plan_round`) allocates strategies per round by Thompson sampling over `hyper["strategies"]` (default: all of `contracts.STRATEGIES`, incl. `guard`), with a floor that gives every strategy at least one arm in any `strategy_floor_every` consecutive rounds. A `guard` directive always targets component `guard`. The `fake` profile's default schedule simply rotates `hyper["strategies"]` over the arms.

## Selection, calibration and envelopes (M7, M4)

`wired_deps` injects `ci_lab.campaign.rrsi_wiring`, backed by `ci_lab.rrsi` and `ci_lab.oes`:

| Field | Implementation |
| --- | --- |
| `schedule` | RRSI Alg. 1 `plan_round`: annealed edit budget `b_max → b_min` (or fixed `budget`), stall detection against δ, exploration/prune of components, Thompson strategy allocation with floors. Directives carry `arm`, `component`, `strategy`, `budget` plus `explore`, `focus`, `avoid`, `stalled`, `strategy_reason`. |
| `select` | RRSI Alg. 2 `rrsi.selection.select`: cost rule `dC ≤ β0 + β1·dS` with a paired-bootstrap CI lower bound when `dS > δ`, the weighted rule `w_s·dS − w_c·dC + w_n·novelty > 0` otherwise, the floor `S' ≥ S* − δ` (best-so-far `S*` from history), and non-compensatory guards (critical safety violations, critic, base commit = incumbent). Pin/split mismatches or a high incumbent missing rate give `rerun`. The verdict keeps the full decision under `rrsi` plus per-arm `attribution`. |
| `calibrate_delta` | `rrsi.stats.aa_delta`: q-quantile (`aa_quantile`) of bootstrapped A/A `|dS|` over per-case means, floored at `1/M`. |
| `confirm_test` | Pre-registered one-sided paired case-level bootstrap (`rrsi.stats.confirm_test`, level α) plus safety non-inferiority on per-case critical-violation rates; `ship` only if significant and safe. |
| `build_envelope` | `ci_lab.oes.build` calibration/round/confirm envelopes (evaluator pin, split hashes, schedule, RRSI params, `SelectionExt` candidates). Record `extensions` (e.g. the guard arm's `com.microsoft.ci.guard`) are merged, the envelope is re-sealed and validated with `ci_lab.oes.validate` against the vendored OES core and extension schemas. An invalid envelope, or a `ship` record whose envelope does not decide `ship`, raises: envelopes are written before decisions and the frontier, so nothing unvalidated is ever promoted. |

RRSI knobs come from `hyper`: `arms` → `n_arms`, `max_rounds` → `T`, `k`, `seed`, `strategies`, `budget` (fixed) or `b_min`/`b_max`, on top of the `rrsi_profile` profile (default `local`); `hyper["rrsi"]` overrides any `rrsi.params.Hyperparams` field (e.g. `{"n_bootstrap": 2000}`).

**Round context.** `schedule` and `select` keep their `CampaignDeps` signatures; the round steps (and `readjudicate`) pass `{**hyper, "_round": {round, eid, delta, incumbent_commit, history}}` (`deps.ROUND_CONTEXT`). `delta` is the immutable calibrated δ from `calibration.json` (schedule's stall test), `history` the earlier `history.jsonl` rows. Implementations that do not need it (the `fake` defaults) ignore it.

**History rows.** Each `history.jsonl` row carries `incumbent_score`, `score_next`, `s_star` (when known) and, per arm, `arm`, `component`, `hypotheses`, `accepted`, `score`, `status` plus the RRSI fields `strategy`, `edits`, `evaluated`, `cost`, `delta_s`, `delta_c`, `novelty`, `admissible` and `reasons` (from the selection trace/attribution). Older rows without these fields still parse: they become `agent` arms with edits built from `component`/`hypotheses` and unmeasured deltas, and an unknown start score disables stall detection for those rounds.

## Durability

- **Checkpoints.** Each workflow checkpoints to `FileCheckpointStorage`, with the declarative state classes allowlisted:
  - round: `CI_RUN_DIR/<eid>/ckpt`
  - arm: `CI_RUN_DIR/<eid>/<arm>/ckpt`
  - calibration: `CI_RUN_DIR/<cid>-cal/ckpt`
  - confirm: `CI_RUN_DIR/<cid>-confirm/ckpt`
- **Run status.** `ckpt/run.status` records `running` or `completed`. `run_or_resume` resumes from the latest checkpoint when a previous run did not complete.
- **Step failures.** Steps raise `StepAborted`, a `BaseException`, because the declarative runner swallows `Exception`s. This stops the workflow at the failing superstep. Callers receive `StepFailed`.
- **Idempotent steps.** Every step first checks its durable marker, such as `slot.json`, `critique_N.json`, `eval.json`, `arm.done`, `selection.json`, `record.done` or `publish.done`.
- **Agents.** Agents are wrapped in `GatedAgent`, so an agent whose `proposal.json` or `analysis.json` already exists is not re-run.
- **External effects.** All ledger commits and every `gh`/`git` mutation go through the Outbox under `contracts.op_id` keys, with reconcile probes.
- **Resuming.** Rerunning `run` after a crash resumes from these checkpoints and markers.

### `run_arms` (C5, C8)

- Launches one `arm_<strategy>.yaml` workflow per arm with asyncio, bounded by `max_parallel_arms`.
- Re-evaluates the incumbent each round as the pseudo-arm `inc`, interleaved with the arms in a seeded random order that is persisted in `run_order.json`.
- Off the Copilot profile, incumbent evals may be served from the tree-keyed cache.
- Then reconciles the `arm.done` markers.
- An arm whose workflow fails `max_arm_attempts` times is recorded as `failed`.

## Tracing and live status (design §12, C27, C36)

- **One trace per round.** `ci.round` (`ci.campaign_id`, `oes.experiment_id`, `rrsi.round`, `ci.profile`, plus `oes.decision`) is a new root (`obs.span(..., new_trace=True)`). Calibration (`ci.calibrate`, with `rrsi.delta_s`) and confirm (`ci.confirm`) are their own traces as well.
- **Children.** `ci.arm` per arm (`oes.variant`, `ci.strategy`, `ci.score`) and `ci.step` per phase (`ci.phase`: every workflow step plus `analyst`/`propose` agent runs).
- **Resume.** A rerun after a crash starts a new root linked to the interrupted trace via `obs.previous_link(run_dir, eid)`; the link source is the `trace` recorded in the status markers.
- **Live markers.** `obs.write_status(run_dir, eid, writer=..., ...)` writes `<eid>/status.d/<writer>.json`: writer `round` (or `campaign` for calibration/confirm) holds `phase`, `state`, `round`, `campaign_id`, `heartbeat_s`, `decision`/`winner`; each arm worker (`v1`, …, `inc`) holds `arms.<arm> = {strategy, state, phase}`. Markers are rewritten at every phase transition and arm state change, and every `heartbeat_s` (default 30, max 60) while a step runs. Readers must aggregate with `obs.read_status`; `campaign status` includes a `live` summary of in-flight rounds.
- **Telemetry.** CLI entry calls `ci_lab.telemetry.setup("campaign", profile=..., run_dir=...)` when M12 is installed (an `ImportError` is ignored) and its `shutdown` on exit.

## Ledger (`experiments/`)

All paths below are under `campaigns/<cid>/`:

| Path | Contents |
| --- | --- |
| `campaign.json` | Campaign settings. |
| `frontier.json` | Current incumbent; updated by compare-and-swap. |
| `calibration.json` | Calibration result; the delta is immutable. |
| `calibration/envelope.json` | Validated OES calibration envelope. |
| `history.jsonl` | One row per round (RRSI history; see above). |
| `rounds/<eid>/{envelope,decisions,evals}.json` | Per-round records; `envelope.json` is a validated OES round envelope. |
| `stack.json` | Accepted layers and the native stack number. |
| `confirm.json`, `confirm/envelope.json` | Confirm decision and its validated OES envelope. |
| `land.json` | Land result. |

The global `holdout-looks.jsonl` sits at the ledger root, keyed by `cid|dataset_hash`. It is written with `ci_lab.ledger.looks.record_look`, which raises `LookBudgetExceeded` past the planned looks. Round decisions are recorded at decision time with `ci_lab.ledger.decisions.record_decisions`, which checks verdicts and emits the `record` step span.

## Publishing (`ci_lab.publish.github`)

`GitHubPublisher` shells out to `gh`/`git` with argv lists, never a shell.

Every dynamic value is validated:

| Value | Rule |
| --- | --- |
| PR number | Digits only |
| SHA | 40 or 64 lowercase hex characters |
| Ref | Matches `^[A-Za-z0-9._/-]+$`, follows git ref rules and passes `git check-ref-format` |
| owner/repo | Each part matches `^[A-Za-z0-9._-]+$` |
| Title/body | No control characters |

Each mutation runs inside `outbox.run_once` as snapshot → revalidate → mutate → verify.

Publishing a round:

1. **Accepted arm:** push `exp/<eid>/<arm>`, then open a PR whose base is the previous accepted layer. The bottom layer's base is `main`.
2. **Native stacks:** at the second accepted layer, `POST repos/{o}/{r}/stacks -F pull_requests[]=…`. Each later layer is appended with `POST repos/{o}/{r}/stacks/{n}/add`. Membership is probed with `GET repos/{o}/{r}/stacks?pull_request=N`.
3. **Losers:** push the archive tag `exp-archive/<eid>/<arm>`; losers get no PR.
4. **Land:** run `gh pr ready` on every layer, then `gh pr merge <top> --merge --auto`.

**Dry-run:** mutating calls are recorded in `calls` and in the optional JSONL journal. Dry-run op ids are namespaced so they never satisfy a real run.

## Scheduled campaigns (`.github/workflows/campaign-scheduled.yml`)

Rounds run weekly (cron `41 9 * * 1`) or on dispatch. The model and the write token never share a job:

| Job | Permissions | Does |
| --- | --- | --- |
| `opt_in` | none | Ends the run with a notice unless the repo variable `CI_HARNESS_ENABLED` is `true` ([template.md](template.md)); every other job depends on it. |
| `gate` | none | Picks the campaign from the dispatch input or the `CAMPAIGN_ID` repo variable (unset: the run is a no-op). Takes rounds from the input, else `CAMPAIGN_ROUNDS`, else 1. Validates both. |
| `round` | `contents: read`, `copilot-requests: write` | Restores `experiments/campaigns/<cid>` from `exp-ledger/<cid>`, then runs `campaign status \|\| new`, `calibrate` and `run --rounds <recorded + N> --defer-publish` (`--rounds` is absolute; rounds the ledger records are skipped on the fresh runner). Uploads the ledger and a git bundle of the arm commits. |
| `publish` | `contents: write`, `pull-requests: write`, environment `campaign-publish` | Runs `python3 -I -B scripts/campaign_publish.py`: system Python, standard library only, no `uv` and no third-party code. Then commits the ledger to `exp-ledger/<cid>` and opens a DRAFT PR for it. |

**Deferred publishing (`ci_lab.publish.deferred`):** `run --defer-publish PATH` swaps in a `DeferredPublisher`. It appends each `publish_round` request to a JSONL file and leaves `stack.json` untouched. `ci-lab campaign publish <cid> --requests PATH` (or the script) replays the requests in order. The replay is idempotent: eids already in `published.jsonl` are skipped.

`campaign_publish.py` trusts nothing from the `round` job. Before publishing it checks:

- At most 20 requests, and every eid is `<cid>-r…`.
- The winner is `None` or one of the heads.
- Heads are full SHAs of commits that arrived in the bundle.
- Each head has exactly one merge base with `origin/main`, no merge commits after it, and at most 200 commits. Every one of those commits (not only the net diff) touches only `src/order_support/harness/`, so the pushed history carries no transient edits elsewhere.
- Title and body pass the publisher's text validation.
- The ledger artifact contains only regular files with safe names, within count and size caps, and its `campaign.json` names this campaign.

`stack.json` and `published.jsonl` are always restored from `exp-ledger/<cid>`, or from the checked-out commit when that branch does not exist (never from the artifact). Only the publish job writes them, so a forged stack cannot redirect PR bases or stack edits. Every PR is a DRAFT, and nothing merges itself. Landing stays a manual `ci-lab campaign land`.

**Incomplete publishes.** `exp-ledger/<cid>` only advances after a full publish. If the publish job fails half-way (for example a PR call fails after the arm branch was pushed), the next `round` job finds `exp/<eid>/*` branches or `exp-archive/<eid>/*` tags for a round the ledger does not record and fails before running any model. Fix it by re-running the failed publish job: the artifact is kept 14 days and the replay is idempotent.

To enable it, set the `CI_HARNESS_ENABLED` repo variable to `true` and the `CAMPAIGN_ID` repo variable (and optionally `CAMPAIGN_ROUNDS`, 1–9), then create the `campaign-publish` environment, ideally with required reviewers. `tests.yml` runs the pytest suite (live/Copilot-marked tests excluded) and the dashboard canvas tests on every PR and on pushes to `main`.

## Injection (`ci_lab.campaign.deps.CampaignDeps`)

The driver codes only against `contracts.py`, and every collaborator from another module is a field on `CampaignDeps`:

| Owner | Fields |
| --- | --- |
| M6 | `ledger`, `outbox`, `provision_slot`, `head_commit`, `harness_tree`, `resolve_incumbent` |
| M8a | `make_agent`, `critique` |
| M10 | `get_strategy`, `strategy_kwargs` |
| M7 | `schedule`, `select`, `calibrate_delta`, `confirm_test` |
| M4 | `build_envelope` |
| M1 | `build_workflow`, `run_or_resume` |
| M8b | `publisher` |

`copilot`/`offline` get M7 and M4 from `ci_lab.campaign.rrsi_wiring`. `ci_lab.campaign.defaults` keeps deterministic stand-ins for the `fake` profile and wiring tests only; they are not a faithful Alg. 2 and their envelopes are not validated OES.

## Open items

- The response shape of the native stacks REST API is parsed defensively. It is unverified against live GitHub.
- The land mechanism is unverified: `gh pr merge --auto` on the top PR of a native stack.
- The `copilot` profile has live evidence for evaluation only: the frozen test sets in `evals/assert/<suite>/test_set.jsonl` were generated by live ASSERT runs with `ORDER_AGENT_PROFILE=copilot` (gpt-5-mini generator and tester, System-1 llama.cpp judge). No live multi-round campaign is recorded in the repo; campaign tests use fakes or a monkeypatched chat client factory.
- Wired deps give `FileLedger` no git committer (files only).
- The ASSERT child registers the `s1` judge provider before judging, and every suite defaults its judge to `s1/llamacpp/qwen3.5-4b` ([judge.md](judge.md)).
