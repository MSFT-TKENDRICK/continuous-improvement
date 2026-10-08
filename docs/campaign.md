# RRSI campaigns (M8b)

A campaign runs RRSI rounds over the harness. Each round is recorded as an OES
experiment and driven by declarative, expression-free MAF workflows. It covers
design §5, §8 and §9 (C5, C6, C8, C14), and design-oes-rrsi-v1 §3 and §5.

```
ci-lab campaign new          <cid> --profile copilot|offline|fake [--hyper k=v ...]
ci-lab campaign calibrate    <cid>     # A/A runs -> immutable delta
ci-lab campaign run          <cid> [--rounds N] [--stop-file PATH]
ci-lab campaign status       <cid>
ci-lab campaign readjudicate <cid> <eid>   # recompute selection from the ledger; exit 1 on drift
ci-lab campaign confirm      <cid>     # sealed held-out, one global look
ci-lab campaign land         <cid>     # requires confirm decision "ship"
```

Common flags:

| Flag | Meaning |
| --- | --- |
| `--run-dir` | Run root. Defaults to `$CI_RUN_DIR`, then `artifacts/ci-runs`. |
| `--ledger-dir` | Ledger root. |
| `--repo owner/name` | GitHub repository for publishing. |
| `--dry-run-publish` | Record the intended `gh`/`git` calls instead of running them. |

The `fake` profile is fully offline: stub domain, fake slots and FakeChatClient agents. Publishing is always dry-run, and all state lives under `<run-dir>/_fake`.

The `copilot` and `offline` profiles exit with code 2 ("integration pending") until `load_deps` is wired to M1/M4/M6/M7/M8a.

## Workflows (`src/ci_lab/workflows/*.yaml`)

The workflows contain only `InvokeFunctionTool` and `InvokeAzureAgent` actions with literal arguments. They use no `=` expressions and no `If`, `ConditionGroup`, `Foreach` or `Goto`. `ci_lab.workflows.assert_expression_free` enforces this.

| file | actions |
| --- | --- |
| `round.yaml` | `begin_round` → Analyst → `run_arms` → `select` → `record` → `publish` |
| `arm.yaml` | `provision_slot` → Proposer → `critique_1` → `repair_1` → `critique_2` → `repair_2` → `critique_final` → `evaluate` → `finalize_arm` |
| `calibrate.yaml` | `aa_runs` → `delta` → `record` |
| `confirm.yaml` | `reserve_look` → `evaluate_heldout` → `decide` → `record` |

Dynamic context never appears in the YAML. The run dir, eid and arm are bound into the registered tool closures (`ci_lab.workflows.steps`), and the YAML passes only `step:` names, `attempt:` numbers and `split:` names.

Repairs are unrolled. `repair_N` is a no-op when `critique_N` passed. Otherwise it re-invokes the proposer through an injected callable.

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

- Launches one `arm.yaml` workflow per arm with asyncio, bounded by `max_parallel_arms`.
- Re-evaluates the incumbent each round as the pseudo-arm `inc`, interleaved with the arms in a seeded random order that is persisted in `run_order.json`.
- Off the Copilot profile, incumbent evals may be served from the tree-keyed cache.
- Then reconciles the `arm.done` markers.
- An arm whose workflow fails `max_arm_attempts` times is recorded as `failed`.

## Ledger (`experiments/`)

All paths below are under `campaigns/<cid>/`:

| Path | Contents |
| --- | --- |
| `campaign.json` | Campaign settings. |
| `frontier.json` | Current incumbent; updated by compare-and-swap. |
| `calibration.json` | Calibration result; the delta is immutable. |
| `calibration/envelope.json` | OES envelope for calibration. |
| `history.jsonl` | One row per round. |
| `rounds/<eid>/{envelope,decisions,evals}.json` | Per-round records. |
| `stack.json` | Accepted layers and the native stack number. |
| `confirm.json`, `confirm/envelope.json` | Confirm decision and its envelope. |
| `land.json` | Land result. |

The global `holdout-looks.jsonl` sits at the ledger root, keyed by `cid|dataset_hash`.

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

## Injection (`ci_lab.campaign.deps.CampaignDeps`)

The driver codes only against `contracts.py`, and every collaborator from another module is a field on `CampaignDeps`:

| Owner | Fields |
| --- | --- |
| M6 | `ledger`, `outbox`, `provision_slot`, `head_commit`, `harness_tree`, `resolve_incumbent` |
| M8a | `make_agent`, `critique` |
| M7 | `schedule`, `select`, `calibrate_delta`, `confirm_test` |
| M4 | `build_envelope` |
| M1 | `build_workflow`, `run_or_resume` |
| M8b | `publisher` |

`ci_lab.campaign.defaults` provides trivial stand-ins for M4 and M7. They are not a faithful Alg. 2.

## Open items

- The response shape of the native stacks REST API is parsed defensively. It is unverified against live GitHub.
- The land mechanism is unverified: `gh pr merge --auto` on the top PR of a native stack.
- The `copilot` and `offline` profiles need the integration wiring in `load_deps`.
