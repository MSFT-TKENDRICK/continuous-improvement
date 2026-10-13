# Use this repository as a template

This repository is a GitHub template for the self-improving harness. A new copy starts with the
repo-root harness tree, five frozen harness ASSERT suites, measured rubrics, a deterministic fake CI
tier, scheduled campaign and sleep workflows, and frozen governance and validation assets.

The Python distribution is `ci-lab-harness`, its import package is `ci_lab`, the CLI is `ci-lab`,
and the only registered domain in the shipped template is `harness`.

## Quick start

1. Choose **Use this template → Create a new repository**. Leave **Include all branches** off.
2. Clone the repository and run `uv sync`.
3. Preview and apply initialization:

   ```powershell
   uv run ci-lab template init --owners "@my-org/agent-owners"
   uv run ci-lab template init --owners "@my-org/agent-owners" --apply
   ```

4. Run `uv run ci-lab template doctor`.
5. Commit the initialization changes through a pull request.
6. Configure branch protection, required Code Owner review, publish environments, and repository
   variables.
7. Set `CI_HARNESS_ENABLED=true` only when scheduled runs should start.

The manual `template-init` workflow performs the same standard-library-only initialization and opens
a pull request. Initialization refuses to modify the template repository itself unless `--force` is
passed.

## What initialization changes

| Path | Change |
|---|---|
| `.github/CODEOWNERS` | Rewrites template-owned rules to the supplied owners while preserving customized rules. |
| `.github/template.yml` | Records the derived repository, owners, template source, version, commit, and date. |
| `experiments/campaigns/` | Removes campaign history that belongs to the template repository. |
| `experiments/sleep/{nights,envelopes,lessons}/` | Removes prior sleep history and proposals. |
| `experiments/sleep/*.pending.jsonl` | Removes unreviewed harvested tasks. |
| `experiments/sleep/state.json` | Resets night counters and history. |
| `lessons/registry.yaml` | Clears lessons learned from template traces. |

By default initialization preserves reviewed harness sleep tasks and the held-out look ledger.
`--reset-state` empties the two harness task files and removes the look ledger when the frozen cases
are intentionally replaced.

Initialization does not remove or rewrite:

- the repo-root `harness/` candidate tree;
- `harness/harness.yaml` and its frozen copy `src/ci_lab/harness_tree/manifest.yaml`;
- `src/ci_lab/governance/policies/harness.acs.yaml`;
- the five `evals/assert/harness_*` suites, measured rubrics, or `evals/datasets/harness.yaml`;
- workflows, schemas, lint rules, and fake CI configuration.

## Repository settings

Shipped automation:

| Workflow | Trigger | Current behavior |
|---|---|---|
| `tests.yml` | pull requests; pushes to `main` | Full non-live pytest suite and Node canvas tests. |
| `lint.yml` | pull requests; pushes to `main` | Structural repository lint. |
| `governance.yml` | governance-related pull requests | Governance tests, `governance doctor`, and lint. |
| `web-chat.yml` | chat/canvas pull requests; pushes to `main` | Rebuilds the pinned CopilotKit bundle and rejects drift. |
| `template-init.yml` | manual | Standard-library initializer; pushes a branch and attempts a draft PR. |
| `campaign-scheduled.yml` | Monday 09:41 UTC; manual | Live Copilot campaign, deferred privileged publish. |
| `sleep-nightly.yml` | daily 07:17 UTC; manual | Usage-gated SkillOpt-Sleep, then privileged publish. |
| `usage-harvest.yml` | daily 05:41 UTC; manual | Converts the latest span artifact into pending-task changes and a draft PR. |
| `governance-native.yml` | Tuesday 06:17 UTC; manual | Informational native ACS build/parity test. |

Scheduled workflows are gated by `CI_HARNESS_ENABLED`. The campaign and sleep workflows also read
the pinned target and judge settings documented below.

| Variable | Purpose |
|---|---|
| `CI_HARNESS_ENABLED` | Enables scheduled campaign, sleep, usage-harvest, and native governance jobs. |
| `CAMPAIGN_ID` | Campaign advanced by the scheduled campaign workflow. |
| `CAMPAIGN_ROUNDS` | Number of rounds requested per scheduled run. |
| `CI_LAB_TARGET_MODEL` | Pinned live harness target model. |
| `CI_LAB_JUDGE_MODEL` | Pinned live System-1 judge model. |
| `CI_S1_LLAMA_URL` | System-1 llama.cpp endpoint used by live harness evaluation. |
| `CI_LAB_SLEEP_TARGET_MODEL` | SkillOpt target model. |
| `CI_LAB_SLEEP_REFLECTOR_MODEL` | SkillOpt reflector model. |
| `SLEEP_USAGE_THRESHOLD` | Reviewed tasks required before a scheduled sleep run. |
| `SLEEP_LESSONS` | Opts into the lessons hook. |
| `CI_TELEMETRY` | Opts into telemetry collection. |

Create `campaign-publish` and `sleep-publish` environments, preferably with required reviewers.
Enable **Require review from Code Owners** on the default branch. Workflows use `GITHUB_TOKEN` with
job-scoped permissions; no repository secret is required by the template defaults.

## Built-in harness target

The only built-in domain is the self-hosted harness. `ci_lab.domain.harness.HarnessDomain` loads
only `harness_*` suites named by `evals/datasets/harness.yaml`, validates candidate trees against the frozen
manifest, copies each case into an isolated writable directory, scrubs credentials, and records
quality plus resource measurements.

Validate a copy with:

```powershell
uv run ci-lab harness validate --dir harness
uv run ci-lab harness metrics --dir harness
uv run pytest tests/ci_lab/template tests/integration/test_reallib_assert.py
uv run ci-lab lint
uv run ci-lab template doctor
```

## Adding a custom domain

There is no configuration-only domain plug-in today. The shipped registry
`ci_lab.domain.DOMAIN_CHOICES` contains only `harness`, and campaign model preflight also validates
that name. Adding a domain is therefore a reviewed code change:

1. Implement `ci_lab.contracts.Domain`: `name`, `surface_globs`, `frozen_globs`,
   `component_globs`, `splits()`, `evaluate()`, and `failures()`.
2. Add its runner, frozen cases, measured rubrics, and dataset manifest. Measurements must be
   evaluator-owned, required values must fail closed, and quality/resource scores must remain
   separate.
3. Register the name in `ci_lab.domain.DOMAIN_CHOICES`/`get_domain` and extend campaign model
   preflight for its target and judge pins.
4. Define a bounded candidate root compatible with `ci_lab.domain.layout.harness_root`; keep
   evaluators, governance, selection, schemas, and workflow automation outside it.
5. Add ACS policy/tool adapters, publish-script path allowlists, CODEOWNERS, lint protections, model
   documentation, template-doctor checks, and fake/offline tests.
6. Update scheduled workflows only after the new domain works through the CLI. The default remains
   `harness` unless deliberately changed in the registry and workflow arguments.

Do not copy campaign arm workflows into the candidate tree. `harness/workflows/` is the current
target-agent workflow surface; arm, round, calibration, and confirmation workflows stay frozen in
`src/ci_lab/workflows/`.
