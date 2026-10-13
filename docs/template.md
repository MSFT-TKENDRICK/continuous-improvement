# Use this repository as a template

This repository is a GitHub template. Create your own copy, point it at your agent, and let the
harness keep improving that agent: RRSI/OES campaigns propose and test harness edits, SkillOpt-Sleep
consolidates skills nightly, and every change lands as a draft PR that a human reviews.

The copy starts with the example agent (`order_support`) and its frozen ASSERT test sets, so the
whole loop works before you change anything. `ci-lab template init` makes the copy yours and
`ci-lab template doctor` tells you what is still missing.

## Contents

* [Quick start](#quick-start)
* [What `template init` changes](#what-template-init-changes)
* [Repository settings](#repository-settings)
* [Swap in your own agent](#swap-in-your-own-agent)
* [How the harness then evolves your agent](#how-the-harness-then-evolves-your-agent)
* [Maintaining the template](#maintaining-the-template)

## Quick start

1. **Create the repository.** On GitHub, choose **Use this template → Create a new repository**.
   Do **not** tick **Include all branches**: GitHub then copies only the default branch, which is
   what you want. The other branches (`exp/*`, `exp-ledger/*`, `exp-archive/*` tags, stacked
   feature branches) are the template's own experiment history and mean nothing in your copy.
   Your new repository starts with a single squashed "Initial commit".
2. **Install.** Clone it and run `uv sync` (Python 3.12, see [harness.md](harness.md)).
3. **Initialize.** Pick the people or team who must review changes to the harness's protected
   paths, then preview and apply:

   ```powershell
   uv run ci-lab template init --owners "@my-org/agent-owners"          # dry run: prints the plan
   uv run ci-lab template init --owners "@my-org/agent-owners" --apply  # writes it
   ```

   The repository name comes from `git remote get-url origin` (or pass `--repo owner/name`).
   Without a terminal, run the **template-init** workflow instead (Actions → template-init → Run
   workflow, input `owners`). It pushes `template-init/<run id>` and opens a PR. Opening the PR
   needs **Allow GitHub Actions to create and approve pull requests** (Settings → Actions →
   General); otherwise open the PR from the pushed branch by hand. PRs opened by `GITHUB_TOKEN`
   do not trigger other workflows, so close and reopen the PR to run `tests.yml` and `lint.yml`.
4. **Check.** `uv run ci-lab template doctor` runs offline readiness checks and prints a fix for
   every failure (exit 1 while anything fails). `--skip-lint` skips the slower lint check.
5. **Commit and merge** the init changes through a PR.
6. **Configure the repository** ([settings](#repository-settings)): branch protection with
   **Require review from Code Owners**, the publish environments, and the repository variables.
7. **Enable the autonomous workflows** by setting the repository variable `CI_HARNESS_ENABLED` to
   `true`. Until then `campaign-scheduled`, `sleep-nightly`, `usage-harvest` and `governance-native` start, print a
   notice ("CI harness not enabled") and stop. `tests.yml`, `lint.yml` and `governance.yml` always run.

`template init` refuses to run in the template repository itself (the repository named in
`.github/template.yml`), and refuses to make the template's owner the owner of your copy.
`--force` overrides both; you should not need it.

## What `template init` changes

Dry run by default; `--apply` writes. Running it again with the same inputs plans nothing, and
running it with new `--owners` moves the owners over. It is pure Python (standard library only), and
makes no network calls.

| Path | Change | Why |
| --- | --- | --- |
| `.github/CODEOWNERS` | owners of every rule → `--owners` | Rules still owned by the template owner would make a stranger the required reviewer of your harness. Rules you already customized are kept and reported. |
| `.github/template.yml` | marker: `role: derived`, `initialized: true`, your repository and owners, the template repository, version (`pyproject.toml`), the commit init ran on, and the date | `doctor`, the CODEOWNERS test and a re-run of init read it. Pass `--template-commit <sha>` to record which template commit you copied. |
| `experiments/campaigns/` | deleted | Campaign ledgers (`stack.json`, `published.jsonl`) name the template's PRs and branches. |
| `experiments/sleep/{nights,envelopes,lessons}/`, `experiments/sleep/*.pending.jsonl` | deleted | Sleep nights, lesson proposals and unreviewed tasks harvested from the template's usage traces. |
| `experiments/sleep/state.json` | reset to night 0 | The night counter and history belong to the template. |
| `lessons/registry.yaml` | emptied | Lessons were learned from the template's traces. |

Kept by default: the example agent, its ASSERT suites and frozen test sets, schemas, lint rules,
guards, the human-reviewed sleep tasks (`experiments/sleep/tasks.jsonl`) and the held-out look
ledger (`experiments/holdout-looks.jsonl`). The look budget is keyed by the test set's hash; the
inherited test sets have already been looked at, so forgetting those looks would overstate how
fresh your held-out data is.

`--reset-state` also empties `tasks.jsonl` (header only) and deletes `holdout-looks.jsonl`. Use it
when you replace the example agent and its frozen test sets anyway. `--keep-example` names the
default explicitly.

Not changed: the JSON schema `$id` URLs under `schemas/` keep the template's repository. They are
identifiers that record where the format was defined, not links your copy has to own. Renaming the
Python project in `pyproject.toml` is optional.

## Repository settings

`doctor` checks that every setting the workflows read is listed here.

**Repository variables** (Settings → Secrets and variables → Actions → Variables):

| Variable | Used by | Meaning |
| --- | --- | --- |
| `CI_HARNESS_ENABLED` | `campaign-scheduled`, `sleep-nightly`, `usage-harvest`, `governance-native` | `true` to opt in. Anything else: the `opt_in` job prints a notice and the workflow does nothing. |
| `CAMPAIGN_ID` | `campaign-scheduled` | Campaign to advance on the weekly schedule (unset: no campaign runs). See [campaign.md](campaign.md). |
| `CAMPAIGN_ROUNDS` | `campaign-scheduled` | Rounds per run, 1–9 (default 1). |
| `SLEEP_USAGE_THRESHOLD` | `sleep-nightly` | New reviewed tasks needed before a night runs (default 1). See [sleep.md](sleep.md). |
| `SLEEP_LESSONS` | `sleep-nightly` | Lessons hook for the night (unset: off). See [lessons.md](lessons.md). |
| `CI_TELEMETRY` | `sleep-nightly` | OTel opt-in for the agent during evals (`auto`, `1`, `true`, `on`; unset: off). See [telemetry.md](telemetry.md). |

**Environments** (Settings → Environments), ideally with required reviewers: `campaign-publish`
(the campaign publish job) and `sleep-publish` (the sleep and usage-harvest publish jobs).

**Secrets:** none. Model calls use the workflow's `GITHUB_TOKEN` with the `copilot-requests`
permission, and publish jobs use `GITHUB_TOKEN` with `contents`/`pull-requests: write`. Never commit
keys; local runs use ambient authentication (see [providers.md](providers.md)).

**Branch protection** on the default branch: require a PR, **Require review from Code Owners**
(CODEOWNERS only blocks merges with this), and the `pytest` (tests) and `lint` checks.

**Actions → General:** allow GitHub Actions to create pull requests if you use the template-init
workflow; the campaign and sleep publish jobs open draft PRs the same way.

## Swap in your own agent

The harness never imports the example agent directly: everything agent-specific goes through a
`Domain` (`ci_lab.contracts.Domain`) and the ASSERT suites. Replace these, in this order:

1. **The agent and its evolvable surface.** Put your agent in its own package (like
   `src/order_support/`) with a data-only harness directory (prompts, skills, tool specs, agent
   config, guards) that the agent loads at runtime. Campaigns and sleep only ever edit that
   directory; code stays frozen.
2. **A `Domain`.** Implement the protocol in `src/ci_lab/contracts.py` next to
   `src/ci_lab/domain/order_support.py`: `name`, `surface_globs` (the harness directory),
   `frozen_globs` (code, evals, experiments, workflows), `component_globs` (which files are the
   `prompt`, `skill`, `client_tool`, `config`, `memory`, `context_mgmt` components), `splits()`,
   `evaluate(...)` (run ASSERT on a harness directory) and `failures(...)`. Construct it where the
   example is constructed (`src/ci_lab/campaign/wiring.py`, `src/ci_lab/sleep/wiring.py`).
3. **ASSERT suites.** Copy a suite under `evals/assert/<suite>/`, rewrite `taxonomy.json` for the
   behaviors you care about, and point `pipeline.inference.target.callable` at your agent's entry
   point (the example uses `order_support.agent:chat`). Keep the System-1 judge
   (`pipeline.judge.model.name: s1/llamacpp/<model>`, [judge.md](judge.md)).
4. **Freeze the test sets.** Generate each suite once with ASSERT
   (`uv run order-support-evals run evals/assert/<suite>/eval_config.yaml`), review the generated
   cases, and copy `artifacts/results/<suite name>/test_set.jsonl` to
   `evals/assert/<suite>/test_set.jsonl`. Campaigns compare arms on these frozen sets; regenerate
   only deliberately, and use `--reset-state` (or delete `experiments/holdout-looks.jsonl`) when you
   do. Delete the example suites you no longer use.
5. **Sleep targets.** Point `src/ci_lab/sleep/targets.yaml` at your agent's skill, memory and
   reviewed tasks, and replace `experiments/sleep/tasks.jsonl` with tasks from your domain.
6. **Publish allowlists.** `scripts/campaign_publish.py` (`DEFAULT_ALLOW`) and
   `scripts/sleep_publish.py` (`SKILLS_PREFIX`) only publish edits under the example's harness
   directory. Point them at yours.
7. **Guards, lint rules, frozen paths.** Adapt the runtime guards in your harness's `guards/`
   directory and `src/ci_lab/guards/domains/`, add domain lint rules under `lint/rules/`
   ([lint.md](lint.md)), and update the frozen lists in
   `.github/extensions/ci-guardrails/policy.mjs` and `.github/CODEOWNERS` together
   (`tests/ci_lab/lint/test_codeowners.py` keeps them in sync).
8. **Check.** `uv run pytest`, `uv run ci-lab lint` and `uv run ci-lab template doctor`.

Doing this inside the harness's own loop works too: ask an agent to make the change in a session
with `CI_ALLOW_CONTRACT_EDIT=1`, review the PR as a code owner, and let the campaigns take over from
there.

## How the harness then evolves your agent

Once `CI_HARNESS_ENABLED` is `true`:

* **Campaigns** ([campaign.md](campaign.md)). Create a campaign (`ci-lab campaign new`), set
  `CAMPAIGN_ID`, and `campaign-scheduled` runs RRSI rounds weekly as OES experiments: several arms
  each edit one harness component, every arm is scored on the frozen ASSERT sets, and the winner is
  published as a draft stacked PR. You review and `ci-lab campaign land` it.
* **Sleep** ([sleep.md](sleep.md)). `sleep-nightly` consolidates the target skill from reviewed
  tasks and opens a draft PR when the candidate passes the suite's acceptance gate.
  `usage-harvest` turns real usage traces into pending tasks; a human reviews them before sleep
  ever sees them.
* **Lessons and guards** ([lessons.md](lessons.md), [lint.md](lint.md)). Recurring failures become
  lessons, lint rules or runtime guards, again only through reviewed PRs.

Nothing merges itself: every change to your agent is a draft PR that a code owner approves.

## Maintaining the template

For the template repository itself (not your copy):

* GitHub's template flag is a repository setting:
  `gh repo edit <owner>/<name> --template` (or Settings → General → Template repository).
* **Use this template** copies only the default branch. Merge stacked branches into the default
  branch before you expect users to get them.
* `.github/template.yml` says `role: template`; keep it that way so `template init` refuses here.
* The scheduled workflows are opt-in in the template too: set `CI_HARNESS_ENABLED=true` on the
  template repository to keep its own campaigns and nights running.
