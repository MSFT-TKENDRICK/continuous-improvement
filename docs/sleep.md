# SkillOpt-Sleep nightly (M9)

`ci_lab.sleep` runs [SkillOpt 0.2.0](../.venv/Lib/site-packages/skillopt_sleep) *SkillOpt-Sleep*
consolidation over the order-support skill once per night in GitHub Actions. The night produces
a digest-pinned **bundle**. A separate privileged job validates the bundle with a stdlib-only
script and opens a **draft** PR. Nothing is ever merged automatically.

## Pipeline

`run_night(cfg, deps)` executes the expression-free declarative workflow
`src/ci_lab/sleep/sleep.yaml`:

```
harvest → consolidate → assert_gate → record → bundle
```

It uses the MAF declarative runner (`ci_lab.maf.workflows`) when that module is present, and
otherwise a sequential fallback. Steps receive their context through closures; the YAML holds
only literal arguments.

| step | what it does |
|---|---|
| `harvest` | Reads reviewed tasks from `experiments/sleep/tasks.jsonl` (`skillopt_sleep.tasks.v1`). Every row must have `reviewed: true`. Also reads AGL journal exports, which must be the **evolve** split; any other split is a hard error (C15). The step dedupes, makes a stable-hash train/val split, validates judge ops (unknown ops are rejected) and strips raw tool output. |
| `consolidate` | Calls `skillopt_sleep.dream.dream_consolidate(OrderSupportSleepBackend, …, gate_mode="on")`. |
| `assert_gate` | Runs an ASSERT evaluation of the incumbent and the candidate. A candidate is accepted only if all three hold: (1) safety violations do not increase (non-compensatory); (2) the bootstrap LCB of Δscore is greater than δ from the latest calibration (C16); (3) the hidden canary trigger tests pass (C12). |
| `record` | Builds the OES sleep envelope and updates `experiments/sleep/state.json`: the night counter, the per-target status, and the task watermark. State persists through the PR, not on the runner (C10). |
| `bundle` | Writes `out/sleep-bundle/{candidate.patch, experiment.json, results.json, manifest.json}`. The manifest records the sha256 digest and size of each file, the base commit SHA, the status, and the `accepted`/`ledger_update` flags. |

`OrderSupportSleepBackend` (`backend.py`) implements the `skillopt_sleep` `Backend` protocol
directly (C3), not through `CliBackend`:

- `attempt` / `attempt_with_tools` run the order-support agent. The candidate skill is written to
  a temporary harness copy at `skills/order-support/SKILL.md`.
- `judge` combines the deterministic safety oracle with an ASSERT-derived scorer. This scorer is a
  diagnostic signal for SkillOpt only.
- `reflect` calls the *SleepReflector* agent. The agent receives typed `FailureRecord`s and returns
  typed `EditRecord`s (C12).

Every external capability is injected through `SleepDeps`, a dataclass of callables. The
`fake` profile wires deterministic fakes. The `copilot` and `offline` profiles probe for the real
modules (`wiring.py`).

`budget.py` enforces hard ceilings on tasks, rollouts, tokens/AIU and wall clock. When a ceiling
is hit it raises `BudgetExceeded`. The night then ends with status `budget_exceeded` and keeps its
partial results.

**M16 lessons hook (`lessons_hook.py`).** When `SleepConfig.lessons_hook=True` (CLI `--lessons`,
or env `SLEEP_LESSONS=1|true|yes|on`; **off by default**), each target runs a `lessons` step right
after `dream_consolidate`. The step works as follows:

1. The night's judged evolve-split rollouts (`backend.judged_rollouts()`: case id, suite, transcript,
   oracle violations, judge rule ids, pass/fail) become `ci_lab.rulespec.Trajectory` records through
   the `ci_lab.lessons` harvest adapters. Each record is sliced by night date and pinned
   `sleep-<profile>`.
2. The records are appended (deduplicated by id) to a **local** trajectory store,
   `<store>/<target>/trajectories.jsonl`. The default store is `<work-dir>/lessons`; use
   `--lessons-dir artifacts/lessons/sleep` (gitignored) to accumulate across local nights.
3. Optional local inputs are harvested into the same store: `--lessons-source usage:DIR`, `agl:`,
   `spans:`, `assert:` or `pr:`. Usage data is reduced to typed features and stays untrusted (B3).
4. `ci_lab.lessons` mines and routes the store.

Only a sanitized, typed summary leaves the hook (§13.6 B8). It holds cluster id, rung, status,
support/family/slice counts, oracle/rubric ids, tool-name n-grams and typed feature slots. It never
holds transcripts, tool arguments, excerpts or member ids. When any target has candidates, the night
adds `experiments/sleep/lessons/<night_id>.json` (`ci_lab.sleep.lessons.v1`, `status: proposed`) to
the bundle. `experiment.json` records per-target counts under `targets.<name>.lessons`.

The publisher opens the usual **draft** PR and notes that lesson candidates are proposals only:
nothing is adopted or enforced automatically. A human confirms clusters with
`ci-lab lessons confirm`, then synthesizes and validates rules in a separate reviewed PR (§13.3).

The hook is fail-soft: an exception is recorded as `{"error": "<type>"}` and the night continues.
Clustering needs support, family and slice convergence, so a single ephemeral CI night usually
yields no candidates. Accumulate a local store (`--lessons-dir`) for useful results.

## Skill targets registry

`src/ci_lab/sleep/targets.yaml` (`ci_lab.sleep.targets.v1`) lists the skill targets. Each target
has a name, a skill path, a memory path, an owner agent, an ASSERT eval suite and a tasks file.
The night iterates over all enabled targets and opens one child span per target. The night is
`accepted` if any target is accepted. Use `--targets FILE` to override the registry. At runtime,
`wiring.SUPPORTED_OWNERS` currently supports only the `order-support` owner, and the publisher
only allows skill paths under `src/order_support/harness/skills/`.

## Usage-driven tasks (§11.3, C22)

`traces.py` defines a `TraceSource` protocol with three implementations:

- `agl:` AGL journal JSONL;
- `spans:<run_dir>`, which reads `telemetry/spans-*.jsonl`;
- `artifacts:<dir>`, a downloaded Actions artifact directory.

Each trace is processed as follows:

1. **Redaction.** Known customer identities, emails, phone numbers, addresses, card numbers, ZIP
   codes and secrets/tokens are redacted. Order ids are hashed to `order-<sha8>`.
2. **Prompt-injection filter.**
3. **Pending task.** The trace becomes a pending row in
   `experiments/sleep/<tasks>.pending.jsonl` with `reviewed: false`, an empty judge and no
   reference.

Pending rows are **never** consumed by SkillOpt-Sleep. A human writes the `reference` and `judge`,
sets `reviewed: true`, moves the row into the reviewed tasks file, and merges. Those steps are
what admit a row.

`ci-lab sleep harvest-usage` writes a `kind: usage` bundle. With `--open-pr` it instead calls `gh`
locally to create the draft PR on `exp/usage-<date>/tasks`.

## CLI

```
ci-lab sleep run --profile copilot|offline|fake --out DIR [--max-tasks N --max-minutes M]
                 [--targets YAML --run-dir DIR --tasks-file F --agl-export F ...]
                 [--lessons --lessons-dir DIR --lessons-source SOURCE:PATH ...]   # $SLEEP_LESSONS
ci-lab sleep dry-run            # harvest + validate only
ci-lab sleep usage-gate         # threshold: --threshold / $SLEEP_USAGE_THRESHOLD (default 1); --force / $SLEEP_FORCE
ci-lab sleep harvest-usage --source agl:P|spans:RUN_DIR|artifacts:DIR [--bundle DIR | --open-pr]
ci-lab sleep redact-spans --run-dir DIR --out FILE
```

## Tracing and live status

Each night is one trace. The root span is `obs.span(SPAN_SLEEP_NIGHT, {ATTR_NIGHT: …},
new_trace=True, links=obs.previous_link(...))`, with one child span per target (`sleep.target`).

Live phase markers are written with `obs.write_status(run_dir, "sleep-<night>", writer="night",
phase=…)`. Read them back with `obs.read_status`.

`ci-lab sleep run` calls `ci_lab.telemetry.setup("sleep", …)` lazily and only warns if
telemetry is missing or fails.

## GitHub Actions (C10)

`.github/workflows/sleep-nightly.yml` runs on cron `17 7 * * *` and on `workflow_dispatch` (no
inputs). Top-level permissions are `{}`, every action is pinned to a full commit SHA, and checkout
always uses `persist-credentials: false`.

1. **`gate`** (`contents: read`) runs `ci-lab sleep usage-gate`. It checks whether the number of
   new reviewed tasks since the last recorded night is at least the threshold
   (`vars.SLEEP_USAGE_THRESHOLD`). A manual dispatch forces the run.
2. **`evaluate`** (`contents: read`, `copilot-requests: write`) does the following:
   - runs `uv sync`;
   - starts a loopback `agl-server` with a random masked key;
   - runs `ci-lab sleep run --profile copilot` with `COPILOT_GITHUB_TOKEN=${{ github.token }}`
     and `SLEEP_LESSONS=${{ vars.SLEEP_LESSONS }}` (the lessons hook; unset means off);
   - uploads two artifacts: `sleep-bundle`, and `sleep-spans` (the redacted span JSONL, which can
     be replayed with `ci-lab telemetry import`).
3. **`publish`** (`environment: sleep-publish`, `contents: write`, `pull-requests: write`) runs
   only if the night was accepted or `ledger_update` is set. It installs nothing and restores no
   caches. It checks out `github.sha`, downloads the bundle, and runs
   `python3 -I scripts/sleep_publish.py`.

`.github/workflows/usage-harvest.yml` runs daily. Its **`harvest`** job (`contents: read`,
`actions: read`) downloads the latest `sleep-spans` artifact, runs `harvest-usage --bundle`, and
uploads `usage-bundle`. Its **`publish`** job is the same stdlib-only validator, which opens a
draft PR on `exp/usage-<date>/tasks`.

### `scripts/sleep_publish.py` (stdlib only)

The publisher refuses the bundle (prints `REFUSED: …` and exits 1) unless all of these hold:

- The bundle contains exactly four regular files within the size limits.
- The manifest format, kind, date, night id and flags are valid.
- Every sha256 digest and size matches.
- `base_sha` equals the checked-out `HEAD`.
- The patch contains only plain edits and new files: no mode, rename, copy, delete or binary
  changes.
- Every path is allowlisted:
  - `kind: sleep` may touch `experiments/sleep/**`, and `src/order_support/harness/skills/**`
    only when the night was accepted;
  - `kind: usage` may touch only `experiments/sleep/*.pending.jsonl`.
- `git apply --numstat` agrees with the parsed paths.
- The checkout is clean.
- `git apply --check` succeeds.

It then runs `git apply --index` on `exp/sleep-<yyyymmdd>-<run_attempt>/cand`, commits with
`Sleep-*` trailers, pushes, and runs `gh pr create --draft`. The PR body is rebuilt from
sanitized fields, never copied from the bundle. If a PR for the branch already exists it is
reused. The publisher **never merges**.
