# SkillOpt-Sleep nightly

`ci_lab.sleep` runs SkillOpt-Sleep 0.2.x over two repo-root harness skills. A night produces a
digest-pinned bundle; a separate privileged job validates the bundle with the standard-library-only
publisher and opens a draft pull request. Nothing merges automatically.

## Targets

`src/ci_lab/sleep/targets.yaml` defines the enabled targets:

| Target | Skill | Owner agent | Tasks |
|---|---|---|---|
| `harness-editing` | `harness/skills/harness-editing/SKILL.md` | `proposer` | `experiments/sleep/harness-editing.jsonl` |
| `trace-triage` | `harness/skills/trace-triage/SKILL.md` | `failure_analyst` | `experiments/sleep/trace-triage.jsonl` |

Both targets use the frozen harness evaluator. Registry validation restricts skills to the intended
repo-root skill paths and tasks to `experiments/sleep/`.

## Pipeline

`run_night(cfg, deps)` executes the expression-free workflow in `src/ci_lab/sleep/sleep.yaml`:

```text
harvest → consolidate → assert_gate → record → bundle
```

- **harvest** reads only reviewed tasks, accepts evolve-split AGL exports, deduplicates records,
  creates a stable train/validation split, validates judge operations, and strips raw tool output.
- **consolidate** runs `skillopt_sleep.dream.dream_consolidate` through `SleepBackend`.
- **assert_gate** compares incumbent and candidate with the frozen harness evaluator. Safety is
  non-compensatory and the calibrated score threshold must pass.
- **record** writes the OES sleep envelope and updates the sleep ledger state.
- **bundle** writes the candidate patch, experiment, results, and manifest with digest and size for
  every file.

The fake profile uses deterministic dependencies. Offline endpoints must be loopback. Budget limits
cover tasks, rollouts, tokens/AIU, and wall time; exceeding a limit keeps partial results and does not
publish a candidate.

## Backend and evaluation

`SleepBackend` implements the SkillOpt backend protocol directly. For each target it runs the
configured harness owner agent against a temporary harness copy containing the candidate skill. The
scorer uses the harness domain, and the reflector receives typed failures and returns typed edits.
Target code cannot supply evaluator measurements.

The optional lessons hook is off by default. When enabled, it converts judged evolve-split rollouts
to typed trajectories, mines a local store, and adds sanitized lesson proposals to the bundle.
Transcripts, tool arguments, excerpts, and member ids do not leave the hook. Proposals are never
adopted automatically.

## Scheduled workflow

`.github/workflows/sleep-nightly.yml` begins with the permissionless `opt_in` gate. It runs only when
`CI_HARNESS_ENABLED=true`.

1. **gate** checks whether enough new reviewed tasks exist; manual dispatch forces a run.
2. **evaluate** resolves the project, runs the harness sleep command, and uploads the bundle and
   redacted span artifact.
3. **publish** runs only for an accepted candidate or ledger update. It downloads the bundle and
   invokes `python3 -I scripts/sleep_publish.py`.

`usage-harvest.yml` downloads the latest successful span artifact, builds a bounded usage bundle,
and opens a separate draft pull request for pending tasks.

## Publisher boundary

`scripts/sleep_publish.py` checks bundle shape, digests, base SHA, patch syntax, allowlisted paths,
`git apply --numstat`, a clean checkout, and `git apply --check`. Accepted sleep patches may change
only the two target skill files plus sleep ledger files. Usage bundles may change only pending task
files. The publisher rebuilds the pull-request body from sanitized fields and never merges.

## Local commands

```powershell
uv run ci-lab sleep dry-run
uv run ci-lab sleep run --profile fake --out out/sleep-bundle
uv run pytest tests/ci_lab/sleep tests/ci_lab/template/test_workflows.py
```

Model pins, telemetry variables, and the scheduled opt-in are documented in
[models.md](models.md), [telemetry.md](telemetry.md), and [template.md](template.md).
