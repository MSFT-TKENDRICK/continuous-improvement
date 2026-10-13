# Task graphs with hidden rubrics (`ci_lab.taskgraph`)

A task graph breaks a goal into **singular deliverables**. Each deliverable is one output (a file,
text, JSON or a patch) with its own dependencies, budget and **sealed rubric**. The scheduler runs
every deliverable on the [agent bus](bus.md): a fresh student attempt per try, a panel of voters, a
deterministic judge and, optionally, adversaries ([adversary.md](adversary.md)). Ready deliverables
run concurrently. The student is never shown its rubric. It sees the rubric's sha256 commitment and,
after a rejection, a sanitized correction.

## Model (`model.py`)

```yaml
id: harness-change-review            # TaskGraph{id, goal, deliverables}
goal: Review a proposed harness change and summarize components and risks.
deliverables:
  - id: inventory                    # task id
    title: Component inventory
    depends_on: [brief]
    context: [{kind: deliverable, ref: brief}]     # kinds: file | deliverable | text
    output: {kind: json, path: out/inventory.json} # kinds: file | text | json | patch (+ schema for json)
    rubric_commitment: 15429bd6…                   # sha256 of the sealed rubric
    budget: {max_attempts: 2, timeout_s: 120}      # defaults: 3 attempts, 600 s, weight {llm: 1.0}
    instructions: |
      …
```

- A `Rubric` has `id`, `version`, `deliverable`, `criteria`, `pass_score` and `canary`.
  `version_id` is `<id>@v<version>`. `commitment` is the sha256 of the rubric's canonical JSON.
- A `Criterion` has `id`, `description`, `measure`
  (`deterministic | assert | metric | s1 | llm`), `check`, `threshold`, `weight` (default 1),
  `required`, and `role` (`quality | resource`). `deterministic`, `assert`, and `metric` criteria
  are **oracles**. Metric criteria are always resources: they are reported separately and never
  folded into the quality score.
- `StudentSpec.of(deliverable)` is the only view a student gets. It has the instructions, output,
  context, budget and commitment, but no rubric.
- `topo_order()` is a Kahn order with ties broken by declaration order. `load_graph` reads YAML or
  JSON.

## Validation (`validate.py`)

`ci-lab graph validate` reports `code`, `where` and `message` for each problem.
- **Graph checks.** Id format, the reserved id `sealed`, duplicate ids, self, unknown or duplicate
  dependencies, cycles, budgets and commitment format. Instructions must be at most 1,200 characters
  and must not name more than one output path (`output.ambiguous`). A `schema` is allowed only for
  `json` outputs. A `deliverable` context must also be a dependency.
- **Rubric checks** (with `--vault`).
  - A commitment must exist in the vault and match its file.
  - Each rubric needs at least one oracle criterion, a 16-hex-digit canary and version ≥ 1.
  - Deterministic checks must be `regex`, `json_schema`, `file_exists`, `command` or `python`.
  - `command` argv[0] must be in `COMMAND_ALLOWLIST` (`python`, `pytest`, `ruff`, `node`, `git`).
  - Soft questions are capped at 300 characters, and vague words ("good", "clean", "robust", …) are
    flagged.
  - `rubric.leak` fires when the deliverable's instructions share an 8-word n-gram with the rubric, or
    contain its canary, a criterion id or a suite name.

## Sealed rubric vault (`vault.py`)

`RubricVault` stores each rubric at `<run-dir>/sealed/<commitment>.json` as canonical JSON, so the
file's sha256 *is* the commitment. Sealing is atomic.
- `open(commitment, role=)` re-hashes the file on every read. It raises `VaultTampered`,
  `VaultMissing`, or `VaultDenied` when `role="student"`.
- `versions(rubric_id)` lists every sealed version, including hardened ones.

As defence in depth, `ci_lab.tools.paths.DENIED_GLOBS = ("**/sealed/**",)`. This makes every file
tool of an arm or student reject vault paths.

## Student firewall

- **`LeakScreen`** (`firewall.py`) detects rubric material in text: canaries, criterion ids, suite
  names, verbatim rubric text (at least 3 words), shared 8-grams, and an optional extra corpus.
  `hits()` names categories and never quotes the material. `redact()` replaces it with `[redacted]`.
- **`sanitize_correction(reasons, rubric, attempt=)`** turns the judge's reasons into a
  `StudentCorrection`. The text starts "The previous attempt was rejected:", has rubric material
  redacted and numbers stripped, and is at most 800 characters. If anything still leaks, it falls back
  to a generic correction.
- **`StudentFirewallMiddleware`** is a pair of MAF middlewares: one screens agent messages, the other
  screens tool results. A hit raises `ContextLeak`, a fail-closed `MiddlewareFailure`.
  [`bus.project.succeed`](bus.md#projection-and-succession-projectpy) installs it for every student.
- **`ci_lab.meta.brief`** applies the same sanitizer to the campaign proposer's documents. It renders
  `analysis`, `critique` and `history` as `StudentCorrection` JSON, redacts other briefs against the
  failure corpus (case ids, suites, scored criteria), and rewrites the `failure_analyst` subagent's
  answers.

## Scheduler (`scheduler.py`)

```python
await run_graph(graph, bus=bus, vault=vault, voters_for=..., student_factory=..., run_id="demo",
                pools=pools, max_parallel=None, challenger=None, hardener=None, quorum=1,
                escalator=None, timeout_s=120.0, hardener_scorer=None, adopt_epoch_patches=True)
```

1. **Manifest.** The scheduler writes the manifest to `<run>/_run`. On resume, the stored graph sha
   must match, and the stored quorum wins.
2. **Epoch adoption** (`adopt_hardened`). Right after the manifest, each pinned rubric is replaced by
   the newest hardened version from earlier runs in the same bus root. A candidate must be the target
   of an accepted `rubric_patch` (task topic or run topic) of another run, reachable from the pinned
   version through accepted patches, sealed in the vault, and newer than the pinned version.
   - Each adoption is a `rubric_patch` on `<run>/_run` with `applies_from_epoch = 0`, written as
     `hardener:epoch-adopt`.
   - A `note` with `data.epoch_adoption` (rubric id → version, possibly empty) closes the decision.
     On resume the decision is re-read from the bus and never recomputed.
   - `adopt_epoch_patches=False` records an empty decision and keeps the pinned rubrics.
3. **Fan-out.** One task per deliverable runs in an `asyncio.TaskGroup`. Each waits for its
   dependencies to settle.
   - If any dependency has no commit, the deliverable writes `abort {reason: "dependency_blocked"}`.
   - Otherwise it takes a slot of the `max_parallel` semaphore and runs its attempts. With
     `max_parallel=None` (the default) every ready deliverable runs at once.
4. **Attempts** run for n = 1 … `budget.max_attempts`. There are no other retries. Each attempt is an
   `effect("attempt", key="<task>@n")`, so a resumed run skips finished attempts and reconciles
   half-finished ones from the bus.
   - The rubric is frozen per attempt. It is the latest accepted `rubric_patch` with
     `applies_from_attempt ≤ n`, else the adopted baseline, else the sealed original.
   - The student (`succeed`, bounded by `budget.timeout_s`) runs concurrently with the challenger.
   - The votes on every proposal run concurrently. Each voter is bounded by `timeout_s` and queued on
     its pool.
   - The judge writes a verdict. A non-commit verdict carries a sanitized correction, which the next
     attempt's projection shows.
   - Each adversary proposal is dueled. Exploits are recorded and handed to the hardener in the
     background. The next attempt waits for that hardening, and `run_graph` waits for any hardening
     still pending before it returns.
5. **Terminal entries.** The task ends in one of these:
   - `commit` (by the judge);
   - `reject {reason: "attempts exhausted"}`;
   - `abort` with reason `timeout`, `error: <ExceptionType>`, `context_leak` or `dependency_blocked`.

   A failing deliverable never cancels its siblings.

`GraphResult.from_bus` derives each task's status (`committed | rejected | aborted | blocked |
pending`), score, attempts, exploits and critical path from the bus alone.

### MAF engine (`maf_engine.py`)

`run_graph_maf` takes the same arguments plus `checkpoint_dir` and builds a MAF workflow.
- There is one `DeliverableExecutor` per task, which runs the same `run_deliverable` as the asyncio
  engine. A start executor fans out to the roots, and a task with several dependencies is a fan-in.
- Every executor always emits a `Committed` or `Blocked` message, so a fan-in fires even when a branch
  fails. `max_iterations` is the task count plus 2.
- Checkpoints are written each superstep to `<CI_RUN_DIR>/<run-id>/ckpt` (default
  `artifacts/runs/<run-id>/ckpt`), not under `--run-dir`. Stale checkpoints are deleted at start.

Resume always replays from the bus. `tests/ci_lab/taskgraph/test_maf_engine.py` checks conformance
with the asyncio engine on chain, diamond, wide, blocked and skewed graphs, plus an exploit and
hardening case. Each task's status, attempts, rubric versions, reason, score and exploit count must
match. The difference is where concurrency comes from: MAF supersteps and fan-in edges instead of
`asyncio.Event`s.

## Students (`students.py`)

- **`fake_student`** answers with the first fenced block of the instructions (or all of them). It is
  for offline smoke runs.
- **`AgentStudentFactory`** runs the `harness/agents/student.yaml` MAF agent in a temporary workspace for
  each attempt, under `<run-dir>/students`. The workspace is seeded with the deliverable's `file`
  context, resolved relative to the graph file. Only the output path is writable, and the agent
  submits with `submit_output`.

## CLI

```text
ci-lab graph validate <graph> [--vault DIR]          # code<TAB>where<TAB>message; exit 1 on any
ci-lab graph seal <rubric.yaml>... --vault DIR       # version_id<TAB>deliverable<TAB>commitment
ci-lab graph run <graph> --run-dir DIR [--run-id ID] [--rubric FILE]... [--max-parallel N]
    [--challenger off|det|llm|both] [--optimizer off|gepa] [--engine asyncio|maf] [--quorum 1]
    [--student fake|agent] [--profile copilot|offline] [--s1-model default] [--json] [--no-telemetry]
    [--no-adopt-hardened]
ci-lab graph show <run-dir>                           # summary + bus heads
```

`graph run` works like this:
- **Paths.** It seals each `--rubric` into `<run-dir>/sealed`, validates the graph against the vault
  (exit 1 on problems), and runs on the bus at `<run-dir>/bus`. It writes
  `<run-dir>/graph_result.json` and, unless `--no-telemetry` is given, spans under
  `<run-dir>/telemetry`.
- **Defaults.** The run id defaults to `rYYYYmmdd-HHMMSS`, the challenger to `off`, the engine to
  `asyncio`, the student to `agent` and the profile to `copilot`.
- **Voters.** Every deliverable gets the deterministic voter. `S1RubricVoter` is added when the rubric
  has `s1` criteria and `--s1-model` isn't `""`. `AgentVoter` is added for `llm` criteria, and
  `AssertVoter` (harness domain) for `assert` criteria.
- **Challenger.** A hardener is attached only when `--challenger` isn't `off`. `--optimizer gepa`
  (needs a challenger) adds GEPA soft-question candidates after template patches; see
  [adversary.md](adversary.md#wiring).
- **Hardened rubrics.** Runs that share a `--run-dir` share its vault and bus, so a new run id adopts
  the accepted hardening of earlier runs there. `--no-adopt-hardened` keeps the pinned rubrics.
- **Exit code.** It exits 2 if `--engine maf` can't be imported or `--optimizer` is set without a
  challenger, and 0 only when every deliverable commits.
## Walkthrough (`examples/taskgraph/`)

[examples/taskgraph/README.md](../examples/taskgraph/README.md) has the commands. In brief, the
example is a diamond graph, `brief` → (`facts`, `risks`) → `summary`. Each rubric mixes regex or JSON
schema oracles with one `s1` criterion. The offline smoke run uses the `fake` student, deterministic
gamers and no s1 judge:

```text
> ci-lab graph run examples/taskgraph/graph.yaml --run-dir runs/tg --run-id demo `
    --rubric examples/taskgraph/rubrics.yaml --student fake --challenger det --s1-model "" --no-telemetry
run demo: ok=True wall_s=1.68 critical_path=['brief', 'risks', 'summary']
  brief            committed  score=1.0 attempts=1 exploits=0
  facts            committed  score=1.0 attempts=1 exploits=0
  risks            committed  score=1.0 attempts=1 exploits=0
  summary          committed  score=1.0 attempts=1 exploits=0
```

`facts` and `risks` run concurrently once `brief` commits. The six gamers' proposals all fail the
regex oracles, and with no s1 voter there are no soft scores to prefer them. So no exploits are
recorded, and the hardener never runs. `--engine maf` gives the same summary.

An exploit needs both soft votes and an independent oracle. Soft votes come from an s1 judge
(`--s1-model default`, which needs the local System-1 backend). In this example only `brief`'s
`b-head` is marked `independent: true`, so only `brief` can record exploits.
