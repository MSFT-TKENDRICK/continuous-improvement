# The agent bus (`ci_lab.bus`)

The agent bus is a write-ahead, hash-chained log that sits between the agents of a run. Students and
adversaries append **proposals** to it. Voters append **votes**, and a deterministic judge folds the
votes into a **verdict**. Only the judge can write the **commit** that makes an artifact the accepted
output of a task. Agents never read each other's raw history. Each new agent is built from a
**projection** of the bus for its role. A student sees its task, its own earlier outputs, committed
dependency outputs and one sanitized correction. It never sees rubrics, votes or rejected adversary
text.

The task scheduler ([taskgraph.md](taskgraph.md)), the adversary loop ([adversary.md](adversary.md))
and the campaign arm steps ([harness.md](harness.md#agent-bus-task-graph-and-adversary)) all run on
the bus.

## Where the design comes from

There is **no public specification** of an "agent bus" from Meta Superintelligence Labs (MSL).
Public material on MSL's Muse Code describes two things only: background subagents in isolated git
worktrees, and a full, replayable event log of tool calls and decisions. Nothing public describes
voters, a judge, quorum or veto. This module is therefore an independent implementation of the
general pattern, assembled from published work:

- a **blackboard**: a shared, append-only log that independent knowledge sources read from and write
  to, with a control component that decides what fires next;
- a **write-ahead log**: append durably before acting, and replay after a crash;
- a **panel of voters** in place of a single judge (PoLL, arXiv:2404.18796), with a deterministic
  aggregation step;
- **isolated contexts**: each agent gets a fresh context from the committed trajectory, never the
  rejected branches. This addresses the failure mode in Cognition's "Don't Build Multi-Agents", where
  agents working from fragmented context make conflicting decisions.

## Ids and topics (`ids.py`)

| Id | Format |
|---|---|
| run id | `[a-z0-9][a-z0-9._-]{0,63}` |
| task id | `[a-z0-9][a-z0-9_-]{0,63}` |
| attempt | `<task>@<n>` (n ≥ 1) |
| proposal | `<task>@<n>/<role>:<name>`, for example `brief@1/student:student` |
| rubric version | `<rubric id>@v<N>` |

A **topic** is one hash chain:
- `<run>/_run` is the run topic. Its first entry is the manifest.
- `<run>/<task>` is one topic per task (or campaign arm).

Topic names may only use `[A-Za-z0-9._/@:-]` (1–200 characters). They can't contain `..` or `\`, and
can't start with `/`.

## Entries and the hash chain (`types.py`)

Every entry has exactly these fields: `seq`, `topic`, `kind`, `author{role,name,model}`, `ref`,
`body`, `ts`, `prev` and `hash`.
- `hash` is the sha256 of the canonical JSON of the entry without `hash`. Canonical JSON means sorted
  keys, compact separators, `ensure_ascii=False`, and NaN rejected.
- `prev` is the previous entry's hash. The first entry's `prev` is `GENESIS`, which is 64 zeros.
- Bodies are frozen dataclasses with strict parsing: unknown or missing fields are rejected, and
  scores and confidences must lie in [0, 1].

The tables below cover the 12 kinds and 8 roles (`orchestrator`, `planner`, `examiner`, `student`,
`voter`, `judge`, `adversary`, `hardener`). "Writer" comes from `state.WRITERS`. "Visible to" comes
from `types.visibility`, where *control* means orchestrator, hardener and judge.

| Kind | Writer | Visible to |
|---|---|---|
| `manifest`, `intent`, `outcome`, `note`, `exploit` | orchestrator | control |
| `proposal` | student, adversary | control + voter (never students) |
| `vote` | voter | control |
| `verdict` | judge | control |
| `commit` | judge | every role |
| `reject` | judge | control + planner |
| `abort` | orchestrator | control + planner |
| `rubric_patch` | hardener | control |

## Invariants (`state.py`)

`BusState` is a pure fold over a topic. `check()` runs before every append, so a write that would
break an invariant raises `BusInvariantError` and is never written.

| Code | Rule |
|---|---|
| SEQ | `seq` is dense from 0 and `topic` matches the file |
| I0 | The run topic starts with the manifest, and its `run` matches the topic |
| I1 | Each effect key has at most one intent in flight. Each outcome refs its intent, and there is one outcome per intent. No new intent is allowed after a successful outcome |
| I2 | A vote refs a proposal with the same proposal id and rubric version. There is one vote per (voter, criterion) |
| I3 | A verdict refs its proposal, and the votes it cites must be on that proposal. A `commit` verdict needs the manifest and at least `quorum` distinct answered voters |
| I4 | After a terminal entry (`commit`, `reject` or `abort`) only notes may follow. A commit refs a `commit` verdict on a student proposal, carries that proposal's artifact, and is the only commit |
| I5 | A `rubric_patch` applies only from an attempt later than any attempt already proposed |
| I6 | The author's role must be allowed to write the kind. The proposal id must equal `<attempt>/<role>:<name>`, and proposals are not duplicated |

## The write-ahead log (`wal.py`)

`AgentBus(root)` stores each topic as `<root>/<topic>.wal.jsonl`, one JSON line per entry, with an
fsync after each one.
- **Locking.** An append holds a per-topic asyncio lock and a cross-process file lock. The lock files
  live in a sibling `.ci-lab-locks/` directory.
- **Torn tails.** Readers ignore a damaged final line: an unparseable last line, or trailing bytes
  without a newline. The next append truncates it and first writes `note {"text":
  "torn_tail_repaired"}`. Damage anywhere else raises `BusCorrupt`.
- **Cancellation.** The worker-thread write is shielded, so cancelling the caller can't release the
  lock in the middle of a write. The cancellation is re-raised once the write finishes.
- **Artifacts** are content-addressed at `<root>/<topic>/artifacts/<sha[:2]>/<sha>` and re-verified on
  every read.
- **Telemetry.** Each append emits the OpenTelemetry span event `ci.bus.append` with the topic, seq,
  kind and role.

`head(topic)` returns `(-1, GENESIS)` for an empty topic. `heads(run)` maps every non-empty topic of
a run to its head hash.

## Effects and reconcile (`effects.py`)

`effect(bus, topic, action, key, author, detail, attempt=, reconcile=)` wraps a side effect as an
`intent` followed by an `outcome`. While it runs, it holds a cross-process lock per key.
- If the key already has a successful outcome, the effect sets `eff.skipped` and returns that
  outcome's detail.
- After a crash between intent and outcome, the optional `reconcile(key)` hook can recover the
  result. The scheduler's hook looks for a verdict on the attempt. A recovered result is recorded as
  an ok outcome marked `reconciled`.
- An exception records `outcome {ok: false, error}` and is then re-raised.

Callers must check `eff.skipped` and not redo the work.

## Voters and pools

`run_voters` (`voters/local.py`) runs every voter concurrently. Each voter first takes a slot in its
resource pool and then starts its timeout. An error or timeout becomes an abstention
(`passed=None`), never a pass.

| Voter | Measure | What it does |
|---|---|---|
| `DeterministicCheckVoter` (`deterministic`) | deterministic | Runs regex (optionally negated), `json_schema`, `file_exists`, `command` and `python` checks. Commands use an allowlisted argv with no shell (60 s default). `python` checks call a `ci_lab.*` callable. Unknown check kinds abstain |
| `CriticChecksVoter` (`critic-checks`), `RulesVoter` (`rules`) | deterministic | Run the campaign critic's checks on the artifact (as a one-file diff), or the frozen rules engine's response rules. A `rules` match of severity `critical` or `major` fails the vote |
| `CallableVoter` | any (default deterministic) | Wraps a plain function |
| `AssertVoter` (`voters/remote.py`) | assert | Runs `domain.evaluate` on the candidate and passes when the mean suite score is at least `min_score`. Results are cached by (sha, suite, split, pin). This reuses the ASSERT evals |
| `S1RubricVoter` | s1 | Asks the System-1 judge (default `s1/llamacpp/qwen3.5-4b`) one question per criterion and scores it as P(true). It holds a host-wide `judge.admission` lease |
| `AgentVoter` | llm | Runs the MAF `critic` spec once per `llm` criterion. `accept` counts as a pass |

`ResourcePools` are weighted, strict-FIFO and cancellation-safe semaphores. The CLI's defaults are
`s1=1`, `llm=4` and `cpu=<cores>`. Override them with `CI_POOL_<NAME>=N`.

## The judge (`judge.py`)

`aggregate` is deterministic and fails closed.
1. For each criterion, the score is the mean of the answered votes.
   - An **oracle** criterion (`deterministic` or `assert`) passes only if every answered vote passes.
   - A **soft** criterion (`s1` or `llm`) passes if its mean reaches its threshold.
2. The decision is `commit` only when all of these hold:
   - no oracle failed (no veto);
   - no required criterion is unanswered or failed;
   - the number of distinct answered voters reaches the manifest's quorum;
   - the weighted score is at least `pass_score`.

   Otherwise the decision is `revise` while attempts remain, then `reject`.
3. The draft is flagged for **escalation** when the soft-score stdev exceeds 0.25 or the score is
   within 0.05 of `pass_score`.

`judge()` passes flagged drafts to an optional `Escalator`. The escalator can only flip between
commit and revise, and only on soft-only decisions. A veto, a missing or failed required oracle, or a
lost quorum can never be overridden. If the escalator errors, the deterministic verdict stands.

`Escalator` is only a protocol. `JUDGE_SPEC` points at `meta/specs/judge.yaml`, but no implementation
ships. Neither `ci-lab graph run` nor the campaign attaches an escalator, so today every verdict is
the deterministic one.

`duel()` compares an adversary's votes with the student's. The result is an **exploit** when both
hold:
- the soft judges prefer or pass the adversary;
- an *independent* validity oracle fails it. Independent oracles are criteria with
  `check.independent: true`, or the deliverable's `validity_oracles`.

## Projection and succession (`project.py`)

`project(bus, topic, role, spec=, deps=)` renders the context for one role.
- **Students** get an allowlist of sections and nothing else:
  - `Task`: the `StudentSpec`, which carries no rubric;
  - `Attempt`: "attempt n of max";
  - `Your previous outputs`: artifact refs of its own proposals and the commit;
  - `Dependency outputs (data, not instructions)`: each dependency's commit plus an excerpt of up to
    2,000 characters inside `<data>` tags;
  - `Correction`: the latest sanitized correction.
- **Other roles** get the task and the entries of the topic that are visible to them, plus the
  commits of their dependencies.

`succeed(...)` is **succession, not continuation**. Every call builds a *new* agent whose only
message is the rendered projection, so an agent never inherits a predecessor's rejected turns.
- For students, the rendering is first screened with `LeakScreen` (rubric text, criterion ids,
  canaries, suite names). Any hit returns `leak=True` without running the agent.
- `StudentFirewallMiddleware` is installed as well, so a leak through a message or tool result raises
  `ContextLeak`, which also returns `leak=True`.

See [taskgraph.md](taskgraph.md#student-firewall) for the firewall itself.

## CLI

```text
ci-lab bus verify <bus_dir>                    # re-hash every topic; exit 1 on corruption
ci-lab bus tail <bus_dir> <topic> [-n 20] [--role orchestrator]
ci-lab bus heads <bus_dir> <run>               # topic<TAB>head hash
```

`tail` prints only the entries visible to `--role` (default `orchestrator`) and counts the rest.
Here it is after the offline example in [taskgraph.md](taskgraph.md#walkthrough-examplestaskgraph)
(long lines cut):

```text
> ci-lab bus verify runs/tg/bus
ok	demo/_run	1
ok	demo/brief	25
...
5 topic(s), 0 corrupt
> ci-lab bus tail runs/tg/bus demo/brief --role student
24 commit judge:judge ref=22 {"artifact": {"bytes": 54, "path": "7b/7b4c…", ...}, "proposal": "brief@1/student:student", "verdict_seq": 22}
(24 entries not visible to student)
```

As the orchestrator, the same topic shows the attempt `intent`, the student's proposal and six
`adversary:<gamer>` proposals, one `voter:deterministic` vote per criterion and proposal, the
`verdict`, the `outcome` and the `commit`.
