# Harness architecture

CI Lab is a Python, harness-only self-improvement system. The target is the repo-root
[`harness/`](../harness/) tree. The control plane under `src/ci_lab/`, evaluation assets under
`evals/`, schemas, and GitHub automation are frozen relative to a candidate.

## Pillars

### Microsoft Agent Framework core

[`ci_lab.maf`](maf.md) loads prompt agents from declarative YAML and executes expression-free
declarative workflows. `FileCheckpointStorage` checkpoints every workflow superstep and
`run_or_resume` resumes from the latest checkpoint. Branching and side effects stay in typed,
idempotent Python tools; PowerFx and .NET are intentionally absent.

The evolvable target includes declarative agents, prompts, skills, target workflows, target-agent
loop limits, client-tool exposure, and MCP exposure. Campaign arm workflows, retries, critique
policy, evaluator parallelism, bus quorum, and judge policy remain frozen under `src/ci_lab/`.

### Optimizers with distinct ownership

The authoritative mapping is `ci_lab.contracts.COMPONENT_OWNERS`:

| Component | Dedicated owner |
|---|---|
| `prompt` | `gepa` |
| `skill` | `skillopt` |
| `guard` | `guard` |
| `agent` | `agl` |
| `loop` | `agl` |
| `workflow` | `agl` |
| `mcp` | `agl` |
| `client_tool` | `agl` |
| `config` | `agl` |
| `context_mgmt` | `agl` |
| `memory` | `agl` |

`strategy_may_edit` gives the general `agent` proposer permission to edit any text component except
`guard`. That general permission does not change dedicated ownership: DSPy/GEPA remains the prompt
optimizer, SkillOpt remains the skill optimizer, the lessons arm owns guards, and AGL owns structural
components.

[`ci_lab.optim`](optim.md) uses DSPy as the LM/prompt layer, GEPA for prompt candidates, and
SkillOpt 0.2.x for skills. [`ci_lab.sleep`](sleep.md) runs the two harness skills through
SkillOpt-Sleep nightly.

[`ci_lab.agl`](agl.md) uses Agent Lightning 1.0.2 as a rollout journal/store and optional loopback
proxy. Its improvement algorithm is CI Lab's `LlmResourceAlgorithm`: a Copilot SDK optimizer model
assigns bounded structural credit and proposes one contained edit. It never imports `verl`, performs
RL training, requires a GPU, optimizes prompts, or optimizes skills. Runtime metrics are emitted
journal-side because Agent Lightning 1.0.2 has no server hook-loading seam.

### ASSERT and System-1 evaluation

[`HarnessDomain`](../src/ci_lab/domain/harness.py) evaluates each case in a separate temporary
writable copy with a scrubbed environment. It validates the candidate before execution and verifies
that the source tree did not change. Candidate output cannot supply evaluator measurements.

The five frozen suites are:

| Suite | Purpose |
|---|---|
| `harness_triage` | Attribute a typed failure to the responsible component and reason code. |
| `harness_proposal` | Make one bounded, parseable edit without case-specific literals. |
| `harness_taskgraph` | Produce one dependency-correct deliverable without rubric leakage. |
| `harness_tool_use` | Use the requested read-only MCP capability in direct or code mode. |
| `harness_injection` | Ignore instructions embedded in evidence and avoid forbidden effects. |

Each rubric contains deterministic quality, a System-1 criterion when judgement is needed, and
evaluator-owned resource criteria. The fake CI profile uses scripted target responses and a
scripted judge. Live tiers require explicit target and judge pins and a reachable System-1 endpoint;
a served target-model mismatch invalidates the trial.

`TaskScore.score` is quality-only. Deterministic correctness/safety and System-1 quality criteria
enter the primary score. Runtime and simplification criteria are stored as resource subscores.
Required missing metrics score zero and record `metric.missing`; the parent evaluator also invalidates
a result missing its required evaluator metrics.

### OES and RRSI election

Every calibration, round, confirmation, and sleep night is represented by an
[OES](oes.md) envelope. [RRSI](rrsi.md) schedules arms and elects a winner against the
contemporaneous incumbent.

The `harness` profile requires resource metrics and applies non-compensatory gates in both RRSI
branches:

- surface complexity growth `dX <= 0.10`;
- mean `(llm_calls + tool_calls)` growth `dCalls <= 0.15`;
- wall-time change is recorded from median `wall_ms` but is diagnostic because `wall_cap=None`.

Missing incumbent or candidate surface/runtime data makes an arm inadmissible. Complexity growth is
also penalized in the weighted branch. Simplicity credit is behavior-gated: the tree must validate,
required agents must exist, critical safety must be non-inferior, and quality change must be
non-negative. Edit-line or output-line reduction alone is not simplicity.

### Agent bus, voters, judge, and adversary

[`ci_lab.bus`](bus.md) is a hash-chained write-ahead log. Students and adversaries append proposals;
deterministic, ASSERT, System-1, rules, critic, or LLM voters append votes; a deterministic judge
folds them into a verdict. Required unanswered criteria, oracle vetoes, lost quorum, and failed
required criteria fail closed. A successor agent receives a sanitized projection, not rejected raw
history.

Campaign critique/evaluate steps use the bus by default. The challenger lane runs after arms and
before selection, records exploits and evaluator proposals, but cannot become a strategy, alter arm
results, or enter RRSI history. Non-template rubric hardening is Wilson-gated; see
[adversary limitations](adversary.md#limitations).

### Governance and deterministic lessons

[`ci_lab.governance`](governance.md) applies AGT identity/audit/SRE primitives and frozen ACS
policies to MAF agents, MCP calls, and campaign launches. It is application-layer policy, not an OS
sandbox.

[`ci_lab.lessons`](lessons.md) reduces trusted failures to typed trajectories, clusters recurring
patterns, and routes them to deterministic enforcement. [`ci_lab.rules`](rules.md) evaluates frozen
RuleSpec YAML. [`ci_lab.guards`](guards.md) can block, redact, or remediate a trajectory.
[`ci_lab.lessons_arm`](lessons-arm.md) evaluates guard changes with paired guard-off/guard-on trials.
Lint, CODEOWNERS, and draft-only publishing keep these changes reviewable.

### Telemetry, Aspire, canvas, and chat

OpenTelemetry spans and local JSONL records are produced by campaign, sleep, taskgraph, MAF, and AGL
paths. [`ci-lab dashboard`](telemetry.md) starts an optional, hash-pinned Aspire dashboard.

The native [`ci-harness-dashboard`](canvas.md) Copilot canvas reads local JSONL, ledgers, OES
envelopes, ASSERT outputs, AGL journals, bus logs, and optionally Aspire's API. Aspire cannot be
embedded or reverse-proxied; the canvas opens it as a top-level page after a user click.

The canvas Chat tab is a local CopilotKit UI over [`ci-lab chat serve`](chat.md). It drafts campaign
configuration and can request a launch. The Python server enforces a single-use, thread-bound human
approval before launch. Chat does not run held-out confirmation; use `ci-lab campaign confirm`
separately.

## Incumbents, candidates, and activation

At round start the campaign copies and hashes the incumbent `harness/**` tree. Analyst, proposer,
repair, and critic use that immutable round snapshot. Candidate evaluation alone loads an arm's
harness directory. This prevents an in-progress arm from changing the context used to create or
judge other arms. An elected candidate becomes active at the next round, not midway through the
current one.

## Trust boundaries

- `harness/harness.yaml` must match `src/ci_lab/harness_tree/manifest.yaml`; a candidate cannot
  change owners, globs, required agents, or caps.
- Harness evaluation copies the candidate and fixtures to a temporary directory, forwards only an
  environment allowlist, and forwards no GitHub token or git credentials.
- MCP server process definitions and maximum limits are frozen. The candidate may only narrow
  exposure and select direct or code mode.
- MCP code mode uses AST restrictions, an isolated interpreter, framed RPC, output/time caps, and
  process-tree termination. It is isolation against accidental or naive misuse, not an OS sandbox.
- MAF checkpoints contain pickle data. Keep them private to one run; do not restore them from
  untrusted artifacts.
- Publishing is draft-only until a human explicitly lands an accepted stack.

## Current limitations

- Skill optimization is tested and workflow-pinned to SkillOpt **0.2.x**
  (`skillopt>=0.2.0,<0.3`).
- MAF declarative agents/workflows are experimental upstream; package bounds reduce, but do not
  remove, schema and loader drift risk.
- The GitHub Copilot SDK provider exposes no reliable temperature or seed control. Fake fixtures,
  journals, and paired evaluation provide determinism instead.
- AGT/ACS middleware and MCP code mode are not OS sandboxes.
- Non-template adversary rubric patches are effectively blocked on small corpora: at
  `epsilon=0.05`, the Wilson gate needs at least 52 held-out honest samples when there are zero new
  false rejections (about 172 honest inputs before the 30% split).
- Agent Lightning's server store is in memory; the local append-only journal remains authoritative
  and can repopulate it.
- MAF background subagent state and chat approval interrupts are in memory and do not survive
  process restart.
- SkillOpt-Sleep nights start fresh rather than resuming an interrupted workflow.
- The canvas is local, loopback, single-user tooling. Its CopilotKit integration uses the
  development-only direct-agent API and keeps no chat history across page reloads.
