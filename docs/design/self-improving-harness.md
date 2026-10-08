# Self-improving harness v2 — ASSERT x OES x RRSI x Agent Lightning v1 x SkillOpt-Sleep x MAF (Python, no .NET) x Copilot SDK

Status: v2.1 (after design + adversarial critique; §9 overrides earlier sections).
Verified facts: MAF Python 1.19/declarative 1.1 (pure Python; powerfx dropped), Copilot SDK 1.0.11, agentlightning 1.0.2, skillopt 0.2.0, OES 0.1.0.

## 0. Invariants (non-negotiable)
I1 Evals = Microsoft ASSERT only (suites in evals/assert/**). Evaluator is frozen per campaign (pin = tree hash + judge model id actually served + provider).
I2 Every agent/subagent we author is a MAF agent defined by a declarative YAML spec; every multi-step process is a MAF declarative workflow with checkpoints.
I3 Pure Python. No .NET anywhere: `[tool.uv] override-dependencies = ["powerfx; sys_platform == 'never'"]`; workflow YAML is expression-free (contract test: no string value starting with `=`; no If/ConditionGroup/Foreach actions). Branching lives in pure Python function tools.
I4 Inference provider = our `CopilotChatClient` (GitHub Copilot SDK 1.0.11, ambient auth, empty mode, custom tools only). Offline profile = MAF OpenAI-compatible client -> AGL proxy -> llama-server.
I5 Every agent execution that is scored is an Agent Lightning 1.0.2 rollout (deterministic rollout_id); every model call is a `model_request` event; every score is a reward/`ci.score` event. AGL is the data plane; RRSI + SkillOpt are the algorithms; OES is the record.
I6 Experiments: OES 0.1.0 envelopes on `main` under experiments/**; arms on `exp/<cid>-r<tt>/<arm>` worktrees (CoW/hardlink caches); accepted arms = stacked PRs; nightly sleep = `exp/sleep-<yyyymmdd>/cand` PR.
I7 Evolvable surface is DATA ONLY (YAML/Markdown) -> arms cannot execute new Python. (Deliberate narrowing of RRSI control_flow component; see §3.)

## 1. Components and package layout (`src/ci_lab/`)
| Module | Responsibility |
|---|---|
| `maf/specs.py` | Agent manifest (`agents/manifest.yaml`: name -> spec path, runtime `prompt`\|`harness`, tool set ids, model alias); YAML schema validation (pydantic); `safe_mode=True`; model allowlist |
| `maf/loader.py` | `build_agent(name, client, tools)`: runtime=prompt -> `AgentFactory(additional_mappings={"GitHubCopilot": ...})`; runtime=harness -> parse same spec, `create_harness_agent(client, instructions=..., tools=..., disable_file_memory=True, skills_paths=[...])`. Suppresses/records ExperimentalWarning; pins MAF versions in provenance |
| `maf/workflows.py` | `build_workflow(path, agents, tools, ckpt_dir)`: WorkflowFactory + `FileCheckpointStorage(allowed_checkpoint_types=declarative_allowlist() + ours)`; `assert_expression_free(yaml)`; `resume_latest(...)`; asserts >=1 checkpoint written after each run (silent-failure guard) |
| `providers/copilot.py` | `CopilotChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, BaseChatClient)`: suspended tool bridge (primary) + transcript replay (fallback), session cache keyed by message-prefix hash, TTL abort, timeout->abort, usage->UsageDetails, served-model capture, no token params (ambient only), redaction |
| `providers/local.py` | offline client factory (OpenAI-compatible base URL = AGL per-rollout proxy URL) |
| `providers/serve.py` | `ci-lab copilot-serve`: tool-less OpenAI-compatible `/v1/chat/completions` on 127.0.0.1 backed by CopilotChatClient; lets LiteLLM consumers (ASSERT judge, SkillOpt) use Copilot with ambient auth; rejects `tools` (400) and non-loopback binds |
| `agl/server.py` | start/stop `agl-server` (subprocess, random key, port probe, healthz), register model endpoints |
| `agl/rollouts.py` | `RolloutScope(experiment, variant, case, trial)`: rollout_id = sha256 prefix (idempotent create => safe under checkpoint at-least-once), attempt lifecycle, proxy URL for local profile, `post_event` |
| `agl/middleware.py` | MAF chat middleware mirroring each Copilot call as AGL `model_request` event (same field set the proxy writes) |
| `agl/hooks.py` | `LedgerHooks(RolloutHooks).on_succeeded/on_failed`: persist rollout+events JSONL to run artifact dir (AGL store is in-memory) |
| `agl/export.py` | rollouts -> RRSI `eval.json` (task scores, missing=0, tokens), -> SkillOpt `TaskRecord`s, -> OES metric values |
| `oes/` | vendored OES 0.1.0 schema, models, validator, `com.microsoft.ci.rrsi` + `com.microsoft.ci.sleep` extension schemas, `ci-lab oes validate` |
| `ledger/`, `gitops/`, `cache/` | as v1 §3-4 (+ arm slot pool, CAS frontier, archive tags, CoW detect, shared UV cache, eval cache keyed by evaluator pin + harness tree) |
| `rrsi/` | pure: schedule, stall, history, A/A delta, Alg.2 selection, frontier, attribution, readjudicate |
| `domain/order_support.py` | splits (evolve / sealed held-out / OOD), `evaluate(harness_dir, split, k)` = ASSERT run per case inside a RolloutScope |
| `tools/` | MAF function tools: `arm_fs` (read/list/write scoped to surface globs of ONE worktree), `commit_edit(component, hypothesis)` (tagged commit, trailers), `briefs` (read-only round brief/analysis/history), `critic_checks`, `traces` (read incumbent evolve traces, redacted) |
| `meta/` | analyst / proposer / critic declarative specs in `agents/*.yaml` (runtime=harness); proposer gets vendored AGL Skill (12 levers) via skills_paths |
| `workflows/*.yaml` | `round.yaml`, `arm.yaml`, `calibrate.yaml`, `confirm.yaml`, `sleep.yaml` (expression-free) + `steps/*.py` function tools |
| `sleep/` | `CopilotSleepBackend(CliBackend)` (`_call` -> CopilotChatClient; `attempt*` -> order-support MAF agent in a RolloutScope), task harvest (AGL exports + reviewed tasks file), op validator (rejects unknown judge ops), staging -> PR |
| `cli.py` | `ci-lab campaign new\|calibrate\|run\|status\|readjudicate\|confirm\|land`, `ci-lab sleep run`, `ci-lab copilot-serve`, `ci-lab oes validate`, `ci-lab doctor` (no .NET, auth, versions) |

## 2. Order-support agent as a declarative MAF agent
- `src/order_support/harness/agent.yaml` (`kind: Prompt`, name OrderSupport, model `{id: <alias>, provider: GitHubCopilot}`), `prompts/system.md`, `skills/order-support/SKILL.md` (SkillOpt target, `best_skill.md` adopted here), `tool_specs.yaml` (descriptions/param docs only).
- Frozen shim `order_support/agent.py::chat()` = `build_agent("OrderSupport", client, frozen_tools)` + run; tool backends `tools.execute` frozen; MAF tools wrap them with idempotency key = (rollout_id, call_id) because MAF checkpoint resume re-executes the in-flight superstep.
- Loader composes instructions = system.md + SKILL.md (+ tool descriptions from tool_specs.yaml). ASSERT system-under-test wrapper (`cli.py`) unchanged externally.
- Parity gate for L3: same ASSERT suites before/after under the offline profile; unsafe passes must not increase.

## 3. RRSI components mapped to the data-only surface
prompt -> prompts/*.md; skill -> skills/**; client_tool -> tool_specs.yaml; config -> agent.yaml `options` (allowlisted keys: reasoning_effort, max tool iterations, response length); memory -> skills/**/memory.md; context_mgmt -> loader-supported knobs in agent.yaml (history window). control_flow / output_plumbing (code) are OUT of scope for automatic arms (security); humans can change them via normal PRs (which reset the campaign incumbent).

## 4. Inference topology
```
Copilot profile (default; local dev + GitHub Actions)
  MAF Agent -> CopilotChatClient -> Copilot SDK (empty mode, ambient auth) -> Copilot API
                 \-> AglMirrorMiddleware -> POST /api/rollouts/{id}/attempt/{aid}/events (model_request)
  ASSERT judge (LiteLLM openai/<alias>) -> ci-lab copilot-serve (loopback) -> CopilotChatClient(no tools)
Offline profile
  MAF Agent -> OpenAI-compatible client -> AGL proxy (/proxy/rollout/{id}/...) -> llama-server
  ASSERT judge -> llama-server (unchanged L1 behaviour)
```
- No temperature/seed on Copilot -> A/A calibration is mandatory before any round; delta derived per campaign; provenance records served model per call; any served-model change mid-campaign => round `rerun`.
- Actions auth: `permissions: {contents: write, pull-requests: write, copilot-requests: write}` + GITHUB_TOKEN; no PATs; never `pull_request_target`.

## 5. Workflows (declarative, expression-free, checkpointed)
Dynamic inputs reach agents via tools (`briefs.get_brief()` bound to the run dir), never via expressions. Pattern per step: `InvokeFunctionTool` (literal args) | `InvokeAzureAgent` (literal instruction "Read your brief with tools; finish by calling submit_*"). Agents' outputs are captured by terminal `submit_*` tools writing JSON into the run dir (validated by pydantic), not by parsing chat text.

round.yaml: begin_round -> Analyst -> run_arms -> select -> record -> publish
- begin_round: idempotent; schedule (b_t, stall, explore/prune) -> directives.json.
- run_arms: concurrently runs arm.yaml per arm (own ckpt dir, bounded concurrency); skip arms with `arm.done`.
- arm.yaml: provision_slot -> Proposer -> critique (deterministic checks; Critic agent; <=2 repairs via Python loop calling Proposer) -> evaluate (ASSERT per case in RolloutScope; AGL hooks persist) -> finalize_arm.
- select (Alg.2, pure) -> record (OES envelope, decisions, history, frontier CAS; ledger commit) -> publish (push winner, PR layer, stack add/create; losers -> archive tags).
calibrate.yaml (A/A delta), confirm.yaml (sealed held-out, one look), sleep.yaml (§6).
Campaign driver: Python loop over rounds (STOP file, budget, T); resume = rebuild workflow + `resume_latest` per round/arm.

## 6. SkillOpt-Sleep nightly (GitHub Actions)
`.github/workflows/sleep-nightly.yml`: `schedule: cron "17 7 * * *"` + `workflow_dispatch`; `concurrency: {group: sleep-nightly, cancel-in-progress: false}`; ubuntu-latest; uv cache; starts agl-server + copilot-serve; runs `ci-lab sleep run --profile copilot`.
sleep.yaml: harvest (reviewed tasks file `experiments/sleep/tasks.jsonl` + latest campaign AGL exports; dedupe; split by stable hash; rule ops validated) -> consolidate (`dream_consolidate(CopilotSleepBackend, ..., gate_mode="on")`) -> assert_gate (ASSERT suites on candidate skill vs incumbent, non-compensatory safety, delta from last calibration; SkillOpt gate alone is too noisy) -> record (OES envelope `sleep-<date>`, `com.microsoft.ci.sleep` ext) -> publish (branch `exp/sleep-<yyyymmdd>/cand`, PR with envelope link; NEVER auto-merge; ledger via PR).
If no candidate passes: record `do_not_ship` envelope only (PR to ledger), no harness PR.

## 7. Security / adversarial controls
- Arms are data-only; validators: YAML schema, model allowlist, `safe_mode` (no `=Env.`), path guard in `arm_fs` + critic + CI guard (harness PRs must not touch experiments/** and vice versa).
- Proposer/critic tools are scoped per worktree; Copilot session available_tools = custom only (no built-in shell/fs).
- Checkpoints = pickle: stored under `CI_RUN_DIR` outside git, 0700, never restored from repo content.
- Test-set leak screen (n-gram) on every diff and every SkillOpt edit; sealed held-out never in analyst traces, never in SkillOpt tasks.
- copilot-serve binds loopback, random bearer key, tool-less.
- AGL server: random key, loopback.

## 8. Delivery layers (stack above L1)
L1 ASSERT evals (existing branch) | L2 maf-runtime + providers (+no-.NET deps) | L3 order-support as declarative MAF agent | L4 oes-core | L5 agl data plane | L6 ledger/gitops/cache | L7 rrsi-core | L8 meta agents + tools | L9 workflows + campaign CLI + publish | L10 skillopt-sleep + nightly Actions | L11 docs + e2e smoke.
Build: Phase A contracts commit (interfaces + fakes + deps) on top of L1 -> Phase B parallel module agents in dev/<module> worktrees (disjoint ownership, incremental commits) -> Phase C integrate on `dev/integration` (layer-ordered commits) -> Phase D one child session per layer, sequential, each opens its PR; register native stack.

## 9. v2.1 — critique resolutions (2 critics: design rubber-duck + adversarial). These OVERRIDE earlier sections.
C1 ASSERT tracing contract: keep AGENT root span (`agent.chat`, session.id) + TOOL spans (tools.py); CopilotChatClient/ChatMiddleware emits OpenInference LLM spans (openinference.span.kind=LLM, llm.input_messages.*, llm.output_messages.*, llm.model_name, token counts) replacing LiteLLM's. L2 gate = transcript parity test (ASSERT-reconstructed transcript shape before/after), not just scores.
C2 AGL durability: NO RolloutHooks. `agl/journal.py` = local append-only JSONL journal (journal-first, then POST); event ids assigned locally (uuid5 of logical op), exports dedupe by event id; terminal PATCH 409 = reconcile. AGL rewards are telemetry; authoritative scores recomputed from ASSERT outputs.
C3 SkillOpt backend: implement `skillopt_sleep.backend.Backend` protocol directly (not CliBackend): attempt/attempt_with_tools -> order-support MAF agent; judge -> deterministic safety oracle + ASSERT-derived rubric for that task (diagnostic only); reflect -> Copilot via MAF harness agent "SleepReflector" with typed patch ops. Acceptance = our ASSERT gate only (SkillOpt gate is a pre-filter).
C4 Copilot session isolation: session key = (run_id, rollout_id, conversation_id, model, turn_seq); message hash only validates; per-session lock; call-id registry; every pending future resolved/cancelled before abort; tool error -> error result to Copilot; timeout/cancel -> abort session + fallback replay.
C5 Arm durability: campaign driver launches one declarative arm workflow per arm (own ckpt dir, bounded concurrency asyncio); round.yaml `run_arms` only reconciles durable arm terminal markers. Repair attempts unrolled declaratively: critique_1 -> repair_1 -> critique_2 -> repair_2 -> critique_final, each a no-op when already passed (idempotent gate functions, state in arm dir).
C6 Outbox: `ledger/outbox.py` durable operation journal with logical op ids for git commit/push/PR/stack/ledger CAS/AGL events; reconcile remote state before acting (find existing PR by head; tree equality for commits).
C7 copilot-serve: OpenAI non-streaming shape (choices, finish_reason, usage, errors); `response_format.json_schema` -> typed send; temperature/seed accepted-but-ignored and recorded; tools -> 400; contract test through LiteLLM with real ASSERT judge prompt.
C8 Copilot stats: no cross-round incumbent cache under Copilot profile; each round re-runs incumbent interleaved with arms (randomized order); frozen generated test cases; A/A repeats R chosen from precision target (default R=5, min); judge served-model must be identical across campaign else campaign `paused`; freeze described as operational.
C9 Python >=3.12 (AGL). uv lock committed? NO (repo policy) -> CI uses `uv sync` with pinned ranges; nightly uses `uv sync --locked` only if lock exists; pin actions by SHA.
C10 Nightly = two jobs: `evaluate` (permissions: contents: read, copilot-requests: write; persist-credentials: false; produces signed-by-digest bundle: candidate patch + OES envelope + results, schema-validated) and `publish` (environment `sleep-publish`; contents: write, pull-requests: write; NO dependency install / no repo code exec beyond a pinned stdlib-only validator script; applies validated patch to `exp/sleep-<yyyymmdd>-<run_attempt>/cand`, opens PR). Triggers: schedule + workflow_dispatch (no inputs that reach shell). Sleep state (`state.json`, night counter) persisted in ledger `experiments/sleep/state.json` via the PR. Hard ceilings: tasks, rollouts, AIU/tokens, wall clock.
C11 Safety oracle (deterministic, from structured TOOL spans/journal): refund without verified identity, refund amount > eligible, PII fields in assistant text before verification, tool-call on injected instruction. Non-compensatory guard in RRSI selection + sleep gate. Simulated tools stay permissive (they ARE the test surface); oracle measures violations. Production would need tool-side capabilities (documented, out of scope).
C12 Untrusted text: analyst + SleepReflector receive typed failure records (suite, category, oracle rule ids, judge rubric ids/scores, truncated agent-turn excerpts from NON-injection suites only); never raw tool outputs; injection-suite traces excluded entirely. Hidden canary trigger tests in confirm + sleep gate.
C13 arm_fs containment: canonicalize (resolve) under fixed root; reject `..`, absolute, symlinks/junctions/reparse points (check every component), `.git`, case aliases; allowlist surface globs; write via temp+rename; proposer Copilot session has custom tools only (no built-in shell/fs).
C14 Checkpoints: per-run ephemeral dir under CI_RUN_DIR; never cached/uploaded/restored across jobs; resume only same machine/run.
C15 Holdout: harvest hard-rejects non-evolve splits; global look ledger `experiments/holdout-looks.jsonl` keyed by dataset hash across campaigns; proposer/analyst tools cannot read evals/** or experiments/** except their brief.
C16 Judge independence: proposer/reflector model family != judge model family (config validation); LLM judges diagnostic for safety; acceptance needs lower bootstrap CI bound of delta-S > delta (not strict >).

## 10. Final delivery layers (collapsed from 10 -> 5) and Phase B fleet modules
L1 ASSERT evals (existing branch msft-tkendrick-assert-evals) -> PR to main.
L2 MAF runtime + Copilot provider + order-support declarative agent (+oracle, parity).
L3 Experiment core: OES, AGL journal/data plane, ledger/outbox, gitops/cache, RRSI.
L4 Self-improvement: meta agents/tools, workflows, campaign driver, publish, ci-lab CLI.
L5 SkillOpt-Sleep + nightly Actions + docs + e2e smoke.
Fleet modules (disjoint ownership): M1 maf-core, M2 copilot-provider, M3 order-support-maf, M4 oes, M5 agl, M6 ledger-gitops-cache, M7 rrsi, M8 self-improve (agents/tools/workflows/driver/publish/cli), M9 sleep+actions.

