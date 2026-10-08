# Governance: AGT + ACS

`ci_lab.governance` puts every MAF agent in the repo, and every campaign launch, behind
deterministic, fail-closed policy. It follows the
[Agent Control Specification](https://github.com/microsoft/agent-governance-toolkit) (ACS, manifest
`0.4.0-alpha.1`) and uses the Agent Governance Toolkit (AGT) packages for identity (`agentmesh`),
the hash-chained audit log (`agentmesh.governance`), execution rings and the kill switch
(`hypervisor`), and error budgets (`agent_sre`).

**AGT is application-layer middleware, not an OS sandbox.** AGT's own `docs/LIMITATIONS.md` says so.
The checks run inside our Python process. A tool that is allowed to run can still do any file or
network I/O its process can do. Real isolation stays with the OS and CI: per-job tokens, draft-only
PRs, `permissions: {}` workflows and CODEOWNERS ([harness.md](harness.md#trust-boundaries)).

## Architecture

```
            CI_GOVERNANCE_MODE=enforce|evaluate_only (invalid => startup error)
                                     |
 policies/*.acs.yaml --load_manifest--> AcsRuntime(dispatcher=adapters.DISPATCHER,
 (order_support, meta_agents,              annotator=adapters.Annotator)
  campaign)                                  |
                                     AgentControl.guard(point, snapshot)
                                     |          |                 |
                     maf.Governance middleware  campaign.check_launch   ci-lab governance eval
                     (agent / chat / function)  (agent_startup)
                                     |
            AuditTrail (AGT AuditEntry hash chain, content-free) --> OES x-ci-governance
            FileApprovalQueue artifacts/governance/approvals/<enforced_identity>.json
```

- **Pure-Python ACS runtime** (`governance.acs`). The upstream engine is Rust/PyO3 and ships no
  wheels, so it doesn't build on Windows boxes without MSVC. Our runtime passes the vendored
  25-case conformance corpus (`tests/ci_lab/governance/acs_conformance`). When
  `agent_control_specification._native` is importable, a parity test compares the two runtimes.
  The weekly `governance-native.yml` job tries to build it on ubuntu.
- **Adapters** (`governance.adapters`). Custom policies are deterministic Python callables
  registered in `DISPATCHER`. The annotator flags injection markers and rubric leaks. The output
  transform redacts PII and secrets. Rego and Cedar policies would need an injected dispatcher;
  none are used.
- **One governed factory** (`governance.maf`). `governed_agent`, `governed_declarative` and
  `governed_harness_agent` install the middleware outermost. The lint rule
  `agt.governed-agent-factory` flags any `Agent(`/`ChatAgent(` construction elsewhere in `src/`.
  These modules build their agents through the factory:
  - `maf/loader.py`
  - `order_support/agent.py`
  - `lessons_arm/agent.py`
  - `meta/spec_loader.py`
  - `chat/agent.py`
  - `sleep/target.py`
  - `sleep/reflector.py`
  - `campaign/fakes.py`
- **Modes.**
  - `enforce`: a deny sets a refusal result and raises MAF's `MiddlewareTermination`.
  - `evaluate_only`: decisions are only audited.
  - The order-support **target** defaults to `evaluate_only`, because its enforcement layer is the
    guard bundle measured by paired experiments. It enforces only when `CI_GOVERNANCE_MODE` is set
    explicitly.
- **Approvals.** A liftable deny (`escalate`) is held in the file queue under its
  `enforced_identity`. The decision is bound to that identity, so an approval can't be replayed
  for a different action. An approval is consumed once.

## ACS intervention points

| ACS point | Where | Snapshot | Used by |
|---|---|---|---|
| `agent_startup` | `governance.campaign.check_launch` from `ci-lab campaign calibrate/run/confirm/publish/land` | `campaign.{id, publish, dry_run, budget_exhausted, arms[].edit_scope}` | `campaign` |
| `input` | `AgentMiddleware` before the run (last user message) | `input.text` | `order_support` |
| `pre_model_call` | `ChatMiddleware` before the model | `model.{id, allowed}`, message count | `meta_agents` |
| `post_model_call` | `ChatMiddleware` after the model (non-streaming) | `response.text` | not configured |
| `pre_tool_call` | `FunctionMiddleware` before the tool | `call.{name, arguments}`, tool `history` | `order_support`, `meta_agents` |
| `post_tool_call` | `FunctionMiddleware` after the tool | `call`, `result` | not configured |
| `output` | `AgentMiddleware` on the final response (and on the stream's final response) | `output.text` | `order_support`, `meta_agents` |
| `agent_shutdown` | not hosted | — | — |

A point that is missing from a manifest is skipped. `ci-lab governance doctor` fails if a configured
point has no policy.

## Policies

| Policy | Rules (reason codes) |
|---|---|
| `order_support` | `issue_refund` needs a prior successful `verify_identity` for the same order (`identity_not_verified`) and a valid amount (`refund_amount_invalid`). An amount over the refund limit escalates for approval (`refund_over_limit`). Inputs with injection markers are denied (`injection_marker`). Output is redacted for PII and secrets (transform). |
| `meta_agents` | Write tools may not touch protected globs (`protected_path`). Only allow-listed models may be called (`model_not_allowed`; `CI_ALLOWED_MODELS` extends the list). Output that leaks the rubric is denied (`rubric_leak`); other output is redacted. |
| `campaign` | A launch is refused when the kill switch is engaged (`kill_switch_engaged`), the SRE error budget is exhausted (`budget_exhausted`), or an arm's edit scope overlaps a protected glob (`protected_scope`). A non-dry-run publish is a liftable deny (`publish_requires_approval`). |

The protected globs are the same in `meta_agents.acs.yaml` and `tools/paths.PROTECTED_WRITE_GLOBS`
(a test asserts it). `ArmFS.write` and the commit tool enforce them too:

- `src/ci_lab/governance/**`
- `src/ci_lab/rules/**`
- `src/ci_lab/guards/**`
- `.github/workflows/**`
- `evals/**`
- `**/sealed/**`
- `**/.lkg/**`
- `**/lessons/**`

## Campaigns, RRSI and OES

- **Launch gate.**
  - Refused or held commands exit with code 3 and print JSON.
  - A held publish prints `ci-lab governance approve <identity>`. Run that, then retry the same
    command.
  - A deferred publish (`run --defer-publish`) and fake-profile or `--dry-run-publish` runs need no
    approval.
- **SRE veto.**
  - `sre.arm_vetoes` replays ledger history into `ArmReliability`. A strategy with 3 failed rounds
    exhausts its error budget.
  - Vetoed arms are removed before the RRSI election. The rationale is recorded as `sre_vetoed`,
    `sre_veto:<arm>` and trace entries.
  - Selection over the remaining arms is unchanged (tested).
  - `Campaign.sre_exhausted()` is true when every strategy is vetoed, and feeds `budget_exhausted`.
- **OES.** Each round envelope gets `extensions["x-ci-governance"] = {audit_head, decisions,
  denies}`. A broken audit chain raises `AuditError`, so no envelope vouches for a tampered trail.

## CLI

```
ci-lab governance doctor              # policies/points, mode, AGT imports, audit chain, kill switch, pending approvals
ci-lab governance eval --policy campaign --point agent_startup --snapshot snap.json   # exit 2 on deny
ci-lab governance audit-verify [--path decisions.jsonl]
ci-lab governance approve <enforced_identity> [--by NAME --reason TEXT]
ci-lab governance deny <enforced_identity>
```

| Setting | Variable | Default |
|---|---|---|
| Audit file | `CI_GOVERNANCE_AUDIT` | `artifacts/governance/audit/decisions.jsonl` |
| Approval queue | `CI_GOVERNANCE_APPROVALS` | `artifacts/governance/approvals` |
| Kill switch | `CI_KILL_SWITCH=1` or the file `artifacts/governance/KILL` | off |

The workflows:

- `governance.yml` runs on PRs: the governance tests (conformance included), the doctor and the repo
  lint.
- `governance-native.yml` runs weekly or on dispatch, gated by `CI_HARNESS_ENABLED` like the other
  scheduled workflows. It builds the native ACS runtime with the runner's Rust toolchain and runs
  the parity test. This job is informational (`continue-on-error`).

## OWASP Top 10 for Agentic Applications (2026)

AGT's own mapping (`docs/compliance/owasp-agentic-top10-architecture.md`) is a **self-assessment**.
It rates 7 risks Full and 3 Partial. Our column says what this repo actually wires up.

| ASI | Risk | AGT (self-rated) | This repo |
|---|---|---|---|
| ASI01 | Agent goal hijack | Full | Partial. Input injection-marker deny (order support), and tool-result text is treated as data (guards, evals). Markers are regex heuristics. |
| ASI02 | Tool misuse | Full | Yes. `pre_tool_call` policies (refund preconditions, protected write paths), and the guards bundle. |
| ASI03 | Identity and privilege abuse | Full | Partial. DIDs per agent role and run via `agentmesh`. They are derived from names; no handshake or credentials. |
| ASI04 | Agentic supply chain | **Partial** (no SBOM or vulnerability scan) | Partial. Pinned actions, no committed lockfile (internal proxy), a lint ban on unwanted stacks. No SBOM. |
| ASI05 | Unexpected code execution | Full | Partial. No `eval`/`exec` tools; the sleep interpreter runs only on request. No OS sandbox. |
| ASI06 | Memory and context poisoning | **Partial** (opt-in `MemoryGuard`) | Partial. Lessons, sealed and `.lkg` paths are write-protected for arms. Lessons still feed later prompts. |
| ASI07 | Insecure inter-agent communication | Full | Not applicable in-process (subagents share one process). No trust-gate handshake. |
| ASI08 | Cascading failures | Full | Yes. SRE error-budget vetoes per arm, the kill switch, campaign budget stops. |
| ASI09 | Human-agent trust exploitation | **Partial** (no universal approval UI) | Partial. Identity-bound file approvals for publish and over-limit refunds; draft PRs only. No UI beyond the CLI. |
| ASI10 | Rogue agents | Full | Partial. The kill switch and the launch gate. No behavior monitor or quarantine. |
| — | Traceability (AGT extension) | Full | Yes. Content-free hash-chained audit, checked by `audit-verify` and stamped into OES envelopes. |

## Bypasses AGT does not close, and our mitigations

AGT's `docs/LIMITATIONS.md` lists these gaps. We mitigate some of them; none are fully closed.

- **Sequence-level ("salami") attacks.** Each action can be individually allowed while the sequence
  is harmful. ACS sees one call at a time. Mitigations:
  - `order_support` policies read the tool `history`, so a refund must follow verification.
  - Campaign arms are reviewed on their **net diff** in a draft PR, not edit by edit, and a human
    merges.
- **Semantic bypass and policy self-editing.** A self-improving harness could weaken its own
  policy. Mitigations:
  - Policies, rules, guards, workflows, evals, sealed, `.lkg` and **lessons** (which compile into
    rules) are write-protected for arms. This is checked at launch scope, in the tool policy, in
    `ArmFS` and in the commit tool.
  - CODEOWNERS requires owner review for governance, rules, guards, lint and workflows.
- **Empty or disabled policy.** AGT allows everything when no policy is loaded. Mitigations:
  - Loading fails at startup if a manifest has no intervention points or `CI_GOVERNANCE_MODE` is
    invalid.
  - `doctor` checks every point in CI.
  - The lint rule keeps every agent behind the governed factory.
- **Provenance and credentials.** Context provenance and credential scope are not tracked by AGT.
  We rely on per-job GitHub tokens and on a model step that never holds a write token
  (`campaign-scheduled.yml`).

## Limitations

- **Approvals.** Approvals use a file queue only and are one-shot. A held action is refused for now
  (suspended); the caller retries the same action after `approve`.
- **Streaming.** Output is checked on the final response only. `post_model_call` is skipped for
  streams.
- **Tool deny.** A function deny terminates the tool loop with a refusal.
- **Tool history.** Tool history is kept in session state and capped at 200 steps.
- **Model check.** The model check is skipped when the agent's model is unspecified.
- **Audit file.** The audit file assumes one writer per path. AGT's entry hash omits
  `policy_decision`; `verify` cross-checks it against the hashed `outcome`, so an allow/transform
  swap is not detectable.
- **SRE vetoes.** Vetoes are keyed by strategy and rebuilt from ledger history. The launch gate
  runs in the CLI, not inside `Campaign.run`.
- **OES.** The extension is on round envelopes only.
