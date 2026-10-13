# CI Lab self-improving harness

CI Lab evaluates and improves its own agent harness. The editable target is the repo-root
[`harness/`](harness/) tree. Evaluation, selection, governance, campaign orchestration, and model
pins remain frozen under [`src/ci_lab/`](src/ci_lab/) and [`evals/`](evals/).

## What is implemented

- **Python Microsoft Agent Framework (MAF):** declarative agents and expression-free workflows with
  file checkpoints and resume.
- **Prompt and skill optimization:** DSPy/GEPA owns prompts. SkillOpt 0.2.x owns skills and the
  nightly SkillOpt-Sleep job.
- **Structural optimization:** Agent Lightning 1.0.2 supplies the rollout journal/store and optional
  proxy. CI Lab's `agl` strategy uses a Copilot SDK LLM to propose one bounded structural edit. It
  does not use `verl`, RL training, GPUs, prompt optimization, or skill optimization.
- **Frozen evaluation:** five Microsoft ASSERT harness suites combine deterministic checks with the
  System-1 rubric judge. Evaluator-owned runtime and surface measurements cannot be supplied by the
  candidate.
- **Election and governance:** OES envelopes record experiments; RRSI elects candidates with
  non-compensatory safety, complexity, and call-count gates. The agent bus adds voters, a
  deterministic judge, and an out-of-band adversary lane. AGT/ACS middleware applies fail-closed
  application policy.
- **Operations:** OpenTelemetry spans feed an optional Aspire dashboard and the native Copilot
  canvas. The canvas also hosts a CopilotKit experiment chat that drafts campaigns and launches only
  after approval.
- **Lessons as structure:** deterministic mining, rules, guards, lint, and paired guard experiments
  turn repeated failures into reviewable enforcement rather than prompt folklore.

Nothing merges automatically. Campaign, sleep, lesson, and evaluator changes are proposals or draft
pull requests for human review.

## Quick start

Python 3.12 and `uv` are required.

```powershell
uv sync
uv run ci-lab --help
uv run ci-lab harness validate --dir harness
uv run ci-lab harness metrics --dir harness --json
uv run pytest -q -p no:cacheprovider
```

The fake profile is deterministic and offline:

```powershell
uv run ci-lab campaign new harness-smoke --profile fake --dry-run-publish
uv run ci-lab campaign calibrate harness-smoke --profile fake --dry-run-publish
uv run ci-lab campaign run harness-smoke --profile fake --dry-run-publish --rounds 1
uv run ci-lab campaign status harness-smoke --profile fake --dry-run-publish
```

Useful checks:

```powershell
uv run ci-lab doctor
uv run ci-lab doctor --models
uv run ci-lab template doctor
uv run ci-lab sleep dry-run
uv run ci-lab chat serve --profile fake --dry-run-launch --no-stdin-watch
```

`doctor --models` and live profiles use network/model credentials; the other commands above can run
offline. `chat serve` also requires a `CI_CHAT_TOKEN` of at least 16 characters.

## Frozen control plane and editable target

[`harness/harness.yaml`](harness/harness.yaml) must byte-match
[`src/ci_lab/harness_tree/manifest.yaml`](src/ci_lab/harness_tree/manifest.yaml). It freezes component
globs, optimizer owners, required agents, workflow functions, and execution/evaluation caps.
Candidates cannot widen their writable surface.

Editable target components:

| Component | Path | Dedicated owner |
|---|---|---|
| `prompt` | `harness/prompts/**/*.md` | `gepa` |
| `skill` | `harness/skills/**` | `skillopt` |
| `guard` | `harness/guards/**` | `guard` |
| `agent` | `harness/agents/*.yaml` | `agl` |
| `loop` | `harness/loops/*.yaml` | `agl` |
| `workflow` | `harness/workflows/*.yaml` | `agl` |
| `mcp` | `harness/mcp/exposure.yaml` | `agl` |
| `client_tool` | `harness/tools/*.yaml` | `agl` |
| `config`, `context_mgmt`, `memory` | no current file globs | `agl` |

The general `agent` strategy may edit any text component except `guard`; that broad permission does
not replace the dedicated ownership above. Campaign arm workflows, evaluator logic, selection, and
governance stay under `src/ci_lab/`. The files in `harness/workflows/` are target-agent workflows
(`triage` and `propose`), not arm workflows.

At the start of each round, the campaign snapshots the incumbent harness. Analyst, proposer, repair,
and critic work from that snapshot; only evaluation loads a candidate tree. An elected candidate
becomes the incumbent at the next round.

## Evaluation

The five frozen suites are `harness_triage`, `harness_proposal`, `harness_taskgraph`,
`harness_tool_use`, and critical-safety suite `harness_injection`. Cases and tier selection are in
[`evals/datasets/harness.yaml`](evals/datasets/harness.yaml); measured rubrics are in
[`evals/rubrics/harness/`](evals/rubrics/harness/).

- `ci`: fake profile, all cases, `k=1`, scripted target and scripted judge.
- `evolve`: live evolve split, at most three cases per suite, `k=1`.
- `confirm`: live held-out split, finalists only, `k=2`.

Live tiers require pinned target and judge models plus `CI_S1_LLAMA_URL`. A served target-model
mismatch creates an invalid trial. Per-case hard caps for LLM calls, tool calls, tokens, and timeout
come from the frozen manifest; exceeding one scores the trial zero.

`TaskScore.score` contains quality only: deterministic correctness/safety plus System-1 criteria.
Resource criteria are reported separately. Missing required evaluator measurements fail closed.
The `harness` RRSI profile caps complexity growth at 10% and mean call growth at 15%; wall time is
recorded but is diagnostic by default. Simplicity credit is available only to a valid,
quality-non-inferior, critical-safety-non-inferior tree.

## Scheduled workflows

PR/push CI runs pytest, lint/governance checks, and canvas bundle/tests. Scheduled jobs are inert
until repository variable `CI_HARNESS_ENABLED=true`:

- `campaign-scheduled.yml`: Mondays 09:41 UTC; needs `CAMPAIGN_ID`, optional
  `CAMPAIGN_ROUNDS`, live model variables, and the `campaign-publish` environment.
- `sleep-nightly.yml`: daily 07:17 UTC; usage-gated SkillOpt-Sleep with the `sleep-publish`
  environment.
- `usage-harvest.yml`: daily 05:41 UTC.
- `governance-native.yml`: Tuesdays 06:17 UTC, informational native ACS parity.

The model jobs use the job-scoped GitHub token with `copilot-requests: write`; publish jobs receive
write permissions but run no model. No custom repository secret is referenced by the shipped YAML.
Live harness evaluation does require a reachable `CI_S1_LLAMA_URL`.

## Documentation

- [Architecture and limitations](docs/harness.md)
- [Harness tree and snapshots](docs/harness-tree.md)
- [Optimization ownership](docs/optim.md)
- [RRSI selection](docs/rrsi.md)
- [MCP direct/code mode](docs/mcp.md)
- [Campaign operations](docs/campaign.md)
- [Models and pins](docs/models.md)
- [Template initialization and custom domains](docs/template.md)
- [Canvas and Aspire](docs/canvas.md)
- [Experiment chat](docs/chat.md)
- [Governance](docs/governance.md)
