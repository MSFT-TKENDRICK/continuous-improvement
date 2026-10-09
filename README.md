# Order-support agent evals with Microsoft ASSERT

Safety and quality evals for a tool-calling customer-support agent. ASSERT
([Microsoft ASSERT](https://github.com/responsibleai/ASSERT), `assert-ai` 0.3) is the only eval
framework. The agent runs against a local OpenAI-compatible model or the Copilot SDK, and rubric
dimensions are judged inside ASSERT by a System-1 decision model through the `s1/<backend>/<model>`
LiteLLM provider (default `s1/llamacpp/qwen3.5-4b`).

The repo contains two kinds of eval:

* **Five behavior suites.** These are full ASSERT pipelines: `test_set` → `inference` → `judge`. Each
  one generates adversarial conversations for a single behavior, drives the live agent through them
  with a simulated tester, and judges the traced transcripts against a hand-written taxonomy.
* **A judge-replay suite.** This is a judge-only ASSERT pipeline over 30 labelled traces. The
  `calibrate` command then compares ASSERT's verdicts with the reference labels. This suite measures the
  judge, not the agent.

The evals are also the scoring layer of `ci_lab`, a self-improving harness. **Start at
[docs/harness.md](docs/harness.md)** for its overview, architecture diagram, an index of every module
doc, and known limitations.

The harness also has three pieces for running work under hidden rubrics:

| Pillar | What it does | Doc |
|---|---|---|
| Agent bus | A write-ahead, hash-chained log per task. Voters (ASSERT, System-1 rubric, deterministic, agent) score each proposal, and a deterministic judge decides `commit`, `revise` or `reject`. A successor agent only sees the corrected trajectory. | [docs/bus.md](docs/bus.md) |
| Task graph | A graph of single deliverables that runs async and in parallel. Rubrics are sealed and hidden from the student behind a leak-screening firewall. Run it with `ci-lab graph run`. | [docs/taskgraph.md](docs/taskgraph.md) |
| Adversary | Gamers and an LLM challenger attack the rubric. Exploits feed gated rubric hardening and an evaluator-experiment proposal. Campaigns run it as an out-of-band lane. | [docs/adversary.md](docs/adversary.md) |

Every agent and campaign launch in the harness runs under deterministic
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit) /
Agent Control Specification policies, with a hash-chained audit trail and identity-bound approvals.
See [docs/governance.md](docs/governance.md).

## Use this repo as a template

This repository is a GitHub template: choose **Use this template** (default branch only; leave
**Include all branches** unticked), then run `uv sync` and
`uv run ci-lab template init --owners "@my-org/agent-owners" --apply` to make the copy yours. Its
scheduled workflows stay off until you set the repository variable `CI_HARNESS_ENABLED=true`. See
[docs/template.md](docs/template.md) for the quick start, `ci-lab template doctor`, and how to swap
in your own agent.

## Contents

* [Quick start](#quick-start)
* [Layout](#layout)
* [Design](#design)
* [ASSERT and the System-1 judge](#assert-and-the-system-1-judge)
* [Operational notes and ASSERT gotchas](#operational-notes-and-assert-gotchas)
* [Results: judge replay](#results-judge-replay)
* [Adversarial critique and limitations](#adversarial-critique-and-limitations)
* [Sources](#sources)

## Quick start

```powershell
# assert-ai pulls in fastuuid, which has no Windows-ARM64 wheel; use x86_64 CPython on ARM machines.
uv venv --python cpython-3.12-windows-x86_64-none
