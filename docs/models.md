# Models: defaults, overrides and the preflight

Every model id the harness can use is listed below, with how an operator selects another one. A model id is part of an experiment's provenance: it is hashed (spec digest, harness tree, evaluator pin) and, for meta agents, checked against an allowlist. The harness never swaps a model on its own. To change a model, set an override. The run records the model it actually used.

Copilot model lists change over time, so a baked-in id can go stale (`Model "claude-sonnet-4.5" is not available`). With the `copilot` profile, a **model preflight** checks every model a command will use before it spends budget. If a model is missing, the command stops with a message that names the model, who uses it, the override to set and the available ids.

## Inventory

| Model id (default) | Used by | Defined in | Override |
|---|---|---|---|
| `claude-sonnet-5` | Meta agents (analyst, proposer, critic, reflector) and their subagents (failure_analyst) | `src/ci_lab/meta/specs/*.yaml`, `manifest.yaml` | `CI_META_MODEL` (allowlisted) |
| `claude-sonnet-5` | Lesson synthesizer (`guard` strategy) | `src/ci_lab/lessons_arm/specs/lesson_synthesizer.yaml` | `CI_META_MODEL` (allowlisted) |
| `gpt-5-mini` | Order agent with `ORDER_AGENT_PROFILE=copilot` | `src/order_support/harness/agent.yaml` (`model.id`) | A committed harness change (hashed in the harness tree) |
| `openai/local` | Order agent with the offline profile | `ORDER_AGENT_MODEL` default | `ORDER_AGENT_MODEL` |
| `gpt-5-mini` | DSPy/GEPA and SkillOpt optimizer LM (copilot profile, through copilot-serve) | `ci_lab.optim.lm.DEFAULT_MODELS` | `CI_LAB_OPTIMIZER_MODEL`; endpoint `CI_COPILOT_SERVE_URL`, `CI_COPILOT_SERVE_KEY[_FILE]` |
| `gpt-5-mini` | Sleep target and reflector (copilot profile) | `ci_lab.sleep.wiring.DEFAULT_MODEL` | `CI_LAB_SLEEP_TARGET_MODEL`, `CI_LAB_SLEEP_REFLECTOR_MODEL` |
| `openai/local` | ASSERT tester and test-set generator | `evals/assert/*/eval_config.yaml` (`default_model`) | `CI_ASSERT_MODEL` in campaigns; `--override default_model.name=<id>` for `order-support-evals run` |
| `s1/llamacpp/qwen3.5-4b` | ASSERT judge | `evals/assert/*/eval_config.yaml` (`judge.model`) | Local llama.cpp judge, not Copilot (see [judge.md](judge.md)) |
| (required `--model`) | `ci-lab copilot-serve` | command line | `--model <id>` |

The GitHub workflows (`campaign-scheduled.yml`, `sleep-nightly.yml`) set no model variables, so they use these defaults. Set the overrides as repository variables or in the workflow `env` to change them.

## Overrides

**`CI_META_MODEL`** replaces `model.id` in every meta-agent spec and in the lesson synthesizer spec. The replacement happens when the spec is loaded, **before** validation and hashing, so the recorded spec, `spec_digest` and run provenance all name the model that actually ran. The value must be in the manifest's `allowed_models` (`claude-sonnet-5`, `claude-sonnet-5.5`, `claude-opus-5`, `gpt-5.5`).

**`CI_ALLOWED_MODELS`** (comma-separated) extends that allowlist. Only the operator environment can extend it. Arms are data-only and cannot change it, so an arm can never pick a model for a meta agent.

**`CI_ASSERT_MODEL`** sets the ASSERT tester for campaign evaluations (`--override default_model.name=<id>` on every case run). It changes the evaluator, so it is folded into the evaluator pin (`evaluator_tree`). Cached results and comparisons never mix testers. A campaign calibrated without it must be recalibrated, or re-created, before rounds with it.

All override values must be plain model ids (no whitespace, `=` or expressions).

## Preflight

The preflight runs only with the `copilot` profile. The `fake` and `offline` profiles need no Copilot models and skip it. Model listings are injectable, so tests never touch the network.

| Command | What is checked |
|---|---|
| `ci-lab campaign calibrate\|run\|confirm` | Before any spend, once per command: the meta agents, the synthesizer (if `guard`), and the optimizer LM (if `gepa`/`skillopt`) against the Copilot account's `list_models()`. The optimizer LM's copilot-serve `GET /models` is checked too. The order agent model and, at `OPENAI_API_BASE`, the ASSERT tester (and `ORDER_AGENT_MODEL` for an offline order agent) are also checked. A failure exits with code 2 and prints `{"error": "model preflight failed", "detail": ...}`. |
| `ci-lab copilot-serve --model <id>` | At startup, `<id>` against `list_models()`. If it is missing, the server prints the available ids and exits (uvicorn startup failure, exit 3). |
| `ci-lab sleep run --profile copilot` | The sleep target and reflector models, after setup and before the night starts. A failure exits 1 with `status=error`. |
| Any `CopilotChatClient` (e.g. `order-support-evals run` with `ORDER_AGENT_PROFILE=copilot`) | Its model, once, right after the Copilot CLI starts. A missing model raises `ModelPreflightError`. A failure to list models is logged and does not block the client. |
| `ci-lab doctor --models` | Opt-in and needs the network: a report of every copilot-profile model above, with its users, override and availability. Exits 1 if any is missing. |

The building blocks are in `ci_lab.providers.models` (`check_copilot_models`, `check_served_models`, `ModelUse`, `ModelPreflightError`) and `ci_lab.campaign.preflight` (`campaign_model_plan`, `make_preflight`, `CampaignDeps.preflight`).

### Example: live campaign through copilot-serve

```powershell
$env:ORDER_AGENT_PROFILE = 'copilot'                # order agent model: harness agent.yaml
$env:OPENAI_API_BASE = 'http://127.0.0.1:8090/v1'   # copilot-serve --model gpt-5-mini
$env:CI_ASSERT_MODEL = 'openai/gpt-5-mini'          # ASSERT tester through copilot-serve
$env:CI_COPILOT_SERVE_URL = 'http://127.0.0.1:8090/v1'
$env:CI_COPILOT_SERVE_KEY_FILE = '.ci/copilot.key'  # gepa/skillopt optimizer LM
# $env:CI_META_MODEL = 'claude-opus-5'              # optional: another allowlisted meta model
ci-lab doctor --models
ci-lab campaign run <cid> --rounds 1 --profile copilot
```
