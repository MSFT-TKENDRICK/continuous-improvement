# Models: defaults, pins, and preflight

Model ids are experiment provenance. Harness target and judge ids are pinned for live evaluation;
meta-agent and optimizer ids are loaded from frozen configuration or operator overrides. The harness
never changes a model silently.

## Inventory

| Use | Default/configuration | Override |
|---|---|---|
| Harness agents | `harness/agents/*.yaml` | `CI_META_MODEL` for meta-agent loading, subject to the allowlist |
| Live harness target | required by live tiers | `CI_LAB_TARGET_MODEL` |
| System-1 judge | suite `pipeline.judge.model` | `CI_LAB_JUDGE_MODEL`; endpoint `CI_S1_LLAMA_URL` |
| DSPy/GEPA and SkillOpt optimizer LM | `ci_lab.optim.lm.DEFAULT_MODELS` | `CI_LAB_OPTIMIZER_MODEL`, `CI_COPILOT_SERVE_URL`, `CI_COPILOT_SERVE_KEY[_FILE]` |
| Sleep target and reflector | `ci_lab.sleep.wiring` defaults | `CI_LAB_SLEEP_TARGET_MODEL`, `CI_LAB_SLEEP_REFLECTOR_MODEL` |
| ASSERT tester/test-set generator | each `eval_config.yaml` `default_model` | suite override for explicit generation runs |
| Copilot serve | required command argument | `ci-lab copilot-serve --model <id>` |

The fake profile uses scripted models and skips network preflight. Live `evolve` and `confirm` tiers
require a target model, a judge model, and the configured System-1 endpoint. A served-model mismatch
invalidates the trial rather than silently accepting a different model.

## Overrides

`CI_META_MODEL` is applied before spec validation and hashing. `CI_ALLOWED_MODELS` can extend the
operator allowlist; candidate assets cannot change the process environment. `CI_LAB_TARGET_MODEL`
and `CI_LAB_JUDGE_MODEL` are folded into the evaluator pin, so cached results never mix model pins.

All override values must be plain model ids without expressions.

## Preflight

The Copilot profile checks requested Copilot models before campaign or sleep work spends budget.
`ci-lab copilot-serve` validates its selected model at startup. `ci-lab doctor --models` performs an
opt-in availability report. Offline mode requires loopback endpoints and does not query Copilot.

Example live pins:

```powershell
$env:CI_LAB_TARGET_MODEL = 'gpt-5-mini'
$env:CI_LAB_JUDGE_MODEL = 's1/llamacpp/qwen3.5-4b'
$env:CI_S1_LLAMA_URL = 'http://127.0.0.1:8081'
ci-lab campaign run <cid> --rounds 1 --profile offline
```

See [judge.md](judge.md) for System-1 backends and [providers.md](providers.md) for network policy.
