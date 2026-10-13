# Experiment chat: `ci-lab chat serve`

`ci_lab.chat` serves an **experiment designer** chat agent over [AG-UI](https://docs.ag-ui.com). The
`ci-harness-dashboard` canvas ([canvas.md](canvas.md)) uses it from a CopilotKit chat. In the chat you
formulate an RRSI campaign ([campaign.md](campaign.md)), review it as an OES experiment (cost estimate, A/A
calibration, held-out looks), and launch it. Launching always needs human approval, and the server enforces it.

## Architecture

```
canvas (CopilotKit chat, webview)
  │  POST /agui on the canvas loopback server (x-ci-token, origin check, 1 MiB body cap)
  ▼
canvas proxy (layer 33)  ── drops Origin, Cookie and the canvas token; adds x-ci-chat-token
  │  http://127.0.0.1:<port>/agui
  ▼
ci-lab chat serve        FastAPI + uvicorn, agent_framework_ag_ui 1.4 add_agent_framework_fastapi_endpoint
  ▼
MAF Agent "experiment_designer" (src/ci_lab/chat/agent.py) + tools (src/ci_lab/chat/tools.py)
  ▼
chat client: CopilotChatClient (GitHub Copilot SDK, ambient auth)  |  FakeDesignerClient (scripted, offline)
  ▼ launch_campaign (after approval)
python -m ci_lab.chat.launch <draft.json>  →  ci-lab campaign new → calibrate → run --rounds N   (local)
gh workflow run campaign-scheduled.yml -f cid=… -f rounds=…                                       (workflow)
```

| Module | Role |
| --- | --- |
| `ci_lab.chat.tools` | `ChatConfig`, `ChatTools` (pure, unit-tested tool logic), `maf_tools()` (the MAF tool wrappers) |
| `ci_lab.chat.agent` | `build_agent()`: a Python MAF `Agent` named and id'd `experiment_designer`, with its instructions |
| `ci_lab.chat.fake` | `FakeDesignerClient`: the deterministic scripted chat client behind `--profile fake` |
| `ci_lab.chat.server` | `create_app()` (auth, origin, `/healthz`, AG-UI endpoint), `serve()` (bind, listening line, stdin watcher) |
| `ci_lab.chat.launch` | `python -m ci_lab.chat.launch <draft.json>`: the detached local launch chain |
| `ci_lab.chat.cli` | `ci-lab chat serve` |

The agent is a Python `Agent`, not a declarative YAML spec ([maf.md](maf.md)). Declarative tool specs cannot
express `approval_mode` or the AG-UI `state_update` results that the tools rely on.

## Process contract

```
CI_CHAT_TOKEN=<random, >= 16 chars> uv run --no-sync ci-lab chat serve [--profile copilot|fake] [--port 0] ...
```

- Without `CI_CHAT_TOKEN`, or with a token shorter than 16 chars, the server exits 2 with `{"error": "..."}` on stderr.
- Hosts other than `127.0.0.1`, `localhost` (bound as `127.0.0.1`) or `::1` are refused with exit 2.
- After the socket is bound and uvicorn has started, the server writes **exactly one** stdout line:
  `{"event":"listening","host":"127.0.0.1","port":<int>,"path":"/agui"}`. With `--port 0` the OS picks the
  port, and this line reports it. All logging goes to stderr. The uvicorn access log is off, and the token is
  never logged.
- When stdin reaches EOF, the server shuts down gracefully and exits 0. This happens when the parent dies or
  closes the pipe, so a crashed canvas cannot leave an orphan server. Pass `--no-stdin-watch` when stdin is
  not a pipe the parent holds, such as `stdin=DEVNULL` or a manual run whose stdin you would close.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Loopback host: `127.0.0.1`, `localhost` or `::1` |
| `--port` | `0` | Port; `0` picks a free one |
| `--profile` | `copilot` | Chat model: `copilot` (Copilot SDK) or `fake` (scripted, no network) |
| `--repo` | none | `OWNER/REPO`, passed to launched campaigns and to `gh workflow run --repo` |
| `--run-dir` | `$CI_RUN_DIR` or `artifacts/ci-runs` | Campaign run root (as `ci-lab campaign --run-dir`) |
| `--dry-run-launch` | off | `launch_campaign` records the argv instead of executing it |
| `--no-stdin-watch` | off | Do not exit on stdin EOF |
| `--campaign-profile` | `--profile` | Profile for locally launched campaigns: `copilot`, `fake` or `offline` |
| `--chat-dir` | `artifacts/chat` | Drafts and launch records |
| `--ledger-dir` | campaign default | Ledger root passed to launched campaigns |
| `--live-publish` | off | Launched local campaigns publish for real. By default they get `--dry-run-publish` |
| `--model` | `$CI_CHAT_MODEL` or `gpt-5-mini` | Copilot model for `--profile copilot` |

The contract specifies the first seven flags. The other five are additions.

## Security model

- **Loopback only.** The server binds `127.0.0.1` or `::1`, and on Windows it uses `SO_EXCLUSIVEADDRUSE`.
- **Token.** `POST /agui` requires `x-ci-chat-token: <CI_CHAT_TOKEN>`. The check uses a constant-time
  compare (`hmac.compare_digest`); a missing or wrong token gets 401. `/healthz` needs no token and returns only
  `{"ok":true}`. The OpenAPI and docs routes are disabled.
- **No browsers.** An ASGI middleware answers **403** to any request that carries an `Origin` header, on every
  path, even with a valid token. A cross-origin page in a browser always sends `Origin` on a POST, so it cannot
  drive the agent. It also cannot read the token. The canvas proxy strips `Origin`.
- **Approval gate.** `launch_campaign` is a MAF tool with `approval_mode="always_require"`. The model can only
  *request* it. AG-UI ends the run with an interrupt, and the tool runs only when a later run resumes that
  interrupt with `approved: true`.
  - An interrupt id is single-use and bound to its thread. A forged id, an id from another thread, a replayed
    resume, a rejected resume or a cancelled resume does not launch. `tests/ci_lab/chat/test_server.py` covers
    each case.
  - It is also the only gated tool. The read-only tools and `draft_campaign` only write a draft file.
- **Launch re-validation.** `launch_campaign(cid)` launches only a stored draft. It re-validates the draft
  first: the id is still free, the ranges hold, and there has been no earlier non-dry-run launch. The approver
  may edit the `cid` argument (see below), but that can only select another validated draft.
- **Safe ids, no shell.** Campaign ids must match `CAMPAIGN_RE` (`[a-z0-9][a-z0-9-]{2,40}`), so `../x`
  and similar ids are rejected. Every subprocess gets an argv list, never `shell=True`. Child processes do
  not inherit `CI_CHAT_TOKEN`.

## Tools

| Tool | Approval | Returns |
| --- | --- | --- |
| `list_strategies` | no | Mutation strategies (`STRATEGIES`) and the evolvable text components |
| `list_suites` | no | Frozen ASSERT suites (`evals/assert/<suite>/test_set.jsonl`) with case counts per split |
| `get_default_hyperparameters` | no | `DEFAULT_HYPER` values, descriptions (from `campaign/defaults.py`) and chat limits |
| `list_campaigns` | no | Campaigns in the ledger/run root plus launch records; missing dirs give an empty list |
| `campaign_status(cid)` | no | Ledger summary and launch record for one campaign |
| `draft_campaign(cid, hyper, rounds, target, rationale)` | no | Validated, normalized draft with cost estimate; sets state `draft` |
| `launch_campaign(cid)` | **always** | `{launched, status, argv, commands, pid?, log?, run_url?, dry_run?}`; sets state `launches` |

`draft_campaign` validation:
- `cid` must be a safe id that is not already used.
- `hyper` takes overrides of `DEFAULT_HYPER` keys only; unknown keys are errors. Values are type-checked, and
  the ranges are:
  - `1 <= arms <= 8`;
  - `aa_repeats >= 2`;
  - `k >= 1`;
  - `strategies` is a non-empty subset of `STRATEGIES`;
  - `b_min <= b_max`;
  - and so on.
- `rounds` must be in `1..max_rounds`. The workflow target also requires `1..9`.
- `target` is `local` or `workflow`.
- `rationale` must be non-empty.

On success, the normalized draft is written to `<chat-dir>/drafts/<cid>.json`.

**Estimate.** `evaluations = (aa_repeats + rounds * arms) * cases * k`, where `cases` is the number of
evolve-split cases in the frozen sets. The result also includes:
- a breakdown with the per-round incumbent re-evaluation (`evaluations_with_incumbent`);
- the cost of one held-out confirm look (`2 * heldout_cases * k`). That look is not part of the launch chain.

**Launch.**
- `local`: spawns `python -m ci_lab.chat.launch <draft.json>` detached, with the server's interpreter. It runs
  `campaign new <cid> --hyper K=V...`, then `calibrate`, then `run --rounds N`, with the same
  `--profile/--run-dir/--ledger-dir/--repo` and with `--dry-run-publish` unless `--live-publish` is set. It
  stops at the first failing step.
  - The output goes to `<chat-dir>/launches/<cid>.log`, and the per-step status goes to
    `<cid>.status.json`.
  - The record `<cid>.json` holds `{cid, target, status, pid, log}`.
- `workflow`: runs `gh workflow run campaign-scheduled.yml -f cid=<cid> -f rounds=<n> [--repo R]`. That
  workflow accepts **only** `cid` and `rounds`, and it runs the copilot profile with default hyperparameters.
  The draft and launch results therefore list any overrides as `ignored_overrides` and include a warning.
- With `--dry-run-launch`, the status is `dry_run`, `launched` is false, and the argv is recorded. A dry-run
  or failed launch does not consume the id.

**Shared state** (AG-UI `STATE_SNAPSHOT`):
```json
{"draft": {"cid": "...", "target": "local", "rounds": 1, "rationale": "...", "hyper": {}, "overrides": {},
           "ignored_overrides": {}, "estimate": {}, "profile": "fake", "commands": [[]], "warnings": []} ,
 "launches": [{"cid": "...", "target": "local", "status": "started|dispatched|dry_run|failed",
               "pid": 0, "log": "...", "run_url": "..."}]}
```
`draft` is `null` until the first successful `draft_campaign`. With a `state_schema` configured, AG-UI also
injects the current state into the model context as a system message.

## Approval / resume protocol (agent_framework_ag_ui 1.4)

**Recorded fixture:** [`tests/fixtures/chat/approval_flow.json`](../tests/fixtures/chat/approval_flow.json).
It holds the real request bodies and parsed SSE events of `--profile fake --dry-run-launch` for these steps:
1. `1_draft`: a draft run.
2. Three launch runs that end in an interrupt: `2_launch_interrupt_{approved,rejected,approved_legacy}`.
3. Their resumes: `3_resume_{approved,rejected,approved_legacy}`.

`protocol` summarizes the field names. `test_approval_fixture.py` fails if the live server's event types or
field paths drift from the fixture. To regenerate it:
`uv run --no-sync python tests/ci_lab/chat/test_approval_fixture.py`.

All fields are camelCase on the wire. The SSE stream is `data: <json>` lines.

1. **Interrupting run.** The model calls `launch_campaign`. The stream emits:
   - `TOOL_CALL_START{toolCallId, toolCallName:"launch_campaign", parentMessageId}`, then `TOOL_CALL_ARGS`,
     then `TOOL_CALL_END`. No `TOOL_CALL_RESULT` follows.
   - `CUSTOM{name:"function_approval_request", value:{id, function_call:{call_id, name, arguments}}}`
   - `MESSAGES_SNAPSHOT{messages}`
   - `RUN_FINISHED{threadId, runId, outcome:{type:"interrupt", interrupts:[{id:"af-call-<hex>",
     reason:"tool_call", message:"Approve running launch_campaign?", toolCallId, responseSchema,
     metadata:{agent_framework:{type:"function_approval_request", function_call:{call_id, name, arguments}}}}]}}`
   - `responseSchema` allows `approved` or `accepted` (boolean, at least one of them required), plus the
     optional edited arguments `cid` / `editedArgs`. Additional properties are not allowed.
   - There is no expiry field.
2. **Resume run.** Send the same `threadId` and a new `runId`. Set `messages` to the interrupted run's last
   `MESSAGES_SNAPSHOT.messages`, and `state` to its last `STATE_SNAPSHOT.snapshot`. Then add:
   - canonical: `"resume": [{"interruptId": "<interrupts[0].id>", "status": "resolved", "payload": {"approved": true}}]`
   - reject: the same with `"payload": {"approved": false}`. `"status": "cancelled"` also rejects.
   - legacy (also accepted): `"resume": [{"interruptId": "<id>", "approved": true}]`
3. **Approved resume.** The stream emits `TOOL_CALL_RESULT{toolCallId, content}` for `launch_campaign` before
   any model text, then a `STATE_SNAPSHOT` with `launches` updated, then the model's summary.
   - **Rejected resume:** no `TOOL_CALL_RESULT`, and the tool is not executed. The model replies that nothing
     was launched.

Pending approvals live in memory in the AG-UI agent, so restarting the server drops them. Every approved resume
also logs `Ignored an approval response ... did not match the active approval occurrence identity` at WARNING
level from `agent_framework`. This is benign: ag-ui has already executed the tool exactly once, and the tests
count executions.

## Profiles

- `fake`: `FakeDesignerClient` is deterministic and needs no network or inference. Tests, the fixture and canvas
  UI development use it. It handles a user message as follows:
  - a message containing `launch` calls `launch_campaign`, which triggers the interrupt;
  - a message containing `draft` calls `draft_campaign`, with the cid, `N arms` and `N rounds` parsed from the
    text (defaults: `chat-demo`, 2 arms, 1 round). `workflow` in the message selects the workflow target;
  - any other message gets a greeting;
  - after a tool result, it summarizes the result. It never claims a launch unless `launched` is true.
- `copilot`: `CopilotChatClient` ([providers.md](providers.md)) on the GitHub Copilot SDK with ambient auth.
  The model comes from `--model`, `$CI_CHAT_MODEL` or `gpt-5-mini`. Its streaming is **buffered**: each model
  turn is computed in full and then emitted, so text appears per turn rather than per token.

## Limitations and contract deviations

- **Resume shape.** The contract sketched `resume: [{interruptId, approved}]`. agent_framework_ag_ui 1.4
  accepts that legacy form, but the canonical form is `{interruptId, status:"resolved", payload:{approved}}`.
  The fixture records both.
- **Origin check scope.** The 403 for requests with an `Origin` header covers `/healthz` as well as `/agui`.
- **Extra flags.** `--campaign-profile`, `--chat-dir`, `--ledger-dir`, `--live-publish` and `--model` are
  additions to the contract's flag list.
- **Buffered streaming** for `--profile copilot` (above).
- **Workflow target** ignores hyperparameter overrides and allows only 1..9 rounds, because those are the only
  inputs `campaign-scheduled.yml` takes.
- **In-memory approvals.** A server restart loses pending interrupts. The canvas should then start a new run.
- **No held-out confirm from chat.** The launch chain stops after `run`. Run `ci-lab campaign confirm`
  separately.
- **Fixture generation.** The fixture is recorded in-process with `fastapi.testclient`. The subprocess test
  (`test_serve_process.py`) covers the real process contract.
