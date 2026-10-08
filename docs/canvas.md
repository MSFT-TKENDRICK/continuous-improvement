# CI Harness Dashboard canvas

A GitHub Copilot app canvas (`ci-harness-dashboard`) that shows the self-improving harness live:
campaigns and their frontier, RRSI rounds and arms, OES experiments and decisions, ASSERT evals,
sleep nights, AGL rollouts, and OpenTelemetry traces from local JSONL files, imported CI runs and
Aspire. It implements design §12 (`docs/design/self-improving-harness.md`) and security controls
C29–C38.

Code: `.github/extensions/ci-harness-dashboard/`. It is plain Node ESM with no npm dependencies
(Node ≥ 20). Only `extension.mjs` imports `@github/copilot-sdk/extension`, which the host resolves.
The one exception is the optional experiment chat. Its prebuilt browser bundle in `ui/chat/` is built
from `web/experiment-chat/` (see [Experiment chat](#experiment-chat)). The extension itself still has
no `package.json` and installs nothing.

## Architecture

```
extension.mjs     thin wiring: joinSession + createCanvas(open/onClose/actions from lib/canvas.mjs)
lib/canvas.mjs    SDK-independent controller: input validation, instance lifecycle, agent actions
lib/server.mjs    per-instance loopback HTTP server: static UI, JSON API, SSE, POST guards, CSP
lib/hub.mjs       one data hub per repo root, refcounted across instances: fs.watch + debounce + poll
lib/sources.mjs   ALL on-disk format knowledge: tolerant, contained, size-capped readers
lib/bus.mjs       agent-bus WAL reader and orchestrator-only projection (Bus tab)
lib/model.mjs     pure aggregation: campaigns, live runs, experiments, span trees, evals, sleep
lib/aspire.mjs    Aspire dashboard API client. It is the only module that sees Aspire secrets.
lib/chat.mjs      experiment-chat backend manager: spawns `ci-lab chat serve`, holds its token
lib/security.mjs  containment, id validation, Host/Origin checks, C29 redaction, secret stripping
ui/               index.html, app.js, app.css: no inline script/style, DOM built via textContent only
ui/chat/          generated CopilotKit chat bundle (do not edit; built from web/experiment-chat)
dev-preview.mjs   run the server in a normal browser without the SDK
test/             node --test suites and fixtures
```

- **Instances.** Every canvas instance (`instanceId`) gets its own server on
  `127.0.0.1:<ephemeral>` with a random per-instance token.
  - `open()` is idempotent. Re-opening an instance returns the same URL. If the input changed,
    the new UI state is pushed to the iframe over SSE.
  - If `repoRoot` changes, the server is recreated.
  - `onClose` stops the server and releases the hub.
- **Hub.** One hub per repo realpath, shared across instances.
  - It watches the directories it knows about with `fs.watch`, coalesces events with a 250 ms
    debounce, and also polls every 5 s because watchers are unreliable on network drives and
    on Windows.
  - It rebuilds the model only when the scanned files' mtime/size signature changes. After a
    rebuild it fans `changed{version}` out to every SSE client.
  - Overlapping refreshes are coalesced, and a manual refresh is never lost.
- **Iframe ↔ server.** The iframe uses `fetch` for JSON and `EventSource('/events')` for push.
  The server sends the SSE events `hello`, `changed`, `ui` and `ping` (heartbeat every 15 s).
  UI state carries a `seq`, so the iframe applies only newer pushes.

## Open input and actions

Open input (`additionalProperties: false`):

```json
{ "repoRoot": "<absolute path>", "view": "overview|live|experiment|traces|evals|sleep|bus|chat|aspire",
  "campaignId": "...", "experimentId": "...", "traceId": "<32 hex>" }
```

- An explicit `repoRoot` is used as given.
- Without one, the session working directory is used and the controller walks up to the nearest
  `.git`.
- A `traceId` implies the Traces view, and an `experimentId` implies the Experiment view.
- Errors are reported as `CanvasError` codes: `invalid_input`, `repo_not_found`, `open_failed`
  and `canvas_not_open`.

| Action | Input | Result |
|---|---|---|
| `refresh` | `{}` | Rescans now and pushes to all viewers. Returns `{ok, version, status}`. |
| `show_view` | `{view}` | Switches the view. Returns `{ok, ui, viewers}`. |
| `select_campaign` | `{campaignId}` | Opens the Experiment view on that campaign. Adds `known`. |
| `focus_experiment` | `{experimentId}` | Opens the Experiment view. Adds `found`, `summary` and `runState`. |
| `focus_trace` | `{traceId}` | Opens the Traces view. Adds `foundLocally`, `name`, `spans` and `errors`. |
| `get_summary` | `{}` | Compact JSON of campaigns, live runs, last decisions, evals, sleep, bus totals, traces and imports. Works without an open instance (uses the session working directory). |
| `dashboard_status` | `{}` | URL, viewer count, UI state, hub roots/watchers/warnings, and Aspire `configured/reachable/version/uiUrl/hint`. Contains no secrets. |
| `chat_status` | `{}` | Experiment-chat backend: `open`, `state` (`starting/ready/crashed/stopped/disabled`), `pid`, `port`, `profile`, `runDir`, `restarts`, `lastError`, `retryInMs`, `viewers`. It never spawns the backend and contains no tokens. To open the chat, use `show_view` with `{view: "chat"}`. |

## Views

- **Overview.** Campaign cards with the incumbent and its score, round count, last decision,
  STOP state and token burn. Also shows live counts, latest decisions, latest evals, the last
  sleep night, imported CI runs, and warnings.
- **Live.** Runs from the run roots. For each run it shows phase, state and age, plus an
  arms × strategy × state grid. A run is **stale** when its `updated` age is more than 2× the
  heartbeat (C36). Stale runs are flagged, not hidden.
- **Experiment.** Campaign frontier, calibration and rounds, or a single OES envelope. Shows arms,
  decision, ΔS/ΔC (score and cost deltas) and confirm/land/stack, and links to the
  experiment's traces.
- **Traces.** A trace list followed by a collapsible span tree with offsets, durations, errors
  and exception types. MAF/GenAI spans appear nested under their parents.
  - Spans come from JSONL files, imported CI runs and Aspire, merged by span id.
  - `gen_ai.*` content attributes (messages, prompts, tool arguments/results, system
    instructions) are redacted unless the span or its resource sets `ci.telemetry.sensitive`,
    `ci.sensitive` or `gen_ai.sensitive_data` to true (C29).
- **Evals.** ASSERT runs with per-dimension distributions (boolean, ordinal and numeric),
  flags, a stale-heartbeat warning, and case counts.
- **Sleep.** Sleep state, nights and their envelopes, and task queue counts.
- **Bus.** Agent-bus write-ahead logs (see [Bus tab](#bus-tab)).
- **Chat.** The experiment chat (see [Experiment chat](#experiment-chat)). It is loaded only when the
  tab is opened. If the bundle or the backend is unavailable, the tab shows an error panel with Retry,
  and the other views are unaffected.
- **Aspire.** Reachability, version and recent traces.
  - "Open Aspire" goes through a user-click POST endpoint that returns the login URL. The iframe
    `window.open`s it and falls back to a copy box if the popup is blocked.
  - Per-trace deep links have the form `/traces/detail/<traceId>`.

Every view has an empty state that says how to produce its data, for example
"run `ci-lab dashboard up`" or "run `ci-lab telemetry pull --run <id>`".

## HTTP API (per instance)

`GET /` · `/app.js` · `/app.css` · `/events` (SSE) · `/api/summary` · `/api/experiments[?campaign=]`
· `/api/experiment/:id` · `/api/live` · `/api/traces[?aspire=0]` · `/api/trace/:id` · `/api/evals`
· `/api/sleep` · `/api/bus` · `/api/aspire` · `/api/chat/status` · `/chat/<file>` (the bundle, allowlisted names).

POST routes: `/api/ui` · `/api/refresh` · `/api/aspire/login` · `/api/chat/start` · `/agui` (the
experiment-chat proxy, body ≤ 1 MiB).

## Security mapping (§12.2)

| Control | Implementation |
|---|---|
| C29 GenAI content | `security.isSensitiveAttr` plus the `sensitiveEnabled` flags. Values are redacted in the model before they reach the API. |
| C30 secrets | `aspire.mjs` keeps `api_key`, `browser_token` and `otlp_key` internal. The `x-api-key` header is used only server-side. `stripSecrets` is applied to action results and status. The login URL is returned only from a token-guarded, user-initiated POST. Tests assert that no secret or instance token appears in any response, action result or log. |
| C31 DNS rebinding / cross-origin | The server binds `127.0.0.1` only. A `Host` other than `127.0.0.1:<port>` or `localhost:<port>` gets 421. POSTs require the `x-ci-token` header (a per-instance random value injected into `index.html` via `<meta>`), JSON content type, an allowed `Origin` and `Sec-Fetch-Site`, and a body ≤ 16 KB. Other methods get 405. Aspire URLs must be loopback. |
| C32 no Aspire iframe/proxy | No reverse proxy. Native views are rendered from OTLP-JSON (the same flattening serves the Aspire API and JSONL). Full Aspire opens top-level on a click. |
| C33 Aspire optional | Every view works without Aspire. The Aspire view and the merged spans degrade to a hint. |
| C34 supply chain | The canvas runtime has zero npm dependencies and downloads nothing. The experiment-chat bundle is built offline from exact pins and a committed lockfile, installed with `--ignore-scripts`, committed, and re-verified by CI (`web-chat.yml`). |
| C35 attacker-controlled files | Realpath containment under the configured roots. Symlinks are rejected. Per-file size caps, JSONL tail caps, and line/file/dir count caps (`LIMITS`) apply. Every parse is wrapped in try/catch, and failures become warnings. Ids are validated (`^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$`; trace ids are 32 hex). The UI uses `textContent` only (no `innerHTML`, no inline handlers). Strict CSP: `default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors *`, plus `nosniff`, `no-referrer`, `no-store` and COOP `same-origin`. |
| C36 staleness | `updated` age > 2 × heartbeat (`CI_DASHBOARD_HEARTBEAT_SEC`, default 30 s) marks a run stale. An ASSERT heartbeat older than 300 s is flagged. A `pid` is read when present but liveness is not probed (stale-by-age only). |
| C37 fs.watch unreliability | Only known dirs are watched. 250 ms debounce, a 5 s poll fallback, and the manual `refresh` action. |
| C38 Node only in host | Zero runtime dependencies, and the canvas itself never writes into the repo. POSTs only change in-memory UI state, trigger a rescan, return the Aspire login URL, or talk to the experiment-chat backend. That backend is a separate process. It writes drafts and, after human approval, launches campaigns (see [Experiment chat](#experiment-chat)). |

## Assumed on-disk layouts

All of this lives in `lib/sources.mjs` (`LAYOUT`, `LIMITS`, and the readers). The readers are
tolerant: they accept snake_case or camelCase, and missing files mean empty data. Adjust
`sources.mjs` only if a producer's layout differs.

- **Ledger** (M7/M8): `experiments/` in the repo, plus `<runRoot>/_fake/experiments` for fake runs.
  - `campaigns/<cid>/` holds `campaign.json`, `frontier.json`, `calibration.json`,
    `history.jsonl`, `confirm.json`, `land.json` and `stack.json`.
  - `calibration/` and `confirm/` are envelope dirs.
  - `rounds/<eid>/` holds `envelope.json` or `experiment.json`, plus `decisions.json`.
  - Envelopes are OES with `extensions["com.microsoft.ci.rrsi"]` and
    `extensions["com.microsoft.ci.sleep"]`.
  - `holdout-looks.jsonl` counts holdout looks.
- **Sleep** (M9): `experiments/sleep/` holds `state.json`, `envelopes/*.json`,
  `nights/<date>/experiment.json`, `tasks.jsonl` (reviewed) and `tasks.pending.jsonl`.
- **Run roots:** `$CI_RUN_DIR`, `artifacts/ci-runs` and `artifacts/runs`.
  - Each `<run>/<experiment_id>/` is one live run.
  - Status comes from `status.d/<writer>.json` (obs v2.3.1, one file per writer) plus legacy
    `status.json`. They are aggregated like `obs.read_status`: docs are sorted by `updated`, later
    writers win on top-level fields, `arms` are merged per arm, and `writers` and
    `trace:{trace_id,span_id}` are kept.
  - Other run files: `begin.json`, `selection.json`, `round.done`, `arms.json`, `STOP`,
    `last_incumbent_eval.json` and `<arm>/arm.done`.
- **Spans:** `<runRoot>/telemetry/spans*.jsonl`, `<run>/telemetry/` and `artifacts/telemetry/`.
  - Each line is either a versioned span record (`ci_lab.telemetry.record`, `schemaVersion: 1`;
    records with a missing or unknown version are counted as rejected, per §12.5 D7) or an
    OTLP-JSON batch (`resourceSpans`).
  - Aspire `/api/telemetry/traces` responses (`{data:{resourceSpans}}`) are flattened by the
    same code.
- **Imported CI runs** (M12 D5): `$CI_IMPORTS_DIR` or `~/.ci-lab/imports/<run_id>/`.
  - `manifest.json` provides `run_id`, `artifact`, `digest`, `schemaVersion`, `files`, `spans`
    and `traces`.
  - The listed `*.jsonl` files (or all `*.jsonl` up to one level deep) are merged into Traces.
- **AGL journal** (M5): `{agl,journal,rollouts}/ro-<hex>.jsonl` under run roots, run dirs and arm
  dirs.
- **ASSERT** (M3): `artifacts/results/<suite>/<run>/` holds `manifest.json`, `metrics.json` and
  `scores.jsonl` (one row per case with dimension values). `<suite>/suite.json` is optional.
- **Aspire state** (M12 `ci-lab dashboard up`): `$CI_DASHBOARD_STATE` or
  `~/.ci-lab/dashboard.json`.
  - Fields: `{pid, version, started, ui_url, api_url, otlp_url, api_key, browser_token}`.
  - Only loopback URLs are used.
- **Agent bus** (`src/ci_lab/bus/`): `<bus>/<run_id>/<task>.wal.jsonl`, one hash-linked entry per
  line; `_run` is the run topic. See [Bus tab](#bus-tab).

### Bus tab

The Bus tab lists every agent-bus run found under `artifacts/` and the run roots
(`ci-lab graph run --run-dir DIR` writes `DIR/bus/`; campaigns write `<run root>/bus/`).

- **Discovery.** A bounded breadth-first search (depth 3, at most 400 dirs, 16 roots, 50 runs,
  64 topics per run, a 2 MiB tail per WAL, 500 rows per topic) for dirs named `bus`. It never
  enters `sealed`, `vault`, `challenger`, `artifacts`, `students` or `telemetry` dirs. Symlinks,
  junctions and dot dirs are skipped, and paths are realpath-contained like every other reader.
- **Runs and topics.** Per topic: entry count, state (`open`, `committed`, `rejected`,
  `aborted`), exploits, rubric patches, and integrity (`ok`, `torn tail`, `corrupt`,
  `tail only`, hidden kinds). The totals feed `get_summary` (`bus`).
- **Entries.** Seq, kind, role, author name, ref, a `key=value` summary and the artifact ref
  (sha256 and bytes). `exploit` rows (red) and `rubric_patch` rows (amber) are highlighted.
- **Orchestrator-only guarantee.** `lib/bus.mjs` projects every entry through a per-kind
  allowlist of orchestrator-visible fields (`types.visibility`) before it leaves the reader.
  Free text is reduced to counts or booleans: vote reasons, verdict criteria and corrections,
  proposal summaries, note data, and intent/outcome detail. Unknown kinds are counted as hidden
  and dropped. Artifact contents, `sealed/` rubric files and the hardener vault are never read,
  so no rubric text can reach the API or the UI.
- **Integrity.** The chain is checked the way `wal._parse` checks it: dense `seq` from 0,
  `prev` equals the previous `hash`, and the topic matches. A torn final line is tolerated.
  Any other damage stops at the verified prefix, marks the topic `corrupt` and adds a warning.
  Hashes are linked but not recomputed, because JS and Python canonical JSON differ for floats.
  Use `ci-lab bus verify` for full verification.

### Aspire notes (spike against Aspire 13.6.1)

- `?limit=` is ignored.
- Integer attributes arrive as `stringValue`.
- An unknown trace returns 404 with body `{}`.
- A missing or wrong `x-api-key` returns 401.
- `/traces/detail/<id>` without a cookie redirects (302) to `/login`.
- The login URL `/login?t=<browser_token>&returnUrl=…` is assumed.

## Experiment chat

The **Chat** tab is a CopilotKit chat with the `experiment_designer` agent served by
`ci-lab chat serve` ([chat.md](chat.md)). In it you formulate an RRSI campaign, review it as an OES
experiment and launch it. A launch always waits for you to click **Approve**, and the Python server
enforces that.

```
Chat tab (iframe, CSP connect-src 'self')
  ui/chat/chat.js  React 19 + CopilotKit 1.76.0 v2 UI + @ag-ui/client HttpAgent
  │  POST /agui   x-ci-token (canvas token from <meta>), same-origin, JSON ≤ 1 MiB
  ▼
canvas instance server (lib/server.mjs)    Host/Origin/Sec-Fetch-Site/token checks as for every POST
  │  strips Origin, Cookie, x-ci-token; adds x-ci-chat-token; streams SSE back unbuffered
  ▼
lib/chat.mjs  one backend per repo root, ref-counted by open instances
  │  spawn (shell:false, cwd = repo root, stdin = pipe, env CI_CHAT_TOKEN + CI_RUN_DIR)
  ▼
uv run --no-sync ci-lab chat serve --profile <p> --run-dir <run root>   127.0.0.1:<port>/agui
  ▼ launch_campaign (after approval)
campaign runs write into <run root> → the hub picks them up → Live / Experiment views
```

- **Backend lifecycle.**
  - The backend starts lazily: when the Chat tab first loads (`POST /api/chat/start`) or on the
    first `/agui` call. Opening the canvas only takes a reference.
  - `lib/chat.mjs` reads exactly one JSON `listening` line from stdout (120 s timeout). Stderr is
    forwarded to the extension log, rate-limited, with token-shaped strings masked.
  - A crash restarts the backend with exponential backoff (1 s → 30 s, at most 5 restarts until it
    has stayed up for 60 s). Then the state is `crashed` until the next request.
  - The process is stopped when the last canvas instance closes and when the extension exits. It
    also exits on its own when its stdin pipe closes, so a killed extension host leaves no orphan.
  - `--run-dir` is the run root the hub watches first (`$CI_RUN_DIR`, else `artifacts/ci-runs`), so
    launched campaigns appear live in the dashboard.
  - On Windows, `uv` is resolved to `uv.exe` on `PATH` and spawned without a shell. If it is
    missing, `/agui` and `/api/chat/start` return **503** `{error:"chat_unavailable"}` with a
    `Retry-After`, and the tab shows the reason.
- **UI.**
  - The chat panel has suggestion chips ("Draft a 1-round, 2-arm campaign", "Explain A/A
    calibration", "What strategies exist?").
  - A **draft card** is rendered from the AG-UI shared state `draft`. It shows cid, target, rounds,
    estimated evaluations and the formula, hyperparameters, warnings and rationale.
  - An **approval card** appears for the launch interrupt.
  - Launches are listed with an **Open in Experiment view** button (`POST /api/ui`).
  - Colors come from the canvas tokens (`--background-color-default`, `--text-color-default`,
    `--font-sans`, …). The light/dark mode follows `data-color-mode` or the OS preference.
- **Interrupts.** agent_framework_ag_ui 1.4 does the following:
  - It ends the launch run with `RUN_FINISHED.outcome = {type:"interrupt", interrupts:[{id, reason:"tool_call", toolCallId, …}]}`,
    after a `CUSTOM function_approval_request` event.
  - CopilotKit's `useInterrupt` renders the approval card. **Approve** or **Reject** calls
    `resolve({approved}, interrupt.id)`, which starts a new run on the same thread. That run carries
    the canonical `resume: [{interruptId, status:"resolved", payload:{approved}}]` plus the last
    messages/state snapshots.
  - An approved resume streams `TOOL_CALL_RESULT` first. A rejected resume launches nothing.
  - The recorded exchange is `test/fixtures/chat/approval_flow.json`, a copy of layer 32's
    `tests/fixtures/chat/approval_flow.json`. To update it, copy the new recording over it; the
    proxy tests replay every step.

### Security model

- **One origin, one token.**
  - The iframe only talks to its own canvas server: CSP `connect-src 'self'` is unchanged.
  - `/agui` gets the same guards as every POST: 421 on a foreign `Host`; 403 on a foreign `Origin` or
    `Sec-Fetch-Site`; 401 without the per-instance `x-ci-token`; 415 unless JSON. It also has its own
    body cap (1 MiB, 413). Other routes keep the 16 KB limit.
  - A page in another origin therefore cannot drive the chat through the proxy.
- **The backend secret never reaches the browser.** `lib/chat.mjs` generates `CI_CHAT_TOKEN` per
  process (24 random bytes, base64url).
  - The proxy adds it as `x-ci-chat-token` and strips `Origin`, `Cookie` and `x-ci-token` before
    forwarding.
  - It is not included in `chat_status`, `/api/chat/status`, errors or logs, and tests assert this.
  - The iframe cannot reach the Python port directly: CSP blocks it, and the backend answers 403 to
    any request with an `Origin` header (including `/healthz`) and 401 without its token.
- **Resource limits.**
  - Request bodies must arrive within 30 s.
  - The server has `headersTimeout` 10 s and `requestTimeout` 60 s (slowloris).
  - At most 4 concurrent chat runs per instance (429).
  - Client aborts are propagated to the upstream request.
- **CSP stays strict.** There is no `'unsafe-inline'`, no `'unsafe-eval'` and no external origin.
  - The bundle is plain same-origin ES modules, and its stylesheet is `ui/chat/chat.css`.
  - Modules that would inject `<style>` tags, use `eval` or fetch remote assets are replaced by
    local stubs at build time (`web/experiment-chat/stubs/`): the markdown renderer, the web
    inspector, A2UI/MCP-apps renderers and protobuf.
  - zod's JIT probe is patched so it never calls `new Function`.
  - `web/experiment-chat/scripts/csp-check.mjs` proves zero CSP violations in headless Microsoft
    Edge. It is not run in CI.
- **No telemetry.** CopilotKit is used without a runtime URL or license key, and with
  `enableInspector={false}`. The bundle contains no cloud endpoint (the build fails if one appears),
  and CSP blocks anything else. The CSP check records zero non-loopback requests.
- **Approval gate.** It is enforced server-side (`approval_mode="always_require"`, single-use,
  thread-bound interrupt ids; see [chat.md](chat.md#security-model)). The UI only relays your
  decision.

### Rebuilding the bundle

```powershell
cd web\experiment-chat
npm ci --ignore-scripts        # exact pins + committed package-lock.json; no install scripts
node build.mjs                 # esbuild → .github/extensions/ci-harness-dashboard/ui/chat/
node --test test               # source tests (draft/launch state from the fixture)
node scripts/csp-check.mjs     # optional: headless Edge, fake backend replaying the fixture
node scripts/csp-check.mjs --command '["uv","run","--no-sync","ci-lab","chat","serve","--profile","fake","--dry-run-launch"]'
```

- `build.mjs` produces minified, tree-shaken ESM with code splitting. It also writes
  `THIRD_PARTY_LICENSES.txt` and the `*.LEGAL.txt` files.
- The build fails if any file is over 1 MiB, or if the output contains a banned vendor name,
  `eval`, a `Function` constructor, `console.log`, or a telemetry host.
- `test/chat-bundle.test.mjs` re-checks the committed files and the gist limits (< 5 MB extension
  total, no `package.json`, no `dist`/`build` folder).
- The build is deterministic. `.github/workflows/web-chat.yml` rebuilds it from the lockfile and
  fails if the committed output differs.
- When you install packages from an internal mirror, rewrite the lockfile's `resolved` URLs back to
  `https://registry.npmjs.org/` before committing. The integrity hashes are identical.

### Licensing

- CopilotKit (`@copilotkit/react-core`), AG-UI (`@ag-ui/client`), React and the other bundled
  packages are MIT or similarly permissive. See `ui/chat/THIRD_PARTY_LICENSES.txt`.
- The agent is wired with CopilotKit's `agents__unsafe_dev_only` prop: an `HttpAgent` passed
  directly, with no CopilotKit runtime. The prop is free, but CopilotKit labels it development-only.
  That fits this canvas, which is local, single-user and loopback.
- CopilotKit's production equivalent, `selfManagedAgents`, requires a CopilotKit Enterprise license.
- The other supported path is a CopilotKit runtime server. Its package pulls in a third-party agent
  framework adapter that this repo does not allow, so it is not used.

### Limitations

- **Pending approvals are in memory.** If the backend restarts, the approval card is stale. Start a
  new message instead ([chat.md](chat.md#limitations-and-contract-deviations)).
- **Buffered text.** With `--profile copilot`, assistant text arrives once per model turn, not per
  token.
- **One backend per repo root.** Every canvas instance on the same root shares the backend and its
  environment: the first instance's `CI_CHAT_*` settings win.
- **Single user and loopback only.** There is no multi-user auth beyond the per-instance token.
- **No chat history across reloads.** Each page load starts a new AG-UI thread.
- **Bundle edits need a rebuild.** Do not hand-edit `ui/chat/`; CI rejects drift.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CI_RUN_DIR` | — | Extra run root, scanned before `artifacts/ci-runs` and `artifacts/runs`. |
| `CI_IMPORTS_DIR` | `~/.ci-lab/imports` | Imported CI telemetry. |
| `CI_DASHBOARD_STATE` | `~/.ci-lab/dashboard.json` | Aspire state file. |
| `CI_DASHBOARD_HEARTBEAT_SEC` | `30` | Heartbeat used for staleness (stale at 2×). |
| `CI_CHAT_DISABLED` | — | `1` disables the experiment chat (no process is ever spawned). |
| `CI_CHAT_PROFILE` | `copilot` | `copilot` or `fake`: the `--profile` of the default chat command. |
| `CI_CHAT_COMMAND` | — | JSON argv array that replaces `uv run --no-sync ci-lab chat serve …` entirely (no `--run-dir` is appended). |
| `CI_CHAT_E2E` | — | `1` enables `test/chat-e2e.test.mjs`, which runs the real backend with `--profile fake --dry-run-launch`. |
| `CI_CHAT_E2E_REPO` | repo root | Checkout to run the e2e backend from. It must contain `ci_lab.chat` and a synced `.venv`. |

## Dev preview and tests

```powershell
cd .github\extensions\ci-harness-dashboard
node dev-preview.mjs --repo C:\path\to\repo [--port 47613] [--view live]   # prints the URL; Ctrl+C stops
node dev-preview.mjs --repo test\fixtures\repo                              # fixture data
node --test                                                                 # all suites (sources, model, security, aspire, hub, server, canvas, chat)
```

- Without `--repo`, dev-preview starts from the current directory and walks up to `.git`.
- Open the printed `http://127.0.0.1:<port>/` URL. Use `127.0.0.1` or `localhost` only; any
  other Host is rejected.
- In the Copilot app, the canvas appears as **CI Harness Dashboard** once the project extension is
  loaded. The agent can drive it with the actions above.
