# CI Harness Dashboard canvas

A GitHub Copilot app canvas (`ci-harness-dashboard`) that shows the self-improving harness live:
campaigns and their frontier, RRSI rounds and arms, OES experiments and decisions, ASSERT evals,
sleep nights, AGL rollouts, and OpenTelemetry traces from local JSONL files, imported CI runs and
Aspire. It implements design §12 (`docs/design/self-improving-harness.md`) and security controls
C29–C38.

Code: `.github/extensions/ci-harness-dashboard/`. It is plain Node ESM with no npm dependencies
(Node ≥ 20). Only `extension.mjs` imports `@github/copilot-sdk/extension`, which the host resolves.

## Architecture

```
extension.mjs     thin wiring: joinSession + createCanvas(open/onClose/actions from lib/canvas.mjs)
lib/canvas.mjs    SDK-independent controller: input validation, instance lifecycle, agent actions
lib/server.mjs    per-instance loopback HTTP server: static UI, JSON API, SSE, POST guards, CSP
lib/hub.mjs       one data hub per repo root, refcounted across instances: fs.watch + debounce + poll
lib/sources.mjs   ALL on-disk format knowledge: tolerant, contained, size-capped readers
lib/model.mjs     pure aggregation: campaigns, live runs, experiments, span trees, evals, sleep
lib/aspire.mjs    Aspire dashboard API client. It is the only module that sees secrets.
lib/security.mjs  containment, id validation, Host/Origin checks, C29 redaction, secret stripping
ui/               index.html, app.js, app.css: no inline script/style, DOM built via textContent only
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
{ "repoRoot": "<absolute path>", "view": "overview|live|experiment|traces|evals|sleep|aspire",
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
| `get_summary` | `{}` | Compact JSON of campaigns, live runs, last decisions, evals, sleep, traces and imports. Works without an open instance (uses the session working directory). |
| `dashboard_status` | `{}` | URL, viewer count, UI state, hub roots/watchers/warnings, and Aspire `configured/reachable/version/uiUrl/hint`. Contains no secrets. |

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
- **Aspire.** Reachability, version and recent traces.
  - "Open Aspire" goes through a user-click POST endpoint that returns the login URL. The iframe
    `window.open`s it and falls back to a copy box if the popup is blocked.
  - Per-trace deep links have the form `/traces/detail/<traceId>`.

Every view has an empty state that says how to produce its data, for example
"run `ci-lab dashboard up`" or "run `ci-lab telemetry pull --run <id>`".

## HTTP API (per instance)

`GET /` · `/app.js` · `/app.css` · `/events` (SSE) · `/api/summary` · `/api/experiments[?campaign=]`
· `/api/experiment/:id` · `/api/live` · `/api/traces[?aspire=0]` · `/api/trace/:id` · `/api/evals`
· `/api/sleep` · `/api/aspire`.

POST routes: `/api/ui` · `/api/refresh` · `/api/aspire/login`.

## Security mapping (§12.2)

| Control | Implementation |
|---|---|
| C29 GenAI content | `security.isSensitiveAttr` plus the `sensitiveEnabled` flags. Values are redacted in the model before they reach the API. |
| C30 secrets | `aspire.mjs` keeps `api_key`, `browser_token` and `otlp_key` internal. The `x-api-key` header is used only server-side. `stripSecrets` is applied to action results and status. The login URL is returned only from a token-guarded, user-initiated POST. Tests assert that no secret or instance token appears in any response, action result or log. |
| C31 DNS rebinding / cross-origin | The server binds `127.0.0.1` only. A `Host` other than `127.0.0.1:<port>` or `localhost:<port>` gets 421. POSTs require the `x-ci-token` header (a per-instance random value injected into `index.html` via `<meta>`), JSON content type, an allowed `Origin` and `Sec-Fetch-Site`, and a body ≤ 16 KB. Other methods get 405. Aspire URLs must be loopback. |
| C32 no Aspire iframe/proxy | No reverse proxy. Native views are rendered from OTLP-JSON (the same flattening serves the Aspire API and JSONL). Full Aspire opens top-level on a click. |
| C33 Aspire optional | Every view works without Aspire. The Aspire view and the merged spans degrade to a hint. |
| C34 supply chain | Not applicable to the canvas: it has zero npm dependencies and downloads nothing. |
| C35 attacker-controlled files | Realpath containment under the configured roots. Symlinks are rejected. Per-file size caps, JSONL tail caps, and line/file/dir count caps (`LIMITS`) apply. Every parse is wrapped in try/catch, and failures become warnings. Ids are validated (`^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$`; trace ids are 32 hex). The UI uses `textContent` only (no `innerHTML`, no inline handlers). Strict CSP: `default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors *`, plus `nosniff`, `no-referrer`, `no-store` and COOP `same-origin`. |
| C36 staleness | `updated` age > 2 × heartbeat (`CI_DASHBOARD_HEARTBEAT_SEC`, default 30 s) marks a run stale. An ASSERT heartbeat older than 300 s is flagged. A `pid` is read when present but liveness is not probed (stale-by-age only). |
| C37 fs.watch unreliability | Only known dirs are watched. 250 ms debounce, a 5 s poll fallback, and the manual `refresh` action. |
| C38 Node only in host | Zero dependencies and read-only: the canvas never writes into the repo. POSTs only change in-memory UI state, trigger a rescan, or return the Aspire login URL. |

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

### Aspire notes (spike against Aspire 13.6.1)

- `?limit=` is ignored.
- Integer attributes arrive as `stringValue`.
- An unknown trace returns 404 with body `{}`.
- A missing or wrong `x-api-key` returns 401.
- `/traces/detail/<id>` without a cookie redirects (302) to `/login`.
- The login URL `/login?t=<browser_token>&returnUrl=…` is assumed.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CI_RUN_DIR` | — | Extra run root, scanned before `artifacts/ci-runs` and `artifacts/runs`. |
| `CI_IMPORTS_DIR` | `~/.ci-lab/imports` | Imported CI telemetry. |
| `CI_DASHBOARD_STATE` | `~/.ci-lab/dashboard.json` | Aspire state file. |
| `CI_DASHBOARD_HEARTBEAT_SEC` | `30` | Heartbeat used for staleness (stale at 2×). |

## Dev preview and tests

```powershell
cd .github\extensions\ci-harness-dashboard
node dev-preview.mjs --repo C:\path\to\repo [--port 47613] [--view live]   # prints the URL; Ctrl+C stops
node dev-preview.mjs --repo test\fixtures\repo                              # fixture data
node --test                                                                 # all suites (sources, model, security, aspire, hub, server, canvas)
```

- Without `--repo`, dev-preview starts from the current directory and walks up to `.git`.
- Open the printed `http://127.0.0.1:<port>/` URL. Use `127.0.0.1` or `localhost` only; any
  other Host is rejected.
- In the Copilot app, the canvas appears as **CI Harness Dashboard** once the project extension is
  loaded. The agent can drive it with the actions above.
