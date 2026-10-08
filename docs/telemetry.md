# Telemetry (`ci_lab.telemetry`)

Implements design §12 (C27–C38) and §12.5 D4/D5/D7. Harness code emits spans through the
`ci_lab.obs` facade (OTel API only). MAF emits GenAI spans. `ci_lab.telemetry` is the single
owner of the TracerProvider and decides where spans go:

| Sink | When | Where |
|---|---|---|
| JSONL span records | `run_dir` given (default `jsonl=True`) | `<run_dir>/telemetry/spans-<pid>.jsonl` |
| OTLP/HTTP (protobuf) | a local Aspire dashboard is running (`aspire="auto"`), or always (`"on"`) | `ci-lab dashboard up` |

The dashboard is optional. Nothing breaks without it, and the JSONL files are the durable record.

## `setup()`: call it first at every entry point

```python
from ci_lab import obs, telemetry

telemetry.setup("night", profile="fake", run_dir=run_dir, campaign_id=cid)  # aspire="auto"
with obs.span(SPAN_NIGHT):
    ...
telemetry.shutdown()  # flush; also registered atexit
```

```text
setup(component, *, profile=None, run_dir=None, aspire="auto"|"on"|"off",
      jsonl=True, sensitive=False, campaign_id=None) -> TelemetryHandle
```

- **Resource:** `service.name=ci-lab.<component>`, `service.instance.id`, `process.pid`,
  `ci.profile`, `ci.campaign_id`, `vcs.ref` (git HEAD, best-effort), and
  `ci.telemetry.sensitive` (whether GenAI content capture was on).
- **Provider install:** done through MAF `configure_otel_providers(exporters=[...],
  enable_sensitive_data=sensitive)` plus `enable_instrumentation()`.
- **Idempotent (D4):** the first call wins and later calls return the same handle. A later
  call that brings a `run_dir`, when no JSONL sink exists yet, attaches one.
- **Fails fast (D4)** with `RuntimeError` if a *non-SDK* global provider is already
  installed, because spans would otherwise vanish silently.
- **Existing SDK provider (C28),** e.g. one installed first by a library: `setup` attaches its
  exporters to that provider instead of replacing it (`handle.owns_provider is False`).
- **`aspire` modes:**
  - `"auto"` reads the dashboard state file and skips silently if the dashboard is absent or dead.
  - `"on"` raises if no dashboard is running.
  - `"off"` never exports to the dashboard.
- **`sensitive=False` (default, C29):** MAF content capture is off. In addition, every
  exporter strips GenAI content defensively, because instrumentation settings can drift:
  - `gen_ai.input.messages` / `output.messages` / `system_instructions` / `tool.call.*`
  - `gen_ai.prompt*` / `gen_ai.completion*` / `gen_ai.*.content|messages`
  - events named `gen_ai.*`

  Use `sensitive=True` only for fake/local profiles.
- **Exporter safety:** exporters never raise into the application.

### Attaching to another provider (Agent Lightning, C28)

If a component must run under a provider it does not own, add our sinks to it:

```python
h = telemetry.setup("agl", run_dir=run_dir)
for p in h.span_processors():      # fresh BatchSpanProcessors over our exporters
    agl_provider.add_span_processor(p)
```

If such a provider is already the global one when `setup()` runs, this happens automatically.
The installed `agentlightning` 1.0.2 has no `tracer` module, so the attachment is generic
rather than AGL-specific.

## Span record v1 (D7)

One JSON object per line. The shape is the same for JSONL, the Aspire `/api/telemetry/*`
responses and OTLP once adapted:

```json
{"schemaVersion":1,"traceId":"<32 hex>","spanId":"<16 hex>","parentSpanId":"<16 hex or ''>",
 "name":"ci.arm","kind":1,"startTimeUnixNano":"<str>","endTimeUnixNano":"<str>",
 "status":{"code":0,"message":""},"attributes":{"flat.key":"scalar or list"},
 "events":[{"name":"…","timeUnixNano":"…","attributes":{}}],
 "links":[{"traceId":"…","spanId":"…","attributes":{}}],
 "resource":{"service.name":"ci-lab.night",…},"scope":{"name":"ci_lab","version":""}}
```

- **Field formats:**
  - `kind` and `status.code` use OTLP numbering (1 = INTERNAL, …; 0 unset / 1 ok / 2 error).
  - Times are decimal strings, because JS numbers lose nanosecond precision.
- **Adapters** (`ci_lab.telemetry.record`): `from_readable_span`, `from_otlp_json` (Aspire API
  or OTLP-JSON), `to_otlp_request` / `from_otlp_request` (protobuf), `validate`.
- **Version check:** a missing or unknown `schemaVersion` raises `SchemaError`.
- **Reading files:** `read_jsonl()` skips torn lines (counted) but rejects wrong versions.
- **Golden fixtures:** `tests/ci_lab/telemetry/fixtures/`.
- **File rotation:** JSONL files rotate at 50 MiB to `spans-<pid>.<n>.jsonl`, keeping 20 backups.

## CLI

`cli.register(subparsers)` adds two command groups. Until `"telemetry"` is listed in
`ci_lab.cli.COMMAND_MODULES`, use `python -m ci_lab.telemetry …`.

```text
ci-lab dashboard up [--trust-new] [--version V] [--rid RID] [--grpc] [--wait S]
ci-lab dashboard down | status | url [--with-token] | open
ci-lab telemetry import <spans-*.jsonl | dir>... [--otlp-url U --otlp-key K] [--batch N] [--keep-sensitive]
ci-lab telemetry pull --run <id> [--repo o/r] [--artifact spans] [--allow-no-digest] [--force] [--no-import]
ci-lab telemetry imports
```

### Dashboard (`ci_lab.telemetry.aspire`)

- **Package:** `Aspire.Dashboard.Sdk.<rid>` (default 13.6.1). It is downloaded from
  `$CI_NUGET_FLAT` (default: internal feed), falling back to nuget.org.
- **Verification:** the download's sha256 is checked against `src/ci_lab/telemetry/aspire.lock.json`.
  - If the RID/version has no pin, `--trust-new` is required. It records the pin (TOFU)
    in that lock file, or in `<install root>/aspire.lock.json` if the package dir is read-only.
- **Install:** zip-slip-safe extraction to `%LOCALAPPDATA%\ci-lab\aspire-dashboard\<ver>\pkg\`
  (`$XDG_DATA_HOME`/`~/.local/share` on Linux, `~/Library/Application Support` on macOS).
  Overrides: `CI_ASPIRE_HOME`, `CI_ASPIRE_RID`.
- **Launch:** `up` starts the dashboard detached on free 127.0.0.1 ports.
  - The UI and API share one port; OTLP/HTTP has its own. gRPC is off unless `--grpc`.
  - Auth: browser-token UI auth and API-key auth for both OTLP (`x-otlp-api-key`) and the
    telemetry API (`x-api-key`). Each secret is generated with `secrets.token_urlsafe(32)`.
  - `AllowedHosts=127.0.0.1;localhost` is set, and the token is suppressed in output.
  - Inherited `OTEL_*`/`Dashboard__*`/`ASPIRE_*`/`ASPNETCORE_*` variables are dropped.
  - Readiness is checked by polling `GET /api/telemetry/resources`.
- **State file:** `$CI_DASHBOARD_STATE` (default `~/.ci-lab/dashboard.json`).
  - Permissions are owner-only: `icacls /inheritance:r /grant:r %USERNAME%:F` on Windows,
    `chmod 600` on POSIX. They are applied before any secret is written.
  - The log is `dashboard.log` next to the state file.
- **`down`** terminates the recorded pid only if its image path equals the recorded
  executable. Otherwise it removes the state and leaves the process alone.
- **No secret output:** `status`, `url`, `up` and `telemetry.current()` never print secrets.
  Only `url --with-token` prints the login URL, and `open` hands it straight to the browser.
- **Querying from code:** `aspire.query("traces" | "traces/<id>" | "spans" | "logs" | "resources")`
  returns the OTLP-JSON (`{"data": {"resourceSpans": [...]}}`).

### Import and pull (D5)

- **`telemetry import`** replays JSONL span records into the running dashboard (or
  `--otlp-url`) as OTLP/HTTP protobuf.
  - Trace/span ids, parents, links and timestamps are preserved.
  - GenAI content is redacted unless `--keep-sensitive` is given.
  - Proxies are bypassed for loopback.
- **`telemetry pull --run <id>`** fetches the run's `spans` artifact via `gh` and imports it:
  1. Lists the run's artifacts with `gh api` and downloads the zip.
  2. Verifies the zip against the artifact's `sha256:` digest. A missing digest needs
     `--allow-no-digest`.
  3. Extracts safely and validates every line's `schemaVersion`.
  4. Caches to `~/.ci-lab/imports/<run_id>/` (`$CI_IMPORTS_DIR`) with a `manifest.json` (run,
     repo, artifact id, digest, sha256, files, span/trace counts, services, timestamps).
  5. Imports into the dashboard if it is running.

  A cached run is reused unless `--force` is given. `telemetry imports` (`pull.list_imports()`)
  lists the manifests for the canvas.

## Tests

- **Offline:** `tests/ci_lab/telemetry/` mocks HTTP, downloads and processes. `setup()` tests
  run in subprocesses because the OTel global provider can be set only once per process.
- **Live:** `pytest -m live tests/ci_lab/telemetry/test_tel_live.py` runs the end-to-end test
  against the real dashboard: `up` → `setup()` + spans → API query → import replay → `down`.
