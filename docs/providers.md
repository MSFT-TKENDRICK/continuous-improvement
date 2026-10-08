# Providers (`ci_lab.providers`)

Module M2 contains three pieces:

- **`CopilotChatClient`** — a Microsoft Agent Framework (MAF) chat client backed by the GitHub Copilot SDK (`github-copilot-sdk`).
- **`make_chat_client`** — a factory that picks a client by profile.
- **`copilot-serve`** — a loopback, OpenAI-compatible server for clients that only speak OpenAI (LiteLLM, ASSERT's judge).

| File | Purpose |
|---|---|
| `copilot.py` | `CopilotChatClient`, the `session_scope` contextvar, `copilot_scope()` |
| `factory.py` | `make_chat_client(profile=..., model=..., purpose=..., rollout=None, **kw)` |
| `serve.py`, `cli.py` | `ci-lab copilot-serve` |
| `fake_sdk.py` | an in-process fake of `CopilotClient`/`CopilotSession` for offline tests |

## CopilotChatClient

```python
from agent_framework import Agent
from ci_lab.providers.copilot import CopilotChatClient, copilot_scope

async with CopilotChatClient(model="gpt-5-mini", on_model_request=print) as client:
    agent = Agent(client=client, instructions="...", tools=[my_tool])
    with copilot_scope("rollout-42"):
        result = await agent.run("...")
```

### Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `model` | required | The Copilot model to request. |
| `reasoning_effort` | `None` | Passed to `create_session` when set. |
| `base_directory` | `None` | SDK state directory. When `None`, a temporary directory is created per client. |
| `timeout_s` | `300` | Deadline for each `get_response` call. |
| `session_ttl_s` | `900` | How long an idle session lives before it is aborted. |
| `sdk_client` | `None` | An injected SDK client (used by tests). The provider starts it but never stops it. |
| `on_model_request` | `None` | A callback that receives one dict per Copilot assistant turn. |
| `session_scope` | `None` | A callable that returns the scope id. |

**Declarative agents.** The client can be built from declarative YAML with `AgentFactory(additional_mappings={"GitHubCopilot": PROVIDER_MAPPING})`; the loader passes `model=<id>`.

**Authentication.** Only ambient GitHub authentication is used: the client runs `CopilotClient(mode="empty", base_directory=...)`. Tokens are never accepted as arguments and never logged.

**Built-in tools.** Copilot's built-in tools are never exposed. Every session is created with:

- `available_tools=ToolSet().add_custom("*")`
- `system_message={"mode": "replace", ...}`, where the content is the MAF instructions plus any system messages
- a permission handler that approves everything (only MAF tools exist, so nothing else can be approved)

### Suspended tool bridge (primary path)

1. MAF `FunctionTool`s become Copilot `Tool`s. Each handler parks on a future keyed by the Copilot `tool_call_id`.
2. When every request in an `assistant.message` batch of `tool_requests` is parked, `_inner_get_response` returns a `ChatResponse` containing one `function_call` content per request. The `call_id` is the Copilot tool call id.
3. MAF's `FunctionInvocationLayer` runs the tools, so middleware, approvals and telemetry all apply. It then calls back with the `function_result` contents.
4. The bridge resolves the matching futures. Exceptions become failure `ToolResult`s whose text is the error. The bridge then waits for the next batch or for `session.idle`, which carries the final text.
5. A `tool_choice="none"` turn (MAF's final turn) fails any new tool request with a "tools disabled" result.

### Session isolation (design C4)

**Session key.** Each session is keyed by `(scope, model, conversation fingerprint)`.

- **Scope.** Taken from the `session_scope` callable. Failing that, from the `ci_lab.providers.copilot.session_scope` contextvar, which `copilot_scope(id)` sets. Failing both, it is `"default"`.
- **Fingerprint.** Either the conversation id, or a hash of the instructions plus the first non-system message.

**Validating results.** When function results arrive, they are matched through a call-id registry. The checks are:

- the session key matches
- the session is in the `waiting_tools` state
- the message-prefix hash matches the history the bridge last returned
- the result ids are a subset of the outstanding batch

**Locking.** Each session has its own asyncio lock.

**Failure handling.** A mismatch, timeout, cancellation or session error kills the session:

1. Every pending future is resolved with a failure result.
2. `session.abort()` is called.
3. `disconnect()` is called.

The next call then falls back to replay.

**Idle sessions.** Idle tool sessions can be reused for a follow-up user turn when the system text, tool schemas and prefix all match. `reap()` runs opportunistically on each call and aborts sessions idle for longer than `session_ttl_s`.

### Fallback: transcript replay

A fresh session is sent the full MAF history as a transcript:

- system text goes into the replaced system message
- tool calls are rendered as `[assistant tool call <id>] name(args)`
- tool results are rendered as `[tool result <id>]`

Replay is used after a crash, a resume, a cache miss or a validation failure. A tool-less call with a single user message is sent as plain text. Tool-less sessions are one-shot: they are disconnected after the answer.

`ChatResponse.additional_properties["copilot_path"]` records which path ran: `bridge`, `followup`, `replay` or `single`.

### Usage, served model and callback

**Usage.** `assistant.usage` events are summed into `UsageDetails`:

- input and output tokens
- cache read and cache write tokens
- reasoning tokens
- `copilot_total_nano_aiu`

**Served model.** `ChatResponse.model` is the model the usage events report as served. The requested model is stored in `additional_properties["requested_model"]`.

**Callback.** `on_model_request` receives one dict per turn, shaped like AGL's `model_request` event:

```
{model, requested_model, server: "copilot", scope, session_id, api_call_id,
 request: {path, message_count, tool_count, tool_results, structured, turn},
 response: {text, tool_calls: [{name, call_id}]}, latency_ms,
 usage: {input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
         reasoning_tokens, total_nano_aiu, cost},
 finish_reason, status: ok|error|timeout}
```

If the callback raises, the exception is logged and swallowed.

### Structured output and ignored options

**Structured output.** `options["response_format"]` may be a pydantic model, an OpenAI `json_schema` or `json_object` format, or a raw JSON schema.

SDK 1.0.11 has **no** `response_schema` or typed send, so the provider does this instead:

1. Adds the schema to the system message.
2. Strips code fences from the reply.
3. Validates the reply with `jsonschema`.
4. If validation fails, makes one repair turn.
5. If the repair also fails, raises `ChatClientInvalidResponseException`.

**Ignored options.** The SDK has no `temperature`, `seed`, `top_p` or similar options. If they are passed, they are ignored and listed in `additional_properties["ignored_options"]`.

### Lifecycle

`close()` and `async with` abort all sessions and stop an owned `CopilotClient`.

A client binds to the event loop it first runs on. If it is later used from a new loop (for example, successive `asyncio.run` calls), it drops its old sessions and restarts its owned SDK client.

## make_chat_client

| Profile | Client |
|---|---|
| `COPILOT` | `CopilotChatClient`. When a rollout is given, its scope is `rollout.rollout_id`. |
| `OFFLINE` | `agent_framework_openai.OpenAIChatCompletionClient`, configured as described below. |
| `FAKE` | `ci_lab.testing.FakeChatClient`. Pass `script=` to set its script. |

For `OFFLINE`:

- `base_url` comes from the `base_url=` argument (for example an AGL proxy). Otherwise it comes from `OPENAI_API_BASE` or `OPENAI_BASE_URL`; if none is set, a `ValueError` is raised.
- The API key comes from the `api_key=` argument, then `OPENAI_API_KEY`, then `"local"`.
- The headers `x-ci-purpose` and `x-ci-rollout-id` are sent, plus a per-request W3C `traceparent` from `obs.carrier()`. The trace header is added only when the factory builds the `async_client`; if you pass your own `async_client=`, the factory doesn't add it.

Any extra `**kw` arguments are passed to the client constructor.

## copilot-serve (design C7)

```
ci-lab copilot-serve --port 8765 --model gpt-5-mini --key-file .ci/copilot.key [--host 127.0.0.1]
```

```python
litellm.completion(model="openai/gpt-5-mini", api_base="http://127.0.0.1:8765/v1",
                   api_key=open(".ci/copilot.key").read(), messages=[...])
```

**Binding and authentication**

- Only loopback hosts are allowed: `127.0.0.0/8`, `::1` or `localhost`. Any other host is refused with exit code 2.
- A random bearer key is written to `--key-file` with mode 0600. The server prints only the key file's path.
- Every request must carry `Authorization: Bearer <key>`; otherwise it gets a 401.

**Endpoints.** `GET /v1/models` and non-streaming `POST /v1/chat/completions`.

- Responses use the OpenAI shape: `id`, `choices[{index, message{role, content}, finish_reason}]`, `usage`, and `model` set to the served model.
- `response_format` with `json_schema` or `json_object` maps to structured output.

**Errors**

| Status | When |
|---|---|
| 400 (OpenAI-style error) | The request uses `tools`, `functions`, `tool_choice` or `stream=true`; asks for `n>1`; sends roles other than system, developer, user or assistant; or sends non-text content |
| 404 | `model` differs from `--model` |
| 504 | Timeout |
| 502 | Invalid structured output or another provider error |

**Ignored parameters.** `temperature`, `seed`, `top_p`, `max_tokens` and similar parameters are accepted but ignored. They are listed in the `x-ci-ignored-params` response header.

**Scope.** Each request runs in its own `copilot_scope`, so requests never share sessions.

**Tracing.** copilot-serve is a long-lived service, so it never inherits a pinned `TRACEPARENT` (design §12.5). Each request is handled inside `obs.use_carrier(request.headers)`, so the request joins the trace of whoever sent its W3C `traceparent` header. Clients add that header per request with `obs.carrier()`; the OFFLINE factory client does this automatically through an httpx request hook. The Copilot CLI subprocess also gets `service_env()`, which is the current environment minus `TRACEPARENT`/`TRACESTATE`.

**Spawning.** `serve.spawn(model=..., key_file=...)` starts `ci-lab copilot-serve` as a child process:

- It picks a free loopback port unless one is given.
- It waits until `/v1/models` answers.
- It returns a `CopilotServeProcess` with `base_url`, `api_key`, `proc` and `log_file` (`<key_file>.log`). The object is a context manager and has `close()`.

**LiteLLM caveat.** LiteLLM itself rejects `temperature != 1` for `gpt-5*` model names before sending the request. Callers need `drop_params=True` or must omit `temperature`.

## Testing

**Offline tests.** `tests/ci_lab/providers` runs offline against `fake_sdk.FakeCopilotClient`. The fake emits the real `copilot.session_events` types (`assistant.usage`, `assistant.message` with `tool_requests`, `session.idle`, `session.error`) and runs tool handlers the way the SDK does.

**Live test.** The live round-trip test is marked `copilot` and is deselected by default:

```
uv run --native-tls pytest -m copilot tests/ci_lab/providers
```
