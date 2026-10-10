# MCP tools (`ci_lab.mcp`)

Agents reach external capabilities only through MCP servers. These servers are declared in a frozen
registry, narrowed by an evolvable overlay, and called through one governed hub.

## Two files

| File | Owner | Contents |
|---|---|---|
| `src/ci_lab/mcp/servers.yaml` (`ci_lab.mcp.v1`) | frozen | Per server: argv `command` (never a shell string; placeholders `{python}` and `{<root>}`), `roots`, `env_allow` and `tools_allow`. Global `code_mode` caps: `timeout_s_max`, `max_output_chars_max` and `allowed_imports`. |
| `harness/mcp/exposure.yaml` (`ci_lab.mcp.exposure.v1`) | evolvable | Per server: `mode: direct\|code` and `tools: {name: description\|null}`, which must be a subset of `tools_allow`. Code-mode `timeout_s`, `max_output_chars` and `imports` must stay within the caps. |

`load_registry()` and `load_exposure(path, registry)` raise `McpConfigError` on anything outside the frozen
declaration: an unknown server or tool, a cap exceeded, a non-allow-listed import, or an unknown key. If a
server is omitted from the exposure, it is not started. If `tools` is omitted, all of `tools_allow` is
exposed.

## `harness` server (read-only)

`python -m ci_lab.mcp.harness_server --root <harness dir> --runs-root <runs dir>` exposes these tools:
`list_components`, `read_component`, `component_metrics`, `trace_summary` and `eval_summary`.

- Every path resolves under `--root`, or under `--runs-root` for trace and eval summaries. Escapes and
  symlinks are rejected.
- The summaries return sanitized counts only, never raw transcripts.
- `component_metrics` uses `ci_lab.metrics.simplicity.surface_metrics` when that module is importable.
  Otherwise it falls back to file, line and character counts.

## `McpHub` and direct mode

```python
async with McpHub(registry, exposure, roots={"harness_root": cand_dir}, before_call=guard) as hub:
    tools = maf_tools(hub)                       # FunctionTool per exposed direct-mode tool
    data = await hub.call_tool("harness", "list_components", {})
```

- Each exposed server is started over stdio with only its `env_allow` variables. The MCP SDK also adds its
  own minimal safe defaults: PATH, SYSTEMROOT, and so on.
- `call_tool` rejects any tool that is not exposed.
- `before_call(server, tool, args)` is the governance hook (sync or async). `None`/`True` allows the
  call; `False`, a reason string or `Denial(reason)` denies it with `McpDenied`; hook exceptions propagate.
- In direct mode, `maf_tools(hub)` names each tool `<server>__<tool>`.

## Code mode

`CodeMode(hub, cfg=None, on_tool_call=None)` exposes a single tool, `run_code(code) -> str`, through
`.maf_tool()`. Its description is generated from typed stubs, for example
`tools.harness.list_components() -> dict`. A snippet can chain several tool calls and `print()` only what
it needs, which keeps large intermediate results out of the model context.

`on_tool_call("server.tool")` fires once for each call made inside a snippet. L7 connects it to
`RunMeter`. `.tool_calls` keeps a local count of the same calls.

### This is an isolation layer, not an OS sandbox

The layer is defence in depth against accidental or naive misuse by model-written code. It does not give
security against a determined attacker. Its layers are:

1. **AST allowlist** (`check_code`).
   - Only the exposure's `imports` and `tools` may be imported. `os`, `sys`, `subprocess`, `socket`,
     `ctypes` and `importlib` are always forbidden.
   - Forbidden: `_`-prefixed attributes, dunders, `getattr` with a non-literal name, `open`/`exec`/`eval`/
     `compile`/`globals`, `format`/`format_map`, and module-valued or frame-introspection attributes.
2. **Fresh interpreter**.
   - The snippet runs in a temporary directory via `sys.executable -I bootstrap.py`, with a minimal
     environment.
   - Builtins are restricted and `__import__` is guarded.
   - The stub directory is inserted into `sys.path` explicitly. The stubs are self-contained and do not
     import `ci_lab`.
3. **Framed RPC**.
   - The bootstrap dups the original fd0 and fd1 for the JSON-lines channel, which uses `{"rpc": ...}`
     envelopes.
   - `sys.stdout` and `sys.stderr` are rebound to capped buffers, so user prints cannot forge or corrupt
     RPC frames.
   - Tool calls go back through `McpHub.call_tool`, so exposure filtering and `before_call` still apply.
4. **Limits**. Output is truncated at `max_output_chars`. At `timeout_s`, the whole process tree is killed
   through a Win32 Job Object (pywin32) or a POSIX process group, falling back to `proc.kill()`.

Errors from tools are raised inside the snippet as `tools.ToolError`. Rejected code is never executed.
