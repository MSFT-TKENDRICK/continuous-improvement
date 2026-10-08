# maf-core (`ci_lab.maf`)

The Microsoft Agent Framework (Python) runtime layer. Every ci_lab agent and workflow is
built here. Nothing in this module needs PowerFx or .NET: `powerfx` must stay uninstalled,
so `agent_framework_declarative` runs with no expression engine (`_get_engine() is None`).

## Agent manifest and specs (`specs.py`)

```yaml
# agents/manifest.yaml (paths are relative to this file and must stay under its directory)
agents:
  OrderSupport:
    spec: order_support.yaml
    runtime: prompt            # prompt | harness
    purpose: target            # contracts.Purpose
    bindings: [lookup_order]   # the only tool bindings this agent may use
    skills_paths: [skills]
```

```yaml
# agents/order_support.yaml: a MAF `kind: Prompt` agent plus our `x-ci` extension
kind: Prompt
name: OrderSupport
description: Answers order questions.
instructions: Optional inline preamble.
model: {id: gpt-5-mini, provider: GitHubCopilot, options: {reasoningEffort: low}}
tools:
  - kind: function
    name: lookup_order
    description: Look up an order.          # descriptions may evolve
    bindings: [{name: lookup_order}]
    parameters: {properties: {order_id: {kind: string, required: true}}}
x-ci:
  instructions_files: [prompts/system.md, skills/refunds.md]
  append_text: Sign off politely.
```

The loader removes `x-ci` and builds `instructions` from these parts, in order:
1. the inline `instructions`;
2. each file in `instructions_files`;
3. `append_text`;
4. the caller's `extra_instructions`.

The parts are joined with blank lines.

Validation rejects a spec, raising `SpecError`, if:
- the model id is not on the allowlist;
- the provider is neither `GitHubCopilot` nor explicitly allowed;
- any key or value, including composed instructions, starts with `=`;
- a tool binding is not in the allowed set;
- an `options` key is not in `reasoningEffort`, `maxOutputTokens` or `allowMultipleToolCalls` (configurable);
- there are unknown keys, non-`function` tools or duplicate tools;
- with `frozen_tool_schemas`, the tool set or any tool's parameter schema changed. Only `description`, `title` and `examples` may differ.

Get frozen schemas with `AgentSpec.tool_schemas()`. Referenced files must be relative, have no `..`, drive or `:` in the path, include no symlinks or junctions, and resolve under the YAML's directory.

## Building agents (`loader.py`)

```python
agent = build_agent("agents/order_support.yaml", client=client,
                    bindings={"lookup_order": lookup_order}, runtime="prompt",
                    allowed_models=["gpt-5-mini"], skills_paths=["skills"])
agents = build_agents_from_manifest("agents/manifest.yaml", client_factory=factory,
                                    profile=profile, bindings=all_bindings,
                                    allowed_models=[...])
```

- **`prompt`** runtime: `AgentFactory(client=..., bindings=..., safe_mode=True).create_agent_from_dict(...)`.
  - The model `id`/`provider` are removed from the dict passed to MAF. Otherwise `AgentFactory` would build its own client instead of using the injected one.
  - Model selection is the client's job. `build_agents_from_manifest` passes `spec.model.id` to the `ChatClientFactory`.
- **`harness`** runtime: `create_harness_agent(client, ...)` with the same instructions, tools and options.
  - File memory is off unless `memory_dir` is given. Web search is disabled.
  - Tool auto-approval is disabled: MAF's approval middleware requires an `AgentSession` on every run.
- **Skills** (`skills_paths`) are data-only. No script extensions are allowed, and loading or reading a skill needs no approval.
- **Bindings** given to `build_agent` are both the implementations and the allowlist. `build_agents_from_manifest` limits each agent to the bindings its manifest entry lists.
- **Provenance:** MAF `ExperimentalWarning`s are suppressed and recorded. `provenance()` returns MAF package versions, the experimental features used and `powerfx_installed`. Each agent's `additional_properties["ci_lab"]` holds the spec digest, runtime, model and provider.
- **No `.env` loading:** `AgentFactory` is created with `env_file_path=os.devnull`, so it never reads a stray `.env`.

## Workflows (`workflows.py`)

```yaml
kind: Workflow
trigger:
  kind: OnConversationStart
  id: demo_wf                     # workflow name / checkpoint key
  actions:
    - {kind: InvokeFunctionTool, id: prep, functionName: prep, arguments: {x: "1"}}
    - {kind: InvokeAzureAgent, id: think, agent: {name: Thinker}, input: {messages: "do it"}}
    - {kind: InvokeFunctionTool, id: finish, functionName: finish, arguments: {}}
```

- `assert_expression_free(text)` rejects:
  - any `=`-prefixed string;
  - the actions `If`, `ConditionGroup`, `Foreach`, `GotoAction`, `BreakLoop`, `ContinueLoop`, `HttpRequestAction` and `InvokeMcpTool`.

  Put branching in Python tools.
- `build_workflow(path, agents=, tools=, checkpoint_dir=)` also rejects:
  - inline `agents:`;
  - unknown agent or tool references;
  - any `kind` other than `Workflow`.

  It returns `(workflow, FileCheckpointStorage)`.
- `declarative_allowlist()` lists every class in `agent_framework_declarative._workflows._declarative_base`. Without these entries MAF only logs a warning and silently drops checkpoints. Pass `checkpoint_types=` for your own state types.
- `run_or_resume(path, message, ...)` resumes from the latest checkpoint for the workflow name (by timestamp, then iteration) or runs fresh. It raises `CheckpointNotWrittenError` unless every executed superstep wrote a checkpoint.
  - Resuming re-runs the superstep after the checkpoint, so tools must be idempotent.
  - Resuming a completed run returns `[]`.

## Observability

maf-core opens no spans of its own: MAF emits the GenAI spans. `run_or_resume` and agent runs execute in the caller's OTel context, and async tasks inherit it. Open a `ci.case` or `ci.step` span with `ci_lab.obs.span(...)` before calling them, and MAF's spans and tool calls will nest under it. `run_or_resume(..., rollout=RolloutKey)` and `record_rollout(rollout, **attrs)` tag the current span with `agl.rollout_id` and `oes.variant`. If no span is recording, they do nothing. This module never installs a tracer provider.

**Trust boundary:** checkpoints contain **pickle** data.
- Keep a checkpoint dir private to one run on one machine (under `CI_RUN_DIR`).
- Never cache, upload, commit or restore it across jobs or branches.
- `secure_dir` creates the dir with mode `0700` on POSIX and refuses symlinks. On Windows the dir inherits its parent's ACL, so place it under a per-user or per-job directory.
