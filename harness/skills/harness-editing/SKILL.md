---
name: harness-editing
description: >-
  How to edit a self-hosted harness tree (harness/) safely. Use when proposing an edit to agent
  specs, prompts, skills, workflows, loops, tool exposure or guards.
---

# Editing the harness tree

- Every file must match a component of the frozen manifest (`harness.yaml`); never edit `harness.yaml`.
- Stay inside the components your directive targets. One hypothesis per edit; keep diffs small.
- `agents/*.yaml`: keep the `name`, the terminal `submit_*` tool and every bound tool; instructions live in
  `prompts/*.md` (`x-ci.instructions_files`). Never add expressions (`=...`).
- `loops/loops.yaml`: `max_nudges`, `max_tool_calls`, `max_turns` per agent and `code_mode.max_runs`; values
  above the frozen caps are rejected.
- `tools/tools.yaml`: can only narrow an agent's bound tools or reword their descriptions; keep the terminal tool.
- `workflows/*.yaml`: only `InvokeAzureAgent` (literal messages) and `InvokeFunctionTool` (`self_check`).
- `skills/<name>/SKILL.md`: frontmatter `name` equal to the directory and a one-paragraph `description`.
- Run `ci-lab harness validate --dir <tree>` before committing.
