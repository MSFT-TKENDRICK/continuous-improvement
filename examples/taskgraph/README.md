# Harness task-graph example

A diamond task graph — `brief` → (`inventory`, `risks`) → `summary` — with sealed rubrics. Each rubric mixes
deterministic oracles (regex / JSON schema) with one System-1 (`s1`) criterion. The deterministic
oracles decide on their own: when no s1 judge is available, the s1 votes abstain.

```powershell
# Check the graph against its sealed rubrics (commitments, leakage, instruction limits)
ci-lab graph seal examples/taskgraph/rubrics.yaml --vault runs/tg/sealed
ci-lab graph validate examples/taskgraph/graph.yaml --vault runs/tg/sealed

# Offline smoke: echo student, deterministic gamers, no s1 judge, no telemetry export
ci-lab graph run examples/taskgraph/graph.yaml --run-dir runs/tg --run-id demo `
  --rubric examples/taskgraph/rubrics.yaml --student fake --challenger det --s1-model "" --no-telemetry

# Real run: MAF student (harness/agents/student.yaml) on the copilot profile; spans in runs/tg/telemetry/
ci-lab graph run examples/taskgraph/graph.yaml --run-dir runs/tg --rubric examples/taskgraph/rubrics.yaml `
  --student agent --challenger both --max-parallel 2

ci-lab graph show runs/tg                         # statuses, scores, critical path, bus heads
ci-lab bus verify runs/tg/bus                     # hash chain of every topic (exit 1 on corruption)
ci-lab bus tail runs/tg/bus demo/brief --role student   # only what a student may see (commits)
ci-lab bus heads runs/tg/bus demo
```

The `fake` student writes the first fenced block of each deliverable's instructions. `graph run` exits
0 when every deliverable commits. Resource pools default to `s1=1, llm=4, cpu=<cores>`; override them
with `CI_POOL_<NAME>=N`. `--engine maf` needs `ci_lab.taskgraph.maf_engine`.
