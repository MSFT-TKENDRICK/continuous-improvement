// Pure-JS tests for the shared-state readers (no npm install needed):
//   node --test web/experiment-chat/test/*.test.mjs
// Input is layer 32's recorded agent_framework_ag_ui exchange, shared with the canvas tests.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { hyperRows, interruptCall, normalizeDraft, normalizeLaunches } from "../src/state.js";

const FLOW = JSON.parse(fs.readFileSync(new URL("../../../.github/extensions/ci-harness-dashboard/test/fixtures/chat/approval_flow.json", import.meta.url), "utf8"));
const lastOf = (step, type) => FLOW.steps[step].events.findLast((e) => e.type === type);

test("draft card fields come from the STATE_SNAPSHOT draft", () => {
    const draft = lastOf("1_draft", "STATE_SNAPSHOT").snapshot.draft;
    const d = normalizeDraft(draft);
    assert.equal(d.cid, "chat-demo");
    assert.equal(d.target, "local");
    assert.equal(d.rounds, 1);
    assert.equal(d.estimatedEvaluations, draft.estimate.evaluations);
    assert.equal(d.formula, draft.estimate.formula);
    assert.ok(Array.isArray(d.warnings));
    assert.ok(d.rationale);
    const rows = hyperRows(d.hyper);
    assert.ok(rows.some(([k, v]) => k === "arms" && v === "2"));
    assert.ok(!rows.some(([k]) => k === "budget" || k === "rrsi"), "null and empty hyperparameters are hidden");
    assert.equal(normalizeDraft(null), null);
    assert.equal(normalizeDraft([1]), null);
});

test("approval card reads the launch call from the interrupt metadata", () => {
    const intr = lastOf("2_launch_interrupt_approved", "RUN_FINISHED").outcome.interrupts[0];
    const { name, args } = interruptCall(intr);
    assert.equal(name, "launch_campaign");
    assert.equal(args.cid, "chat-demo");
    assert.deepEqual(interruptCall({ metadata: { agent_framework: { function_call: { name: "x", arguments: "{bad" } } } }), { name: "x", args: {} });
    const custom = FLOW.steps["2_launch_interrupt_approved"].events.find((e) => e.type === "CUSTOM");
    assert.equal(custom.name, "function_approval_request");
    assert.equal(custom.value.function_call.name, name);
});

test("launch list: approve adds the launch, reject does not; only github.com https run links survive", () => {
    const launches = normalizeLaunches(lastOf("3_resume_approved", "STATE_SNAPSHOT").snapshot.launches);
    assert.deepEqual(launches.map((l) => [l.cid, l.status]), [["chat-demo", "dry_run"]]);
    assert.deepEqual(normalizeLaunches(lastOf("3_resume_rejected", "STATE_SNAPSHOT").snapshot.launches), []);
    const urls = normalizeLaunches([
        { cid: "a", run_url: "https://github.com/o/r/actions/runs/1" },
        { cid: "b", run_url: "javascript:alert(1)" },
        { cid: "c", run_url: "https://evil.example/x" },
        { cid: "", status: "x" },
        "junk",
    ]);
    assert.deepEqual(urls.map((l) => [l.cid, l.runUrl]), [["a", "https://github.com/o/r/actions/runs/1"], ["b", undefined], ["c", undefined]]);
});
