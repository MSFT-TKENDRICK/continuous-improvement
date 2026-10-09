import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Sources } from "../lib/sources.mjs";
import {
    buildModel,
    buildSpanTree,
    campaignView,
    campaignOf,
    experimentDetail,
    flattenOtlp,
    liveView,
    mergeSpans,
    summarize,
    traceList,
    visibleAttributes,
} from "../lib/model.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const FIX = path.join(here, "fixtures");
const REPO = path.join(FIX, "repo");
const env = { CI_IMPORTS_DIR: path.join(FIX, "imports") };
const T = 1791435000;
const TRACE = "0af7651916cd43dd8448eb211c80319c";
const readJson = (...p) => JSON.parse(fs.readFileSync(path.join(FIX, ...p), "utf8"));

let cached;
async function model(opts = {}) {
    cached ??= await new Sources(REPO, { env }).scan();
    return buildModel(cached, { now: T, ...opts });
}

test("campaign view: incumbent, rounds, ΔS/ΔC, budget, stop flag", async () => {
    const m = await model();
    assert.equal(m.campaigns.length, 1);
    const c = m.campaigns[0];
    assert.equal(c.domain, "harness");
    assert.equal(c.incumbent.score, 0.7);
    assert.equal(c.delta, 0.02);
    assert.equal(c.rounds.length, 1);
    const r = c.rounds[0];
    assert.equal(r.eid, "tone-a1-r01");
    assert.equal(r.decision, "ship");
    assert.equal(r.winner, "v1");
    assert.equal(r.deltaS, 0.1);
    assert.equal(r.deltaC, 0);
    assert.equal(r.accepted, 1);
    assert.equal(c.budget.tokensUsed, 17000);
    assert.equal(c.budget.tokenBurn, 0.17);
    assert.equal(c.stopped, true, "STOP file in run dir named after the campaign");
    assert.equal(campaignOf("tone-a1-r12"), "tone-a1");
    assert.equal(campaignOf("tone-a1-confirm"), "tone-a1");
    assert.equal(campaignOf("plain"), null);
    assert.equal(campaignView({ campaignId: "new" }).domain, "harness");
    assert.equal(campaignView({ campaignId: "legacy", domain: "order_support" }).domain, "order_support");
});

test("experiments include rounds, calibration, confirm and sleep nights", async () => {
    const m = await model();
    const kinds = m.experiments.map((e) => e.kind).sort();
    assert.deepEqual(kinds, ["calibration", "confirm", "round", "sleep"]);
    const r1 = m.experiments.find((e) => e.id === "tone-a1-r01");
    assert.equal(r1.campaignId, "tone-a1");
    assert.equal(r1.ciLowerBound, 0.04);
    const d = experimentDetail(m, "tone-a1-r02");
    assert.equal(d.envelope, null);
    assert.equal(d.run.phase, "evaluate");
    assert.deepEqual(d.traceIds, [TRACE]);
    assert.equal(experimentDetail(m, "nope"), null);
});

test("live view: status.d aggregation, arms × strategy × state, C36 staleness", async () => {
    const m = await model();
    const r2 = m.live.find((r) => r.experimentId === "tone-a1-r02");
    assert.equal(r2.state, "running");
    assert.equal(r2.ageSec, 5);
    assert.equal(r2.stale, false);
    assert.equal(r2.round, 2);
    const v1 = r2.arms.find((a) => a.arm === "v1");
    assert.equal(v1.strategy, "gepa");
    assert.equal(v1.state, "running");
    assert.equal(v1.stale, false);
    const v2 = r2.arms.find((a) => a.arm === "v2");
    assert.equal(v2.state, "failed", "arm.done wins over a running status entry");
    assert.equal(v2.stale, false, "done arms are never stale");
    const r1 = m.live.find((r) => r.experimentId === "tone-a1-r01");
    assert.equal(r1.state, "done");
    assert.equal(r1.pr, 42);
    assert.equal(m.live[0].experimentId, "tone-a1-r02", "running first");

    const later = buildModel(cached, { now: T + 61, heartbeatSec: 30 });
    const s2 = later.live.find((r) => r.experimentId === "tone-a1-r02");
    assert.equal(s2.state, "stale", "age 66s > 2 × 30s");
    assert.equal(s2.stale, true);
    const notYet = buildModel(cached, { now: T + 50, heartbeatSec: 30 }).live.find((r) => r.experimentId === "tone-a1-r02");
    assert.equal(notYet.stale, false, "age 55s ≤ 60s");
});

test("live view honours a per-run heartbeat field", () => {
    const runs = [{ experimentId: "x-r01", source: "x", status: { phase: "propose", updated: 100, heartbeat: 300, arms: {} }, arms: [] }];
    assert.equal(liveView(runs, { now: 500 })[0].stale, false);
    assert.equal(liveView(runs, { now: 701 })[0].stale, true);
});

test("JSONL span records: schema v1 accepted, unknown versions rejected, imports tagged", async () => {
    const m = await model();
    assert.equal(m.rejectedSpans, 1);
    const local = m.spans.filter((s) => s.traceId === TRACE);
    assert.equal(local.length, 5);
    const imp = m.spans.find((s) => s.traceId === "1".repeat(32));
    assert.equal(imp.origin, "import:1234567");
    const missing = flattenOtlp([{ traceId: TRACE, spanId: "a".repeat(16), name: "x" }]);
    assert.equal(missing.length, 0);
    assert.equal(missing.rejected, 1);
});

test("golden fixtures from ci_lab.telemetry: JSONL record and Aspire response agree", () => {
    const lines = fs.readFileSync(path.join(FIX, "telemetry", "span_v1.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const a = flattenOtlp(lines, "jsonl");
    const b = flattenOtlp(readJson("telemetry", "aspire_trace.json"), "aspire");
    assert.equal(a.length, 2);
    assert.equal(b.length, 2);
    for (const k of ["traceId", "spanId", "parentSpanId", "name", "durationMs", "error"]) {
        assert.deepEqual(a.map((s) => s[k]), b.map((s) => s[k]), k);
    }
    assert.deepEqual(a[0].attributes, b[0].attributes);
    assert.deepEqual(a[0].resource, b[0].resource);
    assert.deepEqual(a[0].exceptionTypes, ["ValueError"]);
    assert.equal(a[0].links[0].spanId, "0123456789abcdef");
    assert.equal(a[0].durationMs, 1000);
    assert.equal(a[1].parentSpanId, "22cb37a110a6b0b1");
    assert.equal(a[0].parentSpanId, null, "empty parentSpanId means root");
});

test("Aspire spike capture: KeyValue attrs, empty values, error status", () => {
    const spans = flattenOtlp(readJson("aspire", "traces.json"), "aspire");
    assert.equal(spans.length, 2);
    assert.equal(spans[0].attributes["ci.campaign_id"], "demo-a1");
    assert.equal(spans[0].resource["service.name"], "ci-lab.spike");
    assert.equal(spans[0].resource["service.instance.id"], null);
    assert.equal(spans[1].error, true);
    assert.equal(spans[1].status.message, "boom");
    assert.equal(spans[1].durationMs, 1000);
});

test("mergeSpans dedupes local + Aspire copies of the same span", async () => {
    const m = await model();
    const aspire = flattenOtlp(readJson("aspire", "traces.json"), "aspire");
    const merged = mergeSpans(m.spans, aspire);
    assert.equal(merged.filter((s) => s.traceId === TRACE).length, 6, "5 local + 1 Aspire-only (same root id deduped)");
    assert.equal(merged.find((s) => s.spanId === "b7ad6b7169203331").origin, "jsonl", "local copy wins");
});

test("span tree: nesting round → arm → step → agent → chat with durations and errors", async () => {
    const m = await model();
    const t = buildSpanTree(m.spans, TRACE);
    assert.equal(t.roots.length, 1);
    const round = t.roots[0];
    assert.equal(round.name, "ci.round");
    assert.equal(round.category, "harness");
    const chat = round.children[0].children[0].children[0].children[0];
    assert.equal(chat.name, "chat gpt-4.1");
    assert.equal(chat.category, "genai");
    assert.equal(chat.error, true);
    assert.deepEqual(chat.exceptionTypes, ["RateLimitError"]);
    assert.equal(chat.durationMs, 10000);
    assert.equal(chat.offsetMs, 40000);
    assert.equal(t.errors, 1);
    assert.equal(t.genai, 2);
    assert.equal(t.campaignId, "tone-a1");
    assert.equal(buildSpanTree(m.spans, "not-a-trace"), null);
    assert.equal(buildSpanTree(m.spans, "f".repeat(32)), null);
});

test("C29: GenAI content attributes hidden unless sensitive mode is flagged", async () => {
    const m = await model();
    const t = buildSpanTree(m.spans, TRACE);
    const agent = t.roots[0].children[0].children[0].children[0];
    assert.equal(agent.attributes["gen_ai.input.messages"], undefined);
    assert.equal(agent.redacted, 1);
    assert.equal(JSON.stringify(t).includes("4111"), false);
    const chat = agent.children[0];
    assert.equal(chat.attributes["gen_ai.output.messages"], undefined);
    assert.equal(chat.attributes["gen_ai.usage.input_tokens"], "120");
    const sens = visibleAttributes({ attributes: { "gen_ai.input.messages": "hi" }, resource: { "ci.telemetry.sensitive": true } });
    assert.equal(sens.attrs["gen_ai.input.messages"], "hi");
    assert.equal(sens.sensitive, true);
    const long = visibleAttributes({ attributes: { x: "y".repeat(2000) }, resource: {} });
    assert.ok(long.attrs.x.length <= 501);
});

test("span tree survives cycles and orphans", () => {
    const mk = (id, parent) => ({ schemaVersion: 1, traceId: TRACE, spanId: id, parentSpanId: parent, name: id, startTimeUnixNano: "1", endTimeUnixNano: "2" });
    const spans = flattenOtlp([mk("a".repeat(16), "b".repeat(16)), mk("b".repeat(16), "a".repeat(16)), mk("c".repeat(16), "d".repeat(16))]);
    const t = buildSpanTree(spans, TRACE);
    const count = (n) => 1 + n.children.reduce((s, c) => s + count(c), 0);
    assert.equal(t.roots.reduce((s, r) => s + count(r), 0), 3);
    assert.equal(traceList(spans)[0].incomplete, true);
});

test("evals: dimensions, judge agreement, errors, staleness", async () => {
    const m = await model();
    const base = m.evals.find((e) => e.runId === "baseline");
    assert.equal(base.status, "completed");
    assert.equal(base.rows, 4);
    const pv = base.dimensions.find((d) => d.key === "policy_violation");
    assert.equal(pv.type, "boolean");
    const tu = base.dimensions.find((d) => d.key === "tool_use");
    assert.equal(tu.type, "ordinal");
    assert.ok(tu.scale.includes("appropriate"));
    const res = base.dimensions.find((d) => d.key === "resolution");
    assert.equal(res.type, "numeric");
    assert.equal(base.agreement, null);
    const two = m.evals.find((e) => e.runId === "two-judges");
    assert.equal(two.judgeModels.length, 2);
    assert.equal(two.errors, 1);
    assert.ok(two.agreement.byDim.resolution < 1);
    assert.ok(two.flags.some((f) => f.includes("judge error")));
    assert.ok(two.flags.some((f) => f.startsWith("stale")));
    assert.ok(two.flags.some((f) => f.includes("low judge agreement on resolution")));
});

test("sleep view and summary", async () => {
    const m = await model();
    assert.equal(m.sleep.nights.length, 1);
    assert.equal(m.sleep.state.lastStatus, "accepted");
    assert.equal(m.sleep.tasks.pending, 1);
    const s = summarize(m);
    assert.equal(s.campaigns[0].id, "tone-a1");
    assert.equal(s.live.running, 1);
    assert.equal(s.live.done, 1);
    assert.equal(s.imports, 1);
    assert.equal(s.rollouts.total, 1);
    const json = JSON.stringify(s);
    assert.ok(json.length < 4000, `compact summary (${json.length} B)`);
    assert.equal(/browser_token|api_key|otlp_key|4111|secret/i.test(json), false);
});
