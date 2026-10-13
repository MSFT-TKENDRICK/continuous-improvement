import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Sources, aggregateStatus, normalizeEnvelope, summarizeRollout, toEpochSec, clearReadCache } from "../lib/sources.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
export const FIX = path.join(here, "fixtures");
export const REPO = path.join(FIX, "repo");
export const fixtureEnv = { CI_IMPORTS_DIR: path.join(FIX, "imports") };

test("scan discovers ledger, run dirs, results and imports", async () => {
    const s = new Sources(REPO, { env: fixtureEnv });
    const scan = await s.scan();
    assert.equal(scan.roots.ledger.length, 1);
    assert.equal(scan.roots.runs.length, 1);
    assert.equal(scan.roots.results.length, 1);
    assert.equal(scan.roots.imports.length, 1);
    assert.equal(scan.campaigns.length, 1);
    const c = scan.campaigns[0];
    assert.equal(c.campaignId, "tone-a1");
    assert.equal(c.domain, "harness");
    assert.equal(c.frontier.score, 0.7);
    assert.equal(c.budget.tokens, 100000);
    assert.equal(c.history.length, 1, "corrupt history line skipped");
    assert.equal(c.rounds.length, 1);
    assert.equal(c.rounds[0].envelope.id, "tone-a1-r01");
    assert.equal(c.calibration.envelope.kind, "calibration");
    assert.equal(c.confirm.envelope.kind, "confirm");
    assert.equal(scan.sleep.nights.length, 1);
    assert.equal(scan.sleep.tasks.reviewed, 3);
    assert.equal(scan.sleep.tasks.pending, 1);
    assert.equal(scan.holdoutLooks, 1);
    assert.equal(scan.imports.length, 1);
    assert.equal(scan.imports[0].runId, "1234567");
    assert.equal(scan.imports[0].repo, "org/repo");
    assert.equal(scan.imports[0].digestVerified, true);
    assert.equal(scan.imports[0].files, 1, "manifest path escaping the run dir is dropped");
});

test("run dirs aggregate status.d writers like obs.read_status", async () => {
    const s = new Sources(REPO, { env: fixtureEnv });
    const scan = await s.scan();
    const r2 = scan.runs.find((r) => r.experimentId === "tone-a1-r02");
    assert.ok(r2);
    assert.equal(r2.status.phase, "evaluate", "latest writer (v1) wins top-level fields");
    assert.equal(r2.status.traceId, "0af7651916cd43dd8448eb211c80319c", "trace{} from round writer survives");
    assert.deepEqual(r2.status.writers.sort(), ["round", "v1", "v2"]);
    assert.equal(r2.status.arms.v1.state, "running", "legacy status.json arm entry is older and loses");
    assert.equal(r2.status.arms.v2.phase, "propose");
    assert.equal(r2.begin.round, 2);
    const v2 = r2.arms.find((a) => a.arm === "v2");
    assert.equal(v2.done, true);
    assert.equal(v2.status, "failed");
    assert.match(v2.reason, /boom/);
    const r1 = scan.runs.find((r) => r.experimentId === "tone-a1-r01");
    assert.equal(r1.done.decision, "ship");
    assert.equal(r1.done.pr, 42);
    assert.ok(scan.runs.find((r) => r.experimentId === "tone-a1" && r.stopped));
});

test("aggregateStatus mirrors python semantics", () => {
    const out = aggregateStatus([
        { writer: "b", seq: 1, updated: 20, phase: "late", arms: { x: { state: "old", updated: 5 } } },
        null,
        { writer: "a", seq: 4, updated: 10, phase: "early", extra: 1, arms: { x: { state: "new", updated: 9 }, y: { state: "y", updated: 1 } } },
    ]);
    assert.equal(out.phase, "late");
    assert.equal(out.extra, 1);
    assert.equal(out.arms.x.state, "new");
    assert.deepEqual(out.writers, ["a", "b"]);
    assert.equal(out.seq, undefined);
    assert.equal(aggregateStatus([null, undefined]), null);
});

test("spans, rollouts and evals are read with redaction-friendly shapes", async () => {
    const s = new Sources(REPO, { env: fixtureEnv });
    const scan = await s.scan();
    assert.equal(scan.spanLines.length, 7, "5 local spans + 1 future-schema line + 1 import line (broken line dropped)");
    assert.ok(scan.spanLines.some((l) => l.__import === "1234567"));
    assert.equal(scan.rollouts.length, 1);
    const ro = scan.rollouts[0];
    assert.equal(ro.status, "succeeded");
    assert.equal(ro.score, 0.75);
    assert.equal(JSON.stringify(ro).includes("do not show"), false, "rollout input is never surfaced");
    assert.equal(scan.evals.length, 2);
    const base = scan.evals.find((e) => e.runId === "baseline");
    assert.equal(base.manifest.status, "completed");
    assert.equal(base.metrics.calls, 27);
    assert.equal(base.scores.length, 4);
    assert.equal(typeof base.scores[0].dims.policy_violation, "boolean");
    assert.deepEqual(base.scales.tool_use.values, ["appropriate", "unnecessary", "missing_required", "policy_violating"]);
    assert.equal(JSON.stringify(base).includes("REDACTED-JUSTIFICATION"), false, "justifications not loaded");
});

test("readJsonlTail caps bytes and lines; readJson rejects oversize and garbage", async () => {
    const dir = fs.mkdtempSync(path.join(here, ".tmp-"));
    try {
        const p = path.join(dir, "big.jsonl");
        fs.writeFileSync(p, Array.from({ length: 500 }, (_, i) => JSON.stringify({ i })).join("\n") + "\n");
        const s = new Sources(dir, { env: {} });
        await s.resolveRoots();
        const t = await s.readJsonlTail(p, { maxBytes: 200, maxLines: 1000 });
        assert.ok(t.truncated);
        assert.equal(t.rows.at(-1).i, 499);
        assert.ok(t.rows.length < 30);
        const t2 = await s.readJsonlTail(p, { maxLines: 10 });
        assert.equal(t2.rows.length, 10);
        fs.writeFileSync(path.join(dir, "bad.json"), "{nope");
        assert.equal(await s.readJson(path.join(dir, "bad.json")), null);
        fs.writeFileSync(path.join(dir, "big.json"), JSON.stringify({ x: "y".repeat(2000) }));
        assert.equal(await s.readJson(path.join(dir, "big.json"), 100), null);
        assert.ok(s.warnings.some((w) => w.includes("too large")));
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
        clearReadCache();
    }
});

test("symlinks and paths outside the allowed roots are rejected", async (t) => {
    const dir = fs.mkdtempSync(path.join(here, ".tmp-"));
    const outside = fs.mkdtempSync(path.join(here, ".tmp-outside-"));
    try {
        fs.writeFileSync(path.join(outside, "secret.json"), JSON.stringify({ secret: 1 }));
        fs.mkdirSync(path.join(dir, "experiments"));
        let linked = true;
        try {
            fs.symlinkSync(path.join(outside, "secret.json"), path.join(dir, "experiments", "link.json"), "file");
        } catch {
            linked = false;
        }
        try {
            fs.symlinkSync(outside, path.join(dir, "experiments", "campaigns"), "junction");
        } catch {
            /* ignore */
        }
        const s = new Sources(dir, { env: {} });
        await s.resolveRoots();
        if (linked) assert.equal(await s.readJson(path.join(dir, "experiments", "link.json")), null);
        assert.equal(await s.readJson(path.join(outside, "secret.json")), null, "outside root");
        assert.deepEqual(await s.list(path.join(dir, "experiments", "campaigns")), [], "junction/symlink dir not followed");
        const scan = await s.scan();
        assert.equal(scan.campaigns.length, 0);
        if (!linked) t.diagnostic("file symlinks unavailable (no privilege); junction path still tested");
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
        fs.rmSync(outside, { recursive: true, force: true });
    }
});

test("CI_RUN_DIR outside the repo becomes an allowed root", async () => {
    const dir = fs.mkdtempSync(path.join(here, ".tmp-"));
    const runs = fs.mkdtempSync(path.join(here, ".tmp-runs-"));
    try {
        fs.mkdirSync(path.join(runs, "exp-one", "status.d"), { recursive: true });
        fs.writeFileSync(path.join(runs, "exp-one", "status.d", "round.json"), JSON.stringify({ writer: "round", phase: "propose", updated: 1 }));
        const s = new Sources(dir, { env: { CI_RUN_DIR: runs } });
        const scan = await s.scan();
        assert.equal(scan.runs.length, 1);
        assert.equal(scan.runs[0].status.phase, "propose");
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
        fs.rmSync(runs, { recursive: true, force: true });
    }
});

test("normalizers are tolerant", () => {
    assert.equal(normalizeEnvelope(null), null);
    const e = normalizeEnvelope({ experiment: { id: "x" }, variants: [null, { id: "a" }], results: { metricResults: "nope" } });
    assert.equal(e.id, "x");
    assert.equal(e.variants.length, 1);
    assert.deepEqual(e.results, []);
    const env = normalizeEnvelope(JSON.parse(fs.readFileSync(path.join(FIX, "oes", "round.json"), "utf8")));
    assert.equal(env.selection.winner, "v1");
    assert.equal(env.selection.candidates.find((c) => c.variantId === "v1").deltaS, 0.1);
    assert.equal(env.decision.outcome, "ship");
    const sl = normalizeEnvelope(JSON.parse(fs.readFileSync(path.join(FIX, "oes", "sleep.json"), "utf8")));
    assert.equal(sl.kind, "sleep");
    assert.ok(sl.sleep.tasks.total > 0);
    assert.equal(summarizeRollout([], "ro-x", "s").status, "unknown");
    assert.equal(toEpochSec("2026-10-08T00:00:00Z"), Date.parse("2026-10-08T00:00:00Z") / 1000);
    assert.equal(toEpochSec(1791435000000), 1791435000);
});
