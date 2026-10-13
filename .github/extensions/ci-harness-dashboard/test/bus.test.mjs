import { test, after } from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Sources } from "../lib/sources.mjs";
import { buildModel } from "../lib/model.mjs";
import { parseWal, projectEntry, findBusRoots, readBus, ORCHESTRATOR_KINDS } from "../lib/bus.mjs";
import { REPO, scratch } from "./helpers.mjs";

const tmp = scratch("bus");
after(() => fs.rmSync(tmp, { recursive: true, force: true }));

const SECRET = "SEALED_RUBRIC_TEXT_42";
const GENESIS = "0".repeat(64);
const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");

/** Hand-built hash-linked WAL lines (hashes are linked, not recomputed by the reader). */
function wal(topic, entries) {
    let prev = GENESIS;
    return entries
        .map(([kind, role, body, extra = {}], seq) => {
            const hash = sha(`${topic}${seq}${kind}`);
            const e = { seq, topic, kind, author: { role, name: role, model: null }, ref: null, body, ts: "2026-10-08T00:00:00+00:00", prev, hash, ...extra };
            prev = hash;
            return JSON.stringify(e);
        })
        .join("\n") + "\n";
}
function writeBus(repo, rel, run, task, text) {
    const dir = path.join(repo, rel, run);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${task}.wal.jsonl`), text);
}

const LEAKY = [
    ["intent", "orchestrator", { action: "attempt", key: "t@1", attempt: "t@1", detail: { rubric: SECRET } }],
    ["proposal", "student", { proposal_id: "t@1/student:s", summary: SECRET, hidden: SECRET, rubric_version: "r@v1" }],
    ["vote", "voter", { proposal_id: "t@1/student:s", voter: "v", criterion: "c1", passed: false, score: 0, reasons: [SECRET], criteria: SECRET, text: SECRET }],
    ["verdict", "judge", { proposal_id: "t@1/student:s", decision: "reject", score: 0, criteria: { c1: { text: SECRET, passed: false } }, correction: SECRET, votes: [{ reasons: [SECRET] }] }],
    ["exploit", "orchestrator", { attempt: "t@1", gamer: "g", soft_pref: "adversary", oracle_invalid: ["c1"], rubric: SECRET }],
    ["rubric_patch", "hardener", { rubric_id: "r", from_version: "r@v1", to_version: "r@v2", accepted: true, diff: SECRET, rubric: { text: SECRET } }],
    ["note", "orchestrator", { text: SECRET, data: { sealed: SECRET } }],
    ["reject", "judge", { proposal_id: "t@1/student:s", reason: "failed c1", sealed: SECRET }],
];

test("fixture bus: projection, counts, flags and torn tail", async () => {
    const scan = await new Sources(REPO, { env: {} }).scan();
    assert.equal(scan.bus.length, 1);
    const [run] = scan.bus;
    assert.equal(run.root, "artifacts/ci-runs/graph-demo/bus");
    assert.deepEqual(run.topics.map((t) => t.task), ["_run", "facts", "risks"]);
    const facts = run.topics[1];
    assert.equal(facts.entries, 10);
    assert.equal(facts.state, "committed");
    assert.deepEqual([facts.exploits, facts.patches, facts.corrupt, facts.torn], [1, 1, null, false]);
    assert.deepEqual(facts.rows.filter((r) => r.flag).map((r) => [r.kind, r.flag]), [["exploit", "exploit"], ["rubric_patch", "patch"]]);
    const vote = facts.rows.find((r) => r.seq === 4);
    assert.equal(vote.fields.reasons, 1);
    assert.match(vote.summary, /criterion=f-schema passed=true score=0.5/);
    assert.equal(facts.rows.find((r) => r.kind === "commit").artifact.bytes, 14);
    const risks = run.topics[2];
    assert.deepEqual([risks.entries, risks.state, risks.torn, risks.corrupt], [2, "aborted", true, null]);
    assert.match(risks.rows[1].summary, /reason=budget exhausted/);
    assert.deepEqual(scan.warnings, []);
    const m = buildModel(scan, { now: 1791435000 });
    assert.deepEqual(m.bus.totals, { runs: 1, topics: 3, entries: 13, exploits: 1, patches: 1, committed: 1, aborted: 1, rejected: 0, corrupt: 0, torn: 1 });
});

test("orchestrator-only: sealed/hidden body fields never reach the projection", async () => {
    const t = parseWal(wal("r1/t", LEAKY), "r1/t");
    assert.equal(t.entries, LEAKY.length);
    assert.equal(t.state, "rejected");
    assert.ok(!JSON.stringify(t).includes(SECRET), "sentinel leaked into projection");
    assert.equal(t.rows.find((r) => r.kind === "verdict").fields.correction, true);
    assert.equal(t.rows.find((r) => r.kind === "note").fields.data, 1);
    assert.equal(t.rows.find((r) => r.kind === "reject").fields.reason, "failed c1");
    assert.equal(projectEntry({ seq: 0, kind: "mystery", author: {}, body: { x: SECRET } }), null);
    assert.ok(ORCHESTRATOR_KINDS.includes("exploit") && ORCHESTRATOR_KINDS.includes("rubric_patch"));
    const odd = parseWal(wal("r1/u", [["mystery", "orchestrator", { x: SECRET }]]), "r1/u");
    assert.deepEqual([odd.entries, odd.hidden, odd.rows.length], [1, 1, 0]);
    const author = projectEntry({ seq: 0, kind: "note", author: { role: "god", name: `${SECRET} <x>` }, body: {} });
    assert.deepEqual([author.role, author.name], [null, null]);
});

test("corruption: mid-file garbage, seq gap, prev mismatch, wrong topic", () => {
    const good = wal("r1/t", LEAKY.slice(0, 4)).split("\n").filter(Boolean);
    const cases = {
        garbage: [good[0], "{not json", good[2]],
        gap: [good[0], good[2]],
        prev: [good[0], JSON.stringify({ ...JSON.parse(good[1]), prev: sha("x") })],
        topic: [good[0], JSON.stringify({ ...JSON.parse(good[1]), topic: "r1/other" })],
    };
    for (const [name, lines] of Object.entries(cases)) {
        const t = parseWal(lines.join("\n") + "\n", "r1/t");
        assert.ok(t.corrupt, name);
        assert.equal(t.entries, 1, name);
    }
    const torn = parseWal(good.join("\n") + "\n{\"seq\":4,", "r1/t");
    assert.deepEqual([torn.torn, torn.corrupt, torn.entries], [true, null, 4]);
    const tornTerminated = parseWal(good.join("\n") + "\n{\"seq\":4,\n", "r1/t");
    assert.deepEqual([tornTerminated.torn, tornTerminated.corrupt], [true, null]);
    const tail = parseWal(good.slice(2).join("\n") + "\n", "r1/t", { truncated: true });
    assert.deepEqual([tail.entries, tail.corrupt, tail.truncated], [2, null, true]);
    const capped = parseWal(good.join("\n") + "\n", "r1/t", { maxRows: 2 });
    assert.deepEqual([capped.rows.map((r) => r.seq), capped.rowsTruncated], [[2, 3], true]);
});

test("discovery is bounded and never enters sealed/vault/challenger dirs or escapes the repo", async () => {
    const repo = path.join(tmp, "repo");
    const text = wal("r2/t", LEAKY.slice(0, 2));
    writeBus(repo, "artifacts/ci-runs/x/bus", "r2", "t", text);
    for (const skip of ["sealed", "vault", "challenger"]) writeBus(repo, `artifacts/ci-runs/${skip}/bus`, "r2", "t", text);
    writeBus(repo, "artifacts/a/b/c/d/bus", "r2", "t", text);
    writeBus(repo, "artifacts/ci-runs/x/bus", "../evil", "t", text);
    const outside = path.join(tmp, "outside");
    writeBus(outside, "bus", "r3", "t", text);
    let linked = true;
    try {
        fs.symlinkSync(outside, path.join(repo, "artifacts", "ci-runs", "link"), "junction");
    } catch {
        linked = false;
    }
    const src = new Sources(repo, { env: {} });
    await src.resolveRoots();
    const roots = await findBusRoots(src);
    assert.deepEqual(roots.map((r) => src.rel(r)), ["artifacts/ci-runs/x/bus"]);
    const bus = await readBus(src);
    assert.deepEqual(bus.map((r) => r.run), ["r2"]);
    if (linked) assert.ok(fs.existsSync(path.join(repo, "artifacts/ci-runs/link/bus/r3/t.wal.jsonl")), "junction should resolve but be skipped");
    assert.ok(linked || roots.length === 1);
    const corrupt = path.join(repo, "artifacts/ci-runs/x/bus/r2/t.wal.jsonl");
    fs.writeFileSync(corrupt, "{bad\n" + text);
    const warned = new Sources(repo, { env: {} });
    await warned.resolveRoots();
    const [r] = await readBus(warned);
    assert.ok(r.topics[0].corrupt);
    assert.ok(warned.warnings.some((w) => w.includes("bus topic r2/t")));
});
