import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { Hub, acquireHub, releaseHub, activeHubs } from "../lib/hub.mjs";
import { REPO, scratch } from "./helpers.mjs";

const tmp = scratch("hub");
after(() => fs.rmSync(tmp, { recursive: true, force: true }));
const env = { CI_IMPORTS_DIR: path.join(tmp, "no-imports") };
const T = 1791435000;

function copyRepo(name) {
    const dst = path.join(tmp, name);
    fs.cpSync(REPO, dst, { recursive: true });
    return dst;
}

const nextChange = (hub, ms = 4000) =>
    new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error("no change event")), ms);
        hub.once("change", (e) => {
            clearTimeout(t);
            resolve(e);
        });
    });

test("acquireHub shares one hub per repo root and ref-counts it", async () => {
    const root = copyRepo("shared");
    const a = await acquireHub(root, { env, now: () => T, pollMs: 60000 });
    const b = await acquireHub(path.join(root, "."), { env, now: () => T, pollMs: 60000 });
    assert.equal(a, b);
    assert.equal(a.refs, 2);
    assert.ok(a.model, "model is built before acquire resolves");
    assert.equal(a.version, 1);
    assert.equal(a.model.campaigns[0].campaignId, "tone-a1");
    const before = activeHubs();
    releaseHub(a);
    assert.equal(activeHubs(), before, "still referenced");
    releaseHub(b);
    assert.equal(activeHubs(), before - 1);
    assert.equal(a.closed, true);
});

test("file changes are picked up (watch or poll) and fanned out once per real change", async () => {
    const root = copyRepo("watch");
    const hub = await new Hub(root, { env, now: () => T, debounceMs: 20, pollMs: 150 }).start();
    try {
        const run = path.join(root, "artifacts", "ci-runs", "tone-a1-r02", "status.d");
        const changed = nextChange(hub);
        fs.writeFileSync(path.join(run, "v3.json"), JSON.stringify({ writer: "v3", seq: 1, experiment_id: "tone-a1-r02", phase: "mutate", updated: T, arms: { v3: { strategy: "ace", state: "running", phase: "mutate", updated: T } } }));
        const e = await changed;
        assert.ok(e.version >= 2);
        const r = hub.model.live.find((x) => x.experimentId === "tone-a1-r02");
        assert.ok(r.arms.some((a) => a.arm === "v3" && a.strategy === "ace"));
        assert.ok(r.writers.includes("v3"));

        // No further change events when nothing changed (polls keep running).
        const v = hub.version;
        await new Promise((r) => setTimeout(r, 450));
        assert.equal(hub.version, v);

        // A manual refresh always bumps the version so clients refetch.
        const manual = nextChange(hub);
        await hub.refresh("manual");
        assert.equal((await manual).reason, "manual");
        assert.ok(hub.version >= v + 1);
    } finally {
        hub.close();
    }
});

test("a manual refresh that coalesces with a running one still bumps the version once", async () => {
    const root = copyRepo("coalesce-manual");
    const hub = await new Hub(root, { env, now: () => T, watch: false, pollMs: 60000 }).start();
    try {
        const v = hub.version;
        const reasons = [];
        hub.on("change", (e) => reasons.push(e.reason));
        await Promise.all([hub.refresh("poll"), hub.refresh("manual"), hub.refresh("watch")]);
        assert.deepEqual(reasons, ["manual"]);
        assert.equal(hub.version, v + 1);
    } finally {
        hub.close();
    }
});

test("concurrent refreshes coalesce", async () => {
    const root = copyRepo("coalesce");
    const hub = new Hub(root, { env, now: () => T, watch: false, pollMs: 60000 });
    let scans = 0;
    const orig = hub.sources.scan.bind(hub.sources);
    hub.sources.scan = async () => {
        scans++;
        return orig();
    };
    await Promise.all([hub.refresh("a"), hub.refresh("b"), hub.refresh("c"), hub.refresh("d")]);
    assert.ok(scans <= 2, `expected ≤2 scans, got ${scans}`);
    hub.close();
});

test("status() and summary() expose no absolute paths", async () => {
    const root = copyRepo("status");
    const hub = await new Hub(root, { env, now: () => T, watch: false, pollMs: 60000 }).start();
    try {
        const st = hub.status();
        assert.equal(st.repo, "status");
        assert.equal(st.heartbeatSec, 30);
        const s = JSON.stringify([st, hub.summary()]);
        assert.ok(!s.includes(tmp.replaceAll("\\", "\\\\")) && !s.includes(tmp), "no absolute paths");
    } finally {
        hub.close();
    }
});

test("heartbeat comes from CI_DASHBOARD_HEARTBEAT_SEC when set", () => {
    const hub = new Hub(tmp, { env: { CI_DASHBOARD_HEARTBEAT_SEC: "5" }, watch: false });
    assert.equal(hub.heartbeatSec, 5);
    hub.close();
});
