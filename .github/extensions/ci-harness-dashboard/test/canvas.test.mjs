import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { createDashboard, findRepoRoot, OPEN_INPUT_SCHEMA, CANVAS_ID } from "../lib/canvas.mjs";
import { activeHubs } from "../lib/hub.mjs";
import { REPO, FIX, API_KEY, BROWSER_TOKEN, OTLP_KEY, mockAspire, writeState, scratch } from "./helpers.mjs";

class FakeCanvasError extends Error {
    constructor(code, message) {
        super(message);
        this.code = code;
    }
}

const TRACE = "0af7651916cd43dd8448eb211c80319c";
const tmp = scratch("canvas");
const root = path.join(tmp, "repo");
const other = path.join(tmp, "other");
let mock, dash;
const logs = [];

before(async () => {
    fs.cpSync(REPO, root, { recursive: true });
    fs.mkdirSync(path.join(root, ".git"));
    fs.mkdirSync(path.join(other, ".git"), { recursive: true });
    mock = await mockAspire();
    const env = { CI_IMPORTS_DIR: path.join(FIX, "imports"), CI_DASHBOARD_STATE: writeState(tmp, mock.url) };
    dash = createDashboard({ CanvasError: FakeCanvasError, env, log: (m, l) => logs.push([l, m]), hubOptions: { watch: false, pollMs: 60000 } });
});
after(async () => {
    await dash?.closeAll();
    await mock?.close();
    fs.rmSync(tmp, { recursive: true, force: true });
});

const ctx = (instanceId, extra = {}) => ({ sessionId: "s1", extensionId: "project:ci-harness-dashboard", canvasId: CANVAS_ID, instanceId, session: { workingDirectory: path.join(root, "experiments") }, ...extra });
const action = (name, instanceId, input) => dash.actions.find((a) => a.name === name).handler({ ...ctx(instanceId), actionName: name, input });
const getJson = (url) =>
    new Promise((resolve, reject) => {
        http.get(url, (res) => {
            let b = "";
            res.on("data", (c) => (b += c));
            res.on("end", () => resolve(JSON.parse(b)));
        }).on("error", reject);
    });
const noSecrets = (v) => {
    const s = JSON.stringify(v);
    for (const x of [API_KEY, BROWSER_TOKEN, OTLP_KEY]) assert.ok(!s.includes(x), `leaked ${x}`);
};

test("declares the open schema and the seven actions", () => {
    assert.equal(OPEN_INPUT_SCHEMA.additionalProperties, false);
    assert.deepEqual(Object.keys(OPEN_INPUT_SCHEMA.properties), ["repoRoot", "view", "campaignId", "experimentId", "traceId"]);
    assert.deepEqual(
        dash.actions.map((a) => a.name),
        ["refresh", "show_view", "select_campaign", "focus_experiment", "focus_trace", "get_summary", "dashboard_status"],
    );
    for (const a of dash.actions) {
        assert.ok(!a.name.startsWith("canvas."));
        assert.equal(a.inputSchema.additionalProperties, false, a.name);
        assert.equal(typeof a.handler, "function");
        assert.ok(a.description.length > 10);
    }
});

test("findRepoRoot walks up to the directory containing .git", async () => {
    assert.equal(await findRepoRoot(path.join(root, "experiments", "campaigns")), root);
});

test("open resolves the repo from the session working directory and is idempotent", async () => {
    const r1 = await dash.open(ctx("a"));
    assert.match(r1.url, /^http:\/\/127\.0\.0\.1:\d+\/$/);
    assert.match(r1.title, /CI Harness Dashboard — repo/);
    assert.match(r1.status, /experiments/);
    const hubs = activeHubs();
    const r2 = await dash.open(ctx("a"));
    assert.equal(r2.url, r1.url, "re-open reuses the server");
    assert.equal(activeHubs(), hubs);

    const r3 = await dash.open(ctx("b", { input: { view: "traces" } }));
    assert.notEqual(r3.url, r1.url, "each instance gets its own server");
    assert.equal(activeHubs(), hubs, "instances share the repo hub");
    const ui = await getJson(`${r3.url}api/summary`);
    assert.equal(ui.ui.view, "traces");

    // Re-open with different input pushes UI state; same input does not.
    const before = (await dash.instances.get("b")).server.ui.seq;
    await dash.open(ctx("b", { input: { view: "traces" } }));
    assert.equal((await dash.instances.get("b")).server.ui.seq, before);
    await dash.open(ctx("b", { input: { experimentId: "tone-a1-r01" } }));
    const after = (await dash.instances.get("b")).server.ui;
    assert.equal(after.view, "experiment");
    assert.equal(after.experimentId, "tone-a1-r01");
});

test("open rejects bad input with CanvasError codes", async () => {
    const codeOf = async (input) => {
        try {
            await dash.open(ctx("bad", { input }));
            return "ok";
        } catch (e) {
            assert.ok(e instanceof FakeCanvasError);
            return e.code;
        }
    };
    assert.equal(await codeOf({ view: "nope" }), "invalid_input");
    assert.equal(await codeOf({ extra: 1 }), "invalid_input");
    assert.equal(await codeOf({ traceId: "xyz" }), "invalid_input");
    assert.equal(await codeOf({ repoRoot: "relative/path" }), "invalid_input");
    assert.equal(await codeOf({ repoRoot: path.join(tmp, "does-not-exist") }), "repo_not_found");
    assert.equal(await codeOf([1, 2]), "invalid_input");
    assert.equal(dash.instances.has("bad"), false);
});

test("changing repoRoot recreates the instance", async () => {
    const r1 = await dash.open(ctx("c"));
    const r2 = await dash.open(ctx("c", { input: { repoRoot: other } }));
    assert.notEqual(r2.url, r1.url);
    assert.match(r2.title, /— other$/);
    // An explicit repoRoot is used as given (no walk-up to an enclosing .git).
    const sub = path.join(other, "sub");
    fs.mkdirSync(sub, { recursive: true });
    const r3 = await dash.open(ctx("c", { input: { repoRoot: sub } }));
    assert.match(r3.title, /— sub$/);
    await dash.onClose(ctx("c"));
});

test("UI actions push state to the instance and report what they found", async () => {
    await dash.open(ctx("d"));
    const v = await action("show_view", "d", { view: "evals" });
    assert.equal(v.ui.view, "evals");
    const c = await action("select_campaign", "d", { campaignId: "tone-a1" });
    assert.equal(c.ui.view, "experiment");
    assert.equal(c.known, true);
    assert.equal((await action("select_campaign", "d", { campaignId: "zzz" })).known, false);
    const e = await action("focus_experiment", "d", { experimentId: "tone-a1-r01" });
    assert.equal(e.found, true);
    assert.equal(e.summary.id, "tone-a1-r01");
    const t = await action("focus_trace", "d", { traceId: TRACE.toUpperCase() });
    assert.equal(t.ui.view, "traces");
    assert.equal(t.ui.traceId, TRACE);
    assert.equal(t.foundLocally, true);
    const r = await action("refresh", "d", {});
    assert.equal(r.ok, true);
    assert.ok(r.version >= 2);
    for (const [name, input] of [
        ["show_view", { view: "bogus" }],
        ["select_campaign", { campaignId: "../x" }],
        ["focus_experiment", {}],
        ["focus_trace", { traceId: "123" }],
    ]) {
        await assert.rejects(() => action(name, "d", input), (err) => err.code === "invalid_input", name);
    }
    await assert.rejects(() => action("show_view", "not-open", { view: "live" }), (err) => err.code === "canvas_not_open");
});

test("get_summary and dashboard_status work with or without an open instance and leak nothing", async () => {
    const s = await action("get_summary", "d", {});
    assert.equal(s.campaigns[0].id, "tone-a1");
    const s2 = await action("get_summary", "closed-instance", {});
    assert.equal(s2.campaigns[0].id, "tone-a1");

    const st = await action("dashboard_status", "d", {});
    assert.equal(st.open, true);
    assert.equal(st.aspire.reachable, true);
    assert.equal(st.hub.repo, "repo");
    noSecrets(st);
    const inst = await dash.instances.get("d");
    assert.ok(!JSON.stringify(st).includes(inst.server.token), "instance token never returned");
    noSecrets(s);
    const st2 = await action("dashboard_status", "closed-instance", {});
    assert.equal(st2.open, false);
    assert.equal(st2.url, null);
});

test("onClose stops the server and releases the hub", async () => {
    const r = await dash.open(ctx("e"));
    await dash.onClose(ctx("e"));
    assert.equal(dash.instances.has("e"), false);
    await assert.rejects(() => getJson(`${r.url}api/summary`));
    await dash.onClose(ctx("e")); // idempotent
});
