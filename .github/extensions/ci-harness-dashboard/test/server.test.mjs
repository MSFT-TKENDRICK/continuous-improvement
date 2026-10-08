import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { startInstanceServer, normalizeUiState } from "../lib/server.mjs";
import { Hub } from "../lib/hub.mjs";
import { AspireClient } from "../lib/aspire.mjs";
import { REPO, API_KEY, BROWSER_TOKEN, OTLP_KEY, mockAspire, writeState, scratch } from "./helpers.mjs";

const T = 1791435000;
const TRACE = "0af7651916cd43dd8448eb211c80319c";
const tmp = scratch("server");
let hub, aspireMock, srv;

before(async () => {
    const root = path.join(tmp, "repo");
    fs.cpSync(REPO, root, { recursive: true });
    hub = await new Hub(root, { env: { CI_IMPORTS_DIR: path.join(REPO, "..", "imports") }, now: () => T, watch: false, pollMs: 60000 }).start();
    aspireMock = await mockAspire();
    const aspire = new AspireClient({ env: { CI_DASHBOARD_STATE: writeState(tmp, aspireMock.url) } });
    srv = await startInstanceServer({ hub, aspire, instanceId: "inst-1", initialState: { view: "live" }, pingMs: 100 });
});
after(async () => {
    await srv?.close();
    hub?.close();
    await aspireMock?.close();
    fs.rmSync(tmp, { recursive: true, force: true });
});

/** Raw request so we control Host/Origin headers exactly. */
function request(method, p, { headers = {}, body } = {}) {
    return new Promise((resolve, reject) => {
        const req = http.request({ host: "127.0.0.1", port: srv.port, method, path: p, headers: { host: `127.0.0.1:${srv.port}`, ...headers } }, (res) => {
            const chunks = [];
            res.on("data", (c) => chunks.push(c));
            res.on("end", () => {
                const text = Buffer.concat(chunks).toString("utf8");
                let json = null;
                try {
                    json = JSON.parse(text);
                } catch {
                    /* not JSON */
                }
                resolve({ status: res.statusCode, headers: res.headers, text, json });
            });
        });
        req.on("error", reject);
        if (body !== undefined) req.write(typeof body === "string" ? body : JSON.stringify(body));
        req.end();
    });
}
const get = (p, headers) => request("GET", p, { headers });
const post = (p, body = {}, headers = {}) => request("POST", p, { body, headers: { "content-type": "application/json", "x-ci-token": srv.token, ...headers } });
const noSecrets = (text) => {
    for (const s of [API_KEY, BROWSER_TOKEN, OTLP_KEY]) assert.ok(!text.includes(s), `leaked ${s}`);
};

test("listens on loopback with an ephemeral port", () => {
    assert.ok(srv.port > 0);
    assert.equal(srv.url, `http://127.0.0.1:${srv.port}/`);
    assert.equal(srv.ui.view, "live");
});

test("rejects unexpected Host headers (DNS rebinding, C31)", async () => {
    for (const host of ["evil.example.com", `evil.example.com:${srv.port}`, "127.0.0.1", `127.0.0.1:${srv.port + 1}`, `0.0.0.0:${srv.port}`]) {
        const r = await get("/api/summary", { host });
        assert.equal(r.status, 421, host);
        assert.equal(r.json.error.code, "bad_host");
    }
    assert.equal((await get("/api/summary", { host: `localhost:${srv.port}` })).status, 200);
});

test("serves the UI with a strict CSP and the per-instance token", async () => {
    const r = await get("/");
    assert.equal(r.status, 200);
    const csp = r.headers["content-security-policy"];
    for (const d of ["default-src 'self'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "object-src 'none'", "base-uri 'none'"]) assert.ok(csp.includes(d), d);
    assert.ok(!csp.includes("unsafe-inline") && !csp.includes("unsafe-eval"));
    assert.equal(r.headers["x-content-type-options"], "nosniff");
    assert.equal(r.headers["cache-control"], "no-store");
    assert.ok(r.text.includes(`content="${srv.token}"`));
    assert.ok(!r.text.includes("__CI_TOKEN__"));
    assert.ok(!/<script(?![^>]*\bsrc=)[^>]*>/i.test(r.text), "no inline scripts");
    assert.ok(!/<style|style=/i.test(r.text), "no inline styles");
    for (const f of ["/app.js", "/app.css"]) {
        const s = await get(f);
        assert.equal(s.status, 200, f);
        assert.ok(s.headers["content-security-policy"]);
    }
    assert.match((await get("/app.js")).headers["content-type"], /javascript/);
    assert.equal((await get("/../lib/server.mjs")).status, 404);
    assert.equal((await get("/lib/server.mjs")).status, 404);
});

test("the UI never uses innerHTML-style sinks (C32)", () => {
    const js = fs.readFileSync(new URL("../ui/app.js", import.meta.url), "utf8");
    assert.ok(!/\.innerHTML|\.outerHTML|insertAdjacentHTML|document\.write\(|\beval\(|new Function\(/.test(js));
});

test("JSON API endpoints", async () => {
    const s = await get("/api/summary");
    assert.equal(s.status, 200);
    assert.equal(s.json.summary.campaigns[0].id, "tone-a1");
    assert.equal(s.json.imports.length, 1);
    assert.ok(s.headers["content-security-policy"]);

    const ex = await get("/api/experiments");
    assert.ok(ex.json.experiments.some((e) => e.id === "tone-a1-r01"));
    assert.ok((await get("/api/experiments?campaign=tone-a1")).json.experiments.every((e) => e.campaignId === "tone-a1"));

    const d = await get("/api/experiment/tone-a1-r01");
    assert.equal(d.status, 200);
    assert.equal(d.json.envelope.id, "tone-a1-r01");
    assert.equal((await get("/api/experiment/nope-r99")).status, 404);
    assert.equal((await get("/api/experiment/..%2F..%2Fetc")).status, 400);

    const live = await get("/api/live");
    const r02 = live.json.live.find((r) => r.experimentId === "tone-a1-r02");
    assert.ok(r02);
    assert.equal(r02.traceId, TRACE);

    const tr = await get("/api/traces");
    assert.ok(tr.json.traces.some((t) => t.traceId === TRACE));
    assert.equal(tr.json.aspire.included, true);
    const local = await get("/api/traces?aspire=0");
    assert.equal(local.json.aspire.included, false);

    const one = await get(`/api/trace/${TRACE}`);
    assert.equal(one.status, 200);
    assert.ok(one.json.trace.roots.length >= 1);
    assert.equal(one.json.aspireUrl, `${aspireMock.url}/traces/detail/${TRACE}`);
    assert.equal((await get("/api/trace/not-a-trace")).status, 400);
    assert.equal((await get("/api/trace/ffffffffffffffffffffffffffffffff")).status, 404);

    assert.ok((await get("/api/evals")).json.evals.length >= 1);
    assert.ok((await get("/api/sleep")).json.sleep.nights.length >= 1);
    const bus = (await get("/api/bus")).json.bus;
    assert.equal(bus.totals.exploits, 1);
    assert.deepEqual(bus.runs[0].counts.committed, 1);
    assert.equal(s.json.summary.bus.patches, 1);
    const asp = await get("/api/aspire");
    assert.equal(asp.json.aspire.reachable, true);
    assert.equal((await get("/api/nope")).status, 404);

    for (const p of ["/api/summary", "/api/live", "/api/traces", `/api/trace/${TRACE}`, "/api/aspire", "/api/evals", "/api/sleep", "/api/experiments"]) {
        const r = await get(p);
        noSecrets(r.text);
        assert.ok(!r.text.includes(srv.token), `token leaked via ${p}`);
    }
});

test("GenAI content attributes are hidden unless sensitive mode is declared (C29)", async () => {
    const one = await get(`/api/trace/${TRACE}`);
    const walk = (n, acc = []) => (acc.push(n), n.children.forEach((c) => walk(c, acc)), acc);
    const nodes = one.json.trace.roots.flatMap((r) => walk(r));
    assert.ok(nodes.some((n) => n.redacted > 0), "fixture contains redactable content");
    for (const n of nodes) {
        if (n.sensitive) continue;
        for (const k of Object.keys(n.attributes)) assert.ok(!/^gen_ai\.(input|output|prompt|completion|system_instructions|tool\.call\.(arguments|result))/.test(k), k);
    }
});

test("POST requires token, same origin and JSON", async () => {
    assert.equal((await post("/api/ui", { view: "evals" }, { "x-ci-token": "" })).status, 403);
    assert.equal((await post("/api/ui", { view: "evals" }, { "x-ci-token": "wrong" })).status, 403);
    assert.equal((await post("/api/ui", { view: "evals" }, { origin: "http://evil.example.com" })).status, 403);
    assert.equal((await post("/api/ui", { view: "evals" }, { "sec-fetch-site": "cross-site" })).status, 403);
    assert.equal((await post("/api/ui", "view=evals", { "content-type": "application/x-www-form-urlencoded" })).status, 415);
    assert.equal((await post("/api/ui", "x".repeat(17 * 1024))).status, 413);
    assert.equal((await post("/api/ui", "{bad")).status, 400);
    assert.equal((await post("/api/ui", { view: "bogus" })).status, 400);

    const ok = await post("/api/ui", { view: "evals" }, { origin: `http://127.0.0.1:${srv.port}`, "sec-fetch-site": "same-origin" });
    assert.equal(ok.status, 200);
    assert.equal(ok.json.ui.view, "evals");
    assert.equal(srv.ui.view, "evals");

    const ref = await post("/api/refresh");
    assert.equal(ref.status, 200);
    assert.ok(ref.json.version >= 2);
});

test("other methods are rejected", async () => {
    for (const m of ["PUT", "DELETE", "PATCH"]) {
        const r = await request(m, "/api/ui");
        assert.equal(r.status, 405, m);
    }
});

test("Aspire login URL is only handed out on an authenticated POST", async () => {
    assert.equal((await post("/api/aspire/login", { traceId: TRACE }, { "x-ci-token": "nope" })).status, 403);
    assert.equal((await post("/api/aspire/login", { traceId: "zz" })).status, 400);
    const r = await post("/api/aspire/login", { traceId: TRACE });
    assert.equal(r.status, 200);
    const u = new URL(r.json.url);
    assert.equal(u.searchParams.get("t"), BROWSER_TOKEN);
    assert.equal(u.searchParams.get("returnUrl"), `/traces/detail/${TRACE}`);
});

test("SSE: hello with UI state, ui pushes from actions, change notifications and pings", async () => {
    const events = [];
    let buf = "";
    let waiters = [];
    const req = http.get({ host: "127.0.0.1", port: srv.port, path: "/events", headers: { host: `127.0.0.1:${srv.port}` } });
    const res = await new Promise((r) => req.on("response", r));
    assert.equal(res.statusCode, 200);
    assert.match(res.headers["content-type"], /text\/event-stream/);
    assert.ok(res.headers["content-security-policy"]);
    res.setEncoding("utf8");
    res.on("data", (c) => {
        buf += c;
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
            const frame = buf.slice(0, i);
            buf = buf.slice(i + 2);
            const ev = /^event: (.+)$/m.exec(frame)?.[1];
            const data = /^data: (.+)$/m.exec(frame)?.[1];
            if (ev) events.push({ ev, data: data ? JSON.parse(data) : null });
            waiters = waiters.filter((w) => !w());
        }
    });
    const waitFor = (name, pred = () => true) =>
        new Promise((resolve, reject) => {
            const t = setTimeout(() => reject(new Error(`no ${name} event`)), 3000);
            const check = () => {
                const e = events.find((x) => x.ev === name && pred(x.data));
                if (e) {
                    clearTimeout(t);
                    resolve(e.data);
                    return true;
                }
                return false;
            };
            if (!check()) waiters.push(check);
        });
    try {
        const hello = await waitFor("hello");
        assert.ok(hello.ui.view);
        assert.equal(srv.clientCount(), 1);

        const pushed = srv.pushUi({ traceId: TRACE.toUpperCase() });
        assert.equal(pushed.view, "traces", "traceId implies the traces view");
        const ui = await waitFor("ui", (d) => d.traceId === TRACE);
        assert.equal(ui.seq, pushed.seq);
        assert.equal(srv.pushUi({ experimentId: "tone-a1-r01" }).view, "experiment");
        assert.throws(() => srv.pushUi({ view: "nope" }), RangeError);
        assert.throws(() => srv.pushUi({ traceId: "xyz" }), RangeError);

        await post("/api/refresh");
        const ch = await waitFor("changed", (d) => d.reason === "manual");
        assert.ok(ch.version >= 2);
        await waitFor("ping");
        for (const e of events) noSecrets(JSON.stringify(e));
    } finally {
        req.destroy();
    }
    const deadline = Date.now() + 3000;
    while (srv.clientCount() !== 0 && Date.now() < deadline) await new Promise((r) => setTimeout(r, 25));
    assert.equal(srv.clientCount(), 0);
});

test("normalizeUiState validates ids", () => {
    assert.deepEqual(normalizeUiState({ view: "sleep", campaignId: "", traceId: null }), { view: "sleep", campaignId: null, traceId: null });
    assert.throws(() => normalizeUiState({ experimentId: "../x" }), RangeError);
    assert.throws(() => normalizeUiState({ campaignId: "a b" }), RangeError);
    assert.deepEqual(normalizeUiState(null), {});
});

test("503 until the hub has a model; close() is clean", async () => {
    const cold = new Hub(path.join(tmp, "repo"), { watch: false });
    const s = await startInstanceServer({ hub: cold });
    try {
        const r = await new Promise((resolve) =>
            http.get({ host: "127.0.0.1", port: s.port, path: "/api/summary", headers: { host: `localhost:${s.port}` } }, (res) => {
                res.resume();
                resolve(res.statusCode);
            }),
        );
        assert.equal(r, 503);
    } finally {
        await s.close();
        cold.close();
    }
    assert.equal(cold.listenerCount("change"), 0);
});
