// /agui proxy, chat status/start routes and chat bundle serving. The fake upstream replays the
// recorded AG-UI exchange from layer 32 (fixtures/chat/approval_flow.json; refresh it by copying
// tests/fixtures/chat/approval_flow.json from the Python side after re-running its recorder).
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { startInstanceServer, AGUI_BODY_MAX } from "../lib/server.mjs";
import { ChatUnavailable } from "../lib/chat.mjs";
import { Hub } from "../lib/hub.mjs";
import { FIX, REPO, scratch } from "./helpers.mjs";

const FLOW = JSON.parse(fs.readFileSync(path.join(FIX, "chat", "approval_flow.json"), "utf8"));
const UP_TOKEN = "upstream-chat-token-SECRET-0001";
const tmp = scratch("chatproxy");
let hub, srv, upstream;
const seen = [];
let mode = "replay";
let release = null;
let upstreamClosed = null;
let ensureImpl;

const chat = {
    ensure: () => ensureImpl(),
    status: () => ({ state: "ready", pid: 1, port: upstream.address().port, profile: "fake", lastError: null }),
};

before(async () => {
    const root = path.join(tmp, "repo");
    fs.cpSync(REPO, root, { recursive: true });
    hub = await new Hub(root, { env: {}, now: () => 1791435000, watch: false, pollMs: 60000 }).start();
    upstream = http.createServer((req, res) => {
        const chunks = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
            seen.push({ headers: req.headers, body: Buffer.concat(chunks).toString("utf8") });
            if (req.headers["x-ci-chat-token"] !== UP_TOKEN) {
                res.writeHead(401, { "content-type": "application/json" });
                return res.end('{"detail":"missing or bad x-ci-chat-token"}');
            }
            if (req.headers.origin) {
                res.writeHead(403, { "content-type": "application/json" });
                return res.end("{}");
            }
            res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "set-cookie": "a=b", "x-upstream": "1" });
            const body = JSON.parse(seen.at(-1).body);
            const raw = JSON.stringify(body);
            const step = Object.keys(FLOW.steps).find((k) => JSON.stringify(FLOW.steps[k].request.body) === raw) ?? (body.resume ? "3_resume_approved" : "2_launch_interrupt_approved");
            const events = FLOW.steps[step].events;
            if (mode === "hold") {
                res.write(`data: ${JSON.stringify(events[0])}\n\n`);
                req.socket.on("close", () => upstreamClosed?.());
                res.on("close", () => upstreamClosed?.());
                release = () => {
                    for (const e of events.slice(1)) res.write(`data: ${JSON.stringify(e)}\n\n`);
                    res.end();
                };
                return;
            }
            res.write(": keepalive\n\n");
            for (const e of events) res.write(`data: ${JSON.stringify(e)}\n\n`);
            res.end();
        });
    });
    await new Promise((r) => upstream.listen(0, "127.0.0.1", r));
    ensureImpl = async () => ({ host: "127.0.0.1", port: upstream.address().port, path: "/agui", token: UP_TOKEN });
    srv = await startInstanceServer({ hub, chat, instanceId: "chat-1", pingMs: 60000 });
});
after(async () => {
    await srv?.close();
    hub?.close();
    await new Promise((r) => upstream.close(r));
    fs.rmSync(tmp, { recursive: true, force: true });
});

function request(method, p, { headers = {}, body, onResponse } = {}) {
    return new Promise((resolve, reject) => {
        const req = http.request({ host: "127.0.0.1", port: srv.port, method, path: p, headers: { host: `127.0.0.1:${srv.port}`, ...headers } }, (res) => {
            if (onResponse) return onResponse(res, req, resolve);
            const chunks = [];
            res.on("data", (c) => chunks.push(c));
            res.on("end", () => {
                const text = Buffer.concat(chunks).toString("utf8");
                let json = null;
                try {
                    json = JSON.parse(text);
                } catch {
                    /* SSE or text */
                }
                resolve({ status: res.statusCode, headers: res.headers, text, json });
            });
        });
        req.on("error", reject);
        if (body !== undefined) req.write(typeof body === "string" || Buffer.isBuffer(body) ? body : JSON.stringify(body));
        req.end();
    });
}
const auth = () => ({ "content-type": "application/json", "x-ci-token": srv.token });
const agui = (body, headers = {}) => request("POST", "/agui", { body, headers: { ...auth(), accept: "text/event-stream", ...headers } });
const sse = (text) => text.split("\n\n").map((f) => f.trim()).filter((f) => f.startsWith("data: ")).map((f) => JSON.parse(f.slice(6)));
const launchBody = () => FLOW.steps["2_launch_interrupt_approved"].request.body;
const resumeBody = () => FLOW.steps["3_resume_approved"].request.body;

test("fixture is the canonical MAF approval exchange the bundle codes against", () => {
    const last = FLOW.steps["2_launch_interrupt_approved"].events.at(-1);
    assert.equal(last.type, "RUN_FINISHED");
    assert.equal(last.outcome.type, "interrupt");
    const intr = last.outcome.interrupts[0];
    assert.equal(intr.reason, "tool_call");
    assert.equal(intr.metadata.agent_framework.function_call.name, "launch_campaign");
    assert.deepEqual(Object.keys(resumeBody().resume[0]).sort(), ["interruptId", "payload", "status"]);
    assert.equal(resumeBody().resume[0].payload.approved, true);
});

test("/agui streams the upstream SSE through and strips Origin, Cookie and the canvas token", async () => {
    seen.length = 0;
    const r = await agui(launchBody(), { origin: `http://127.0.0.1:${srv.port}`, cookie: "session=abc", "sec-fetch-site": "same-origin" });
    assert.equal(r.status, 200);
    assert.match(r.headers["content-type"], /^text\/event-stream/);
    assert.match(r.headers["content-security-policy"], /connect-src 'self'/);
    assert.equal(r.headers["set-cookie"], undefined, "upstream headers are not forwarded");
    assert.equal(r.headers["x-upstream"], undefined);
    const events = sse(r.text);
    assert.deepEqual(events, FLOW.steps["2_launch_interrupt_approved"].events);
    const up = seen.at(-1);
    assert.equal(up.headers["x-ci-chat-token"], UP_TOKEN);
    for (const h of ["origin", "cookie", "x-ci-token", "sec-fetch-site"]) assert.equal(up.headers[h], undefined, `${h} forwarded`);
    assert.deepEqual(JSON.parse(up.body), launchBody());
    assert.ok(!r.text.includes(UP_TOKEN));
});

test("/agui resume run carries the canonical resume array upstream", async () => {
    seen.length = 0;
    const r = await agui(resumeBody());
    assert.equal(r.status, 200);
    assert.ok(sse(r.text).some((e) => e.type === "TOOL_CALL_RESULT"));
    assert.deepEqual(JSON.parse(seen.at(-1).body).resume, resumeBody().resume);
});

test("every recorded step (draft, approve, reject, legacy) round-trips byte-for-byte through the proxy", async () => {
    for (const [name, step] of Object.entries(FLOW.steps)) {
        seen.length = 0;
        const r = await agui(step.request.body);
        assert.equal(r.status, 200, name);
        assert.deepEqual(sse(r.text), step.events, name);
        assert.deepEqual(JSON.parse(seen[0].body), step.request.body, name);
        assert.equal(seen[0].headers.origin, undefined, name);
        const types = step.events.map((e) => e.type);
        if (name.startsWith("2_")) {
            assert.ok(step.events.some((e) => e.type === "CUSTOM" && e.name === "function_approval_request"), name);
            assert.equal(step.events.at(-1).outcome?.type, "interrupt", name);
        }
        if (name.startsWith("3_resume_approved")) {
            const firstResult = types.indexOf("TOOL_CALL_RESULT");
            assert.ok(firstResult >= 0 && types.slice(0, firstResult).every((t) => t === "RUN_STARTED" || t === "STATE_SNAPSHOT" || t === "MESSAGES_SNAPSHOT"), `${name}: result first`);
        }
        if (name.startsWith("3_")) {
            const prev = FLOW.steps[name.replace("3_resume", "2_launch_interrupt")];
            const b = step.request.body;
            assert.equal(b.threadId, prev.request.body.threadId, `${name}: same thread`);
            assert.notEqual(b.runId, prev.request.body.runId, `${name}: new run`);
            assert.deepEqual(b.messages, prev.events.findLast((e) => e.type === "MESSAGES_SNAPSHOT").messages, `${name}: messages from snapshot`);
            assert.deepEqual(b.state, prev.events.findLast((e) => e.type === "STATE_SNAPSHOT").snapshot, `${name}: state from snapshot`);
            assert.equal(b.resume[0].interruptId, prev.events.at(-1).outcome.interrupts[0].id, `${name}: interrupt id`);
        }
        if (name.startsWith("3_resume_rejected")) assert.ok(!types.includes("TOOL_CALL_RESULT"), `${name}: no launch`);
    }
});

test("/agui enforces Host, Origin, sec-fetch-site and the canvas token like every POST", async () => {
    seen.length = 0;
    assert.equal((await request("POST", "/agui", { body: launchBody(), headers: { "content-type": "application/json" } })).status, 403);
    assert.equal((await agui(launchBody(), { "x-ci-token": "nope" })).status, 403);
    assert.equal((await agui(launchBody(), { origin: "http://evil.example" })).status, 403);
    assert.equal((await agui(launchBody(), { origin: "null" })).status, 403);
    assert.equal((await agui(launchBody(), { "sec-fetch-site": "cross-site" })).status, 403);
    assert.equal((await agui(launchBody(), { host: "evil.example" })).status, 421);
    assert.equal((await request("GET", "/agui")).status, 404);
    assert.equal(seen.length, 0, "nothing reached the backend");
});

test("/agui accepts bodies up to 1 MiB (other POSTs keep 16 KB), rejects larger and non-JSON", async () => {
    const big = { ...launchBody(), forwardedProps: { pad: "x".repeat(200 * 1024) } };
    assert.equal((await agui(big)).status, 200);
    assert.equal((await request("POST", "/api/ui", { body: { view: "chat", pad: "x".repeat(20 * 1024) }, headers: auth() })).status, 413);
    assert.equal((await agui({ pad: "x".repeat(AGUI_BODY_MAX + 10) })).status, 413);
    assert.equal((await agui("view=chat", { "content-type": "text/plain" })).status, 415);
    assert.equal((await agui("{nope")).status, 400);
    assert.equal((await agui("[1,2]")).status, 400);
});

test("/agui is unbuffered and a client abort cancels the upstream request", async () => {
    mode = "hold";
    const closed = new Promise((r) => (upstreamClosed = r));
    const first = await request("POST", "/agui", {
        body: launchBody(),
        headers: { ...auth(), accept: "text/event-stream" },
        onResponse: (res, req, resolve) => {
            res.once("data", (chunk) => {
                resolve(String(chunk));
                req.destroy();
            });
        },
    });
    assert.equal(sse(first)[0].type, "RUN_STARTED", "first event arrives while upstream is still open");
    await Promise.race([closed, new Promise((_, rej) => setTimeout(() => rej(new Error("upstream not cancelled")), 2000))]);
    release = null;
    mode = "replay";
});

test("/agui returns 503 JSON when the backend cannot start, 502 when it refuses the proxy", async () => {
    ensureImpl = async () => {
        throw new ChatUnavailable("chat_command_not_found", "cannot find uv on PATH (install uv, or set CI_CHAT_COMMAND)");
    };
    let r = await agui(launchBody());
    assert.equal(r.status, 503);
    assert.equal(r.json.error.code, "chat_command_not_found");
    assert.match(r.json.error.message, /install uv/);
    ensureImpl = async () => {
        throw new ChatUnavailable("chat_restarting", "retrying", 4200);
    };
    r = await agui(launchBody());
    assert.equal(r.status, 503);
    assert.equal(r.headers["retry-after"], "5");
    ensureImpl = async () => ({ host: "127.0.0.1", port: upstream.address().port, path: "/agui", token: "wrong-token-but-long-enough" });
    r = await agui(launchBody());
    assert.equal(r.status, 502);
    assert.equal(r.json.error.code, "chat_upstream_auth");
    assert.ok(!r.text.includes("wrong-token"));
    ensureImpl = async () => ({ host: "127.0.0.1", port: 1, path: "/agui", token: UP_TOKEN });
    r = await agui(launchBody());
    assert.equal(r.status, 502);
    ensureImpl = async () => ({ host: "127.0.0.1", port: upstream.address().port, path: "/agui", token: UP_TOKEN });
});

test("chat status/start routes expose no secrets; a canvas without a backend answers 503", async () => {
    const st = await request("GET", "/api/chat/status");
    assert.equal(st.status, 200);
    assert.equal(st.json.chat.state, "ready");
    assert.ok(!st.text.includes(UP_TOKEN));
    assert.equal((await request("POST", "/api/chat/start", { body: {}, headers: { "content-type": "application/json" } })).status, 403);
    const started = await request("POST", "/api/chat/start", { body: {}, headers: auth() });
    assert.equal(started.status, 200);
    assert.ok(!started.text.includes(UP_TOKEN));
    const bare = await startInstanceServer({ hub, instanceId: "no-chat", pingMs: 60000 });
    try {
        const r = await new Promise((resolve, reject) => {
            const req = http.request({ host: "127.0.0.1", port: bare.port, method: "POST", path: "/agui", headers: { host: `127.0.0.1:${bare.port}`, "content-type": "application/json", "x-ci-token": bare.token } }, (res) => {
                const c = [];
                res.on("data", (d) => c.push(d));
                res.on("end", () => resolve({ status: res.statusCode, json: JSON.parse(Buffer.concat(c).toString()) }));
            });
            req.on("error", reject);
            req.end("{}");
        });
        assert.equal(r.status, 503);
        assert.equal(r.json.error.code, "chat_unavailable");
    } finally {
        await bare.close();
    }
});

test("chat bundle files are served with JS/CSS types from ui/chat; traversal and unknown names 404", async () => {
    const js = await request("GET", "/chat/chat.js");
    assert.equal(js.status, 200);
    assert.match(js.headers["content-type"], /^text\/javascript/);
    assert.match(js.text, /export\s*\{/);
    const css = await request("GET", "/chat/chat.css");
    assert.equal(css.status, 200);
    assert.match(css.headers["content-type"], /^text\/css/);
    assert.match((await request("GET", "/chat/THIRD_PARTY_LICENSES.txt")).headers["content-type"], /^text\/plain/);
    for (const p of ["/chat/../app.js", "/chat/..%2fapp.js", "/chat/%2e%2e/lib/server.mjs", "/chat/sub/x.js", "/chat/missing.js", "/chat/chat.mjs", "/chat/.hidden.js", "/chat/x.js%00.css", "/chat/"]) {
        const r = await request("GET", p);
        if (p === "/chat/../app.js") {
            // URL parsing collapses dot segments to /app.js: still only the allowlisted UI file.
            assert.ok(r.status === 200 && /text\/javascript/.test(r.headers["content-type"]), p);
            continue;
        }
        assert.equal(r.status, 404, p);
    }
});
