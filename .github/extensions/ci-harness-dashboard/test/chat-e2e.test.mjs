// Opt-in end-to-end test against the real Python backend (`ci-lab chat serve --profile fake
// --dry-run-launch`). Skipped unless CI_CHAT_E2E=1. CI_CHAT_E2E_REPO selects the checkout that
// provides `ci_lab.chat` (defaults to this repository); it needs a synced uv environment.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ChatBackend } from "../lib/chat.mjs";
import { startInstanceServer } from "../lib/server.mjs";
import { Hub } from "../lib/hub.mjs";
import { scratch } from "./helpers.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = process.env.CI_CHAT_E2E_REPO ? path.resolve(process.env.CI_CHAT_E2E_REPO) : path.resolve(HERE, "..", "..", "..", "..");
const enabled = process.env.CI_CHAT_E2E === "1";

function post(srv, body) {
    return new Promise((resolve, reject) => {
        const req = http.request(
            {
                host: "127.0.0.1",
                port: srv.port,
                method: "POST",
                path: "/agui",
                headers: { host: `127.0.0.1:${srv.port}`, origin: `http://127.0.0.1:${srv.port}`, "content-type": "application/json", accept: "text/event-stream", "x-ci-token": srv.token },
            },
            (res) => {
                let text = "";
                res.setEncoding("utf8");
                res.on("data", (c) => (text += c));
                res.on("end", () => {
                    const events = text
                        .split(/\r?\n\r?\n/)
                        .map((f) => f.split(/\r?\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n"))
                        .filter(Boolean)
                        .map((d) => JSON.parse(d));
                    resolve({ status: res.statusCode, events, text });
                });
            },
        );
        req.on("error", reject);
        req.end(JSON.stringify(body));
    });
}
const last = (events, type) => events.findLast((e) => e.type === type);

test("real backend: draft, launch interrupt, approve resume (dry run) through the canvas proxy", { skip: !enabled && "set CI_CHAT_E2E=1 (needs uv and ci_lab.chat)", timeout: 180000 }, async () => {
    const tmp = scratch("chate2e");
    const runDir = path.join(tmp, "runs");
    const logs = [];
    const env = {
        ...process.env,
        CI_CHAT_COMMAND: JSON.stringify(["uv", "run", "--no-sync", "ci-lab", "chat", "serve", "--profile", "fake", "--dry-run-launch", "--run-dir", runDir, "--chat-dir", path.join(tmp, "chat"), "--ledger-dir", path.join(tmp, "ledger")]),
    };
    const chat = new ChatBackend({ repoRoot: REPO, env, log: (m) => logs.push(m), options: { listenTimeoutMs: 120000 } });
    chat.refs = 1;
    fs.mkdirSync(path.join(tmp, "repo"), { recursive: true });
    const hub = await new Hub(path.join(tmp, "repo"), { env: {}, watch: false, pollMs: 60000 }).start();
    const srv = await startInstanceServer({ hub, chat, instanceId: "e2e", pingMs: 60000 });
    try {
        const r1 = await post(srv, { threadId: "e2e-thread", runId: "e2e-1", messages: [{ id: "u1", role: "user", content: "Please draft chat-demo: 2 arms, 1 round, local." }], state: {}, tools: [], context: [], forwardedProps: {} });
        assert.equal(r1.status, 200, r1.text.slice(0, 300));
        assert.equal(chat.status().state, "ready");
        const draft = last(r1.events, "STATE_SNAPSHOT")?.snapshot?.draft;
        assert.equal(draft?.cid, "chat-demo");

        const msgs1 = last(r1.events, "MESSAGES_SNAPSHOT").messages;
        const r2 = await post(srv, { threadId: "e2e-thread", runId: "e2e-2", messages: [...msgs1, { id: "u2", role: "user", content: "launch chat-demo" }], state: last(r1.events, "STATE_SNAPSHOT").snapshot, tools: [], context: [], forwardedProps: {} });
        assert.equal(r2.status, 200);
        assert.ok(r2.events.some((e) => e.type === "CUSTOM" && e.name === "function_approval_request"));
        const outcome = last(r2.events, "RUN_FINISHED").outcome;
        assert.equal(outcome.type, "interrupt");
        const interrupt = outcome.interrupts[0];

        const r3 = await post(srv, {
            threadId: "e2e-thread",
            runId: "e2e-3",
            messages: last(r2.events, "MESSAGES_SNAPSHOT").messages,
            state: last(r2.events, "STATE_SNAPSHOT").snapshot,
            tools: [],
            context: [],
            forwardedProps: {},
            resume: [{ interruptId: interrupt.id, status: "resolved", payload: { approved: true } }],
        });
        assert.equal(r3.status, 200);
        assert.ok(r3.events.some((e) => e.type === "TOOL_CALL_RESULT"), "approved resume runs launch_campaign");
        const launches = last(r3.events, "STATE_SNAPSHOT")?.snapshot?.launches ?? [];
        assert.ok(launches.some((l) => l.cid === "chat-demo"));
        assert.ok(!JSON.stringify(chat.status()).includes(chat.token ?? "\0"));
        assert.ok(!logs.join("\n").includes(chat.token ?? "\0"));
    } finally {
        await srv.close();
        hub.close();
        await chat.stop();
        fs.rmSync(tmp, { recursive: true, force: true });
    }
});
