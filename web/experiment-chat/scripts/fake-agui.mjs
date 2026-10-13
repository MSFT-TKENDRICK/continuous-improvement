#!/usr/bin/env node
// Fake `ci-lab chat serve` for browser checks: honours the process contract (CI_CHAT_TOKEN,
// loopback, one JSON listening line on stdout, exit on stdin EOF, 403 on Origin, 401 on bad token)
// and replays layer 32's recorded agent_framework_ag_ui exchange.
//   CI_CHAT_COMMAND='["node","web/experiment-chat/scripts/fake-agui.mjs"]'
import fs from "node:fs";
import http from "node:http";
import { fileURLToPath } from "node:url";

const FIXTURE = fileURLToPath(new URL("../../../.github/extensions/ci-harness-dashboard/test/fixtures/chat/approval_flow.json", import.meta.url));
const FLOW = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
const TOKEN = process.env.CI_CHAT_TOKEN ?? "";
if (TOKEN.length < 16) {
    process.stderr.write(JSON.stringify({ error: "CI_CHAT_TOKEN missing or shorter than 16 chars" }) + "\n");
    process.exit(2);
}

const userText = (m) => (typeof m?.content === "string" ? m.content : Array.isArray(m?.content) ? m.content.map((p) => p?.text ?? "").join(" ") : "");

function pickStep(body) {
    const r = Array.isArray(body.resume) ? body.resume[0] : null;
    if (r) {
        const approved = r.status !== "cancelled" && (r.payload?.approved ?? r.approved) === true;
        return approved ? "3_resume_approved" : "3_resume_rejected";
    }
    const last = [...(body.messages ?? [])].reverse().find((m) => m.role === "user");
    return /\blaunch\b/i.test(userText(last)) ? "2_launch_interrupt_approved" : "1_draft";
}

// The recording used fixed thread/run ids; rewrite them to the caller's so clients can match runs.
// Snapshots carry the recorded history; splice in the caller's own messages (their ids differ) and
// keep only what the recorded turn added after its last user message.
function retarget(event, body) {
    const e = structuredClone(event);
    if ("threadId" in e) e.threadId = body.threadId;
    if ("runId" in e) e.runId = body.runId;
    if (e.type === "MESSAGES_SNAPSHOT" && Array.isArray(e.messages)) {
        const mine = body.messages ?? [];
        const known = new Set(mine.map((m) => m.id));
        const lastUser = e.messages.map((m) => m.role).lastIndexOf("user");
        e.messages = [...mine, ...e.messages.slice(lastUser + 1).filter((m) => !known.has(m.id))];
    }
    return e;
}

const server = http.createServer((req, res) => {
    if (req.headers.origin) {
        res.writeHead(403, { "content-type": "application/json" });
        return res.end('{"detail":"browser origins are not allowed"}');
    }
    if (req.method === "GET" && req.url === "/healthz") {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end('{"ok":true}');
    }
    if (req.method !== "POST" || req.url !== "/agui") {
        res.writeHead(404);
        return res.end();
    }
    if (req.headers["x-ci-chat-token"] !== TOKEN) {
        res.writeHead(401, { "content-type": "application/json" });
        return res.end('{"detail":"missing or bad x-ci-chat-token"}');
    }
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
        let body;
        try {
            body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        } catch {
            res.writeHead(422);
            return res.end();
        }
        const step = pickStep(body);
        process.stderr.write(`fake-agui: ${step}\n`);
        res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" });
        for (const event of FLOW.steps[step].events) {
            res.write(`data: ${JSON.stringify(retarget(event, body))}\n\n`);
            await new Promise((r) => setTimeout(r, 15));
        }
        res.end();
    });
});

server.listen(0, "127.0.0.1", () => {
    process.stdout.write(JSON.stringify({ event: "listening", host: "127.0.0.1", port: server.address().port, path: "/agui" }) + "\n");
});
process.stdin.resume();
process.stdin.on("end", () => server.close(() => process.exit(0)));
process.stdin.on("error", () => process.exit(0));
