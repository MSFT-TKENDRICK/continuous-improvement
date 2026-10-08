import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import path from "node:path";
import { ChatBackend, ChatUnavailable, acquireChat, releaseChat, activeChats, chatCommand, chatRunDir, parseListening, resolveExecutable } from "../lib/chat.mjs";

const ROOT = path.resolve("repo-root-for-chat-tests");
const LISTEN = '{"event":"listening","host":"127.0.0.1","port":43210,"path":"/agui"}\n';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Child-process stand-in: the test drives stdout/stderr/exit; kill() and stdin EOF exit it. */
class FakeChild extends EventEmitter {
    constructor(pid, { exitOnEof = true } = {}) {
        super();
        this.pid = pid;
        this.exitCode = null;
        this.killed = false;
        this.stdinEnded = false;
        this.stdout = new PassThrough();
        this.stderr = new PassThrough();
        this.stdin = new PassThrough();
        this.stdin.on("finish", () => {
            this.stdinEnded = true;
            if (exitOnEof) setImmediate(() => this.exit(0));
        });
        this.stdin.resume();
    }
    kill() {
        this.killed = true;
        setImmediate(() => this.exit(null, "SIGTERM"));
        return true;
    }
    exit(code, signal = null) {
        if (this.exitCode !== null || this.exited) return;
        this.exited = true;
        this.exitCode = code ?? 1;
        this.emit("exit", code, signal);
    }
}

function harness({ env = {}, options = {}, onSpawn, platform = "linux" } = {}) {
    const calls = [];
    const logs = [];
    const spawn = (cmd, args, opts) => {
        const child = new FakeChild(1000 + calls.length);
        calls.push({ cmd, args, opts, child });
        onSpawn?.(child, calls.length);
        return child;
    };
    const backend = new ChatBackend({
        repoRoot: ROOT,
        env: { PATH: "", ...env },
        log: (m, level) => logs.push({ m, level }),
        spawn,
        platform,
        options: { listenTimeoutMs: 500, backoffBaseMs: 20, backoffMaxMs: 200, stopGraceMs: 50, ...options },
    });
    return { backend, calls, logs };
}
const listenSoon = (child) => setImmediate(() => child.stdout.write(LISTEN));

test("default command: uv run ci-lab chat serve with the profile and the hub's run root", () => {
    const run = chatRunDir(ROOT, {});
    assert.equal(run, path.join(ROOT, "artifacts", "ci-runs"));
    assert.equal(chatRunDir(ROOT, { CI_RUN_DIR: "runs/x" }), path.join(ROOT, "runs", "x"));
    assert.deepEqual(chatCommand({}, run).argv, ["uv", "run", "--no-sync", "ci-lab", "chat", "serve", "--profile", "copilot", "--run-dir", run]);
    assert.equal(chatCommand({ CI_CHAT_PROFILE: "fake" }, run).argv[7], "fake");
    assert.match(chatCommand({ CI_CHAT_PROFILE: "evil --x" }, run).error, /CI_CHAT_PROFILE/);
    assert.deepEqual(chatCommand({ CI_CHAT_COMMAND: '["node","fake.mjs"]' }, run), { argv: ["node", "fake.mjs"], custom: true });
    for (const bad of ["node fake.mjs", "[]", "[1]", '[""]', "{}"]) assert.match(chatCommand({ CI_CHAT_COMMAND: bad }, run).error, /CI_CHAT_COMMAND/);
});

test("Windows resolves a bare program to an .exe on PATH (shell:false cannot run .cmd shims)", () => {
    const dirs = ["C:\\nope", "C:\\tools\\uv"];
    const exists = (p) => p === path.join("C:\\tools\\uv", "uv.exe");
    assert.equal(resolveExecutable("uv", { platform: "win32", env: { PATH: dirs.join(";") }, exists }), path.join("C:\\tools\\uv", "uv.exe"));
    assert.equal(resolveExecutable("uv", { platform: "win32", env: { Path: dirs.join(";") }, exists }), path.join("C:\\tools\\uv", "uv.exe"));
    assert.equal(resolveExecutable("uv", { platform: "win32", env: { PATH: "C:\\nope" }, exists }), null);
    assert.equal(resolveExecutable("uv", { platform: "linux", env: {} }), "uv");
});

test("listening line parse accepts only the loopback contract shape", () => {
    assert.deepEqual(parseListening(LISTEN.trim()), { host: "127.0.0.1", port: 43210, path: "/agui" });
    assert.equal(parseListening('{"event":"listening","host":"localhost","port":1}').host, "127.0.0.1");
    for (const bad of ["hello", '{"event":"ready","host":"127.0.0.1","port":1}', '{"event":"listening","host":"0.0.0.0","port":1}', '{"event":"listening","host":"127.0.0.1","port":70000}', '{"event":"listening","host":"127.0.0.1","port":"1"}']) {
        assert.throws(() => parseListening(bad));
    }
});

test("lazy start: one spawn for concurrent callers, shell:false, repo cwd, stdin pipe, random token", async () => {
    const { backend, calls } = harness({ env: { CI_CHAT_PROFILE: "fake" }, onSpawn: (c) => setImmediate(() => {
        c.stdout.write(LISTEN.slice(0, 20));
        setImmediate(() => c.stdout.write(LISTEN.slice(20)));
    }) });
    assert.equal(calls.length, 0, "nothing spawned before first use");
    assert.equal(backend.status().state, "stopped");
    const [a, b] = await Promise.all([backend.ensure(), backend.ensure()]);
    assert.equal(calls.length, 1);
    assert.deepEqual(a, b);
    assert.equal(a.port, 43210);
    assert.ok(a.token.length >= 16);
    const { cmd, args, opts } = calls[0];
    assert.equal(cmd, "uv");
    assert.deepEqual(args.slice(0, 7), ["run", "--no-sync", "ci-lab", "chat", "serve", "--profile", "fake"]);
    assert.equal(opts.shell, false);
    assert.equal(opts.cwd, ROOT);
    assert.deepEqual(opts.stdio, ["pipe", "pipe", "pipe"]);
    assert.equal(opts.env.CI_CHAT_TOKEN, a.token);
    assert.equal(opts.env.CI_RUN_DIR, path.join(ROOT, "artifacts", "ci-runs"));
    assert.equal(backend.status().state, "ready");
    assert.equal(await backend.ensure().then((e) => e.token), a.token, "ready backend is reused");
    await backend.stop();
    assert.ok(calls[0].child.stdinEnded, "stop closes stdin (contract: exit on EOF)");
    assert.equal(backend.status().state, "stopped");
});

test("listening timeout rejects with 503-style error and kills the child", async () => {
    const { backend, calls } = harness({ options: { listenTimeoutMs: 40 } });
    await assert.rejects(backend.ensure(), (e) => e instanceof ChatUnavailable && e.code === "chat_start_timeout" && e.status === 503);
    assert.ok(calls[0].child.killed || calls[0].child.stdinEnded);
    await sleep(10);
    assert.equal(backend.status().state, "crashed");
});

test("garbage on stdout before the listening line fails the start", async () => {
    const { backend } = harness({ onSpawn: (c) => setImmediate(() => c.stdout.write("Installed 3 packages\n")) });
    await assert.rejects(backend.ensure(), (e) => e.code === "chat_bad_listening");
});

test("crash restarts with exponential backoff while instances are open, up to maxRestarts", async () => {
    const { backend, calls } = harness({ options: { maxRestarts: 2 }, onSpawn: (c) => listenSoon(c) });
    backend.refs = 1;
    await backend.ensure();
    const t0 = Date.now();
    calls[0].child.exit(1);
    assert.equal(backend.status().state, "crashed");
    assert.ok(backend.status().retryInMs > 0);
    await assert.rejects(backend.ensure(), (e) => e.code === "chat_restarting" && e.retryAfterMs > 0);
    while (calls.length < 2) await sleep(5);
    assert.ok(Date.now() - t0 >= 15, "first restart waits for the base backoff");
    while (backend.status().state !== "ready") await sleep(5);
    const t1 = Date.now();
    calls[1].child.exit(1);
    while (calls.length < 3) await sleep(5);
    assert.ok(Date.now() - t1 >= 35, "second restart doubles the delay");
    while (backend.status().state !== "ready") await sleep(5);
    calls[2].child.exit(1);
    await sleep(300);
    assert.equal(calls.length, 3, "no restart beyond maxRestarts");
    assert.equal(backend.status().state, "crashed");
    assert.equal(backend.status().restarts, 2);
    await backend.stop();
});

test("exit code 2 (bad startup input) is not auto-restarted; no restarts without viewers", async () => {
    const { backend, calls } = harness({ onSpawn: (c) => setImmediate(() => {
        c.stderr.write('{"error": "CI_CHAT_TOKEN must be set to at least 16 characters"}\n');
        c.exit(2);
    }) });
    backend.refs = 1;
    await assert.rejects(backend.ensure(), (e) => e.code === "chat_exited" && /at least 16/.test(e.message));
    await sleep(80);
    assert.equal(calls.length, 1);
    const h2 = harness({ onSpawn: (c) => listenSoon(c) });
    await h2.backend.ensure();
    h2.calls[0].child.exit(1);
    await sleep(80);
    assert.equal(h2.calls.length, 1, "refs = 0: crash is not restarted");
});

test("spawn ENOENT surfaces as chat_command_not_found", async () => {
    const { backend } = harness({ onSpawn: (c) => setImmediate(() => c.emit("error", Object.assign(new Error("spawn uv ENOENT"), { code: "ENOENT" }))) });
    await assert.rejects(backend.ensure(), (e) => e.code === "chat_command_not_found" && /install uv/.test(e.message));
    assert.equal(backend.status().state, "crashed");
});

test("missing uv.exe on Windows fails fast without spawning", async () => {
    const { backend, calls } = harness({ platform: "win32" });
    await assert.rejects(backend.ensure(), (e) => e.code === "chat_command_not_found");
    assert.equal(calls.length, 0);
});

test("CI_CHAT_DISABLED=1 and a bad CI_CHAT_COMMAND never spawn", async () => {
    const off = harness({ env: { CI_CHAT_DISABLED: "1" } });
    assert.equal(off.backend.status().state, "disabled");
    await assert.rejects(off.backend.ensure(), (e) => e.code === "chat_disabled");
    const bad = harness({ env: { CI_CHAT_COMMAND: "uv run" } });
    await assert.rejects(bad.backend.ensure(), (e) => e.code === "chat_config");
    assert.equal(off.calls.length + bad.calls.length, 0);
});

test("token never appears in status or logs; stderr is rate-limited", async () => {
    const { backend, calls, logs } = harness({ options: { logBurst: 5, logWindowMs: 60_000 }, onSpawn: (c) => listenSoon(c) });
    const ep = await backend.ensure();
    const child = calls[0].child;
    child.stderr.write(`INFO token is ${ep.token}\nheader x-ci-chat-token: ${ep.token}\n`);
    for (let i = 0; i < 50; i++) child.stderr.write(`line ${i}\n`);
    await sleep(10);
    const all = JSON.stringify(logs) + JSON.stringify(backend.status());
    assert.ok(!all.includes(ep.token), "token leaked");
    assert.ok(logs.some((l) => l.m.includes("***")));
    assert.ok(logs.filter((l) => l.m.startsWith("chat:")).length <= 5, "stderr burst limit");
    const st = backend.status();
    assert.deepEqual(Object.keys(st).sort(), ["lastError", "pid", "port", "profile", "program", "readySince", "restarts", "retryInMs", "runDir", "state", "viewers"]);
    await backend.stop();
});

test("pool: one backend per repo root, ref-counted; last release stops the child", async () => {
    const children = [];
    const spawn = () => {
        const c = new FakeChild(7);
        children.push(c);
        listenSoon(c);
        return c;
    };
    const opts = { env: { PATH: "" }, spawn, platform: "linux", options: { stopGraceMs: 50 } };
    const before = activeChats();
    const a = acquireChat(ROOT, opts);
    const b = acquireChat(ROOT, opts);
    assert.equal(a, b);
    assert.equal(a.refs, 2);
    await a.ensure();
    await releaseChat(a);
    assert.equal(children[0].stdinEnded, false, "still in use by the other instance");
    await releaseChat(b);
    assert.equal(children[0].stdinEnded, true, "last close ends stdin and stops the child");
    assert.equal(activeChats(), before);
    assert.equal(a.status().state, "stopped");
});

test("stop escalates to kill when the child ignores stdin EOF", async () => {
    let child;
    const backend = new ChatBackend({
        repoRoot: ROOT, env: { PATH: "" }, platform: "linux", options: { stopGraceMs: 20 },
        spawn: () => {
            child = new FakeChild(9, { exitOnEof: false });
            listenSoon(child);
            return child;
        },
    });
    await backend.ensure();
    await backend.stop();
    assert.ok(child.killed);
    assert.equal(backend.status().state, "stopped");
});
