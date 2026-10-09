// Guards on the committed CopilotKit bundle (web/experiment-chat → ui/chat): gist-share size limits,
// banned vendor names, CSP-hostile code and telemetry endpoints. Rebuild with `node build.mjs` there.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CHAT_ASSET_RE, VIEWS } from "../lib/server.mjs";

const EXT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CHAT = path.join(EXT, "ui", "chat");
const MB = 1024 * 1024;
const BANNED = new RegExp(["lang", "(chain|smith|graph)"].join(""), "i");

function walk(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
        const p = path.join(dir, d.name);
        if (d.isDirectory()) return d.name === "node_modules" || d.name.startsWith(".tmp-") ? [] : walk(p);
        return [p];
    });
}

test("dashboard UI lists the same views as the server and loads the chat lazily into its own host", () => {
    const js = fs.readFileSync(path.join(EXT, "ui", "app.js"), "utf8");
    const html = fs.readFileSync(path.join(EXT, "ui", "index.html"), "utf8");
    const block = js.match(/const VIEWS = \[([\s\S]*?)\];/)[1];
    assert.deepEqual([...block.matchAll(/\["([a-z]+)",/g)].map((m) => m[1]), VIEWS);
    assert.match(js, /import\("\/chat\/chat\.js"\)/, "bundle loads only on demand");
    assert.ok(!/<script[^>]+\/chat\//.test(html), "index.html must not load the bundle eagerly");
    assert.match(html, /id="chat-host"[^>]*hidden/);
    assert.match(js, /chatErrorPanel/);
});

test("chat bundle exists, every file is under 1 MB, flat and servable by the allowlist", () => {
    const files = fs.readdirSync(CHAT, { withFileTypes: true });
    assert.ok(files.some((f) => f.name === "chat.js") && files.some((f) => f.name === "chat.css"), "run node build.mjs in web/experiment-chat");
    for (const f of files) {
        assert.ok(f.isFile(), `${f.name}: no subdirectories in ui/chat`);
        const size = fs.statSync(path.join(CHAT, f.name)).size;
        assert.ok(size < MB, `${f.name} is ${size} bytes (>= 1 MB)`);
        assert.match(`/chat/${f.name}`, CHAT_ASSET_RE, `${f.name} would not be served`);
    }
});

test("committed chat bundle defaults its suggestions and draft label to harness", () => {
    const js = fs.readdirSync(CHAT)
        .filter((name) => name.endsWith(".js"))
        .map((name) => fs.readFileSync(path.join(CHAT, name), "utf8"))
        .join("\n");
    assert.match(js, /Improve the harness/);
    assert.match(js, /Inspect harness metrics/);
    assert.match(js, /label:"Domain"/);
    assert.match(js, /\?\?"harness"/);
});

test("chat bundle has no banned vendor names, console logging, eval or telemetry endpoints", () => {
    for (const name of fs.readdirSync(CHAT)) {
        const text = fs.readFileSync(path.join(CHAT, name), "utf8");
        assert.ok(!BANNED.test(text), `${name} contains a banned vendor name`);
        if (!name.endsWith(".js")) continue;
        assert.ok(!/\bconsole\.(?:log|info|debug)\s*\(/.test(text), `${name}: console.log/info/debug`);
        assert.ok(!/\beval\s*\(/.test(text), `${name}: eval`);
        assert.ok(!/\bnew\s+Function\s*\(/.test(text), `${name}: new Function`);
        for (const host of ["api.cloud.copilotkit.ai", "segment.io", "segment.com", "scarf.sh", "telemetry.copilotkit"]) {
            assert.ok(!text.includes(host), `${name}: ${host}`);
        }
    }
});

test("extension stays shareable: total under 5 MB, no package.json, no dist/build folders", () => {
    const files = walk(EXT).filter((f) => !f.includes(`${path.sep}test${path.sep}.tmp-`));
    const total = files.reduce((n, f) => n + fs.statSync(f).size, 0);
    assert.ok(total < 5 * MB, `extension is ${total} bytes`);
    for (const f of files) {
        const rel = path.relative(EXT, f).split(path.sep);
        assert.notEqual(rel.at(-1), "package.json", rel.join("/"));
        assert.ok(!rel.slice(0, -1).some((d) => d === "dist" || d === "build"), rel.join("/"));
        assert.ok(fs.statSync(f).size < MB, `${rel.join("/")} >= 1 MB`);
    }
});
