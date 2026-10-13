import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import {
    CSP,
    hostAllowed,
    isLoopbackUrl,
    isSafeId,
    isSensitiveAttr,
    isTraceId,
    isWithin,
    newToken,
    originAllowed,
    sensitiveEnabled,
    stripSecrets,
    tokenEquals,
} from "../lib/security.mjs";

test("C31 host and origin checks are exact", () => {
    assert.ok(hostAllowed("127.0.0.1:4000", 4000));
    assert.ok(hostAllowed("localhost:4000", 4000));
    assert.ok(!hostAllowed("127.0.0.1:4001", 4000));
    assert.ok(!hostAllowed("evil.example:4000", 4000));
    assert.ok(!hostAllowed("127.0.0.1.nip.io:4000", 4000));
    assert.ok(!hostAllowed(undefined, 4000));
    assert.ok(originAllowed(undefined, 4000));
    assert.ok(originAllowed("http://127.0.0.1:4000", 4000));
    assert.ok(!originAllowed("null", 4000));
    assert.ok(!originAllowed("http://evil.example", 4000));
});

test("CSP forbids inline script/style and remote connects", () => {
    assert.match(CSP, /script-src 'self'(;|$)/);
    assert.match(CSP, /style-src 'self'(;|$)/);
    assert.match(CSP, /connect-src 'self'(;|$)/);
    assert.doesNotMatch(CSP, /unsafe-inline|unsafe-eval/);
});

test("tokens", () => {
    const a = newToken();
    assert.ok(a.length >= 32);
    assert.notEqual(a, newToken());
    assert.ok(tokenEquals(a, a));
    assert.ok(!tokenEquals(a, a.slice(1)));
    assert.ok(!tokenEquals("", ""));
    assert.ok(!tokenEquals(a, undefined));
});

test("ids, urls and containment", () => {
    assert.ok(isSafeId("tone-a1-r01"));
    assert.ok(!isSafeId("../x"));
    assert.ok(!isSafeId("a/b"));
    assert.ok(!isSafeId(".hidden"));
    assert.ok(isTraceId("0af7651916cd43dd8448eb211c80319c"));
    assert.ok(!isTraceId("0af7"));
    assert.ok(isLoopbackUrl("http://127.0.0.1:18888"));
    assert.ok(isLoopbackUrl("http://localhost:1"));
    assert.ok(!isLoopbackUrl("http://example.com"));
    assert.ok(!isLoopbackUrl("file:///c:/x"));
    const root = path.resolve("C:/repo");
    assert.ok(isWithin(root, path.join(root, "a", "b")));
    assert.ok(isWithin(root, root));
    assert.ok(!isWithin(root, path.resolve("C:/repo2/x")));
    assert.ok(!isWithin(root, path.resolve("C:/other")));
});

test("C29 sensitive attribute detection matches ci_lab.telemetry.jsonl", () => {
    for (const k of [
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.prompt",
        "gen_ai.prompt.0.content",
        "gen_ai.completion.0.content",
        "gen_ai.choice.message.content",
    ]) {
        assert.ok(isSensitiveAttr(k), k);
    }
    for (const k of ["gen_ai.operation.name", "gen_ai.request.model", "gen_ai.usage.input_tokens", "ci.score"]) assert.ok(!isSensitiveAttr(k), k);
    assert.ok(sensitiveEnabled({}, { "ci.telemetry.sensitive": true }));
    assert.ok(sensitiveEnabled({ "ci.telemetry.sensitive": "true" }));
    assert.ok(!sensitiveEnabled({ "ci.telemetry.sensitive": false }, {}));
});

test("stripSecrets removes secret-looking keys at any depth", () => {
    const out = stripSecrets({ ok: 1, api_key: "k", nested: { browser_token: "t", list: [{ otlp_key: "o", fine: 2 }] }, Authorization: "x" });
    assert.deepEqual(out, { ok: 1, nested: { list: [{ fine: 2 }] } });
});
