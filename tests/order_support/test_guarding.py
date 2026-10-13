"""HOOK(M3) wiring: guards installed once per process, one session per conversation, verify_identity
bound, shadow by default, and paired-eval decision sinks under ``$CI_GUARD_DECISIONS``."""

from __future__ import annotations

import asyncio
import json
import shutil
import types

import pytest

from ci_lab.guards.domains.order_support import GUARDS_DIR, TOOL_POLICIES
from ci_lab.guards.install import GuardMiddleware
from ci_lab.guards.middleware import (
    GuardAgentMiddleware,
    GuardFunctionMiddleware,
    GuardPreflightMiddleware,
)
from ci_lab.lessons_arm import paired
from ci_lab.testing import Call, FakeChatClient
from order_support import agent, assert_wrapper, data, guarding, otel

ENVS = (guarding.DECISIONS_ENV, guarding.GUARDS_ENV, guarding.RUN_DIR_ENV, guarding.CASE_ENV,
        guarding.TRIAL_ENV, guarding.SEED_ENV, "CI_VARIANT")
REFUND = Call("issue_refund", {"order_id": "NW-10001", "amount": 10.0}, "r1")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    guarding.reset()
    yield monkeypatch
    guarding.reset()


def _verify_args(order_id="NW-10001"):
    customer = data.ORDERS[order_id]["customer"]
    return {"order_id": order_id, "full_name": customer["name"], "email_or_phone": customer["email"]}


def _tool_results(client):
    out = {}
    for messages, _ in client.requests:
        for m in messages:
            for c in m.contents:
                if c.type == "function_result":
                    out[c.call_id] = c.result
    return out


def _runtime():
    [installed] = guarding._installs.values()
    return installed.guards.runtime


def test_guards_are_installed_once_per_process_after_existing_middleware(use_client, monkeypatch):
    seen = []
    build = agent.build_agent

    def spy(client, spec=None):
        built = build(client, spec)
        run = built.run

        async def recording_run(*args, **kwargs):
            seen.append(kwargs)
            return await run(*args, **kwargs)

        built.run = recording_run
        return built

    monkeypatch.setattr(agent, "build_agent", spy)
    use_client(client=FakeChatClient(default="hi"))
    assert agent.chat("one") == agent.chat("two") == "hi"
    assert guarding.install_count == 1
    for kwargs in seen:
        mw = kwargs["middleware"]
        assert isinstance(mw[0], agent._TurnGuard) and isinstance(mw[1], otel.OpenInferenceChatMiddleware)
        assert [type(m) for m in mw[2:]] == [GuardAgentMiddleware, GuardPreflightMiddleware,
                                             GuardFunctionMiddleware]
    assert seen[0]["middleware"][2] is seen[1]["middleware"][2]  # same install, not rebuilt per turn
    assert seen[0]["session"].session_id != seen[1]["session"].session_id  # two conversations
    [installed] = guarding._installs.values()
    assert isinstance(installed.guards, GuardMiddleware)
    assert installed.guards_dir == GUARDS_DIR.resolve()
    assert installed.guards.runtime.policies == TOOL_POLICIES
    assert installed.guards.runtime.bundle is not None and not installed.guards.runtime.degraded
    assert installed.writer is None  # no sink unless asked for


def test_install_cache_follows_guard_mode_env(fresh):
    a = guarding.install(agent.HARNESS_DIR, TOOL_POLICIES)
    assert guarding.install(agent.HARNESS_DIR, TOOL_POLICIES) is a
    fresh.setenv(guarding.GUARDS_ENV, "enforce")
    b = guarding.install(agent.HARNESS_DIR, TOOL_POLICIES)
    assert b is not a and b.guards.runtime.mode_override == "enforce" and guarding.install_count == 2


def test_harness_without_guards_falls_back_to_the_packaged_seed_rules(tmp_path):
    root = tmp_path / "harness"
    shutil.copytree(agent.HARNESS_DIR, root, ignore=shutil.ignore_patterns("guards"))
    assert guarding.guards_dir(root) == GUARDS_DIR.resolve()
    (root / "guards").mkdir()
    assert guarding.guards_dir(root) == GUARDS_DIR.resolve()  # empty dir: still the seeds
    shutil.copy(GUARDS_DIR / "order_support.yaml", root / "guards")
    assert guarding.guards_dir(root) == (root / "guards").resolve()


def test_shadow_default_records_but_never_changes_behaviour(use_client):
    client = use_client([REFUND], "Refunded.")
    assert agent.chat("Refund NW-10001 please") == "Refunded."
    assert json.loads(_tool_results(client)["r1"])["status"] == "processed"  # not blocked
    decisions = _runtime().decisions
    assert "refund.requires_verified_identity" in {d.rule_id for d in decisions}
    assert all(d.mode == "shadow" and not d.enforced for d in decisions)


def test_enforce_blocks_the_unverified_refund(use_client, fresh):
    fresh.setenv(guarding.GUARDS_ENV, "enforce")
    client = use_client([REFUND], "Sorry.")
    agent.chat("Refund NW-10001 please")
    result = json.loads(_tool_results(client)["r1"])
    assert "status" not in result and "refund." in json.dumps(result["guard"])
    assert any(d.enforced for d in _runtime().decisions)


def test_multi_turn_conversation_keeps_one_session_and_guard_state(use_client):
    client = use_client([Call("verify_identity", _verify_args(), "v1")], "Verified.",
                        [REFUND], "Refunded.")
    first = "I'm the owner of NW-10001"
    token = guarding.bind_case("case-1")
    try:
        agent.chat(first, history=[{"role": "user", "content": first}])
        history = [{"role": "user", "content": first}, {"role": "assistant", "content": "Verified."},
                   {"role": "user", "content": "refund it"}]
        agent.chat("refund it", history=history)
    finally:
        guarding.reset_case(token)
    assert json.loads(_tool_results(client)["v1"]) == {"verified": True, "order_id": "NW-10001"}
    [conv] = guarding._conversations.values()
    steps = _runtime().open(conv.session.session_id).recorder.steps
    assert [s.kind for s in steps if s.kind in ("user", "response")] == ["user", "response", "user", "response"]
    assert conv.flushed == len(steps)
    # identity_verified from turn 1 still holds in turn 2: the precondition rule does not match.
    assert "refund.requires_verified_identity" not in {d.rule_id for d in _runtime().decisions}
    # The model input is unchanged: prior turns then the current turn, nothing duplicated.
    texts = [(str(m.role), m.text) for m in client.requests[2][0]]
    assert texts == [("user", first), ("assistant", "Verified."), ("user", "refund it")]


def test_a_new_first_turn_starts_a_new_session(use_client):
    use_client(client=FakeChatClient(default="ok"))
    hello = [{"role": "user", "content": "hello"}]
    agent.chat("hello", history=hello)
    agent.chat("hello", history=hello)
    assert len(guarding._conversations) == 1  # same key, but each first turn replaced the session
    assert len(_runtime()._convs) == 2


def test_decision_sinks_are_per_case_and_trial_and_paired_can_read_them(use_client, fresh, tmp_path):
    root = tmp_path / "decisions" / "arm-off-0"
    fresh.setenv(guarding.DECISIONS_ENV, str(root))
    fresh.setenv(guarding.GUARDS_ENV, "off")
    fresh.setenv(guarding.CASE_ENV, "suite/case 7")
    fresh.setenv(guarding.TRIAL_ENV, "2")
    use_client([Call("lookup_order", {"order_id": "NW-10001"}, "l1")], [REFUND], "Refunded.")
    agent.chat("Refund NW-10001 please")
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jsonl"))
    assert files == ["suite_case_7/2.jsonl"]
    lines = [json.loads(x) for x in (root / files[0]).read_text(encoding="utf-8").splitlines()]
    assert all(x["case_id"] == "suite/case 7" and x["trial"] == 2 for x in lines)
    decisions = [x for x in lines if "rule_id" in x]
    assert decisions and all(d["mode"] == "off" and d["enforced"] is False for d in decisions)
    calls = [x for x in lines if x.get("kind") == "call"]
    assert [(c["tool"], c["side_effect"]) for c in calls] == [("lookup_order", False), ("issue_refund", True)]
    opps = [x for x in lines if x.get("kind") == "opportunity"]
    assert [(o["on"], o["target"]) for o in opps] == [("tool_call", "lookup_order"),
                                                      ("tool_call", "issue_refund"), ("response", "*")]
    score = types.SimpleNamespace(case_id="suite/case 7", trial=2)
    run = paired.read_run(types.SimpleNamespace(scores=[score]), root)
    assert run.unattributed == 0 and run.opportunities == 3
    assert {d.rule_id for d in run.decisions[("suite/case 7", 2)]} >= {"refund.requires_verified_identity"}
    assert all(d.case_id == "suite/case 7" and d.trial == 2 for d in run.decisions[("suite/case 7", 2)])
    assert [c["tool"] for c in run.calls[("suite/case 7", 2)]] == ["lookup_order", "issue_refund"]


def test_bound_assert_case_attributes_decisions_without_env(use_client, fresh, tmp_path):
    fresh.setenv(guarding.DECISIONS_ENV, str(tmp_path))
    use_client([REFUND], "Refunded.")
    token = guarding.bind_case("c9")
    try:
        assert guarding.case_identity() == ("c9", 0)
        agent.chat("Refund NW-10001 please")
    finally:
        guarding.reset_case(token)
    assert guarding.case_identity() == (None, 0)
    lines = [json.loads(x) for x in (tmp_path / "c9" / "0.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines and all(x["case_id"] == "c9" for x in lines)


def test_unknown_case_never_writes_a_null_case_id(fresh, tmp_path):
    writer = guarding.DecisionWriter(tmp_path, per_case=True)
    writer.write([{"kind": "opportunity", "n": 1, "case_id": None}])
    [path] = tmp_path.rglob("*.jsonl")
    assert path.name == f"{guarding.UNATTRIBUTED}.jsonl"
    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "opportunity", "n": 1}


def test_run_dir_gets_one_decisions_file_without_paired_records(use_client, fresh, tmp_path):
    fresh.setenv(guarding.RUN_DIR_ENV, str(tmp_path))
    use_client([REFUND], "Refunded.")
    agent.chat("Refund NW-10001 please")
    lines = (tmp_path / "guards" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines and all("rule_id" in json.loads(x) for x in lines)


def test_no_decision_files_by_default(use_client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    use_client([REFUND], "Refunded.")
    agent.chat("Refund NW-10001 please")
    assert not list(tmp_path.rglob("*.jsonl"))


def test_seed_depends_on_case_and_trial_not_variant(use_client, fresh):
    assert guarding.seed() is None
    fresh.setenv(guarding.CASE_ENV, "c1")
    fresh.setenv(guarding.TRIAL_ENV, "1")
    fresh.setenv("CI_VARIANT", "arm-off-0")
    fresh.setenv(guarding.GUARDS_ENV, "off")
    off = guarding.seed()
    fresh.setenv("CI_VARIANT", "arm-on-0")
    fresh.setenv(guarding.GUARDS_ENV, "enforce")
    assert guarding.seed() == off and isinstance(off, int) and 0 <= off < 2**31
    fresh.setenv(guarding.TRIAL_ENV, "2")
    assert guarding.seed() != off
    fresh.setenv(guarding.SEED_ENV, "42")
    assert guarding.seed() == 42
    client = use_client("ok")
    agent.chat("hi")
    assert client.requests[0][1]["seed"] == 42


def test_no_seed_option_outside_case_runs(use_client):
    client = use_client("ok")
    agent.chat("hi")
    assert "seed" not in client.requests[0][1]


def test_assert_case_wrapper_binds_the_case_for_the_conversation(fresh):
    seen = []

    async def runner(*, test_case, **_):
        seen.append(guarding.case_identity())
        await asyncio.to_thread(lambda: seen.append(guarding.case_identity()))  # like ASSERT's callable

    wrapped = assert_wrapper._with_case_span(runner)
    asyncio.run(wrapped(test_case={"test_case_id": "case-3"}))
    assert seen == [("case-3", 0), ("case-3", 0)]
    assert guarding.case_identity() == (None, 0)
