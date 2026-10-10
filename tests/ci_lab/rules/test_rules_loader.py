"""Loader: schema, templates, RE2, duplicates, conflicts, transactional load, LKG (N1)."""

import json
import shutil
import textwrap

import pytest

from ci_lab.rules import (
    RuleLoadError,
    build_bundle,
    default_templates,
    load_bundle,
    load_templates,
    load_with_lkg,
    write_lkg,
)
from ci_lab.rulespec import MAX_GUARD_BLOCKS_PER_TURN, RuleSpec, bundle_digest

RULE = """
- id: {id}
  version: 1
  rung: R2
  on: tool_call
  target: write_file
  require: {require}
  action: {action}
  template: {template}
  slots: {slots}
"""
ARG_EQ = "{{kind: arg, path: current.args.currency, op: eq, value: {v}}}"
SLOTS = "{tool: write_file}"
USD = ARG_EQ.format(v="USD")


def write(tmp_path, name, *rules, header="schema_version: 1\nrules:"):
    p = tmp_path / name
    p.write_text(header + "".join(textwrap.indent(r, "  ") for r in rules), encoding="utf-8")
    return p


def rule(id="t.one", require=USD, action="block", template="count.exceeded", slots=SLOTS):
    return RULE.format(id=id, require=require, action=action, template=template, slots=slots)


def problems(fn):
    with pytest.raises(RuleLoadError) as ei:
        fn()
    return "\n".join(ei.value.problems)


def test_seed_bundle_digest_matches_contract(seeds):
    assert seeds.digest == bundle_digest(list(seeds.rules))
    assert len(seeds.sources) == 1 and seeds.source_digests[0].startswith("sha256:")


def test_template_catalog_is_complete_and_data_free():
    t = default_templates()
    need = {"precondition.missing", "precondition.same_subject", "arg.out_of_range", "arg.not_allowed",
            "arg.exceeds_prior", "sequence.required", "access.before_inspection", "pii.redacted",
            "response.blocked", "guard.terminal",
            # M17 synthesizer ids (exact slot names)
            "precondition.prior_call", "precondition.state_flag", "arg.constraint",
            "amount.not_exceed_prior", "response.redact_pattern"}
    assert need <= set(t)
    assert t["precondition.prior_call"].slots == ["tool", "prior_tool", "subject"]
    assert t["precondition.state_flag"].slots == ["tool", "flag", "subject", "via_tool"]
    assert t["arg.constraint"].slots == ["tool", "arg", "constraint"]
    assert t["amount.not_exceed_prior"].slots == ["tool", "arg", "prior_tool", "prior_field"]
    assert t["response.redact_pattern"].slots == ["pattern_class", "flag"]
    assert t["guard.terminal"].slots == [] and MAX_GUARD_BLOCKS_PER_TURN >= 1
    for spec in t.values():
        assert not any(ch.isdigit() for ch in spec.message + spec.fix), spec.id  # no data / literals
        assert "@" not in spec.message + spec.fix
        assert spec.fix.split()[0][0].isupper()  # imperative sentence


def test_yaml_on_key_and_duplicate_keys(tmp_path):
    ok = write(tmp_path, "ok.yaml", rule())
    assert load_bundle([ok]).rules[0].on == "tool_call"
    dup = tmp_path / "dup.yaml"
    dup.write_text("schema_version: 1\nschema_version: 1\nrules: []\n", encoding="utf-8")
    assert "duplicate key" in problems(lambda: load_bundle([dup]))


def test_unknown_keys_and_bad_schema_rejected(tmp_path):
    p = write(tmp_path, "x.yaml", rule() + "  surprise: 1\n")
    assert "surprise" in problems(lambda: load_bundle([p]))
    p2 = write(tmp_path, "y.yaml", rule(require="{kind: eval, code: '1'}"))
    assert "y.yaml" in problems(lambda: load_bundle([p2]))
    ex = tmp_path / "ex.yaml"
    ex.write_text("extractors:\n  - {flag: f_x, tool: t, result_path: result.ok, subject: args.id, extra: 1}\n",
                  encoding="utf-8")
    assert "extra" in problems(lambda: load_bundle([], [ex]))


def test_unknown_template_and_slot_mismatch(tmp_path):
    p = write(tmp_path, "t.yaml",
              rule(id="t.a", template="no.such.template"),
              rule(id="t.b", require=ARG_EQ.format(v="EUR"), slots="{}"),
              rule(id="t.c", require="{kind: arg, path: current.args.x, op: exists}",
                   slots="{tool: write_file, extra: y}"))
    msg = problems(lambda: load_bundle([p]))
    assert "unknown template id 'no.such.template'" in msg
    assert "missing ['tool']" in msg and "unexpected ['extra']" in msg


def test_duplicate_ids_across_files_and_transactional(tmp_path):
    a = write(tmp_path, "a.yaml", rule())
    b = write(tmp_path, "b.yaml", rule())
    msg = problems(lambda: load_bundle([a, b]))
    assert "duplicate rule id 't.one'" in msg and "a.yaml" in msg
    bad = write(tmp_path, "c.yaml", rule(id="t.two", template="nope"))
    with pytest.raises(RuleLoadError):
        load_bundle([a, bad])  # one bad file => nothing loads


def test_conflicts_detected(tmp_path):
    p = write(tmp_path, "c.yaml",
              rule(id="t.usd"), rule(id="t.eur", require=ARG_EQ.format(v="EUR")),
              rule(id="t.same", action="warn"),  # identical predicates to t.usd, different action
              rule(id="t.in", require="{kind: arg, path: current.args.currency, op: in, value: [GBP, JPY]}"))
    msg = problems(lambda: load_bundle([p]))
    assert "'t.eur' and 't.usd'" in msg or "'t.usd' and 't.eur'" in msg
    assert "identical predicates but different action" in msg
    assert "not in" in msg
    # disjoint `when` scopes are fine
    w = write(tmp_path, "w.yaml",
              rule(id="t.big") + "  when: {kind: arg, path: current.args.amount, op: gt, value: 100}\n",
              rule(id="t.small", require=ARG_EQ.format(v="EUR"))
              + "  when: {kind: arg, path: current.args.amount, op: le, value: 100}\n")
    assert len(load_bundle([w]).rules) == 2


def test_state_conflict_and_unknown_flag(mk):
    with pytest.raises(RuleLoadError, match="required both"):
        mk({"require": {"kind": "state", "flag": "access_verified"}},
           {"require": {"kind": "state", "flag": "access_verified", "value": False}})
    with pytest.raises(RuleLoadError, match="no extractor"):
        mk({"require": {"kind": "state", "flag": "made_up"}})


@pytest.mark.parametrize("pattern", [r"(\w+)\1", r"(?=a)b", r"(?<!x)y", r"a{1001}", "(", r"\p{Nope}"])
def test_re2_rejects_unsupported_patterns(mk, pattern):
    with pytest.raises(RuleLoadError, match="RE2 rejected"):
        mk({"require": {"kind": "arg", "path": "current.args.x", "op": "matches", "value": pattern}})


def test_redact_rule_needs_text_in_require(mk):
    with pytest.raises(RuleLoadError, match="no text predicate"):
        mk({"rung": "R3", "target": "*", "action": "redact", "template": "pii.redacted", "slots": {},
            "require": {"kind": "state", "flag": "access_verified"}})


def test_exists_value_must_be_bool(mk):
    with pytest.raises(RuleLoadError, match="exists"):
        mk({"require": {"kind": "arg", "path": "current.args.x", "op": "exists", "value": "yes"}})


def test_custom_templates(tmp_path):
    t = tmp_path / "tmpl.yaml"
    t.write_text("templates:\n  - {id: my.t, message: 'Bad {tool}.', fix: 'Fix {tool}.', slots: [tool]}\n",
                 encoding="utf-8")
    cat = load_templates(t)
    r = RuleSpec.model_validate({"id": "t.x", "version": 1, "rung": "R2", "on": "tool_call", "target": "x",
                                 "require": {"kind": "arg", "path": "current.args.a", "op": "exists"},
                                 "action": "block", "template": "my.t", "slots": {"tool": "x"}})
    b = build_bundle([r], templates=cat)
    assert b.render(r) == ("Bad x.", "Fix x.")
    with pytest.raises(RuleLoadError):
        build_bundle([r])  # not in the trusted default catalog


# ---------------------------------------------------------------- LKG (N1)


def _guards(tmp_path, seeds_path):
    g = tmp_path / "harness" / "guards"
    g.mkdir(parents=True)
    shutil.copy(seeds_path, g / "seeds.yaml")
    return g


def test_lkg_write_load_and_degrade(tmp_path, request):
    fx = request.path.parent / "fixtures"
    g = _guards(tmp_path, fx / "seeds.yaml")
    ex = [fx / "extractors.yaml"]
    lock = g / "BUNDLE.lock"
    b, degraded = load_with_lkg(g, lock, ex)
    assert not degraded
    assert write_lkg(g, b) == lock
    meta = json.loads(lock.read_text(encoding="utf-8"))
    assert meta["digest"] == b.digest and list(meta["files"]) == ["seeds.yaml"]
    assert (g / ".lkg" / "seeds.yaml").read_bytes() == (g / "seeds.yaml").read_bytes()

    (g / "broken.yaml").write_text("schema_version: 1\nrules: [{id: X}]\n", encoding="utf-8")
    b2, degraded = load_with_lkg(g, lock, ex)
    assert degraded and b2.digest == b.digest

    (g / ".lkg" / "seeds.yaml").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuleLoadError) as ei:
        load_with_lkg(g, lock, ex)
    msg = "\n".join(ei.value.problems)
    assert "broken.yaml" in msg and "does not match the lock" in msg


def test_lkg_rejects_digest_mismatch_and_bad_names(tmp_path, request):
    fx = request.path.parent / "fixtures"
    g = _guards(tmp_path, fx / "seeds.yaml")
    ex = [fx / "extractors.yaml"]
    b, _ = load_with_lkg(g, g / "BUNDLE.lock", ex)
    lock = write_lkg(g, b)
    meta = json.loads(lock.read_text(encoding="utf-8"))
    (g / "seeds.yaml").write_text("not: [valid", encoding="utf-8")
    lock.write_text(json.dumps({**meta, "digest": "sha256:0"}), encoding="utf-8")
    with pytest.raises(RuleLoadError, match="digest does not match"):
        load_with_lkg(g, lock, ex)
    lock.write_text(json.dumps({**meta, "files": {"../seeds.yaml": "x"}}), encoding="utf-8")
    with pytest.raises(RuleLoadError, match="bad file name"):
        load_with_lkg(g, lock, ex)
    lock.unlink()
    with pytest.raises(RuleLoadError, match="lock missing"):
        load_with_lkg(g, lock, ex)


def test_write_lkg_refuses_changed_source(tmp_path, request):
    fx = request.path.parent / "fixtures"
    g = _guards(tmp_path, fx / "seeds.yaml")
    b, _ = load_with_lkg(g, g / "BUNDLE.lock", [fx / "extractors.yaml"])
    with (g / "seeds.yaml").open("a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    with pytest.raises(RuleLoadError, match="changed since"):
        write_lkg(g, b)
    assert not (g / "BUNDLE.lock").exists()


def test_empty_guards_dir_is_an_empty_bundle(tmp_path):
    b, degraded = load_with_lkg(tmp_path / "missing", tmp_path / "BUNDLE.lock")
    assert b.rules == () and not degraded
