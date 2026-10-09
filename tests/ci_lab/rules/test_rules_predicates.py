"""Path resolver + type-strict predicate semantics."""

import pytest

from ci_lab.rules.engine import MISSING, cap_text, compare, parse_path, strict_eq, walk
from ci_lab.rulespec import TEXT_MAX_BYTES


def test_parse_and_walk_paths():
    assert parse_path("current.args.a[0].b") == ("current.args", ("a", 0, "b"))
    assert parse_path("prior.result.items[12]") == ("prior.result", ("items", 12))
    assert parse_path("current.args") == ("current.args", ())
    for bad in ("current.argsx", "current.args.", "current.args[x]", "current.args.1a", "other.args.a", "args..a"):
        with pytest.raises(ValueError):
            parse_path(bad)
    doc = {"a": [{"b": 1}], "m": {"k": None}}
    assert walk(doc, ("a", 0, "b")) == 1
    assert walk(doc, ("m", "k")) is None
    assert walk(doc, ("a", 1)) is MISSING
    assert walk(doc, ("a", "b")) is MISSING          # name segment on a list
    assert walk(doc, ("m", 0)) is MISSING            # index segment on a dict
    assert walk({"s": "abc"}, ("s", 0)) is MISSING   # strings are not indexable


def test_strict_comparisons_never_coerce():
    assert strict_eq(1, 1.0) and strict_eq("a", "a") and strict_eq(None, None) and strict_eq(True, True)
    assert not strict_eq(True, 1) and not strict_eq(0, False) and not strict_eq("1", 1) and not strict_eq([1], [1])
    assert compare("le", 5, 5.0) and compare("lt", "a", "b")
    assert not compare("gt", True, 0) and not compare("lt", "1", 2) and not compare("le", None, 1)
    assert not compare("ne", "1", 1)  # type mismatch => false, even for ne
    assert compare("ne", 1, 2) and not compare("eq", MISSING, MISSING)


def test_text_cap_never_splits_codepoints():
    s = "é" * TEXT_MAX_BYTES
    c = cap_text(s)
    assert len(c.encode()) <= TEXT_MAX_BYTES and set(c) == {"é"}
    assert cap_text("short") == "short"


OPS = [
    # (op, value, args, expected)
    ("eq", 5, {"x": 5}, True),
    ("eq", 5, {"x": "5"}, False),
    ("eq", 1, {"x": True}, False),
    ("eq", True, {"x": True}, True),
    ("eq", None, {"x": None}, True),
    ("eq", None, {}, False),
    ("ne", 5, {"x": 6}, True),
    ("ne", 5, {"x": "6"}, False),
    ("ne", 5, {}, False),
    ("gt", 10, {"x": 10.5}, True),
    ("gt", 10, {"x": True}, False),
    ("ge", "b", {"x": "b"}, True),
    ("lt", 3, {"x": "2"}, False),
    ("le", 3, {"x": 3}, True),
    ("in", ["a", 1], {"x": "a"}, True),
    ("in", ["a", 1], {"x": True}, False),
    ("in", [1, 2], {"x": "1"}, False),
    ("nin", ["a", "b"], {"x": "c"}, True),
    ("nin", ["a", "b"], {"x": "a"}, False),
    ("nin", ["a", "b"], {"x": 3}, False),   # type mismatch => false
    ("nin", ["a"], {}, False),              # missing => false
    ("nin", ["a"], {"x": {"k": 1}}, False),
    ("exists", None, {"x": None}, True),
    ("exists", True, {}, False),
    ("exists", False, {}, True),
    ("exists", False, {"x": 0}, False),
    ("matches", r"^A\d+$", {"x": "A1001"}, True),
    ("matches", r"^A\d+$", {"x": 1001}, False),
    ("matches", "b", {"x": "abc"}, True),   # RE2 search semantics
]


@pytest.mark.parametrize(("op", "value", "args", "expected"), OPS)
def test_arg_ops(mk, fires, tb, op, value, args, expected):
    b = mk({"require": {"kind": "not", "of": {"kind": "arg", "path": "current.args.x", "op": op, "value": value}}})
    # require = NOT pred  =>  rule fires exactly when pred holds
    assert bool(fires(b, [], tb.pending("write_file", **args))) is expected


def test_nested_paths_and_indexes(mk, fires, tb):
    b = mk({"require": {"kind": "arg", "path": "current.args.items[1].sku", "op": "eq", "value": "S2"}})
    assert fires(b, [], tb.pending("write_file", items=[{"sku": "S1"}, {"sku": "S2"}])) == []
    assert fires(b, [], tb.pending("write_file", items=[{"sku": "S1"}])) == ["t.rule0"]  # missing => fires


def test_nested_all_any_not(mk, fires, tb):
    pred = {"kind": "any", "of": [
        {"kind": "all", "of": [{"kind": "arg", "path": "current.args.a", "op": "eq", "value": 1},
                               {"kind": "not", "of": {"kind": "arg", "path": "current.args.b", "op": "exists"}}]},
        {"kind": "arg", "path": "current.args.c", "op": "in", "value": ["x", "y"]}]}
    b = mk({"require": pred})
    assert fires(b, [], tb.pending("write_file", a=1)) == []
    assert fires(b, [], tb.pending("write_file", a=1, b=0)) == ["t.rule0"]
    assert fires(b, [], tb.pending("write_file", a=1, b=0, c="y")) == []
    assert fires(b, [], tb.pending("write_file")) == ["t.rule0"]


def test_when_gates_firing_and_target_filter(mk, fires, tb):
    b = mk({"when": {"kind": "arg", "path": "current.args.amount", "op": "gt", "value": 100},
            "require": {"kind": "arg", "path": "current.args.approved", "op": "eq", "value": True}},
           {"target": "*", "action": "warn", "require": {"kind": "arg", "path": "current.args.ok", "op": "exists"}})
    assert fires(b, [], tb.pending("write_file", amount=50)) == ["t.rule1"]
    assert fires(b, [], tb.pending("write_file", amount=500)) == ["t.rule0", "t.rule1"]
    assert fires(b, [], tb.pending("write_file", amount=500, approved=True, ok=1)) == []
    assert fires(b, [], tb.pending("read_file", amount=500)) == ["t.rule1"]  # target filter
