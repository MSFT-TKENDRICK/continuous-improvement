from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Any

import pytest

from ci_lab.taskgraph.model import (
    Budget,
    Criterion,
    Deliverable,
    OutputSpec,
    Rubric,
    StudentSpec,
    TaskGraph,
    canonical_json,
    dump_graph,
    load_graph,
)


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_graph_round_trips_through_files(tmp_path, graph_factory, suffix: str) -> None:
    g = graph_factory()
    assert TaskGraph.from_json(g.to_json()) == g
    assert load_graph(dump_graph(g, tmp_path / f"g{suffix}")) == g


def test_rubric_and_spec_round_trip(rubric_factory, graph_factory) -> None:
    r = rubric_factory()
    assert Rubric.from_json(r.to_json()) == r
    spec = StudentSpec.of(graph_factory().deliverable("triage"))
    assert StudentSpec.from_json(spec.to_json()) == spec


def _graph_json(graph_factory) -> dict[str, Any]:
    return graph_factory().to_json()


@pytest.mark.parametrize("where", ["top", "deliverable", "output", "context", "budget"])
def test_unknown_fields_rejected(graph_factory, where: str) -> None:
    data = _graph_json(graph_factory)
    target = {"top": data, "deliverable": data["deliverables"][0], "output": data["deliverables"][0]["output"],
              "context": data["deliverables"][0]["context"][0], "budget": data["deliverables"][0]["budget"]}[where]
    target["rubric"] = "smuggled"
    with pytest.raises(ValueError, match="unknown field"):
        TaskGraph.from_json(data)


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("canary"),
    lambda d: d.update(extra=1),
    lambda d: d["criteria"][0].update(secret="x"),
    lambda d: d["criteria"][0].update(measure="vibes"),
    lambda d: d["criteria"][0].update(threshold="high"),
    lambda d: d["criteria"][0].update(required="yes"),
    lambda d: d.update(version=True),
    lambda d: d.update(criteria="c-format"),
])
def test_rubric_from_json_is_strict(rubric_factory, mutate) -> None:
    data = rubric_factory().to_json()
    mutate(data)
    with pytest.raises(ValueError):
        Rubric.from_json(data)


def test_bad_literals_and_types_rejected(graph_factory) -> None:
    for patch in ({"kind": "video"}, {"path": 3}, {"schema": "x"}):
        data = _graph_json(graph_factory)
        data["deliverables"][0]["output"].update(patch)
        with pytest.raises(ValueError):
            TaskGraph.from_json(data)
    with pytest.raises(ValueError):
        TaskGraph.from_json([])


def test_commitment_is_sha256_of_canonical_json(rubric_factory) -> None:
    r = rubric_factory()
    expected = hashlib.sha256(canonical_json(r.to_json()).encode("utf-8")).hexdigest()
    assert r.commitment() == expected and len(expected) == 64
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert rubric_factory(pass_score=1).commitment() == rubric_factory(pass_score=1.0).commitment()
    assert rubric_factory(version=2).commitment() != r.commitment()
    assert rubric_factory(canary="fedcba9876543210").commitment() != r.commitment()


def test_rubric_views(rubric_factory) -> None:
    r = rubric_factory()
    assert [c.id for c in r.oracles()] == ["c-format", "c-suite"]
    assert [c.id for c in r.soft()] == ["c-cites"]
    assert r.version_id == "summary-rubric@v1"
    assert r.criteria[0].oracle and not r.criteria[2].oracle


def test_models_are_deeply_immutable(rubric_factory) -> None:
    r = rubric_factory()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.pass_score = 0.1  # type: ignore[misc]
    with pytest.raises(TypeError):
        r.criteria[0].check["pattern"] = ".*"  # type: ignore[index]
    with pytest.raises(TypeError):
        Budget().weight["llm"] = 5  # type: ignore[index]
    assert isinstance(r.criteria[2].to_json()["check"]["options"], list)


def test_topo_order_and_dependents(graph_factory) -> None:
    g = graph_factory()
    assert g.topo_order() == ("summary", "triage", "audit", "reply")
    assert g.dependents("summary") == ("triage", "audit")
    assert g.dependents("reply") == ()
    with pytest.raises(KeyError):
        g.deliverable("nope")


def _with_deps(g: TaskGraph, **deps: tuple[str, ...]) -> TaskGraph:
    return dataclasses.replace(g, deliverables=tuple(
        dataclasses.replace(d, depends_on=deps.get(d.id, d.depends_on)) for d in g.deliverables))


def test_topo_order_rejects_cycles_unknown_and_duplicates(graph_factory) -> None:
    g = graph_factory()
    with pytest.raises(ValueError, match="cycle"):
        _with_deps(g, summary=("reply",)).topo_order()
    with pytest.raises(ValueError, match="unknown"):
        _with_deps(g, audit=("ghost",)).topo_order()
    with pytest.raises(ValueError, match="duplicate"):
        dataclasses.replace(g, deliverables=g.deliverables + g.deliverables[:1]).topo_order()


_WORDS = ("ledger", "refund", "escalate", "carrier", "invoice", "parcel", "warranty", "courier", "voucher")


def _random_rubric(rng: random.Random, i: int) -> Rubric:
    def phrase(n: int) -> str:
        return " ".join(rng.choice(_WORDS) + str(rng.randint(10, 99)) for _ in range(n))

    criteria = tuple(Criterion(f"crit-{i}-{k}", phrase(6), m, {"question": phrase(9), "type": "noul"}
                               if m == "s1" else {"suite": f"suite-{phrase(1)}", "split": "dev", "min_score": 0.5},
                               0.5) for k, m in enumerate(("assert", "s1", "s1")))
    return Rubric(f"rub-{i}", 1, "summary", criteria, 0.5, f"{rng.getrandbits(64):016x}")


@pytest.mark.parametrize("seed", range(6))
def test_student_spec_carries_no_rubric_material(graph_factory, seed: int) -> None:
    rng = random.Random(seed)
    rubric = _random_rubric(rng, seed)
    deliverable = dataclasses.replace(graph_factory().deliverable("triage"), rubric_commitment=rubric.commitment())
    spec = StudentSpec.of(deliverable)
    assert {f.name for f in dataclasses.fields(StudentSpec)} == {
        "id", "title", "instructions", "output", "context", "depends_on", "budget", "rubric_commitment"}
    surfaces = (spec.render(), canonical_json(spec.to_json()))
    secrets = [rubric.canary, rubric.id, rubric.version_id]
    for c in rubric.criteria:
        secrets += [c.id, c.description, *(v for v in c.check.values() if isinstance(v, str) and len(v) > 3)]
    for text in surfaces:
        assert not [s for s in secrets if s in text]
    assert rubric.commitment() in surfaces[0]


def test_student_render_is_deterministic(graph_factory) -> None:
    d = graph_factory().deliverable("triage")
    d = dataclasses.replace(d, output=OutputSpec("json", "out/triage.json", {"type": "object", "a": [1]}),
                            budget=Budget(max_tokens=900))
    text = StudentSpec.of(d).render()
    assert text == StudentSpec.of(Deliverable.from_json(d.to_json())).render()
    assert "## Instructions\nTriage each escalation" in text
    assert '- schema: {"a":[1],"type":"object"}' in text and "- max_tokens: 900" in text
    assert "## Depends on\n- summary" in text and "- file: data/log.txt" in text
