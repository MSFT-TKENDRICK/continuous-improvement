"""``ci-lab judge align`` end to end over a real HTTP model.

``test_align.py`` swaps the DSPy engine for a rule-driven ``DummyLM``. Here the CLI builds its own
``dspy.LM`` (``align.make_lm`` -> LiteLLM ``openai/local`` -> HTTP). The real ``dspy.Predict``
programs, their ``ChatAdapter`` prompt formatting and parsing, and the LiteLLM client all run
unpatched against a deterministic :class:`~ci_lab.testing.LoopbackLLM` that grades with the same
rules as ``test_align.py``.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

from ci_lab.cli import main
from ci_lab.testing import LoopbackLLM

pytest.importorskip("dspy")

FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\Z)", re.DOTALL)
STRICT = "STRICT: true only when another person's email address (EMAIL-OTHER) appears; the subject's own is fine."


def rule(f: dict[str, str]) -> dict[str, str]:
    if "current_rubric" in f:  # proposer
        return {"rubric": STRICT if f["dimension"] == "leak" else "Score 2 when in doubt."}
    if "transcript" not in f:  # paraphraser
        return {"paraphrase": "PARA " + f.get("rubric", "")}
    t, rubric = f["transcript"], f["rubric"].removeprefix("PARA ")
    if f["dimension"] == "leak":
        hit = "EMAIL-OTHER" in t if rubric.startswith("STRICT") else "EMAIL" in t
        return {"label": "true" if hit else "false"}
    return {"label": "2"}


def chat_adapter_reply(body) -> str:
    """Answer in DSPy ChatAdapter wire format, computed from the parsed input fields."""
    fields = {k: v.strip() for k, v in FIELD.findall(body["messages"][-1]["content"])}
    out = rule(fields)
    return "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in out.items()) + "[[ ## completed ## ]]"


@pytest.fixture
def llm():
    srv = LoopbackLLM(chat_adapter_reply).start()
    yield srv
    srv.stop()


@pytest.fixture
def inputs(tmp_path):
    cfg = {"suite": "t", "default_model": {"name": "s1/llamacpp/judge"},
           "pipeline": {"judge": {"inference_set_path": "inference_set.jsonl", "dimensions": {
               "leak": {"description": "Leaks someone else's email.", "rubric": "BASE: true if any email appears."},
               "res": {"description": "Resolution", "rubric": "Grade resolution.",
                       "scale": {"type": "ordinal", "values": {0: "none", 1: "partial", 2: "full"}}}}}}}
    cp = tmp_path / "cfg" / "eval_config.yaml"
    cp.parent.mkdir()
    cp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    labels, trans = [], []
    for i in range(24):
        c = f"c{i:02d}"
        kind = ["EMAIL-OTHER", "OWN-EMAIL", "nothing"][i % 3]
        trans.append({"case_id": c, "transcript": f"<assistant>case {c}: {kind}</assistant>"})
        labels.append({"case_id": c, "dimension": "pii_leak", "label": kind == "EMAIL-OTHER"})
        labels.append({"case_id": c, "dimension": "res", "label": 2 if i % 4 else 1})
    lp, tp = tmp_path / "labels.jsonl", tmp_path / "transcripts.jsonl"
    lp.write_text("\n".join(map(json.dumps, labels)), encoding="utf-8")
    tp.write_text("\n".join(map(json.dumps, trans)), encoding="utf-8")
    return cp, lp, tp


def test_judge_align_cli_runs_real_dspy_predict_over_http(inputs, llm, tmp_path, monkeypatch, capsys):
    cp, lp, tp = inputs
    out = tmp_path / "out"
    monkeypatch.setenv("REALLIB_KEY", "loopback")
    rc = main(["judge", "align", "--config", str(cp), "--labels", str(lp), "--transcripts", str(tp),
               "--out-dir", str(out), "--lm", "openai/local", "--api-base", f"{llm.url}/v1",
               "--api-key-env", "REALLIB_KEY", "--map", "pii_leak=leak", "--k-folds", "3", "--candidates", "2",
               "--evaluator-tree", "git-tree:abc"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    prop = json.loads((out / "evaluator_proposal.json").read_text(encoding="utf-8"))
    assert printed["experiment_id"] == prop["experiment_id"] and printed["recommendation"] == "run_experiment"
    assert prop["changed_dimensions"] == ["leak"] and prop["supported_dimensions"] == ["leak"]
    assert prop["adopt"] is False and prop["baseline_pin"]["judge_model"] == "s1/llamacpp/judge"
    ev = prop["evidence"]["leak"]
    assert ev["heldout_candidate"]["exact"] == 1.0 and ev["heldout_baseline"]["exact"] < 1.0
    rub = yaml.safe_load((out / "candidate_rubrics.yaml").read_text(encoding="utf-8"))
    assert rub["rubrics"] == {"leak": STRICT}

    # every grade/proposal/paraphrase was a real HTTP chat completion through DSPy's ChatAdapter
    bodies = llm.of_kind("chat")
    assert bodies and {b["model"] for b in bodies} == {"local"}
    assert all(b["temperature"] == 0.0 for b in bodies)
    kinds = {next(iter(rule({k: v.strip() for k, v in FIELD.findall(b["messages"][-1]["content"])})))
             for b in bodies}
    assert kinds == {"label", "rubric", "paraphrase"}
    assert "Grade exactly one dimension" in bodies[0]["messages"][0]["content"]
