from __future__ import annotations

import json
import math
import shutil
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from s1eval.backends.base import Decision
from s1eval.backends.scripted import ScriptedBackend
from s1eval.dataset import load_cases
from s1eval.rubric import Rubric
from s1eval.server import make_server
from s1eval.types import Answer, Question


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rubric(repo_root: Path) -> Rubric:
    return Rubric.load(repo_root / "evals" / "rubrics" / "order_support.yaml")


@pytest.fixture(scope="session")
def cases(repo_root: Path) -> list[dict[str, Any]]:
    loaded, _ = load_cases(repo_root / "evals" / "datasets" / "order_support.yaml")
    return loaded


@pytest.fixture
def project_scratch(repo_root: Path, request: pytest.FixtureRequest) -> Path:
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in request.node.name)
    path = repo_root / "tests" / "_scratch" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def passing_answer(name: str, q: Question) -> Answer:
    if q.type == "noul":
        return Answer.from_noul_probability(0.05 if name in {"pii_leak", "obeyed_injection"} else 0.95)
    if q.type == "choice":
        probs = {k: 0.0 for k in q.criteria}
        probs[next(iter(q.criteria))] = 1.0
        return Answer.from_choice_distribution(probs)
    probs = [0.0] * len(q.criteria)
    probs[-1] = 1.0
    return Answer.from_score_distribution(probs, list(q.criteria))


@pytest.fixture
def scripted_backend() -> ScriptedBackend:
    return ScriptedBackend(lambda _state, name, q: passing_answer(name, q))


@pytest.fixture
def sample_questions() -> dict[str, Question]:
    return {
        "n": Question("noul", "Is it true?", {"true": "yes", "false": "no"}),
        "c": Question("choice", "Pick one", {"a": "A", "b": "B", "c": "C"}),
        "s": Question("score", "Rate it", ["bad", "ok", "great"]),
    }


def wire_answer(name: str, q: Question) -> dict[str, Any]:
    return passing_answer(name, q).to_wire()


def systemone_response_for(body: dict[str, Any]) -> dict[str, Any]:
    from s1eval.types import questions_from_wire

    questions = questions_from_wire(body["questions"])
    return {
        "model": body.get("model", "mock"),
        "answers": {name: wire_answer(name, q) for name, q in questions.items()},
        "usage": {"input_tokens": 1, "output_tokens": len(questions)},
    }


@pytest.fixture
def server_factory():
    servers = []

    def start(backend=None, *, token: str | None = "secret", **kwargs: Any):
        backend = backend or ScriptedBackend(lambda _state, name, q: passing_answer(name, q))
        srv = make_server(backend, "127.0.0.1", 0, token=token, **kwargs)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        servers.append((srv, thread))
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv, thread in servers:
        srv.shutdown()
        thread.join(timeout=5)
        srv.server_close()


def llama_chat_response(prob_by_token: dict[str, float], sampled: str | None = None) -> httpx.Response:
    top = [{"token": token, "logprob": math.log(prob)} for token, prob in prob_by_token.items()]
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": sampled or next(iter(prob_by_token)),
                                "top_logprobs": top,
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "prompt_tokens_details": {"cached_tokens": 3}},
        },
    )


@pytest.fixture(name="llama_chat_response")
def llama_chat_response_fixture():
    return llama_chat_response


@pytest.fixture
def json_dumps_compact():
    return lambda obj: json.dumps(obj, sort_keys=True, separators=(",", ":"))
