from __future__ import annotations

import json
import socket

import httpx
import pytest

from s1eval.backends.scripted import ScriptedBackend
from s1eval.server import MAX_BODY_BYTES, make_server
from s1eval.types import Answer


def _headers(token: str = "secret", **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", **extra}


def test_server_official_typesafe_sdk_parses_all_answer_types(server_factory):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    srv, base_url = server_factory(token="sdk-token")
    with TypeSafeClient(api_key="sdk-token", base_url=base_url, timeout=5) as client:
        res = client.system_one(
            state={"final_response": "ok"},
            questions={
                "n": Noul(instructions="true?"),
                "c": Choice(instructions="pick", criteria={"a": None, "b": None}),
                "s": Score(instructions="rate", criteria=["bad", "ok", "great"]),
            },
            model=srv.model_name,
        )
    assert 0 <= res.nouls["n"].noul <= 1
    assert res.choices["c"].choice in {"a", "b"}
    assert res.scores["s"].score == pytest.approx(2.0)


def test_systemone_backend_round_trips_real_rubric_through_server(server_factory, rubric, cases):
    """Our own TypeSafe client -> hardened server -> judge, over real HTTP, for every rubric question."""
    from s1eval.backends.systemone import SystemOneBackend
    from s1eval.runner import run_cases

    srv, base_url = server_factory(token="loop-token")
    be = SystemOneBackend(base_url, srv.model_name, "loop-token", max_retries=0)
    recs = run_cases(be, rubric, cases[:3])
    assert len(recs) == 3
    for r in recs:
        assert "error" not in r or not r["error"]
        assert set(r["answers"]) == set(rubric.questions)
        assert all(a["status"] == "ok" for a in r["answers"].values())


def test_server_raw_http_guards_and_shapes(server_factory):
    _srv, base_url = server_factory(token="secret")
    good = {
        "model": "m",
        "state": {"x": "y"},
        "questions": {"n": {"type": "noul", "instructions": "true?"}},
    }
    with httpx.Client(base_url=base_url, timeout=5) as client:
        assert client.get("/health", headers=_headers()).json() == {"status": "ok"}
        models = client.get("/v1/models", headers=_headers()).json()
        assert set(models["models"][0]) == {"name", "description", "release_date"}

        for method in ("GET", "OPTIONS", "PUT"):
            r = client.request(method, "/v1/systemone", headers=_headers())
            assert r.status_code == 405

        assert client.post("/v1/systemone", content=json.dumps(good), headers=_headers(**{"Content-Type": "text/plain"})).status_code == 415
        assert client.post("/v1/systemone", json=good, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/v1/systemone", json=good).status_code == 401
        assert client.post("/v1/systemone", json=good, headers=_headers(Authorization="Bearer secret", Host="evil.test")).status_code == 421

        too_many = {**good, "questions": {f"q{i}": {"type": "noul"} for i in range(33)}}
        r = client.post("/v1/systemone", json=too_many, headers=_headers())
        assert r.status_code == 422
        assert "detail" in r.json()

        invalid = {**good, "questions": {"bad name": {"type": "noul"}}}
        r = client.post("/v1/systemone", json=invalid, headers=_headers())
        assert r.status_code == 422
        assert "detail" in r.json()

        big = "x" * (MAX_BODY_BYTES + 1)
        assert client.post("/v1/systemone", content=big, headers=_headers(**{"Content-Type": "application/json"})).status_code == 413

        token_text = json.dumps(client.post("/v1/systemone", json=good, headers={"Authorization": "Bearer wrong"}).json())
        assert "secret" not in token_text


def test_server_missing_content_length_and_chunked_are_rejected(server_factory):
    srv, _base_url = server_factory(token=None)
    host, port = srv.server_address
    body = b'{"model":"m","state":{},"questions":{"n":{"type":"noul"}}}'

    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1:" + str(port).encode() + b"\r\nContent-Type: application/json\r\n\r\n" + body)
        data = s.recv(512)
    assert b"411" in data

    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(
            b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1:"
            + str(port).encode()
            + b"\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
        )
        data = s.recv(512)
    assert b"411" in data


def test_server_abstain_default_502_and_opt_in(server_factory):
    be = ScriptedBackend(lambda _state, _name, q: Answer.non_answer(q.type, "abstain", reason="uncertain"))
    _srv, base_url = server_factory(be, token=None)
    body = {"model": "m", "state": {}, "questions": {"n": {"type": "noul"}}}
    with httpx.Client(base_url=base_url, timeout=5) as client:
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "judge_abstained"
        r = client.post("/v1/systemone", json=body, headers={"X-S1Eval-Allow-Abstain": "1"})
        assert r.status_code == 200
        assert r.json()["answers"] == {}
        assert r.json()["s1eval"]["abstained"] == {"n": "abstain"}


def test_make_server_refuses_unsafe_remote_binds():
    be = ScriptedBackend()
    with pytest.raises(ValueError, match="non-loopback"):
        make_server(be, "0.0.0.0", 0)
    with pytest.raises(ValueError, match="requires --token"):
        make_server(be, "0.0.0.0", 0, insecure_allow_remote=True)
