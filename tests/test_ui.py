"""#27: `prompteval ui`, a local page over runs, judgements and criteria."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from prompteval.compare.pairwise import judge_pairwise, save_judgement
from prompteval.eval import save_run
from prompteval.ui.server import serve
from tests.test_pairwise import _judge, _run


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    a = replace(_run("v1", ["bad"] * 4), finished_at="2026-11-12T10:00:00")
    b = replace(_run("v2", ["GOOD"] * 4), finished_at="2026-11-12T11:00:00")
    save_run(a)
    save_run(b)
    save_judgement(judge_pairwise(a, b, client=_judge("quality"), model="gpt-4.1-mini"), a, b)
    (tmp_path / "criteria").mkdir()
    (tmp_path / "criteria" / "kundli.md").write_text("# PROMPTEVAL-DRAFT: x\n1. rule\n")
    return tmp_path


@pytest.fixture
def server(project: Path) -> Iterator[tuple[int, str]]:
    srv, _ = serve(project, port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    page = _request(port, "GET", "/")[1]
    token = page.split('const TOKEN = "', 1)[1].split('"', 1)[0]
    yield port, token
    srv.shutdown()
    srv.server_close()


def _request(
    port: int, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None
) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    resp = conn.getresponse()
    return resp.status, resp.read().decode()


def test_page_and_state(server: tuple[int, str]) -> None:
    port, token = server
    status, page = _request(port, "GET", "/")
    assert status == 200 and "<title>prompteval</title>" in page and len(token) > 20
    state = json.loads(_request(port, "GET", "/api/state")[1])
    assert [r["tag"] for r in state["runs"]] == ["v2", "v1"]  # newest first
    assert state["judgements"][0]["baseline"] == "v1"
    assert state["criteria"] == [{"name": "kundli.md", "draft": True}]


def test_compare_includes_the_judgement_and_its_reasons(server: tuple[int, str]) -> None:
    port, _ = server
    res = json.loads(_request(port, "GET", "/api/compare?a=v1&b=v2")[1])
    names = [d["name"] for d in res["report"]["scorer_deltas"]]
    assert "pairwise_preference" in names
    pw = next(d for d in res["report"]["scorer_deltas"] if d["name"] == "pairwise_preference")
    assert pw["status"] == "ok" and pw["delta"] == pytest.approx(0.5)
    detail = res["judgement"]["details"]["e0"]
    assert detail["first_order"] == "v2" and detail["second_order"] == "v2"
    assert len(res["a"]["examples"]) == 4 and res["b"]["examples"][0]["output"] == "GOOD"


def test_criteria_edit_needs_the_token_and_stays_inside_criteria(
    server: tuple[int, str], project: Path
) -> None:
    port, token = server
    body = {"text": "1. new rule"}
    assert _request(port, "PUT", "/api/criteria/kundli.md", body)[0] == 403  # no token
    hdr = {"X-Prompteval-Token": token}
    assert _request(port, "PUT", "/api/criteria/kundli.md", body, hdr)[0] == 200
    assert (project / "criteria" / "kundli.md").read_text() == "1. new rule\n"
    # A path in the name is reduced to a file name inside criteria/.
    assert _request(port, "PUT", "/api/criteria/..%2F..%2Fevil.md", body, hdr)[0] == 200
    assert not (project.parent / "evil.md").exists()
    assert (project / "criteria" / "evil.md").exists()
    assert _request(port, "PUT", "/api/criteria/x.md", {"text": " "}, hdr)[0] == 400
    got = json.loads(_request(port, "GET", "/api/criteria/kundli")[1])
    assert got["text"] == "1. new rule\n"


def test_other_hosts_are_refused(server: tuple[int, str]) -> None:
    port, token = server
    for method in ("GET", "PUT"):
        status, _ = _request(
            port,
            method,
            "/api/state" if method == "GET" else "/api/criteria/k.md",
            {"text": "x"} if method == "PUT" else None,
            {"Host": "evil.example:80", "X-Prompteval-Token": token},
        )
        assert status == 403


def test_errors_are_json_not_crashes(server: tuple[int, str]) -> None:
    port, _ = server
    assert _request(port, "GET", "/api/run/nope")[0] == 404
    assert _request(port, "GET", "/api/compare?a=v1&b=nope")[0] == 404
    assert _request(port, "GET", "/api/nothing")[0] == 404


def test_the_page_ships_in_the_package() -> None:
    from importlib.resources import files

    page = files("prompteval.ui").joinpath("index.html").read_text()
    assert "__TOKEN__" in page and "https://" not in page  # nothing fetched from the internet
