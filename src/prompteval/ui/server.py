"""`prompteval ui`: a local page for reading runs, comparisons and judge verdicts (#27).

Standard library only, no build step and nothing fetched from the internet:
the page is one HTML file shipped in the package. It reads what the CLI
already wrote (`.prompteval/runs`, `.prompteval/judgements`, `criteria/`) and
can write exactly one thing: a criteria file, through the editor.

It never calls a model. Re-running or re-judging spends money, so the page
shows the command to copy instead of doing it behind a button.

Safety, since it is a web server on your machine:
- it listens on 127.0.0.1 only;
- every request must carry a Host header naming that address, so a web page
  elsewhere cannot reach it through DNS rebinding;
- the one write (PUT /api/criteria/NAME) also needs a per-launch token that
  only the served page knows, and NAME is reduced to a plain file name inside
  `criteria/`.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import asdict, is_dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from prompteval import criteria as criteria_mod
from prompteval.compare import compute_comparison
from prompteval.compare.pairwise import (
    PAIRWISE_SCORER,
    baseline_with_pairwise,
)
from prompteval.eval.persistence import DEFAULT_RUNS_DIR, load_run
from prompteval.eval.runner import ExampleResult, RunResult, ScoreOutcome

MAX_CRITERIA_BYTES = 200_000


def _as_dict(value: Any) -> dict[str, Any]:
    cleaned = _clean(value)
    if not isinstance(cleaned, dict):
        raise TypeError(f"expected an object, got {type(cleaned).__name__}")
    return cleaned


def _clean(value: Any) -> Any:
    """JSON-safe: dataclasses to dicts, NaN and infinity to null."""
    if is_dataclass(value) and not isinstance(value, type):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Project:
    """What the page can see: one folder where prompteval runs."""

    def __init__(self, root: Path, runs_dir: Path | None = None) -> None:
        self.root = root
        self.runs_dir = runs_dir or (root / DEFAULT_RUNS_DIR)
        self.judgements_dir = self.runs_dir.parent / "judgements"
        self.criteria_dir = root / criteria_mod.CRITERIA_DIR

    # ---- reads ------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.runs_dir.glob("*.json")) if self.runs_dir.is_dir() else []:
            try:
                run = load_run(path.stem, runs_dir=self.runs_dir)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            ok = [e for e in run.examples if e.error is None]
            scorers: dict[str, list[float]] = {}
            for e in ok:
                for s in e.scores:
                    scorers.setdefault(s.name, []).append(s.score)
            out.append(
                {
                    "tag": run.tag,
                    "eval_name": run.eval_name,
                    "model": run.model,
                    "prompt_path": run.prompt_path,
                    "criteria": run.criteria,
                    "finished_at": run.finished_at,
                    "examples": len(run.examples),
                    "errors": len(run.examples) - len(ok),
                    "total_cost": run.total_cost if run.priced else None,
                    "judge_cost": run.total_judge_cost,
                    "scores": {k: sum(v) / len(v) for k, v in scorers.items()},
                }
            )
        out.sort(key=lambda r: str(r["finished_at"] or ""), reverse=True)
        return out

    def run(self, tag: str) -> dict[str, Any]:
        run = load_run(tag, runs_dir=self.runs_dir)
        return _as_dict(run) | {"total_cost": run.total_cost, "judge_cost": run.total_judge_cost}

    def judgements(self) -> list[dict[str, Any]]:
        found = []
        folder = self.judgements_dir
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found.append(
                {
                    k: data.get(k)
                    for k in ("baseline", "candidate", "judged_at", "model", "judged_by")
                }
            )
        return found

    def judgement(self, a: str, b: str) -> dict[str, Any] | None:
        folder = self.judgements_dir
        for path in folder.glob("*.json") if folder.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("baseline") == a and data.get("candidate") == b:
                return dict(data)
        return None

    def compare(self, a: str, b: str) -> dict[str, Any]:
        run_a = load_run(a, runs_dir=self.runs_dir)
        run_b = load_run(b, runs_dir=self.runs_dir)
        judged = self.judgement(a, b)
        if judged and isinstance(judged.get("scores"), dict):
            scores = {str(k): float(v) for k, v in judged["scores"].items()}
            run_b = _with_scores(run_b, scores)
            run_a = baseline_with_pairwise(run_a)
        report = compute_comparison(run_a, run_b)
        return {
            "report": _clean(report),
            "judgement": judged,
            "a": _as_dict(run_a) | {"total_cost": run_a.total_cost},
            "b": _as_dict(run_b) | {"total_cost": run_b.total_cost},
        }

    def criteria(self) -> list[dict[str, Any]]:
        out = []
        folder = self.criteria_dir
        for path in sorted(folder.glob("*")) if folder.is_dir() else []:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            out.append(
                {
                    "name": path.name,
                    "draft": any(
                        line.startswith(criteria_mod.DRAFT_MARKER) for line in text.splitlines()
                    ),
                }
            )
        return out

    def criteria_file(self, name: str) -> Path:
        safe = Path(unquote(name)).name  # a bare file name, never a path
        if not safe or safe.startswith("."):
            raise ValueError("not a criteria file name")
        if not safe.endswith((".md", ".txt")):
            safe += ".md"
        return self.criteria_dir / safe

    # ---- the one write ----------------------------------------------------

    def save_criteria(self, name: str, text: str) -> Path:
        path = self.criteria_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return path


def _with_scores(run: RunResult, scores: dict[str, float]) -> RunResult:
    examples: list[ExampleResult] = [
        replace(e, scores=[*e.scores, ScoreOutcome(PAIRWISE_SCORER, scores[e.id])])
        if e.error is None and e.id in scores
        else e
        for e in run.examples
    ]
    return replace(run, examples=examples)


def make_handler(project: Project, token: str, port_ref: list[int]) -> type[BaseHTTPRequestHandler]:
    page = files("prompteval.ui").joinpath("index.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "prompteval-ui"

        def log_message(self, format: str, *args: Any) -> None:  # quiet by default
            return

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            port = port_ref[0]
            return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, status: int = 200) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "wrong host")
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if not parts:
                    body = page.replace("__TOKEN__", token).replace(
                        "__ROOT__", json.dumps(str(project.root))[1:-1]
                    )
                    return self._send(200, body.encode(), "text/html; charset=utf-8")
                if parts == ["api", "state"]:
                    return self._json(
                        {
                            "root": str(project.root),
                            "runs": project.runs(),
                            "judgements": project.judgements(),
                            "criteria": project.criteria(),
                        }
                    )
                if parts[:2] == ["api", "run"] and len(parts) == 3:
                    return self._json(project.run(parts[2]))
                if parts == ["api", "compare"] and q.get("a") and q.get("b"):
                    return self._json(project.compare(q["a"], q["b"]))
                if parts[:2] == ["api", "criteria"] and len(parts) == 3:
                    path = project.criteria_file(parts[2])
                    if not path.is_file():
                        return self._error(HTTPStatus.NOT_FOUND, f"no {path.name}")
                    return self._json({"name": path.name, "text": path.read_text(encoding="utf-8")})
            except FileNotFoundError as err:
                return self._error(HTTPStatus.NOT_FOUND, str(err))
            except (ValueError, KeyError, TypeError) as err:
                return self._error(HTTPStatus.BAD_REQUEST, str(err))
            return self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_PUT(self) -> None:
            if not self._host_ok():
                return self._error(HTTPStatus.FORBIDDEN, "wrong host")
            if not secrets.compare_digest(self.headers.get("X-Prompteval-Token", ""), token):
                return self._error(HTTPStatus.FORBIDDEN, "missing or wrong token")
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
            if parts[:2] != ["api", "criteria"] or len(parts) != 3:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_CRITERIA_BYTES:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too large")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                text = data.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("criteria text is empty")
                path = project.save_criteria(parts[2], text)
            except (ValueError, json.JSONDecodeError) as err:
                return self._error(HTTPStatus.BAD_REQUEST, str(err))
            return self._json({"saved": path.name})

    return Handler


def serve(
    root: Path, runs_dir: Path | None = None, port: int = 8765
) -> tuple[ThreadingHTTPServer, str]:
    """A server bound to 127.0.0.1 (port 0 picks a free one) and its URL."""
    token = secrets.token_urlsafe(24)
    port_ref = [port]
    handler = make_handler(Project(root, runs_dir), token, port_ref)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    port_ref[0] = server.server_address[1]
    return server, f"http://127.0.0.1:{port_ref[0]}/"
