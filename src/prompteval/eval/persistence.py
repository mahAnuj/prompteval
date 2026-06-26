"""Persist + load RunResults as JSON files in `.prompteval/runs/<tag>.json`.

Why JSON files (not SQLite or a server)?
- prompteval is a single-developer tool by design for v1
- Git-friendly (commit runs/ if you want history)
- Zero setup (no migrations, no schema)
- Trivial to inspect with `jq` or grep
- If v1.x ever needs query performance, swap to SQLite — the data shape
  doesn't change

Latest-wins-on-rerun. If you want history, save under unique tags
(e.g. `baseline-v1`, `baseline-v2`) and rely on git for archival.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prompteval.cost import CostBreakdown
from prompteval.eval.runner import ExampleResult, RunResult, ScoreOutcome

#: Default location for run files, relative to the user's project root.
DEFAULT_RUNS_DIR = Path(".prompteval/runs")


def save_run(result: RunResult, runs_dir: Path | None = None) -> Path:
    """Write `result` to `<runs_dir>/<tag>.json`. Returns the written path.

    Creates `runs_dir` (and any parents) if missing. Overwrites an existing
    file for the same tag — re-running with the same tag replaces the prior
    snapshot. Caller can rotate via unique tags if they want history.
    """
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{_safe_filename(result.tag)}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return path


def load_run(tag: str, runs_dir: Path | None = None) -> RunResult:
    """Reconstruct a RunResult from disk. Raises FileNotFoundError if missing."""
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    path = runs_dir / f"{_safe_filename(tag)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No run found for tag {tag!r} in {runs_dir}. "
            f"Run `prompteval run --tag {tag} --prompt ...` to create one."
        )
    raw = json.loads(path.read_text())
    return _result_from_dict(raw)


def _result_from_dict(raw: dict[str, Any]) -> RunResult:
    """Inverse of `dataclasses.asdict` for RunResult — the dataclasses are
    nested, so we reconstruct manually rather than wave a wand."""
    examples = [
        ExampleResult(
            id=e["id"],
            input=e["input"],
            expected=e.get("expected", {}),
            output=e["output"],
            scores=[
                ScoreOutcome(
                    name=s["name"],
                    score=s["score"],
                    reasoning=s.get("reasoning"),
                    metadata=s.get("metadata"),
                )
                for s in e["scores"]
            ],
            cost=CostBreakdown(**e["cost"]),
            latency_s=e["latency_s"],
            error=e.get("error"),
        )
        for e in raw["examples"]
    ]
    return RunResult(
        tag=raw["tag"],
        eval_name=raw["eval_name"],
        model=raw["model"],
        prompt_path=raw["prompt_path"],
        prompt_text=raw["prompt_text"],
        started_at=raw["started_at"],
        finished_at=raw["finished_at"],
        examples=examples,
    )


def _safe_filename(tag: str) -> str:
    """Strip path separators + control chars from a tag.

    Defense in depth: even if a malicious tag tries to escape the runs dir
    via `../../etc/passwd`, the resulting filename stays inside `runs_dir`.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    return cleaned.strip("_.") or "untagged"
