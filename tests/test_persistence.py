"""Tests for run persistence — save_run + load_run round trip."""

from __future__ import annotations

import json
from pathlib import Path

from prompteval import CostBreakdown, ExampleResult, RunResult, load_run, save_run
from prompteval.eval.persistence import _safe_filename
from prompteval.eval.runner import ScoreOutcome


def _make_result(tag: str = "baseline") -> RunResult:
    """Build a small, complete RunResult for round-trip testing."""
    cost = CostBreakdown(
        model="gpt-4o-mini",
        uncached_input_tokens=80,
        cached_input_tokens=20,
        output_tokens=40,
        reasoning_tokens=0,
        uncached_input_cost=0.000012,
        cached_input_cost=0.0000015,
        output_cost=0.000024,
        total_cost=0.0000375,
    )
    examples = [
        ExampleResult(
            id="ex-1",
            input="hi",
            expected={"target": "hi"},
            output="hi",
            scores=[
                ScoreOutcome(name="exact_match", score=1.0),
                ScoreOutcome(name="judge", score=0.8, reasoning="great"),
            ],
            cost=cost,
            latency_s=0.42,
        ),
        ExampleResult(
            id="ex-2",
            input="bye",
            expected={"target": "bye"},
            output="bye",
            scores=[
                ScoreOutcome(name="exact_match", score=1.0),
                ScoreOutcome(name="judge", score=0.9),
            ],
            cost=cost,
            latency_s=0.55,
        ),
    ]
    return RunResult(
        tag=tag,
        eval_name="my-eval",
        model="gpt-4o-mini",
        prompt_path="prompts/v1.txt",
        prompt_text="be helpful",
        started_at="2026-06-26T15:00:00+00:00",
        finished_at="2026-06-26T15:00:01+00:00",
        examples=examples,
    )


def test_save_creates_file_at_expected_path(tmp_path: Path) -> None:
    result = _make_result(tag="baseline")
    path = save_run(result, runs_dir=tmp_path)
    assert path == tmp_path / "baseline.json"
    assert path.exists()


def test_saved_file_is_valid_json(tmp_path: Path) -> None:
    result = _make_result()
    path = save_run(result, runs_dir=tmp_path)
    data = json.loads(path.read_text())
    assert data["tag"] == "baseline"
    assert data["eval_name"] == "my-eval"
    assert len(data["examples"]) == 2


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = _make_result()
    save_run(original, runs_dir=tmp_path)
    loaded = load_run("baseline", runs_dir=tmp_path)

    assert loaded.tag == original.tag
    assert loaded.eval_name == original.eval_name
    assert loaded.model == original.model
    assert loaded.prompt_text == original.prompt_text
    assert len(loaded.examples) == len(original.examples)
    assert loaded.examples[0].id == original.examples[0].id
    assert loaded.examples[0].scores[0].name == "exact_match"
    assert loaded.examples[0].scores[1].reasoning == "great"
    # Aggregates work post-reload
    assert loaded.total_cost == original.total_cost
    assert loaded.scorer_means == original.scorer_means


def test_save_overwrites_existing_file_for_same_tag(tmp_path: Path) -> None:
    """Latest-wins semantics — re-running with the same tag overwrites prior."""
    save_run(_make_result(tag="t"), runs_dir=tmp_path)
    save_run(_make_result(tag="t"), runs_dir=tmp_path)
    # Just one file with that tag
    matches = list(tmp_path.glob("t*.json"))
    assert len(matches) == 1


def test_load_missing_tag_raises_friendly_error(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="No run found for tag 'nope'"):
        load_run("nope", runs_dir=tmp_path)


def test_safe_filename_strips_path_separators() -> None:
    """Defense in depth: a malicious tag shouldn't escape the runs dir."""
    assert "/" not in _safe_filename("../../etc/passwd")
    assert _safe_filename("../../etc/passwd").endswith("etc_passwd")


def test_safe_filename_handles_empty_input() -> None:
    assert _safe_filename("") == "untagged"
    assert _safe_filename("___") == "untagged"


def test_save_creates_runs_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "runs"
    save_run(_make_result(), runs_dir=nested)
    assert (nested / "baseline.json").exists()
