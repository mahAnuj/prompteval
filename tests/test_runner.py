"""Tests for the Eval runner — the orchestrator that ties prompt + dataset +
scorers + LLM call + cost computation together.

OpenAI is fully mocked here. The default_runner is exercised against a
MagicMock chat.completions.create call so we don't hit the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from prompteval import Eval, Example, run_eval, scorer
from prompteval.eval.runner import (
    _call_scorer,
    _load_dataset,
    _sum_usages,
    default_runner,
)
from prompteval.eval.scorer import ScorerResult

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _mock_openai_client(
    response_text: str = "ok",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
) -> Any:
    """Build a MagicMock OpenAI client returning a single canned response."""
    client = MagicMock()
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    if cached_tokens:
        usage.prompt_tokens_details = MagicMock(cached_tokens=cached_tokens)
    else:
        usage.prompt_tokens_details = None
    usage.completion_tokens_details = None

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=response_text))]
    completion.usage = usage
    client.chat.completions.create.return_value = completion
    return client


def _write_dataset(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _write_prompt(tmp_path: Path, text: str = "You are helpful.") -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text(text)
    return p


# Real scorers used across tests (proper @scorer decoration so dispatch works).


@scorer
def always_one(output: str) -> float:
    return 1.0


@scorer
def needs_expected(output: str, expected: dict[str, Any]) -> float:
    return 1.0 if output == expected.get("target") else 0.0


@scorer
def needs_all_three(input: str, output: str, expected: dict[str, Any]) -> float:
    # Sanity: scorer receives all three when declared.
    assert isinstance(input, str)
    assert isinstance(output, str)
    assert isinstance(expected, dict)
    return 0.5


@scorer
def returns_result(output: str) -> ScorerResult:
    return ScorerResult(score=0.7, reasoning="partial", metadata={"k": "v"})


@scorer
def bad_score_range(output: str) -> float:
    return 1.5  # invalid: out of [0, 1]


@scorer
def raises(output: str) -> float:
    raise RuntimeError("scorer blew up")


# ---------------------------------------------------------------------------
# Eval class validation
# ---------------------------------------------------------------------------


def test_eval_requires_at_least_one_scorer() -> None:
    with pytest.raises(ValueError, match="no scorers"):
        Eval(name="empty", dataset="x.jsonl", scorers=[])


def test_eval_rejects_undecorated_function_in_scorers() -> None:
    def not_a_scorer(output: str) -> float:
        return 1.0

    with pytest.raises(TypeError, match="not a @scorer"):
        Eval(name="bad", dataset="x.jsonl", scorers=[not_a_scorer])


def test_eval_accepts_valid_scorers() -> None:
    eval_def = Eval(name="good", dataset="x.jsonl", scorers=[always_one])
    assert eval_def.scorers == [always_one]
    assert eval_def.model == "gpt-4o-mini"  # default


# ---------------------------------------------------------------------------
# default_runner
# ---------------------------------------------------------------------------


def test_default_runner_returns_single_usage() -> None:
    client = _mock_openai_client(
        response_text="hello world",
        prompt_tokens=120,
        completion_tokens=30,
    )
    ex = Example(id="e1", input="hi", expected={})
    result = default_runner(ex, "be polite", "gpt-4o-mini", client)

    assert result.output == "hello world"
    assert len(result.usages) == 1
    assert result.usages[0].prompt_tokens == 120
    assert result.usages[0].completion_tokens == 30


def test_default_runner_extracts_cached_tokens() -> None:
    client = _mock_openai_client(
        response_text="x",
        prompt_tokens=200,
        completion_tokens=10,
        cached_tokens=80,
    )
    ex = Example(id="e1", input="hi", expected={})
    result = default_runner(ex, "system", "gpt-4o", client)

    assert result.usages[0].cached_tokens == 80


def test_default_runner_handles_missing_usage_gracefully() -> None:
    """If OpenAI returns no usage block, we still produce a RunnerResult
    (with zeroed counts) rather than crashing."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="hi"))]
    completion.usage = None
    client.chat.completions.create.return_value = completion

    ex = Example(id="e1", input="hi", expected={})
    result = default_runner(ex, "sys", "gpt-4o-mini", client)
    assert result.output == "hi"
    assert result.usages[0].prompt_tokens == 0


def test_default_runner_handles_none_content() -> None:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=None))]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 0
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None
    completion.usage = usage
    client.chat.completions.create.return_value = completion

    ex = Example(id="e1", input="hi", expected={})
    result = default_runner(ex, "sys", "gpt-4o-mini", client)
    assert result.output == ""


# ---------------------------------------------------------------------------
# _load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_happy_path(tmp_path: Path) -> None:
    p = _write_dataset(
        tmp_path,
        [
            {"id": "a", "input": "1", "expected": {"x": 1}},
            {"id": "b", "input": "2", "expected": {}},
        ],
    )
    examples = _load_dataset(p)
    assert len(examples) == 2
    assert examples[0].id == "a"
    assert examples[1].expected == {}


def test_load_dataset_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text('{"id": "a", "input": "x"}\n\n# comment line\n{"id": "b", "input": "y"}\n')
    examples = _load_dataset(p)
    assert [e.id for e in examples] == ["a", "b"]


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_dataset(tmp_path / "nope.jsonl")


def test_load_dataset_invalid_json_raises_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "ds.jsonl"
    p.write_text('{"id": "a", "input": "x"}\nNOT JSON\n')
    with pytest.raises(ValueError, match=":2:"):
        _load_dataset(p)


def test_load_dataset_missing_required_field_raises(tmp_path: Path) -> None:
    p = _write_dataset(tmp_path, [{"id": "a"}])  # no `input`
    with pytest.raises(ValueError, match="missing required field 'input'"):
        _load_dataset(p)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = _write_dataset(tmp_path, [{"id": "a", "input": "1"}, {"id": "a", "input": "2"}])
    with pytest.raises(ValueError, match="duplicate id"):
        _load_dataset(p)


def test_load_dataset_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("\n# only a comment\n")
    with pytest.raises(ValueError, match="dataset is empty"):
        _load_dataset(p)


# ---------------------------------------------------------------------------
# _call_scorer (signature dispatch)
# ---------------------------------------------------------------------------


def test_call_scorer_dispatches_only_declared_params() -> None:
    ex = Example(id="e", input="hi", expected={"target": "hi"})
    outcome = _call_scorer(needs_expected, ex, output="hi")
    assert outcome.name == "needs_expected"
    assert outcome.score == 1.0


def test_call_scorer_passes_input_when_declared() -> None:
    ex = Example(id="e", input="hi", expected={})
    outcome = _call_scorer(needs_all_three, ex, output="anything")
    assert outcome.score == 0.5


def test_call_scorer_handles_scorer_result() -> None:
    ex = Example(id="e", input="hi", expected={})
    outcome = _call_scorer(returns_result, ex, output="anything")
    assert outcome.score == 0.7
    assert outcome.reasoning == "partial"
    assert outcome.metadata == {"k": "v"}


def test_call_scorer_rejects_out_of_range_score() -> None:
    ex = Example(id="e", input="hi", expected={})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _call_scorer(bad_score_range, ex, output="x")


# ---------------------------------------------------------------------------
# _sum_usages
# ---------------------------------------------------------------------------


def test_sum_usages_single_is_passthrough() -> None:
    from prompteval import Usage

    u = Usage(prompt_tokens=100, completion_tokens=50, cached_tokens=20)
    assert _sum_usages([u]) is u


def test_sum_usages_aggregates_fields() -> None:
    from prompteval import Usage

    a = Usage(prompt_tokens=100, completion_tokens=50, cached_tokens=20, reasoning_tokens=5)
    b = Usage(prompt_tokens=200, completion_tokens=70, cached_tokens=40, reasoning_tokens=10)
    summed = _sum_usages([a, b])
    assert summed.prompt_tokens == 300
    assert summed.completion_tokens == 120
    assert summed.cached_tokens == 60
    assert summed.reasoning_tokens == 15


# ---------------------------------------------------------------------------
# run_eval — end-to-end orchestration
# ---------------------------------------------------------------------------


def test_run_eval_end_to_end_happy_path(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {"id": "a", "input": "x"},
            {"id": "b", "input": "y"},
        ],
    )
    prompt = _write_prompt(tmp_path, "be helpful")
    eval_def = Eval(
        name="end-to-end",
        dataset=dataset,
        scorers=[always_one, needs_expected],
        model="gpt-4o-mini",
    )
    client = _mock_openai_client(response_text="ok", prompt_tokens=80, completion_tokens=20)

    result = run_eval(eval_def, prompt_path=prompt, tag="my-tag", client=client)

    assert result.tag == "my-tag"
    assert result.eval_name == "end-to-end"
    assert result.model == "gpt-4o-mini"
    assert result.prompt_path == str(prompt)
    assert result.prompt_text == "be helpful"
    assert len(result.examples) == 2

    # Per-example scoring: always_one returns 1.0; needs_expected returns 0 (no match)
    for ex in result.examples:
        names = {s.name for s in ex.scores}
        assert names == {"always_one", "needs_expected"}
        always_one_score = next(s.score for s in ex.scores if s.name == "always_one")
        assert always_one_score == 1.0

    # Cost > 0 because gpt-4o-mini has positive input/output rates
    assert result.total_cost > 0
    assert result.scorer_means["always_one"] == 1.0


def test_run_eval_records_per_example_errors_without_aborting(tmp_path: Path) -> None:
    """If one scorer raises, that example records an error but the run completes."""
    dataset = _write_dataset(
        tmp_path,
        [
            {"id": "a", "input": "1"},
            {"id": "b", "input": "2"},
            {"id": "c", "input": "3"},
        ],
    )
    prompt = _write_prompt(tmp_path)
    eval_def = Eval(name="e", dataset=dataset, scorers=[raises])
    client = _mock_openai_client()

    result = run_eval(eval_def, prompt_path=prompt, tag="t", client=client)

    assert len(result.examples) == 3
    assert all(ex.error is not None for ex in result.examples)
    # Aggregates ignore errored examples.
    assert result.total_cost == 0.0
    assert result.scorer_means == {}


def test_run_eval_model_override_replaces_eval_default(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [{"id": "a", "input": "1"}])
    prompt = _write_prompt(tmp_path)
    eval_def = Eval(name="e", dataset=dataset, scorers=[always_one], model="gpt-4o-mini")
    client = _mock_openai_client()

    result = run_eval(eval_def, prompt_path=prompt, tag="t", model="gpt-4o", client=client)

    assert result.model == "gpt-4o"
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"


def test_run_eval_progress_callback_called_per_example(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [{"id": "a", "input": "1"}, {"id": "b", "input": "2"}])
    prompt = _write_prompt(tmp_path)
    eval_def = Eval(name="e", dataset=dataset, scorers=[always_one])
    client = _mock_openai_client()

    calls: list[tuple[int, int, str, str | None]] = []

    def progress(i: int, n: int, ex_id: str, latency: float, error: str | None) -> None:
        calls.append((i, n, ex_id, error))

    run_eval(eval_def, prompt_path=prompt, tag="t", client=client, progress=progress)
    assert calls == [(1, 2, "a", None), (2, 2, "b", None)]


def test_run_eval_aggregates_scorer_means(tmp_path: Path) -> None:
    """3 examples; always_one returns 1.0 each; mean should be 1.0."""
    dataset = _write_dataset(
        tmp_path,
        [{"id": f"e{i}", "input": "x"} for i in range(3)],
    )
    prompt = _write_prompt(tmp_path)
    eval_def = Eval(name="e", dataset=dataset, scorers=[always_one])
    client = _mock_openai_client()

    result = run_eval(eval_def, prompt_path=prompt, tag="t", client=client)
    assert result.scorer_means["always_one"] == 1.0
