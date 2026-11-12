"""#19: a pairwise judge compares two runs' answers directly, with no scorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.compare import compute_comparison, render_markdown, render_text
from prompteval.compare.pairwise import (
    PAIRWISE_SCORER,
    baseline_with_pairwise,
    default_judge,
    judge_pairwise,
    pairwise_notes,
    with_pairwise,
)
from prompteval.cost import Usage, compute_cost
from prompteval.eval import save_run
from prompteval.eval.runner import ExampleResult, RunResult


def _run(tag: str, outputs: list[str], model: str = "gpt-4o-mini") -> RunResult:
    return RunResult(
        tag=tag,
        eval_name="e",
        model=model,
        prompt_path="p.txt",
        prompt_text=tag,
        started_at="",
        finished_at="",
        examples=[
            ExampleResult(
                id=f"e{i}",
                input=f"question {i}",
                expected={},
                output=out,
                scores=[],
                cost=compute_cost(model, Usage(prompt_tokens=100, completion_tokens=50)),
                latency_s=1.0,
            )
            for i, out in enumerate(outputs)
        ],
    )


def _judge(rule: str) -> Any:
    """A fake judge. rule: 'quality' picks the answer containing GOOD; 'first'
    always says A (pure position bias); 'garbage' replies nonsense."""
    client = MagicMock()

    def create(**kwargs: Any) -> Any:
        user = kwargs["messages"][-1]["content"]
        a_part = user.split("Answer A:\n", 1)[1].split("\n\nAnswer B:\n", 1)[0]
        b_part = user.split("\n\nAnswer B:\n", 1)[1]
        if rule == "first":
            reply = "A"
        elif rule == "garbage":
            reply = "I think both are fine"
        elif "GOOD" in a_part and "GOOD" not in b_part:
            reply = "A"
        elif "GOOD" in b_part and "GOOD" not in a_part:
            reply = "B"
        else:
            reply = "TIE"
        done = MagicMock()
        done.choices = [MagicMock(message=MagicMock(content=reply))]
        done.usage = MagicMock(prompt_tokens=1_000_000, completion_tokens=0)
        return done

    client.chat.completions.create.side_effect = create
    return client


def _compare(a: RunResult, b: RunResult, rule: str) -> tuple[Any, Any]:
    result = judge_pairwise(a, b, client=_judge(rule), model="gpt-4.1-mini")
    report = compute_comparison(baseline_with_pairwise(a), with_pairwise(b, result))
    return result, report


def test_better_candidate_wins_every_example() -> None:
    a, b = _run("v1", ["bad"] * 20), _run("v2", ["GOOD"] * 20)
    result, report = _compare(a, b, "quality")
    assert set(result.scores.values()) == {1.0}
    assert result.order_disagreements == 0
    d = next(x for x in report.scorer_deltas if x.name == PAIRWISE_SCORER)
    assert d.status == "ok" and d.delta == pytest.approx(0.5)


def test_worse_candidate_is_a_regression() -> None:
    a, b = _run("v1", ["GOOD"] * 20), _run("v2", ["bad"] * 20)
    _, report = _compare(a, b, "quality")
    assert report.recommendation.startswith("don't ship v2")
    assert PAIRWISE_SCORER in report.recommendation


def test_position_bias_becomes_ties_not_wins() -> None:
    a, b = _run("v1", ["x"] * 10), _run("v2", ["y"] * 10)
    result, report = _compare(a, b, "first")
    assert result.order_disagreements == 10
    assert set(result.scores.values()) == {0.5}
    d = next(x for x in report.scorer_deltas if x.name == PAIRWISE_SCORER)
    assert d.delta == pytest.approx(0.0)


def test_unreadable_replies_count_as_ties() -> None:
    result, _ = _compare(_run("v1", ["x"] * 3), _run("v2", ["y"] * 3), "garbage")
    assert result.unreadable == 6 and set(result.scores.values()) == {0.5}


def test_judge_cost_and_notes() -> None:
    a, b = _run("v1", ["bad"] * 4), _run("v2", ["GOOD"] * 4)
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4.1-mini")
    # 8 calls x 1M input tokens at gpt-4.1-mini's input price.
    per_m = compute_cost(
        "gpt-4.1-mini", Usage(prompt_tokens=1_000_000, completion_tokens=0)
    ).total_cost
    assert result.cost == pytest.approx(8 * per_m)
    note = " ".join(pairwise_notes(result, a, b))
    assert "gpt-4.1-mini" in note and "judged twice with the answers swapped" in note


def test_default_judge_avoids_the_compared_models_and_warns_otherwise() -> None:
    a, b = _run("v1", ["x"], model="gpt-4.1-mini"), _run("v2", ["y"], model="gpt-4o")
    assert default_judge(a, b) == "gpt-4o-mini"
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4o")
    assert any("also one of the models being compared" in n for n in pairwise_notes(result, a, b))


def test_errored_examples_are_not_judged() -> None:
    from dataclasses import replace

    a, b = _run("v1", ["bad"] * 3), _run("v2", ["GOOD"] * 3)
    b = replace(b, examples=[replace(b.examples[0], error="Timeout"), *b.examples[1:]])
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4.1-mini")
    assert set(result.scores) == {"e1", "e2"}


def test_no_quality_measure_is_never_called_holds() -> None:
    report = compute_comparison(_run("v1", ["x"] * 5), _run("v2", ["y"] * 5))
    assert report.quality_verdict.startswith("not evaluated")
    assert report.recommendation.startswith("inconclusive — nothing measures quality")
    assert "--pairwise" in report.recommendation


def test_cli_compare_pairwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    save_run(_run("v1", ["bad"] * 20))
    save_run(_run("v2", ["GOOD"] * 20))
    with patch("prompteval.eval.runner.OpenAI", return_value=_judge("quality")):
        result = CliRunner().invoke(
            main, ["compare", "v1", "v2", "--pairwise", "--markdown", "r.md"]
        )
    assert result.exit_code == 0, result.output
    assert PAIRWISE_SCORER in result.output
    assert "Note: Pairwise judge gpt-4.1-mini" in result.output
    assert "Pairwise judge" in (tmp_path / "r.md").read_text()


def test_reports_render_notes() -> None:
    a, b = _run("v1", ["bad"] * 5), _run("v2", ["GOOD"] * 5)
    result, report = _compare(a, b, "quality")
    from dataclasses import replace

    report = replace(report, notes=pairwise_notes(result, a, b))
    assert "Note: Pairwise judge" in render_text(report)
    assert "> Pairwise judge" in render_markdown(report)
    json.dumps([report.recommendation])  # plain strings throughout
