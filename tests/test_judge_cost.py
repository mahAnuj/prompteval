"""#15: LLM-judge calls are priced and reported apart from the prompt's cost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from prompteval import Eval, llm_judge, run_eval, scorer
from prompteval.compare import compute_comparison, render_markdown, render_text
from prompteval.eval import load_run, save_run
from tests.test_runner import _mock_openai_client


def _judge_client(prompt_tokens: int = 1_000_000, completion_tokens: int = 0) -> Any:
    client = MagicMock()
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    usage.prompt_tokens_details = None
    done = MagicMock()
    done.choices = [MagicMock(message=MagicMock(content="0.8"))]
    done.usage = usage
    client.chat.completions.create.return_value = done
    return client


def _eval(tmp_path: Path, judge_model: str = "gpt-4o-mini") -> tuple[Eval, Path]:
    judge = _judge_client()

    @scorer
    def tone(output: str) -> float:
        return llm_judge("Rate 0-1.", output, model=judge_model, client=judge)

    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join(json.dumps({"id": f"e{i}", "input": "x"}) for i in range(3)) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    return Eval(name="e", dataset=ds, scorers=[tone]), prompt


def test_judge_calls_are_priced_per_example_and_kept_out_of_prompt_cost(tmp_path: Path) -> None:
    eval_def, prompt = _eval(tmp_path)
    result = run_eval(eval_def, prompt, "t", client=_mock_openai_client())
    # 1M judge input tokens at gpt-4o-mini's $0.15 per 1M, per example.
    assert [round(e.judge_cost or 0, 6) for e in result.examples] == [0.15, 0.15, 0.15]
    assert round(result.total_judge_cost or 0, 6) == 0.45
    assert result.total_cost < 0.01  # the prompt's own cost is unaffected


def test_unpriced_judge_model_is_reported_not_zero(tmp_path: Path) -> None:
    eval_def, prompt = _eval(tmp_path, judge_model="some-private-judge")
    result = run_eval(eval_def, prompt, "t", client=_mock_openai_client())
    assert result.total_judge_cost is None
    assert all(e.error is None for e in result.examples)


def test_judge_cost_survives_reload_and_shows_in_reports(tmp_path: Path) -> None:
    eval_def, prompt = _eval(tmp_path)
    a = run_eval(eval_def, prompt, "a", client=_mock_openai_client())
    b = run_eval(eval_def, prompt, "b", client=_mock_openai_client())
    save_run(a, runs_dir=tmp_path / "runs")
    assert round(load_run("a", runs_dir=tmp_path / "runs").total_judge_cost or 0, 6) == 0.45
    report = compute_comparison(a, b)
    assert (
        "Judge cost (scoring overhead, not in the cost verdict): a $0.4500, b $0.4500"
        in render_text(report)
    )
    assert "Judge cost" in render_markdown(report)


def test_no_judge_line_without_a_judge(tmp_path: Path) -> None:
    from tests.test_compare_decisions import _run

    report = compute_comparison(
        _run("a", {"q": [1.0] * 3}, [1] * 3), _run("b", {"q": [1.0] * 3}, [1] * 3)
    )
    assert "Judge cost" not in render_text(report)
