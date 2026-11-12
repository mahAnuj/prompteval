"""#17: a model with no known price runs with --unpriced; cost is never shown as $0."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prompteval import Eval, run_eval, scorer
from prompteval.cli import main
from prompteval.compare import (
    compute_comparison,
    evaluate_gates,
    parse_gate_spec,
    render_markdown,
    render_text,
)
from prompteval.cost import UnknownModelError
from prompteval.eval.runner import RunResult
from tests.test_cli_run import _mock_openai_client as cli_client
from tests.test_cli_run import _write_eval_dir
from tests.test_runner import _mock_openai_client

MODEL = "some-private-model"


@scorer
def ok(output: str) -> float:
    return 1.0


def _runs(tmp_path: Path) -> tuple[RunResult, RunResult]:
    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join(json.dumps({"id": f"e{i}", "input": "x"}) for i in range(20)) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    ev = Eval(name="e", dataset=ds, scorers=[ok])
    a = run_eval(
        ev, prompt, "base", client=_mock_openai_client(prompt_tokens=100, completion_tokens=50)
    )
    b = run_eval(
        ev,
        prompt,
        "private",
        model=MODEL,
        unpriced=True,
        client=_mock_openai_client(prompt_tokens=100, completion_tokens=25),
    )
    return a, b


def test_unknown_model_still_refused_without_the_flag(tmp_path: Path) -> None:
    ds = tmp_path / "d.jsonl"
    ds.write_text(json.dumps({"id": "a", "input": "x"}) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    with pytest.raises(UnknownModelError):
        run_eval(
            Eval(name="e", dataset=ds, scorers=[ok], model=MODEL),
            prompt,
            "t",
            client=_mock_openai_client(),
        )


def test_unpriced_run_records_tokens_and_marks_itself(tmp_path: Path) -> None:
    _, b = _runs(tmp_path)
    assert b.price_source == "unpriced" and not b.priced
    assert b.total_cost == 0
    assert b.examples[0].cost.output_tokens == 25


def test_compare_does_not_call_zero_cost_a_saving(tmp_path: Path) -> None:
    a, b = _runs(tmp_path)
    report = compute_comparison(a, b)
    assert report.unpriced_tags == ("private",)
    assert report.cost_verdict.startswith("not evaluated: no price for private")
    assert report.recommendation.startswith("quality holds; cost not evaluated")
    out = {t.name: t for t in report.token_deltas}
    assert out["output tokens"].delta_pct == pytest.approx(-50.0)
    text = render_text(report)
    assert "unpriced: no price for private" in text and "output tokens / example" in text
    assert "unpriced" in render_markdown(report)


def test_cost_gate_fails_closed_on_an_unpriced_run(tmp_path: Path) -> None:
    a, b = _runs(tmp_path)
    report = compute_comparison(a, b)
    for mode in ("bound", "significant"):
        breaches = evaluate_gates(report, parse_gate_spec("cost+10%"), mode=mode)
        assert len(breaches) == 1 and "cannot be evaluated" in breaches[0].detail
    assert evaluate_gates(report, parse_gate_spec("quality-5%")) == []


def test_cli_run_with_unpriced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=2)
    monkeypatch.chdir(tmp_path)
    with patch("prompteval.eval.runner.OpenAI", return_value=cli_client()):
        refused = CliRunner().invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "t",
                "--eval-file",
                str(eval_file),
                "--model",
                MODEL,
            ],
        )
        allowed = CliRunner().invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "t",
                "--eval-file",
                str(eval_file),
                "--model",
                MODEL,
                "--unpriced",
            ],
        )
    assert refused.exit_code != 0
    assert allowed.exit_code == 0, allowed.output
    assert "unpriced (tokens: 100 in, 40 out)" in allowed.output


def test_cli_sweep_with_an_unpriced_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=20)
    monkeypatch.chdir(tmp_path)
    with patch("prompteval.eval.runner.OpenAI", return_value=cli_client()):
        result = CliRunner().invoke(
            main,
            [
                "sweep",
                "--prompt",
                str(prompt),
                "--models",
                f"gpt-4o-mini,{MODEL}",
                "--eval-file",
                str(eval_file),
                "--unpriced",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "running unpriced" in result.output
    assert "quality holds, unpriced" in result.output
    assert "Cheapest model that holds quality" not in result.output  # never from $0
