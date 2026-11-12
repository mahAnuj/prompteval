"""#9: the same prompt on several models, each compared to a baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.compare.sweep import compute_sweep, render_sweep_markdown, render_sweep_text
from tests.test_cli_run import _write_eval_dir
from tests.test_compare_decisions import _run


def _model_aware_client() -> Any:
    """Every model returns 50 prompt + 20 completion tokens and the text "ok"."""
    client = MagicMock()

    def create(**kwargs: Any) -> Any:
        usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        usage.prompt_tokens_details = None
        usage.completion_tokens_details = None
        done = MagicMock()
        done.choices = [MagicMock(message=MagicMock(content="ok"))]
        done.usage = usage
        return done

    client.chat.completions.create.side_effect = create
    return client


def test_cheaper_model_that_holds_quality_is_the_answer() -> None:
    q = [1.0] * 20
    report = compute_sweep(
        {
            "big": _run("big", {"q": q}, [1000] * 20),
            "small": _run("small", {"q": q}, [100] * 20),
            "mid": _run("mid", {"q": q}, [500] * 20),
        },
        "big",
    )
    assert report.answer.startswith("Cheapest model that holds quality: small (-90% cost vs big)")
    assert report.rows[0].verdict == "baseline"
    text = render_sweep_text(report)
    assert "small" in text and "big (baseline)" in text
    md = render_sweep_markdown(report)
    assert md.startswith("**Cheapest model that holds quality: small")
    assert "| model |" in md


def test_a_cheaper_model_that_loses_quality_is_not_the_answer() -> None:
    report = compute_sweep(
        {
            "big": _run("big", {"q": [0.9] * 30}, [1000] * 30),
            "small": _run("small", {"q": [0.4] * 30}, [100] * 30),
        },
        "big",
    )
    assert report.rows[1].verdict == "worse quality"
    assert report.answer.startswith("No model is cheaper than big")


def test_thin_evidence_is_inconclusive() -> None:
    rng = np.random.default_rng(1)
    a = (rng.random(12) < 0.8).astype(float)
    b = (rng.random(12) < 0.7).astype(float)
    report = compute_sweep(
        {"big": _run("big", {"q": a}, [1000] * 12), "small": _run("small", {"q": b}, [100] * 12)},
        "big",
    )
    assert report.rows[1].verdict.startswith("inconclusive")
    assert "Add examples" in report.answer


def test_more_models_means_stricter_decisions() -> None:
    q = [1.0] * 10
    runs = {m: _run(m, {"q": q}, [100] * 10) for m in ("a", "b", "c", "d")}
    report = compute_sweep(runs, "a")
    levels = {r.report.decision_level for r in report.rows if r.report}
    assert levels == {1 - 0.05 / 3}


def test_sweep_requires_a_known_baseline() -> None:
    q = [1.0] * 3
    with pytest.raises(ValueError, match="Baseline"):
        compute_sweep({"a": _run("a", {"q": q}, [1] * 3), "b": _run("b", {"q": q}, [1] * 3)}, "z")


def _invoke(tmp_path: Path, eval_file: Path, prompt: Path, *extra: str) -> Any:
    return CliRunner().invoke(
        main,
        [
            "sweep",
            "--prompt",
            str(prompt),
            "--models",
            "gpt-4o,gpt-4o-mini",
            "--eval-file",
            str(eval_file),
            *extra,
        ],
    )


def test_cli_sweep_answers_and_reuses_saved_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=20)
    monkeypatch.chdir(tmp_path)
    client = _model_aware_client()
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        first = _invoke(tmp_path, eval_file, prompt, "--markdown", "sweep.md")
    assert first.exit_code == 0, first.output
    assert "Cheapest model that holds quality: gpt-4o-mini (-94% cost vs gpt-4o)" in first.output
    assert (tmp_path / "sweep.md").read_text().startswith("**Cheapest model")
    calls = client.chat.completions.create.call_count
    assert calls == 40

    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        second = _invoke(tmp_path, eval_file, prompt)
    assert second.exit_code == 0, second.output
    assert "reusing saved run" in second.output
    assert client.chat.completions.create.call_count == calls  # nothing re-run, nothing paid

    prompt.write_text("a different prompt")  # a change invalidates the saved runs
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        third = _invoke(tmp_path, eval_file, prompt)
    assert third.exit_code == 0, third.output
    assert client.chat.completions.create.call_count == calls + 40


def test_cli_sweep_checks_every_price_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=2)
    monkeypatch.chdir(tmp_path)
    client = _model_aware_client()
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        result = CliRunner().invoke(
            main,
            [
                "sweep",
                "--prompt",
                str(prompt),
                "--models",
                "gpt-4o,not-priced",
                "--eval-file",
                str(eval_file),
            ],
        )
    assert result.exit_code != 0
    assert "--price input=X,output=Y" in result.output
    client.chat.completions.create.assert_not_called()


def test_cli_sweep_uses_endpoints_from_prices_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=3)
    (eval_file.parent / "prices.yaml").write_text(
        "models:\n"
        "  - name: llama-3.3-70b-versatile\n"
        "    input_per_1m: 0.59\n"
        "    output_per_1m: 0.79\n"
        "    base_url: https://api.groq.com/openai/v1\n"
        "    api_key_env: GROQ_API_KEY\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    with patch("prompteval.eval.runner.OpenAI", return_value=_model_aware_client()) as made:
        result = CliRunner().invoke(
            main,
            [
                "sweep",
                "--prompt",
                str(prompt),
                "--models",
                "gpt-4o-mini,llama-3.3-70b-versatile",
                "--eval-file",
                str(eval_file),
            ],
        )
    assert result.exit_code == 0, result.output
    made.assert_any_call(
        base_url="https://api.groq.com/openai/v1", api_key="gsk_test", max_retries=5
    )
