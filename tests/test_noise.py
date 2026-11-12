"""#14: repeats per example, a deterministic judge, and the noise command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prompteval import Eval, llm_judge, run_eval, scorer
from prompteval.cli import main
from tests.test_cli_run import _write_eval_dir
from tests.test_runner import _mock_openai_client


def _cycling_client(outputs: list[str]) -> Any:
    client = _mock_openai_client()
    done = client.chat.completions.create.return_value
    seq = iter(outputs * 100)

    def create(**kwargs: Any) -> Any:
        reply = MagicMock()
        reply.choices = [MagicMock(message=MagicMock(content=next(seq)))]
        reply.usage = done.usage
        return reply

    client.chat.completions.create.side_effect = create
    return client


@scorer
def says_yes(output: str) -> float:
    return 1.0 if output == "yes" else 0.0


def _dataset(tmp_path: Path, n: int = 2) -> tuple[Path, Path]:
    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join(json.dumps({"id": f"e{i}", "input": "q"}) for i in range(n)) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    return ds, prompt


def test_repeats_average_scores_and_report_cost_per_call(tmp_path: Path) -> None:
    ds, prompt = _dataset(tmp_path, n=1)
    once = run_eval(
        Eval(name="e", dataset=ds, scorers=[says_yes]), prompt, "a", client=_cycling_client(["yes"])
    )
    client = _cycling_client(["yes", "no", "yes", "no"])
    rep = run_eval(
        Eval(name="e", dataset=ds, scorers=[says_yes]), prompt, "b", client=client, repeats=4
    )
    assert client.chat.completions.create.call_count == 4
    assert rep.examples[0].scores[0].score == 0.5
    assert rep.examples[0].output == "yes"  # the first answer is kept
    assert rep.examples[0].cost.total_cost == pytest.approx(once.examples[0].cost.total_cost)
    assert rep.repeats == 4


def test_judge_defaults_to_temperature_zero() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="1"))
    ]
    llm_judge("rubric", "text", client=client)
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.0
    llm_judge("rubric", "text", client=client, temperature=None)
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_compare_warns_when_repeats_differ() -> None:
    from dataclasses import replace

    from prompteval.compare import compute_comparison, render_text
    from tests.test_compare_decisions import _run

    a = _run("a", {"q": [1.0] * 3}, [1] * 3)
    b = replace(_run("b", {"q": [1.0] * 3}, [1] * 3), repeats=3)
    assert "different number of times (1 vs 3)" in render_text(compute_comparison(a, b))


def test_noise_command_reports_a_floor_and_warns_on_a_noisy_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=20)
    monkeypatch.chdir(tmp_path)
    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()):
        steady = CliRunner().invoke(
            main, ["noise", "--prompt", str(prompt), "--eval-file", str(eval_file)]
        )
    assert steady.exit_code == 0, steady.output
    assert "changes smaller than ±0.000 look like noise" in steady.output
    assert "No scorer moved beyond noise" in steady.output

    # A scorer that flips between the two runs is flagged.
    (eval_file.parent / "eval.py").write_text(
        "from prompteval import Eval, scorer\n"
        "import itertools\n"
        "_calls = itertools.count()\n"
        "@scorer\n"
        "def flaky(output: str) -> float:\n"
        "    return 1.0 if next(_calls) < 20 else 0.0\n"
        "eval = Eval(name='cli-test', dataset='evals/dataset.jsonl', scorers=[flaky])\n"
    )
    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()):
        noisy = CliRunner().invoke(
            main,
            ["noise", "--prompt", str(prompt), "--eval-file", str(eval_file), "--concurrency", "1"],
        )
    assert noisy.exit_code == 0, noisy.output
    assert "Warning: flaky scored differently on an unchanged prompt" in noisy.output
