"""Tests for `prompteval run` CLI — end-to-end with the OpenAI client mocked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prompteval.cli import main


def _write_eval_dir(tmp_path: Path, dataset_rows: int = 2) -> tuple[Path, Path]:
    """Build a minimal evals/ tree (eval.py, dataset.jsonl, prompts/v1.txt)
    in `tmp_path` and return (eval_file_path, prompt_path)."""
    evals = tmp_path / "evals"
    evals.mkdir()
    prompts = evals / "prompts"
    prompts.mkdir()

    (prompts / "v1.txt").write_text("be helpful")

    rows = [
        {"id": f"ex-{i}", "input": f"hello {i}", "expected": {"target": "ok"}}
        for i in range(dataset_rows)
    ]
    (evals / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    (evals / "eval.py").write_text(
        "from prompteval import Eval, scorer\n"
        "\n"
        "@scorer\n"
        "def always_pass(output: str) -> float:\n"
        "    return 1.0\n"
        "\n"
        "eval = Eval(\n"
        "    name='cli-test',\n"
        "    dataset='evals/dataset.jsonl',\n"
        "    scorers=[always_pass],\n"
        "    model='gpt-4o-mini',\n"
        ")\n"
    )
    return evals / "eval.py", prompts / "v1.txt"


def _mock_openai_client() -> Any:
    """Mock OpenAI client returning a small canned completion."""
    client = MagicMock()
    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 20
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="ok"))]
    completion.usage = usage
    client.chat.completions.create.return_value = completion
    return client


def test_run_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=3)
    monkeypatch.chdir(tmp_path)

    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "baseline",
                "--eval-file",
                str(eval_file),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Running 3 examples × gpt-4o-mini" in result.output  # noqa: RUF001
    assert "=== baseline ===" in result.output
    assert "always_pass" in result.output
    assert "saved to:" in result.output

    saved = tmp_path / ".prompteval" / "runs" / "baseline.json"
    assert saved.exists()

    data = json.loads(saved.read_text())
    assert data["tag"] == "baseline"
    assert len(data["examples"]) == 3
    assert all(ex["error"] is None for ex in data["examples"])


def test_run_model_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    mock_client = _mock_openai_client()
    with patch("prompteval.eval.runner.OpenAI", return_value=mock_client):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "alt",
                "--eval-file",
                str(eval_file),
                "--model",
                "gpt-4o",
            ],
        )

    assert result.exit_code == 0, result.output
    # OpenAI was called with the override model
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"


def test_run_missing_eval_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a valid prompt file so Click's path-exists check on --prompt
    succeeds; this lets the missing --eval-file path surface its own error."""
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.txt"
    prompt.write_text("be helpful")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--prompt", str(prompt), "--tag", "x", "--eval-file", "nope.py"],
    )
    assert result.exit_code != 0
    assert "nope.py" in result.output


def test_run_eval_file_without_eval_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Write an eval.py that has NO Eval instance.
    eval_file = tmp_path / "evals" / "eval.py"
    eval_file.parent.mkdir()
    eval_file.write_text("x = 42\n")

    prompt = tmp_path / "p.txt"
    prompt.write_text("be helpful")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--prompt", str(prompt), "--tag", "x", "--eval-file", str(eval_file)],
    )
    assert result.exit_code != 0
    assert "No Eval instance found" in result.output


def test_run_eval_file_import_error_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    eval_file = tmp_path / "evals" / "eval.py"
    eval_file.parent.mkdir()
    eval_file.write_text("import nonexistent_module_xyz\n")

    prompt = tmp_path / "p.txt"
    prompt.write_text("be helpful")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--prompt", str(prompt), "--tag", "x", "--eval-file", str(eval_file)],
    )
    assert result.exit_code != 0
    assert "Failed to load" in result.output


def test_run_progress_shows_each_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One progress line per example, with the example id visible."""
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=3)
    monkeypatch.chdir(tmp_path)

    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--prompt", str(prompt), "--tag", "baseline", "--eval-file", str(eval_file)],
        )

    assert "[1/3] ex-0" in result.output
    assert "[2/3] ex-1" in result.output
    assert "[3/3] ex-2" in result.output
