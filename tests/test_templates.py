"""#10: prompt templates with variables, and request settings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prompteval import Eval, run_eval, scorer
from prompteval.cli import main
from prompteval.eval.runner import Example, build_messages, fill_prompt, prompt_variables
from tests.test_cli_run import _mock_openai_client, _write_eval_dir


@scorer
def ok(output: str) -> float:
    return 1.0


def test_plain_prompt_keeps_system_and_user_shape() -> None:
    msgs = build_messages("You are helpful.", Example(id="a", input="hi"))
    assert msgs == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]


def test_json_in_a_plain_prompt_is_not_a_template() -> None:
    prompt = 'Reply as JSON like {"answer": "...", "nested": {"x": 1}}.'
    assert prompt_variables(prompt) == []
    assert build_messages(prompt, Example(id="a", input="hi"))[0]["content"] == prompt


def test_template_is_filled_from_input_and_vars() -> None:
    prompt = "Summarise this ticket for {customer}: {input}"
    msgs = build_messages(prompt, Example(id="a", input="Broken item", vars={"customer": "Asha"}))
    assert msgs == [{"role": "user", "content": "Summarise this ticket for Asha: Broken item"}]


def test_escaped_braces_stay_literal() -> None:
    assert (
        fill_prompt("Use {{name}} literally, fill {x}", {"x": 1}) == "Use {name} literally, fill 1"
    )


def test_missing_variable_names_itself() -> None:
    with pytest.raises(KeyError, match="'customer'"):
        fill_prompt("Hi {customer}", {"input": "x"})


def test_missing_variable_fails_that_example_only(tmp_path: Path) -> None:
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"id": "a", "input": "x", "vars": {"customer": "Asha"}})
        + "\n"
        + json.dumps({"id": "b", "input": "y"})
        + "\n"
    )
    prompt = tmp_path / "p.txt"
    prompt.write_text("Hi {customer}: {input}")
    result = run_eval(
        Eval(name="e", dataset=ds, scorers=[ok]), prompt, "t", client=_mock_openai_client()
    )
    assert result.examples[0].error is None
    assert result.examples[1].error is not None and "customer" in result.examples[1].error


def test_vars_must_be_an_object(tmp_path: Path) -> None:
    ds = tmp_path / "d.jsonl"
    ds.write_text(json.dumps({"id": "a", "input": "x", "vars": ["no"]}) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("x")
    with pytest.raises(ValueError, match="'vars' must be an object"):
        run_eval(
            Eval(name="e", dataset=ds, scorers=[ok]), prompt, "t", client=_mock_openai_client()
        )


def test_params_are_sent_and_recorded(tmp_path: Path) -> None:
    ds = tmp_path / "d.jsonl"
    ds.write_text(json.dumps({"id": "a", "input": "x"}) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    client = _mock_openai_client()
    result = run_eval(
        Eval(name="e", dataset=ds, scorers=[ok], params={"temperature": 0, "max_tokens": 50}),
        prompt,
        "t",
        client=client,
        params={"max_tokens": 20},  # run-level value wins
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0 and kwargs["max_tokens"] == 20
    assert result.params == {"temperature": 0, "max_tokens": 20}


def test_cli_param_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=1)
    monkeypatch.chdir(tmp_path)
    client = _mock_openai_client()
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        result = CliRunner().invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "t",
                "--eval-file",
                str(eval_file),
                "--param",
                "temperature=0",
                "--param",
                'response_format={"type": "json_object"}',
                "--param",
                "stop=END",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["stop"] == "END"


def test_compare_warns_when_settings_differ() -> None:
    from dataclasses import replace

    from prompteval.compare import compute_comparison, render_text
    from tests.test_compare_decisions import _run

    a = _run("v1", {"q": [1.0] * 3}, [1] * 3)
    b = replace(_run("v2", {"q": [1.0] * 3}, [1] * 3), params={"max_tokens": 100})
    assert "request settings differ" in render_text(compute_comparison(a, b))
