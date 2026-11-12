"""#8: any OpenAI-compatible provider, with user-supplied prices."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.cost import (
    UnknownModelError,
    Usage,
    compute_cost,
    get_pricing,
    load_price_file,
    parse_price_spec,
    resolve_pricing,
)
from prompteval.eval.runner import _make_client
from tests.test_cli_run import _mock_openai_client, _write_eval_dir

GROQ = "llama-3.3-70b-versatile"


def test_parse_price_spec() -> None:
    p = parse_price_spec(GROQ, "input=0.59, output=0.79")
    assert (p.input_per_1m, p.output_per_1m, p.cached_input_per_1m) == (0.59, 0.79, 0.59)
    assert parse_price_spec(GROQ, "input=1,output=2,cached=0.5").cached_input_per_1m == 0.5


@pytest.mark.parametrize(
    "spec", ["input=1", "output=2", "input=x,output=1", "in=1,out=2", "input=-1,output=1"]
)
def test_parse_price_spec_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_price_spec(GROQ, spec)


def test_load_price_file(tmp_path: Path) -> None:
    f = tmp_path / "prices.yaml"
    f.write_text(
        "models:\n"
        f"  - name: {GROQ}\n"
        "    input_per_1m: 0.59\n"
        "    output_per_1m: 0.79\n"
        "    source: https://groq.com/pricing\n"
    )
    prices = load_price_file(f)
    assert prices[GROQ].output_per_1m == 0.79
    assert prices[GROQ].notes == "https://groq.com/pricing"


@pytest.mark.parametrize(
    "body",
    ["nope: 1", "models:\n  - input_per_1m: 1\n", "models:\n  - name: m\n    input_per_1m: 1\n"],
)
def test_load_price_file_rejects_bad_files(tmp_path: Path, body: str) -> None:
    f = tmp_path / "prices.yaml"
    f.write_text(body)
    with pytest.raises(ValueError):
        load_price_file(f)


def test_user_prices_win_and_unknown_models_explain_how_to_price_them() -> None:
    custom = {"gpt-4o": parse_price_spec("gpt-4o", "input=1,output=1")}
    assert resolve_pricing("gpt-4o", custom).input_per_1m == 1
    assert resolve_pricing("gpt-4o-mini", custom) == get_pricing("gpt-4o-mini")
    with pytest.raises(UnknownModelError, match="--price input=X,output=Y"):
        resolve_pricing(GROQ, custom)


def test_compute_cost_uses_given_pricing() -> None:
    p = parse_price_spec(GROQ, "input=1,output=2")
    cost = compute_cost(GROQ, Usage(prompt_tokens=1_000_000, completion_tokens=500_000), p)
    assert cost.total_cost == pytest.approx(2.0)


def test_client_for_a_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = _make_client("https://api.groq.com/openai/v1", "GROQ_API_KEY")
    assert str(client.base_url).startswith("https://api.groq.com/openai/v1")
    assert client.api_key == "gsk_test"


def test_client_reports_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        _make_client("https://api.groq.com/openai/v1", "GROQ_API_KEY")


def test_local_server_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _make_client("http://localhost:11434/v1", None)
    assert str(client.base_url).startswith("http://localhost:11434/v1")


def test_cli_runs_an_unlisted_model_with_a_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()) as made:
        result = CliRunner().invoke(
            main,
            [
                "run",
                "--prompt",
                str(prompt),
                "--tag",
                "groq",
                "--eval-file",
                str(eval_file),
                "--model",
                GROQ,
                "--base-url",
                "https://api.groq.com/openai/v1",
                "--api-key-env",
                "GROQ_API_KEY",
                "--price",
                "input=1000000,output=0",
            ],
        )
    assert result.exit_code == 0, result.output
    made.assert_called_once_with(base_url="https://api.groq.com/openai/v1", api_key="gsk_test")
    data = json.loads((tmp_path / ".prompteval/runs/groq.json").read_text())
    # 50 prompt tokens at $1M per 1M tokens = $50 per example.
    assert data["examples"][0]["cost"]["total_cost"] == pytest.approx(50.0)
    assert data["base_url"] == "https://api.groq.com/openai/v1"
    assert data["price_source"].startswith("user-supplied")


def test_cli_reads_prices_yaml_next_to_the_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=1)
    (eval_file.parent / "prices.yaml").write_text(
        f"models:\n  - name: {GROQ}\n    input_per_1m: 0.59\n    output_per_1m: 0.79\n"
    )
    monkeypatch.chdir(tmp_path)
    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()):
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
                "--model",
                GROQ,
            ],
        )
    assert result.exit_code == 0, result.output


def test_cli_unknown_model_without_a_price_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file, prompt = _write_eval_dir(tmp_path, dataset_rows=1)
    monkeypatch.chdir(tmp_path)
    with patch("prompteval.eval.runner.OpenAI", return_value=_mock_openai_client()) as made:
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
                "--model",
                GROQ,
            ],
        )
    assert result.exit_code != 0
    assert "--price input=X,output=Y" in result.output
    made.return_value.chat.completions.create.assert_not_called()
