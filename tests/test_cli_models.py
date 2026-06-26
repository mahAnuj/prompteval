"""Tests for `prompteval models` CLI subcommands.

Smoke-test the table output, the JSON output, and the error path.
The cost math itself is tested in test_cost_compute.py.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from prompteval.cli import main
from prompteval.cost import PRICING


def test_models_list_human_readable_shows_every_model() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "list"])
    assert result.exit_code == 0, result.output
    for name in PRICING:
        assert name in result.output


def test_models_list_json_is_parseable_and_has_expected_keys() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "list", "--json"])
    assert result.exit_code == 0, result.output

    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == len(PRICING)
    for entry in data:
        for key in ("name", "input_per_1m", "cached_input_per_1m", "output_per_1m"):
            assert key in entry, entry


def test_models_price_known_shows_details() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "price", "gpt-4o-mini"])
    assert result.exit_code == 0, result.output
    assert "gpt-4o-mini" in result.output
    assert "Input:" in result.output
    assert "Cached input:" in result.output
    assert "Output:" in result.output
    # Discount percentage should show up for a model with caching.
    assert "%" in result.output


def test_models_price_json_round_trips() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "price", "gpt-4o", "--json"])
    assert result.exit_code == 0, result.output

    data = json.loads(result.output)
    assert data["name"] == "gpt-4o"
    assert data["input_per_1m"] == 2.50


def test_models_price_unknown_exits_with_friendly_error() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "price", "not-a-model"])
    assert result.exit_code != 0
    # The error message itself should include the catalog (not just a stack trace).
    for known in PRICING:
        assert known in result.output


def test_models_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "price" in result.output
