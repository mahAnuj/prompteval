"""Tests for `prompteval scorer list` + `prompteval scorer copy`."""

from __future__ import annotations

from click.testing import CliRunner

from prompteval.cli import main
from prompteval.eval import stock


def test_scorer_list_shows_every_stock_scorer() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "list"])
    assert result.exit_code == 0, result.output
    # Each stock scorer name should appear in the output.
    for name in dir(stock):
        obj = getattr(stock, name)
        if getattr(obj, "__prompteval_scorer__", False):
            assert name in result.output


def test_scorer_list_shows_first_line_of_each_docstring() -> None:
    """Sanity: the output isn't just bare names — descriptions help users pick."""
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "list"])
    assert result.exit_code == 0
    # `exact_match` first-line doc starts with "1.0 iff"
    assert "1.0 iff" in result.output


def test_scorer_copy_prints_source_of_known_scorer() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "copy", "exact_match"])
    assert result.exit_code == 0, result.output
    assert "@scorer" in result.output
    assert "def exact_match" in result.output


def test_scorer_copy_unknown_exits_with_friendly_error() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "copy", "not_a_real_scorer"])
    assert result.exit_code != 0
    assert "Unknown stock scorer" in result.output
    # The error should list available stock scorers so user can fix in one shot.
    assert "exact_match" in result.output


def test_scorer_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "copy" in result.output


def test_scorer_copy_output_is_syntactically_valid_python() -> None:
    """If copy ever drops a closing paren or whatever, this catches it."""
    runner = CliRunner()
    result = runner.invoke(main, ["scorer", "copy", "mentions_required_terms"])
    assert result.exit_code == 0
    # compile() raises SyntaxError if the output is broken Python.
    # The output is just the function source, so we wrap it in a module-level
    # check by trying to compile it as a statement.
    compile(result.output, "<copied>", "exec")
