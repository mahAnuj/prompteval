"""Tests for `prompteval compare <tag-a> <tag-b>`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from prompteval import CostBreakdown, ExampleResult, RunResult, save_run
from prompteval.cli import main
from prompteval.eval.runner import ScoreOutcome


def _cost(amount: float) -> CostBreakdown:
    return CostBreakdown(
        model="gpt-4o-mini",
        uncached_input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        reasoning_tokens=0,
        uncached_input_cost=amount / 2,
        cached_input_cost=0.0,
        output_cost=amount / 2,
        total_cost=amount,
    )


def _build_run(tag: str, n: int, score: float, cost_per_example: float) -> RunResult:
    examples = [
        ExampleResult(
            id=f"ex-{i}",
            input=f"in-{i}",
            expected={},
            output=f"out-{i}",
            scores=[ScoreOutcome(name="acc", score=score)],
            cost=_cost(cost_per_example),
            latency_s=1.0,
            error=None,
        )
        for i in range(n)
    ]
    return RunResult(
        tag=tag,
        eval_name="t",
        model="gpt-4o-mini",
        prompt_path="prompts/p.txt",
        prompt_text="be helpful",
        started_at="2026-06-26T00:00:00+00:00",
        finished_at="2026-06-26T00:00:01+00:00",
        examples=examples,
    )


def test_compare_happy_path(tmp_path: Path) -> None:
    """Two runs persisted to disk → CLI loads + compares + prints report."""
    runs_dir = tmp_path / "runs"
    save_run(_build_run("baseline", n=20, score=0.9, cost_per_example=0.02), runs_dir=runs_dir)
    save_run(_build_run("cheap", n=20, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "baseline", "cheap", "--runs-dir", str(runs_dir)])
    assert result.exit_code == 0, result.output
    assert "=== baseline vs cheap ===" in result.output
    assert "Recommendation:" in result.output
    assert "ship cheap" in result.output.lower()


def test_compare_missing_tag_friendly_error(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    save_run(_build_run("only", n=5, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "only", "nope", "--runs-dir", str(runs_dir)])
    assert result.exit_code != 0
    assert "No run found for tag 'nope'" in result.output


def test_compare_insufficient_pairing_reports_friendly_error(tmp_path: Path) -> None:
    """If runs share <2 examples, compare should fail with a friendly message."""
    runs_dir = tmp_path / "runs"
    save_run(_build_run("a", n=1, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)
    save_run(_build_run("b", n=1, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "a", "b", "--runs-dir", str(runs_dir)])
    assert result.exit_code != 0
    assert "Need at least 2" in result.output


def test_compare_help_describes_command() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "TAG_A" in result.output
    assert "TAG_B" in result.output
    assert "--runs-dir" in result.output
    assert "--html" in result.output
    assert "--fail-on" in result.output


def test_compare_writes_html_when_html_flag_given(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    save_run(_build_run("baseline", n=20, score=0.9, cost_per_example=0.02), runs_dir=runs_dir)
    save_run(_build_run("cheap", n=20, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)
    html_out = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            "baseline",
            "cheap",
            "--runs-dir",
            str(runs_dir),
            "--html",
            str(html_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert html_out.exists()
    body = html_out.read_text()
    assert body.startswith("<!doctype html>")
    assert "baseline" in body and "cheap" in body
    assert f"HTML report: {html_out}" in result.output


def test_compare_fail_on_passes_when_no_breach(tmp_path: Path) -> None:
    """Cost cut + quality holds → both gates pass → exit 0."""
    runs_dir = tmp_path / "runs"
    save_run(_build_run("a", n=30, score=0.9, cost_per_example=0.02), runs_dir=runs_dir)
    save_run(_build_run("b", n=30, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["compare", "a", "b", "--runs-dir", str(runs_dir), "--fail-on", "cost+10%,quality-5%"],
    )
    assert result.exit_code == 0, result.output
    assert "GATE FAILED" not in result.output


def test_compare_fail_on_exits_1_when_cost_breach(tmp_path: Path) -> None:
    """Cost jumps 50% with cost+10% gate → exit 1 + GATE FAILED message."""
    runs_dir = tmp_path / "runs"
    save_run(_build_run("a", n=30, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)
    save_run(_build_run("b", n=30, score=0.9, cost_per_example=0.015), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(
        main, ["compare", "a", "b", "--runs-dir", str(runs_dir), "--fail-on", "cost+10%"]
    )
    assert result.exit_code == 1
    assert "GATE FAILED" in result.output
    assert "cost regressed" in result.output


def test_compare_fail_on_invalid_spec_fails_fast(tmp_path: Path) -> None:
    """Malformed gate spec → ClickException, not stack trace."""
    runs_dir = tmp_path / "runs"
    save_run(_build_run("a", n=5, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)
    save_run(_build_run("b", n=5, score=0.9, cost_per_example=0.01), runs_dir=runs_dir)

    runner = CliRunner()
    result = runner.invoke(
        main, ["compare", "a", "b", "--runs-dir", str(runs_dir), "--fail-on", "speed-10%"]
    )
    assert result.exit_code != 0
    assert "Invalid --fail-on clause" in result.output


def test_compare_uses_default_runs_dir_when_not_specified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `--runs-dir` isn't given, look in `./.prompteval/runs/`."""
    monkeypatch.chdir(tmp_path)
    default_dir = tmp_path / ".prompteval" / "runs"
    save_run(_build_run("a", n=5, score=0.9, cost_per_example=0.01), runs_dir=default_dir)
    save_run(_build_run("b", n=5, score=0.9, cost_per_example=0.02), runs_dir=default_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["compare", "a", "b"])
    assert result.exit_code == 0, result.output
    assert "=== a vs b ===" in result.output
