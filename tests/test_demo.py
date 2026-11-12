"""#11: `prompteval demo` shows a full report with no key, no network, no setup."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from prompteval.cli import main


def test_demo_prints_a_labelled_report_and_next_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # works from any directory: the runs ship in the package
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(main, ["demo", "--html", "demo.html"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Illustrative data:")
    assert "=== baseline vs short-prompt ===" in result.output
    assert "Recommendation:   ship short-prompt" in result.output
    assert "prompteval init" in result.output
    assert (tmp_path / "demo.html").read_text().startswith("<!doctype html>")
