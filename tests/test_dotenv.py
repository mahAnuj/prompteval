"""API keys from .env: `prompteval init` tells users to put them there."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompteval.dotenv import load_env_files, parse_env


def test_parse_env_handles_the_common_shapes() -> None:
    text = (
        "# comment\n"
        "OPENAI_API_KEY=sk-abc\n"
        "export GROQ_API_KEY='gsk xyz'\n"
        'QUOTED="a # not a comment"\n'
        "TRAILING=value # a comment\n"
        "not a line\n"
        "1BAD=x\n"
    )
    assert parse_env(text) == {
        "OPENAI_API_KEY": "sk-abc",
        "GROQ_API_KEY": "gsk xyz",
        "QUOTED": "a # not a comment",
        "TRAILING": "value",
    }


def test_load_never_overrides_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("PE_TEST_A=from-file\nPE_TEST_B=from-file\nPE_TEST_C=sk-...\n")
    monkeypatch.setenv("PE_TEST_A", "from-shell")
    monkeypatch.delenv("PE_TEST_B", raising=False)
    monkeypatch.delenv("PE_TEST_C", raising=False)
    loaded = load_env_files((env, tmp_path / "missing.env"))
    import os

    assert os.environ["PE_TEST_A"] == "from-shell"
    assert os.environ["PE_TEST_B"] == "from-file"
    assert "PE_TEST_C" not in os.environ  # the template's placeholder is ignored
    assert loaded == ["PE_TEST_B"]
    monkeypatch.delenv("PE_TEST_B")


def test_cli_reads_evals_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from click.testing import CliRunner

    from prompteval.cli import main

    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / ".env").write_text("PE_TEST_KEY=from-evals-env\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PE_TEST_KEY", raising=False)
    assert CliRunner().invoke(main, ["hello"]).exit_code == 0
    assert os.environ.get("PE_TEST_KEY") == "from-evals-env"
    monkeypatch.delenv("PE_TEST_KEY")
