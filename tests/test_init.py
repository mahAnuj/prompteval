"""Tests for `prompteval init` and the bootstrap helper.

Coverage targets:
- Files created at expected paths, with the right rename (env.example → .env.example).
- Refuses to write into a non-empty target without --force.
- --force does write; existing unrelated files are left alone.
- Template contents are actually meaningful (jsonl parses, eval.py imports the
  v0.1 surface, .env.example points at OpenAI not Anthropic).
- CLI entry point works end-to-end (--dir, --force, plain invocation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.init import bootstrap

EXPECTED_FILES = {
    ".env.example",
    "dataset.jsonl",
    "eval.py",
    "prompts/v1.txt",
    "prompts/v2.txt",
}


def _relative_files(target: Path) -> set[str]:
    return {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}


def test_bootstrap_creates_expected_files(tmp_path: Path) -> None:
    target = tmp_path / "evals"
    result = bootstrap(target)

    assert _relative_files(target) == EXPECTED_FILES
    assert result.files_written == len(EXPECTED_FILES)
    assert result.target == target


def test_bootstrap_writes_dotenv_with_leading_dot(tmp_path: Path) -> None:
    """Source-side env.example must land as .env.example on disk."""
    target = tmp_path / "evals"
    bootstrap(target)
    assert (target / ".env.example").exists()
    assert not (target / "env.example").exists()


def test_bootstrap_refuses_non_empty_target(tmp_path: Path) -> None:
    target = tmp_path / "evals"
    target.mkdir()
    (target / "user-thing.txt").write_text("don't touch me")

    with pytest.raises(FileExistsError, match="already exists"):
        bootstrap(target)


def test_bootstrap_allows_empty_existing_dir(tmp_path: Path) -> None:
    """An empty pre-created dir should be safe to write into (mkdir -p semantics)."""
    target = tmp_path / "evals"
    target.mkdir()
    bootstrap(target)
    assert (target / "dataset.jsonl").exists()


def test_bootstrap_force_writes_into_non_empty(tmp_path: Path) -> None:
    target = tmp_path / "evals"
    target.mkdir()
    keepsake = target / "user-thing.txt"
    keepsake.write_text("don't touch me")

    bootstrap(target, force=True)

    assert (target / "dataset.jsonl").exists()
    # We don't delete files we didn't write — safer default.
    assert keepsake.exists()
    assert keepsake.read_text() == "don't touch me"


def test_dataset_is_valid_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "evals"
    bootstrap(target)

    lines = (target / "dataset.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 3, "Dataset should have a few examples to feel real"
    for line in lines:
        record = json.loads(line)
        assert {"id", "input", "expected"} <= set(record), record
        assert isinstance(record["expected"], dict)


def test_eval_template_references_v01_api(tmp_path: Path) -> None:
    """The eval.py template documents the v0.1 public API contract."""
    target = tmp_path / "evals"
    bootstrap(target)

    eval_py = (target / "eval.py").read_text()
    assert "from prompteval import" in eval_py
    assert "Eval" in eval_py
    assert "scorer" in eval_py
    assert "llm_judge" in eval_py
    # Default model — confirmed in IMPLEMENTATION_PLAN.md decisions log.
    assert "gpt-4o-mini" in eval_py


def test_env_example_points_at_openai_not_anthropic(tmp_path: Path) -> None:
    """If this regresses we've quietly flipped providers — fail loudly."""
    target = tmp_path / "evals"
    bootstrap(target)

    env_text = (target / ".env.example").read_text()
    assert "OPENAI_API_KEY" in env_text
    assert "ANTHROPIC" not in env_text


def test_prompts_are_distinct(tmp_path: Path) -> None:
    """v1 and v2 must be different — they're the whole point of the demo."""
    target = tmp_path / "evals"
    bootstrap(target)

    v1 = (target / "prompts" / "v1.txt").read_text()
    v2 = (target / "prompts" / "v2.txt").read_text()
    assert v1 != v2
    # v2 should be the shorter one (it's the "cheaper" variant in the story).
    assert len(v2) < len(v1)


def test_cli_init_creates_evals_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["init"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "evals").exists()
    assert (tmp_path / "evals" / "dataset.jsonl").exists()
    assert "Created" in result.output
    assert "Next:" in result.output


def test_cli_init_respects_custom_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--dir", "my-evals"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "my-evals").exists()
    assert not (tmp_path / "evals").exists()


def test_cli_init_refuses_existing_non_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "stale.txt").write_text("x")
    runner = CliRunner()

    result = runner.invoke(main, ["init"])

    assert result.exit_code != 0
    assert "already exists" in result.output


def test_cli_init_force_overrides_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "stale.txt").write_text("x")
    runner = CliRunner()

    result = runner.invoke(main, ["init", "--force"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "evals" / "dataset.jsonl").exists()
