"""#12: the GitHub Action's script, run for real against stub `prompteval` and `gh`.

A temporary git repository holds a base and a head commit; the stubs record
how they were called, so the tests check what the Action would do on a PR
without network, an API key or GitHub.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "action" / "run.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

PROMPTEVAL_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG/prompteval.log"
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--markdown" ]]; then j=$((i+1)); echo "### stub report: $1" > "${!j}"; fi
done
[[ "$1" == "compare" || "$1" == "sweep" ]] && exit "${STUB_EXIT:-0}"
exit 0
"""

GH_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG/gh.log"
if [[ "$*" == *"--paginate"* ]]; then echo "${STUB_EXISTING_COMMENT:-}"; fi
exit 0
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Path | str]:
    repo = tmp_path / "repo"
    (repo / "evals" / "prompts").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "evals" / "prompts" / "v1.txt").write_text("Be helpful.\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("prompteval", PROMPTEVAL_STUB), ("gh", GH_STUB)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "log"
    log.mkdir()
    return {"repo": repo, "base": base, "bin": bin_dir, "log": log, "tmp": tmp_path}


def _run(ctx: dict[str, Path | str], **env: str) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "PATH": f"{ctx['bin']}{os.pathsep}{os.environ['PATH']}",
        "STUB_LOG": str(ctx["log"]),
        "RUNNER_TEMP": str(ctx["tmp"]),
        "GITHUB_STEP_SUMMARY": str(Path(str(ctx["tmp"])) / "summary.md"),
        "GITHUB_REPOSITORY": "owner/repo",
        "INPUT_PROMPT": "evals/prompts/v1.txt",
        "PR_BASE_SHA": str(ctx["base"]),
        "PR_NUMBER": "42",
        **env,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=ctx["repo"], env=full, capture_output=True, text=True
    )


def _log(ctx: dict[str, Path | str], name: str) -> str:
    path = Path(str(ctx["log"])) / f"{name}.log"
    return path.read_text() if path.exists() else ""


def test_changed_prompt_is_compared_and_commented(repo: dict[str, Path | str]) -> None:
    Path(str(repo["repo"]), "evals/prompts/v1.txt").write_text("Be brief and helpful.\n")
    result = _run(repo, INPUT_FAIL_ON="cost+10%")
    assert result.returncode == 0, result.stderr
    calls = _log(repo, "prompteval").splitlines()
    assert calls[0].startswith("run --prompt") and "--tag baseline" in calls[0]
    assert "--tag candidate" in calls[1]
    assert calls[2].startswith("compare baseline candidate") and "--fail-on cost+10%" in calls[2]
    gh = _log(repo, "gh")
    assert "--method POST repos/owner/repo/issues/42/comments" in gh
    assert "<!-- prompteval-action:evals/prompts/v1.txt -->" in gh
    assert "stub report: compare" in Path(str(repo["tmp"]), "summary.md").read_text()


def test_unchanged_prompt_spends_nothing(repo: dict[str, Path | str]) -> None:
    result = _run(repo)
    assert result.returncode == 0, result.stderr
    assert _log(repo, "prompteval") == ""
    assert "unchanged from the base" in _log(repo, "gh")


def test_existing_comment_is_updated_not_duplicated(repo: dict[str, Path | str]) -> None:
    Path(str(repo["repo"]), "evals/prompts/v1.txt").write_text("Changed.\n")
    result = _run(repo, STUB_EXISTING_COMMENT="987")
    assert result.returncode == 0, result.stderr
    gh = _log(repo, "gh")
    assert "--method PATCH repos/owner/repo/issues/comments/987" in gh
    assert "--method POST" not in gh


def test_gate_failure_fails_the_step_after_commenting(repo: dict[str, Path | str]) -> None:
    Path(str(repo["repo"]), "evals/prompts/v1.txt").write_text("Changed.\n")
    result = _run(repo, STUB_EXIT="1", INPUT_FAIL_ON="quality-5%")
    assert result.returncode == 1
    assert "--method POST" in _log(repo, "gh")


def test_new_prompt_with_no_baseline_is_explained(repo: dict[str, Path | str]) -> None:
    Path(str(repo["repo"]), "evals/prompts/v2.txt").write_text("New.\n")
    result = _run(repo, INPUT_PROMPT="evals/prompts/v2.txt")
    assert result.returncode == 0, result.stderr
    assert _log(repo, "prompteval") == ""
    assert "does not exist at the base" in _log(repo, "gh")


def test_sweep_mode(repo: dict[str, Path | str]) -> None:
    result = _run(repo, INPUT_MODELS="gpt-4o,gpt-4o-mini", INPUT_COMMENT="false")
    assert result.returncode == 0, result.stderr
    assert _log(repo, "prompteval").startswith(
        "sweep --prompt evals/prompts/v1.txt --models gpt-4o,gpt-4o-mini"
    )
    assert _log(repo, "gh") == ""  # comment disabled


def test_no_comment_outside_a_pull_request(repo: dict[str, Path | str]) -> None:
    Path(str(repo["repo"]), "evals/prompts/v1.txt").write_text("Changed.\n")
    result = _run(repo, PR_NUMBER="")
    assert result.returncode == 0, result.stderr
    assert _log(repo, "gh") == ""
