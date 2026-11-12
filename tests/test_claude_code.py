"""The Claude Code provider (`--base-url claude-code`), and fixes found by running it for real."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from prompteval import Eval, llm_judge, run_eval, scorer
from prompteval.compare import compute_comparison
from prompteval.eval import stock
from prompteval.eval.runner import _make_client
from prompteval.providers.claude_code import CLAUDE_CODE, ClaudeCodeClient, ClaudeCodeError


def _fake_cli(
    result: str = "OK", cost: float = 0.0003, is_error: bool = False, returncode: int = 0
) -> Any:
    calls: list[dict[str, Any]] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append({"cmd": cmd, **kwargs, "cwd_files": os.listdir(kwargs["cwd"])})
        body = {
            "result": result,
            "is_error": is_error,
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 1300,
                "cache_read_input_tokens": 100,
                "output_tokens": 4,
            },
        }
        return SimpleNamespace(stdout=json.dumps(body), stderr="", returncode=returncode)

    return run, calls


def _create(client: ClaudeCodeClient, **kw: Any) -> Any:
    return client.chat.completions.create(model="claude-haiku-5-5", **kw)


def test_command_is_minimal_and_runs_in_an_empty_folder() -> None:
    run, calls = _fake_cli()
    with (
        patch("prompteval.providers.claude_code.shutil.which", return_value="/usr/bin/claude"),
        patch("prompteval.providers.claude_code.subprocess.run", side_effect=run),
    ):
        out = _create(
            ClaudeCodeClient(),
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
            temperature=0,  # ignored: the CLI has no temperature
        )
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1] == "Be brief."  # replaces Claude Code's own prompt
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5"
    assert "--no-session-persistence" in cmd and "Hi" not in cmd  # prompt goes on stdin
    assert calls[0]["input"] == "Hi"
    assert calls[0]["cwd_files"] == []  # empty folder: no CLAUDE.md can leak in
    assert out.choices[0].message.content == "OK"
    assert out.usage.prompt_tokens == 1402 and out.usage.prompt_tokens_details.cached_tokens == 100
    assert out.reported_cost_usd == pytest.approx(0.0003)


def test_json_mode_asks_for_json_and_strips_fences() -> None:
    run, calls = _fake_cli(result='```json\n{"a": 1}\n```')
    with (
        patch("prompteval.providers.claude_code.shutil.which", return_value="claude"),
        patch("prompteval.providers.claude_code.subprocess.run", side_effect=run),
    ):
        out = _create(
            ClaudeCodeClient(),
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )
    assert json.loads(out.choices[0].message.content) == {"a": 1}
    assert "single JSON object" in calls[0]["cmd"][calls[0]["cmd"].index("--system-prompt") + 1]


def test_errors_are_clear() -> None:
    with (
        patch("prompteval.providers.claude_code.shutil.which", return_value=None),
        pytest.raises(ClaudeCodeError, match="not installed"),
    ):
        _create(ClaudeCodeClient(), messages=[{"role": "user", "content": "x"}])
    run, _ = _fake_cli(result="rate limited", is_error=True)
    with (
        patch("prompteval.providers.claude_code.shutil.which", return_value="claude"),
        patch("prompteval.providers.claude_code.subprocess.run", side_effect=run),
        pytest.raises(ClaudeCodeError, match="rate limited"),
    ):
        _create(ClaudeCodeClient(), messages=[{"role": "user", "content": "x"}])


def test_make_client_and_reported_cost_flow_into_runs(tmp_path: Path) -> None:
    assert isinstance(_make_client(CLAUDE_CODE, None), ClaudeCodeClient)
    run, _ = _fake_cli(result="yes", cost=0.002)

    @scorer
    def judged(output: str) -> float:
        return llm_judge(
            "Rate 0-1. Reply with only a number.",
            output,
            model="claude-sonnet-5-5",
            client=ClaudeCodeClient(),
        )

    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join(json.dumps({"id": f"e{i}", "input": "q"}) for i in range(3)) + "\n")
    prompt = tmp_path / "p.txt"
    prompt.write_text("sys")
    with (
        patch("prompteval.providers.claude_code.shutil.which", return_value="claude"),
        patch("prompteval.providers.claude_code.subprocess.run", side_effect=_judge_or_answer(run)),
    ):
        # A Claude model the price catalog doesn't know runs anyway: the CLI reports its cost.
        result = run_eval(
            Eval(
                name="e",
                dataset=ds,
                scorers=[judged],
                model="claude-haiku-5-5",
                base_url=CLAUDE_CODE,
            ),
            prompt,
            "t",
        )
    assert result.price_source == "reported by Claude Code (API list prices)" and result.priced
    assert result.total_cost == pytest.approx(3 * 0.002)
    assert result.total_judge_cost == pytest.approx(3 * 0.001)  # the judge's own reported cost


def _judge_or_answer(answer_run: Any) -> Any:
    def run(cmd: list[str], **kwargs: Any) -> Any:
        if cmd[cmd.index("--model") + 1] == "claude-sonnet-5-5":  # the judge
            body = {
                "result": "0.8",
                "is_error": False,
                "total_cost_usd": 0.001,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
            return SimpleNamespace(stdout=json.dumps(body), stderr="", returncode=0)
        return answer_run(cmd, **kwargs)

    return run


# ---- fixes found in the real run (2026-11-12) -----------------------------------------


def test_required_terms_ignore_hyphens_dashes_and_spacing() -> None:
    reply = (
        "You can return any item within 30 days of delivery. Delivery takes 3\u20135  working days."
    )
    expected = {"must_mention": ["30-day", "3-5 working days"]}
    assert stock.mentions_required_terms(reply, expected) == 1.0
    assert (
        stock.mentions_required_terms(reply, {"must_mention": ["photo"]}) == 0.0
    )  # a real miss still misses


def test_a_metric_that_only_improved_is_not_called_unclear() -> None:
    from tests.test_compare_decisions import _run

    a = np.zeros(8)
    b = np.zeros(8)
    b[0] = 1.0  # one example better, none worse (the real run's length_between)
    report = compute_comparison(_run("v1", {"q": a}, [1] * 8), _run("v2", {"q": b}, [1] * 8))
    assert report.scorer_deltas[0].status == "ok"


def test_unclear_message_names_the_metric_the_limit_belongs_to() -> None:
    from tests.test_compare_decisions import _run

    rng = np.random.default_rng(3)
    wide = (rng.random(8) < 0.5).astype(float)
    report = compute_comparison(
        _run("v1", {"narrow": np.full(8, 0.8), "wide": wide}, [1] * 8),
        _run("v2", {"narrow": np.full(8, 0.79), "wide": 1 - wide}, [1] * 8),
    )
    assert report.recommendation.startswith("inconclusive — wide could be up to")
