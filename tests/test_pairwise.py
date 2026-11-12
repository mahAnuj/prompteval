"""#19: a pairwise judge compares two runs' answers directly, with no scorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.compare import compute_comparison, render_markdown, render_text
from prompteval.compare.pairwise import (
    PAIRWISE_SCORER,
    baseline_with_pairwise,
    default_judge,
    judge_pairwise,
    pairwise_notes,
    with_pairwise,
)
from prompteval.cost import Usage, compute_cost
from prompteval.eval import save_run
from prompteval.eval.runner import ExampleResult, RunResult


def _run(tag: str, outputs: list[str], model: str = "gpt-4o-mini") -> RunResult:
    return RunResult(
        tag=tag,
        eval_name="e",
        model=model,
        prompt_path="p.txt",
        prompt_text=tag,
        started_at="",
        finished_at="",
        examples=[
            ExampleResult(
                id=f"e{i}",
                input=f"question {i}",
                expected={},
                output=out,
                scores=[],
                cost=compute_cost(model, Usage(prompt_tokens=100, completion_tokens=50)),
                latency_s=1.0,
            )
            for i, out in enumerate(outputs)
        ],
    )


def _judge(rule: str) -> Any:
    """A fake judge. rule: 'quality' picks the answer containing GOOD; 'first'
    always says A (pure position bias); 'garbage' replies nonsense."""
    client = MagicMock()

    def create(**kwargs: Any) -> Any:
        user = kwargs["messages"][-1]["content"]
        a_part = user.split("Answer A:\n", 1)[1].split("\n\nAnswer B:\n", 1)[0]
        b_part = user.split("\n\nAnswer B:\n", 1)[1]
        if rule == "first":
            reply = "A"
        elif rule == "garbage":
            reply = "I think both are fine"
        elif "GOOD" in a_part and "GOOD" not in b_part:
            reply = "A"
        elif "GOOD" in b_part and "GOOD" not in a_part:
            reply = "B"
        else:
            reply = "TIE"
        done = MagicMock()
        done.choices = [MagicMock(message=MagicMock(content=reply))]
        done.usage = MagicMock(prompt_tokens=1_000_000, completion_tokens=0)
        return done

    client.chat.completions.create.side_effect = create
    return client


def _compare(a: RunResult, b: RunResult, rule: str) -> tuple[Any, Any]:
    result = judge_pairwise(a, b, client=_judge(rule), model="gpt-4.1-mini")
    report = compute_comparison(baseline_with_pairwise(a), with_pairwise(b, result))
    return result, report


def test_better_candidate_wins_every_example() -> None:
    a, b = _run("v1", ["bad"] * 20), _run("v2", ["GOOD"] * 20)
    result, report = _compare(a, b, "quality")
    assert set(result.scores.values()) == {1.0}
    assert result.order_disagreements == 0
    d = next(x for x in report.scorer_deltas if x.name == PAIRWISE_SCORER)
    assert d.status == "ok" and d.delta == pytest.approx(0.5)


def test_worse_candidate_is_a_regression() -> None:
    a, b = _run("v1", ["GOOD"] * 20), _run("v2", ["bad"] * 20)
    _, report = _compare(a, b, "quality")
    assert report.recommendation.startswith("don't ship v2")
    assert PAIRWISE_SCORER in report.recommendation


def test_position_bias_becomes_ties_not_wins() -> None:
    a, b = _run("v1", ["x"] * 10), _run("v2", ["y"] * 10)
    result, report = _compare(a, b, "first")
    assert result.order_disagreements == 10
    assert set(result.scores.values()) == {0.5}
    d = next(x for x in report.scorer_deltas if x.name == PAIRWISE_SCORER)
    assert d.delta == pytest.approx(0.0)


def test_unreadable_replies_count_as_ties() -> None:
    result, _ = _compare(_run("v1", ["x"] * 3), _run("v2", ["y"] * 3), "garbage")
    assert result.unreadable == 6 and set(result.scores.values()) == {0.5}


def test_judge_cost_and_notes() -> None:
    a, b = _run("v1", ["bad"] * 4), _run("v2", ["GOOD"] * 4)
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4.1-mini")
    # 8 calls x 1M input tokens at gpt-4.1-mini's input price.
    per_m = compute_cost(
        "gpt-4.1-mini", Usage(prompt_tokens=1_000_000, completion_tokens=0)
    ).total_cost
    assert result.cost == pytest.approx(8 * per_m)
    note = " ".join(pairwise_notes(result, a, b))
    assert "gpt-4.1-mini" in note and "judged twice with the answers swapped" in note


def test_default_judge_avoids_the_compared_models_and_warns_otherwise() -> None:
    a, b = _run("v1", ["x"], model="gpt-4.1-mini"), _run("v2", ["y"], model="gpt-4o")
    assert default_judge(a, b) == "gpt-4o-mini"
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4o")
    assert any("also one of the models being compared" in n for n in pairwise_notes(result, a, b))


def test_errored_examples_are_not_judged() -> None:
    from dataclasses import replace

    a, b = _run("v1", ["bad"] * 3), _run("v2", ["GOOD"] * 3)
    b = replace(b, examples=[replace(b.examples[0], error="Timeout"), *b.examples[1:]])
    result = judge_pairwise(a, b, client=_judge("quality"), model="gpt-4.1-mini")
    assert set(result.scores) == {"e1", "e2"}


def test_no_quality_measure_is_never_called_holds() -> None:
    report = compute_comparison(_run("v1", ["x"] * 5), _run("v2", ["y"] * 5))
    assert report.quality_verdict.startswith("not evaluated")
    assert report.recommendation.startswith("inconclusive — nothing measures quality")
    assert "--pairwise" in report.recommendation


def test_cli_compare_pairwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    save_run(_run("v1", ["bad"] * 20))
    save_run(_run("v2", ["GOOD"] * 20))
    with patch("prompteval.eval.runner.OpenAI", return_value=_judge("quality")):
        result = CliRunner().invoke(
            main, ["compare", "v1", "v2", "--pairwise", "--markdown", "r.md"]
        )
    assert result.exit_code == 0, result.output
    assert PAIRWISE_SCORER in result.output
    assert "Note: Pairwise judge gpt-4.1-mini" in result.output
    assert "Pairwise judge" in (tmp_path / "r.md").read_text()


def test_reports_render_notes() -> None:
    a, b = _run("v1", ["bad"] * 5), _run("v2", ["GOOD"] * 5)
    result, report = _compare(a, b, "quality")
    from dataclasses import replace

    report = replace(report, notes=pairwise_notes(result, a, b))
    assert "Note: Pairwise judge" in render_text(report)
    assert "> Pairwise judge" in render_markdown(report)
    json.dumps([report.recommendation])  # plain strings throughout


# ---- #23: the judge gets a checklist and the assistant's rules -------------------------


def _capturing_judge(reply: str = "TIE") -> tuple[Any, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    client = MagicMock()

    def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        done = MagicMock()
        done.choices = [MagicMock(message=MagicMock(content=reply))]
        done.usage = MagicMock(prompt_tokens=10, completion_tokens=1)
        return done

    client.chat.completions.create.side_effect = create
    return client, seen


def test_judge_sees_the_checklist_and_the_baseline_rules() -> None:
    from dataclasses import replace

    a = replace(
        _run("v1", ["x"]),
        prompt_text="Returns are accepted within 30 days. Never promise faster delivery.",
    )
    b = _run("v2", ["y"])
    client, seen = _capturing_judge()
    result = judge_pairwise(a, b, client=client, model="gpt-4.1-mini")
    system = seen[0]["messages"][0]["content"]
    user = seen[0]["messages"][1]["content"]
    assert "Checklist:" in system and "Follows the assistant's instructions" in system
    assert "invents no facts, prices, policies" in system and "Length is not quality" in system
    assert "The instructions the assistant was given" in user and "within 30 days" in user
    assert "v1's prompt as the assistant's rules" in result.shown
    assert "6-point checklist" in " ".join(pairwise_notes(result, a, b))


def test_your_criteria_replace_the_checklist_and_the_report_names_them() -> None:
    a, b = _run("v1", ["x"]), _run("v2", ["y"])
    client, seen = _capturing_judge()
    result = judge_pairwise(
        a,
        b,
        client=client,
        model="gpt-4.1-mini",
        context="Policy: refunds within 14 days only.",
        criteria="1. Uses only the birth details given.\n2. No fatalistic predictions.",
        criteria_label="criteria/prompt-create-kundli.md",
    )
    system = seen[0]["messages"][0]["content"]
    user = seen[0]["messages"][1]["content"]
    assert "refunds within 14 days only" in user
    assert "Criteria (written by the person who owns this assistant)" in system
    assert "No fatalistic predictions" in system
    assert "Checklist:" not in system  # replaced, not appended to
    assert "Length is not quality" in system  # the ground rules stay
    assert result.shown == "your --judge-context"
    note = " ".join(pairwise_notes(result, a, b))
    assert "applied your criteria (criteria/prompt-create-kundli.md)" in note


def _kundli_project(tmp_path: Path, eval_name: str = "e") -> None:
    from dataclasses import replace

    save_run(replace(_run("v1", ["bad"] * 5), eval_name=eval_name))
    save_run(replace(_run("v2", ["GOOD"] * 5), eval_name=eval_name))
    (tmp_path / "criteria").mkdir()
    (tmp_path / "criteria" / "prompt-create-kundli.md").write_text(
        "# PROMPTEVAL-DRAFT: review me\n1. Uses only the birth details given.\n"
    )


def test_cli_criteria_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _kundli_project(tmp_path)
    (tmp_path / "rules.txt").write_text("Refunds within 14 days.")
    client, seen = _capturing_judge()
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        result = CliRunner().invoke(
            main,
            [
                "compare",
                "v1",
                "v2",
                "--pairwise",
                "--judge-context",
                "rules.txt",
                "--criteria",
                "prompt-create-kundli",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Judging against your criteria in criteria/prompt-create-kundli.md" in result.output
    assert "still a DRAFT" in result.output
    system = seen[0]["messages"][0]["content"]
    assert "Uses only the birth details given." in system and "PROMPTEVAL-DRAFT" not in system
    assert "Refunds within 14 days." in seen[0]["messages"][1]["content"]


def test_cli_criteria_found_from_the_eval_name_and_typos_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _kundli_project(tmp_path, eval_name="prompt-create-kundli")
    client, _ = _capturing_judge()
    with patch("prompteval.eval.runner.OpenAI", return_value=client):
        found = CliRunner().invoke(main, ["compare", "v1", "v2", "--pairwise"])
        typo = CliRunner().invoke(
            main, ["compare", "v1", "v2", "--pairwise", "--criteria", "prompt-create-kundly"]
        )
    assert found.exit_code == 0, found.output
    assert "criteria/prompt-create-kundli.md" in found.output
    assert typo.exit_code != 0 and "prompt-create-kundli.md" in typo.output  # lists what exists


def test_runs_remember_the_eval_criteria(tmp_path: Path) -> None:
    from dataclasses import replace

    from prompteval.eval import load_run

    save_run(replace(_run("v1", ["x"]), criteria="prompt-create-kundli"), runs_dir=tmp_path)
    assert load_run("v1", runs_dir=tmp_path).criteria == "prompt-create-kundli"


def test_criteria_rubric_and_new_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prompteval import criteria_rubric, load_criteria

    monkeypatch.chdir(tmp_path)
    made = CliRunner().invoke(main, ["criteria", "new", "prompt-create-kundli"])
    assert made.exit_code == 0, made.output
    assert (tmp_path / "criteria" / "prompt-create-kundli.md").exists()
    again = CliRunner().invoke(main, ["criteria", "new", "prompt-create-kundli"])
    assert again.exit_code != 0  # never overwrites
    (tmp_path / "criteria" / "prompt-create-kundli.md").write_text("1. Cites the chart.\n")
    assert load_criteria("prompt-create-kundli") == "1. Cites the chart."
    rubric = criteria_rubric("prompt-create-kundli")
    assert "1. Cites the chart." in rubric and "number between 0 and 1" in rubric
    listed = CliRunner().invoke(main, ["criteria", "list"])
    assert "criteria/prompt-create-kundli.md" in listed.output
