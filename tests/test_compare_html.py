"""Tests for `render_html(report)` — single self-contained HTML output."""

from __future__ import annotations

import re

from prompteval import CostBreakdown, ExampleResult, RunResult
from prompteval.compare import compute_comparison, render_html
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


def _example(id_: str, score: float, cost: float = 0.01) -> ExampleResult:
    return ExampleResult(
        id=id_,
        input=f"in-{id_}",
        expected={},
        output=f"out-{id_}",
        scores=[ScoreOutcome(name="acc", score=score)],
        cost=_cost(cost),
        latency_s=1.0,
        error=None,
    )


def _run(tag: str, score: float, cost: float, n: int = 20) -> RunResult:
    return RunResult(
        tag=tag,
        eval_name="t",
        model="gpt-4o-mini",
        prompt_path="prompts/p.txt",
        prompt_text="be helpful",
        started_at="2026-06-26T00:00:00+00:00",
        finished_at="2026-06-26T00:00:01+00:00",
        examples=[_example(f"e{i}", score, cost) for i in range(n)],
    )


def test_html_is_self_contained_doctype_and_inline_style() -> None:
    """No <link>, no <script src>, no external assets — emails/CI artifacts must work offline."""
    a = _run("baseline", score=0.9, cost=0.02)
    b = _run("cheap", score=0.9, cost=0.01)
    html = render_html(compute_comparison(a, b))
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "<link " not in html
    assert "<script src" not in html


def test_html_contains_tags_and_metric_values() -> None:
    a = _run("baseline", score=0.9, cost=0.02)
    b = _run("cheap", score=0.85, cost=0.01)
    html = render_html(compute_comparison(a, b))
    # Tag names appear in the title and header
    assert "baseline" in html
    assert "cheap" in html
    # The scorer name shows up
    assert "acc" in html
    # Cost values render with the $-prefix from value_fmt
    assert "$" in html


def test_html_escapes_tag_names() -> None:
    """Tag names from the CLI become attacker-controlled in long-lived reports — escape them."""
    a = _run("<script>alert(1)</script>", score=0.9, cost=0.01)
    b = _run("safe", score=0.9, cost=0.01)
    html = render_html(compute_comparison(a, b))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_html_ship_recommendation_uses_ship_class() -> None:
    """Cost cut with no quality change → ship badge gets the 'ship' (green) class."""
    a = _run("baseline", score=0.9, cost=0.02, n=30)
    b = _run("cheap", score=0.9, cost=0.005, n=30)
    html = render_html(compute_comparison(a, b))
    assert re.search(r"class=['\"]rec ship['\"]", html)


def test_html_dont_ship_uses_dont_ship_class() -> None:
    """Quality regression → don't-ship badge gets the 'dont-ship' (red) class."""
    a = _run("baseline", score=0.9, cost=0.01, n=30)
    b = _run("worse", score=0.5, cost=0.01, n=30)
    html = render_html(compute_comparison(a, b))
    assert re.search(r"class=['\"]rec dont-ship['\"]", html)


def test_html_includes_recommendation_text() -> None:
    a = _run("baseline", score=0.9, cost=0.02, n=30)
    b = _run("cheap", score=0.9, cost=0.005, n=30)
    report = compute_comparison(a, b)
    html = render_html(report)
    assert report.recommendation in html
