"""Tests for `parse_gate_spec` and `evaluate_gates` — the brains of `--fail-on`."""

from __future__ import annotations

import pytest

from prompteval import CostBreakdown, ExampleResult, RunResult
from prompteval.compare import (
    GateClause,
    GateSpecError,
    compute_comparison,
    evaluate_gates,
    parse_gate_spec,
)
from prompteval.eval.runner import ScoreOutcome

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


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


def _ex(id_: str, score: float, cost: float) -> ExampleResult:
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


def _run(tag: str, score: float, cost: float, n: int = 30) -> RunResult:
    return RunResult(
        tag=tag,
        eval_name="t",
        model="gpt-4o-mini",
        prompt_path="prompts/p.txt",
        prompt_text="be helpful",
        started_at="2026-06-26T00:00:00+00:00",
        finished_at="2026-06-26T00:00:01+00:00",
        examples=[_ex(f"e{i}", score, cost) for i in range(n)],
    )


# ---------------------------------------------------------------------------
# parse_gate_spec
# ---------------------------------------------------------------------------


def test_parse_single_cost_clause() -> None:
    [clause] = parse_gate_spec("cost+10%")
    assert clause == GateClause(metric="cost", threshold_pct=10.0)


def test_parse_single_quality_clause() -> None:
    [clause] = parse_gate_spec("quality-5%")
    assert clause == GateClause(metric="quality", threshold_pct=-5.0)


def test_parse_combined_spec() -> None:
    clauses = parse_gate_spec("cost+10%,quality-5%")
    assert clauses == [
        GateClause(metric="cost", threshold_pct=10.0),
        GateClause(metric="quality", threshold_pct=-5.0),
    ]


def test_parse_tolerates_whitespace_and_case() -> None:
    clauses = parse_gate_spec(" COST + 10 % , Quality - 5 % ")
    assert clauses == [
        GateClause(metric="cost", threshold_pct=10.0),
        GateClause(metric="quality", threshold_pct=-5.0),
    ]


def test_parse_accepts_decimal_threshold() -> None:
    [clause] = parse_gate_spec("cost+2.5%")
    assert clause.threshold_pct == pytest.approx(2.5)


def test_parse_rejects_unknown_metric() -> None:
    with pytest.raises(GateSpecError, match="Invalid --fail-on clause"):
        parse_gate_spec("speed-10%")


def test_parse_rejects_missing_sign() -> None:
    with pytest.raises(GateSpecError):
        parse_gate_spec("cost10%")


def test_parse_rejects_empty_spec() -> None:
    with pytest.raises(GateSpecError, match="Empty"):
        parse_gate_spec("")


def test_parse_rejects_all_blank_clauses() -> None:
    with pytest.raises(GateSpecError, match="Empty"):
        parse_gate_spec(" , , ")


# ---------------------------------------------------------------------------
# evaluate_gates — cost clauses
# ---------------------------------------------------------------------------


def test_significant_cost_increase_above_threshold_breaches() -> None:
    """+50% significant cost regression with gate cost+10% → breach."""
    a = _run("a", score=0.9, cost=0.01)
    b = _run("b", score=0.9, cost=0.015)  # +50%
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("cost+10%"))
    assert len(breaches) == 1
    assert "cost regressed" in breaches[0].detail
    assert "gate was cost+10%" in breaches[0].detail


def test_cost_within_threshold_passes() -> None:
    """+5% rise vs gate of cost+10% → no breach (under the bar)."""
    a = _run("a", score=0.9, cost=0.0100)
    b = _run("b", score=0.9, cost=0.0105)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("cost+10%"))
    assert breaches == []


def test_cost_improvement_never_breaches_positive_clause() -> None:
    """Cheaper b vs cost+10% → no breach — the gate only fires on regressions."""
    a = _run("a", score=0.9, cost=0.02)
    b = _run("b", score=0.9, cost=0.01)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("cost+10%"))
    assert breaches == []


def test_non_significant_cost_change_does_not_breach() -> None:
    """No actual change → not significant → can't breach any gate."""
    a = _run("a", score=0.9, cost=0.01)
    b = _run("b", score=0.9, cost=0.01)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("cost+1%"))
    assert breaches == []


# ---------------------------------------------------------------------------
# evaluate_gates — quality clauses
# ---------------------------------------------------------------------------


def test_significant_quality_drop_above_threshold_breaches() -> None:
    """Score 0.9 → 0.5 (drop of 0.4) vs quality-5% (0.05 abs) → breach."""
    a = _run("a", score=0.9, cost=0.01)
    b = _run("b", score=0.5, cost=0.01)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("quality-5%"))
    assert len(breaches) == 1
    assert "acc" in breaches[0].detail
    assert "gate was quality-5%" in breaches[0].detail


def test_quality_drop_below_threshold_passes() -> None:
    """0.90 → 0.88 (drop of 0.02) vs quality-5% (need >0.05) → no breach."""
    a = _run("a", score=0.90, cost=0.01)
    b = _run("b", score=0.88, cost=0.01)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("quality-5%"))
    assert breaches == []


def test_quality_improvement_does_not_breach() -> None:
    a = _run("a", score=0.7, cost=0.01)
    b = _run("b", score=0.9, cost=0.01)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("quality-5%"))
    assert breaches == []


def test_combined_spec_reports_both_breaches() -> None:
    """Cost up AND quality down → both clauses breached, both surfaced."""
    a = _run("a", score=0.9, cost=0.01)
    b = _run("b", score=0.4, cost=0.02)
    report = compute_comparison(a, b)
    breaches = evaluate_gates(report, parse_gate_spec("cost+10%,quality-5%"))
    assert len(breaches) == 2
    metrics = {b.clause.metric for b in breaches}
    assert metrics == {"cost", "quality"}
