"""Tests for the compare module — paired deltas, bootstrap CIs, verdicts, rendering.

The whole wedge depends on this being correct, so the tests are paranoid:
- Hand-verified easy cases (constant deltas, monotone improvements)
- Edge cases: pairing by id, errored examples excluded, missing scorers
- Verdict logic: regression detection, ship/don't-ship recommendation paths
- Rendering: matches the README's killer-output shape
"""

from __future__ import annotations

import numpy as np
import pytest

from prompteval import CostBreakdown, ExampleResult, RunResult
from prompteval.compare import compute_comparison, render_text
from prompteval.compare.core import (
    MetricDelta,
    _bootstrap_mean_ci,
    _bootstrap_percent_change_ci,
    _common_scorer_names,
    _format_p,
    _make_verdicts,
    _pair_examples,
    _paired_p_value,
)
from prompteval.eval.runner import ScoreOutcome

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _cost(amount: float) -> CostBreakdown:
    """Tiny CostBreakdown for tests; only `total_cost` matters for compare math."""
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


def _example(
    id_: str,
    score_map: dict[str, float],
    cost_amount: float = 0.01,
    latency: float = 1.0,
    error: str | None = None,
) -> ExampleResult:
    return ExampleResult(
        id=id_,
        input=f"input-{id_}",
        expected={},
        output=f"output-{id_}",
        scores=[ScoreOutcome(name=n, score=s) for n, s in score_map.items()],
        cost=_cost(cost_amount),
        latency_s=latency,
        error=error,
    )


def _run(
    tag: str,
    examples: list[ExampleResult],
    model: str = "gpt-4o-mini",
) -> RunResult:
    return RunResult(
        tag=tag,
        eval_name="test-eval",
        model=model,
        prompt_path="prompts/p.txt",
        prompt_text="be helpful",
        started_at="2026-06-26T00:00:00+00:00",
        finished_at="2026-06-26T00:00:01+00:00",
        examples=examples,
    )


def _identical_runs(n: int = 10) -> tuple[RunResult, RunResult]:
    """Two runs with identical examples (same ids, scores, costs, latencies)."""
    examples_a = [
        _example(f"ex-{i}", {"acc": 0.8}, cost_amount=0.01, latency=1.0) for i in range(n)
    ]
    examples_b = [
        _example(f"ex-{i}", {"acc": 0.8}, cost_amount=0.01, latency=1.0) for i in range(n)
    ]
    return _run("a", examples_a), _run("b", examples_b)


# ---------------------------------------------------------------------------
# _pair_examples
# ---------------------------------------------------------------------------


def test_pair_examples_finds_common_ids() -> None:
    a = _run("a", [_example("x", {}), _example("y", {}), _example("z", {})])
    b = _run("b", [_example("y", {}), _example("z", {}), _example("w", {})])
    common, idx_a, idx_b = _pair_examples(a, b)
    assert common == ["y", "z"]
    assert idx_a == [1, 2]
    assert idx_b == [0, 1]


def test_pair_examples_excludes_errored() -> None:
    a = _run(
        "a",
        [_example("x", {}), _example("y", {}, error="boom")],
    )
    b = _run("b", [_example("x", {}), _example("y", {})])
    common, _, _ = _pair_examples(a, b)
    # 'y' errored in run_a, so not paired
    assert common == ["x"]


# ---------------------------------------------------------------------------
# _bootstrap_mean_ci / _bootstrap_percent_change_ci / _paired_p_value
# ---------------------------------------------------------------------------


def test_bootstrap_mean_ci_brackets_true_mean() -> None:
    """For a sample with known mean, the 95% CI should contain it."""
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.5, scale=0.1, size=100)
    low, high = _bootstrap_mean_ci(sample)
    assert low < 0.5 < high


def test_bootstrap_mean_ci_handles_single_value() -> None:
    """One-sample edge: CI collapses to the value (no resampling possible)."""
    sample = np.array([0.42])
    low, high = _bootstrap_mean_ci(sample)
    assert low == 0.42
    assert high == 0.42


def test_bootstrap_percent_change_ci_brackets_true_pct() -> None:
    """Known 50% reduction (1.0 → 0.5) should produce a CI bracketing -50%."""
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    low, high = _bootstrap_percent_change_ci(a, b)
    # With identical values bootstrap is degenerate — CI collapses to exact value
    assert low == pytest.approx(-50.0, abs=0.01)
    assert high == pytest.approx(-50.0, abs=0.01)


def test_paired_p_value_significant_for_clear_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.05, 50)
    b = rng.normal(0.7, 0.05, 50)  # clear improvement
    p = _paired_p_value(a, b)
    assert p < 0.001


def test_paired_p_value_high_for_no_difference() -> None:
    """Two identical samples: no signal, p should be high (or 1.0 if degenerate)."""
    a = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    p = _paired_p_value(a, b)
    assert p == 1.0  # degenerate-equal case returns 1.0 (documented)


def test_paired_p_value_handles_short_samples() -> None:
    assert _paired_p_value(np.array([0.5]), np.array([0.6])) == 1.0
    assert _paired_p_value(np.array([]), np.array([])) == 1.0


# ---------------------------------------------------------------------------
# _format_p
# ---------------------------------------------------------------------------


def test_format_p_tiny_uses_lt_marker() -> None:
    assert _format_p(0.0001) == "<0.001"
    assert _format_p(0.00001) == "<0.001"


def test_format_p_moderate_three_decimals() -> None:
    assert _format_p(0.004) == "0.004"


def test_format_p_normal_two_decimals() -> None:
    assert _format_p(0.21) == "0.21"


# ---------------------------------------------------------------------------
# _common_scorer_names
# ---------------------------------------------------------------------------


def test_common_scorer_names_intersects_across_runs() -> None:
    a = _run("a", [_example("x", {"acc": 1.0, "tone": 0.8})])
    b = _run("b", [_example("x", {"acc": 1.0})])
    assert _common_scorer_names(a, b) == ["acc"]


def test_common_scorer_names_empty_when_no_overlap() -> None:
    a = _run("a", [_example("x", {"acc": 1.0})])
    b = _run("b", [_example("x", {"tone": 0.8})])
    assert _common_scorer_names(a, b) == []


# ---------------------------------------------------------------------------
# compute_comparison — end-to-end
# ---------------------------------------------------------------------------


def test_compute_identical_runs_produces_zero_delta() -> None:
    a, b = _identical_runs(n=10)
    report = compute_comparison(a, b)
    assert report.n_paired == 10
    assert len(report.scorer_deltas) == 1
    acc = report.scorer_deltas[0]
    assert acc.delta == pytest.approx(0.0, abs=1e-9)
    assert acc.value_a == acc.value_b == pytest.approx(0.8, abs=1e-9)
    # Cost identical → no significant change
    assert not report.cost_delta.significant
    # Recommendation should be inconclusive (no quality change AND no cost change)
    assert "inconclusive" in report.recommendation.lower()


def test_compute_clear_cost_reduction_with_quality_holding() -> None:
    """v2 costs half as much per example; quality identical → ship recommendation."""
    n = 20
    a = _run(
        "baseline",
        [_example(f"e{i}", {"acc": 0.85}, cost_amount=0.02, latency=1.5) for i in range(n)],
    )
    b = _run(
        "cheap",
        [_example(f"e{i}", {"acc": 0.85}, cost_amount=0.01, latency=1.5) for i in range(n)],
    )
    report = compute_comparison(a, b)

    # Cost delta is ~-50%
    assert report.cost_delta.delta_pct is not None
    assert report.cost_delta.delta_pct == pytest.approx(-50.0, abs=0.01)
    assert report.cost_delta.significant
    # Quality unchanged
    assert report.scorer_deltas[0].delta == pytest.approx(0.0, abs=1e-9)
    # Recommendation
    assert "ship cheap" in report.recommendation.lower()


def test_compute_significant_quality_regression_triggers_dont_ship() -> None:
    """v2 has lower scores AND cheaper cost — recommendation should still be don't-ship."""
    n = 30
    rng = np.random.default_rng(123)
    a_scores = rng.uniform(0.85, 0.95, n)
    b_scores = a_scores - 0.20  # 20-pt drop, will be highly significant
    a = _run(
        "baseline",
        [_example(f"e{i}", {"acc": float(a_scores[i])}, cost_amount=0.02) for i in range(n)],
    )
    b = _run(
        "broken",
        [_example(f"e{i}", {"acc": float(b_scores[i])}, cost_amount=0.01) for i in range(n)],
    )
    report = compute_comparison(a, b)
    assert report.scorer_deltas[0].significant
    assert report.scorer_deltas[0].delta < 0
    assert "don't ship broken" in report.recommendation.lower()
    assert "regression on acc" in report.recommendation.lower()


def test_compute_pairs_only_common_examples() -> None:
    """Examples present in only one run should be silently excluded."""
    a = _run(
        "a",
        [
            _example("shared-1", {"acc": 1.0}, cost_amount=0.01),
            _example("shared-2", {"acc": 1.0}, cost_amount=0.01),
            _example("a-only", {"acc": 1.0}, cost_amount=0.01),
        ],
    )
    b = _run(
        "b",
        [
            _example("shared-1", {"acc": 1.0}, cost_amount=0.01),
            _example("shared-2", {"acc": 1.0}, cost_amount=0.01),
            _example("b-only", {"acc": 1.0}, cost_amount=0.01),
        ],
    )
    report = compute_comparison(a, b)
    assert report.n_paired == 2


def test_compute_excludes_errored_examples_from_pairing() -> None:
    """An example errored in either run shouldn't appear in the paired set."""
    a = _run(
        "a",
        [
            _example("ok", {"acc": 1.0}),
            _example("err-in-a", {"acc": 1.0}, error="boom"),
        ],
    )
    b = _run(
        "b",
        [
            _example("ok", {"acc": 1.0}),
            _example("err-in-a", {"acc": 1.0}),  # not errored in b, but errored in a
        ],
    )
    # Only 1 paired example — should raise (n<2 is undefined)
    with pytest.raises(ValueError, match="only 1 paired"):
        compute_comparison(a, b)


def test_compute_raises_on_no_paired_examples() -> None:
    a = _run("a", [_example("x", {"acc": 1.0})])
    b = _run("b", [_example("y", {"acc": 1.0})])
    with pytest.raises(ValueError, match="0 paired"):
        compute_comparison(a, b)


def test_compute_handles_only_common_scorers() -> None:
    """If runs have different scorer sets, only common ones appear in the report."""
    a = _run(
        "a",
        [_example(f"e{i}", {"acc": 0.9, "tone": 0.7}) for i in range(5)],
    )
    b = _run(
        "b",
        [
            _example(f"e{i}", {"acc": 0.9, "speed": 0.8})  # no tone
            for i in range(5)
        ],
    )
    report = compute_comparison(a, b)
    names = [d.name for d in report.scorer_deltas]
    assert names == ["acc"]


# ---------------------------------------------------------------------------
# _make_verdicts
# ---------------------------------------------------------------------------


def _delta(
    name: str = "m",
    delta: float = 0.0,
    delta_pct: float | None = 0.0,
    p: float = 0.5,
) -> MetricDelta:
    return MetricDelta(
        name=name,
        value_a=0.0,
        value_b=0.0,
        delta=delta,
        delta_pct=delta_pct,
        ci_low=delta - 0.05,
        ci_high=delta + 0.05,
        p_value=p,
        significant=p < 0.05,
    )


def test_verdicts_ship_path() -> None:
    quality_v, cost_v, rec = _make_verdicts(
        "v2",
        scorer_deltas=[_delta("acc", delta=0.0, p=0.5)],
        cost_delta=_delta("cost", delta_pct=-37, p=0.0001),
    )
    assert "no significant regression" in quality_v
    assert "37% reduction" in cost_v
    assert "ship v2" in rec


def test_verdicts_dont_ship_on_regression() -> None:
    _, _, rec = _make_verdicts(
        "v2",
        scorer_deltas=[_delta("acc", delta=-0.2, p=0.001)],
        cost_delta=_delta("cost", delta_pct=-50, p=0.0001),
    )
    assert "don't ship v2" in rec
    assert "regression on acc" in rec


def test_verdicts_expensive_warning() -> None:
    _, _, rec = _make_verdicts(
        "v2",
        scorer_deltas=[_delta("acc", delta=0.0, p=0.5)],
        cost_delta=_delta("cost", delta_pct=+30, p=0.001),
    )
    assert "more expensive" in rec


def test_verdicts_inconclusive_when_nothing_significant() -> None:
    _, _, rec = _make_verdicts(
        "v2",
        scorer_deltas=[_delta("acc", delta=0.0, p=0.7)],
        cost_delta=_delta("cost", delta_pct=2, p=0.6),
    )
    assert "inconclusive" in rec


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------


def test_render_text_includes_all_sections() -> None:
    a, b = _identical_runs(n=5)
    report = compute_comparison(a, b)
    text = render_text(report)
    # Anchor lines from the README's killer-output shape
    assert "=== a vs b ===" in text
    assert "Paired examples: 5" in text
    assert "p-value" in text
    assert "acc" in text  # the scorer
    assert "total cost" in text
    assert "avg latency" in text
    assert "Quality verdict:" in text
    assert "Cost verdict:" in text
    assert "Recommendation:" in text


def test_render_text_shows_significance_marker_in_p_column() -> None:
    """Very-significant cost change should show `<0.001`."""
    n = 30
    a = _run(
        "baseline",
        [_example(f"e{i}", {"acc": 0.9}, cost_amount=0.02) for i in range(n)],
    )
    b = _run(
        "cheap",
        [_example(f"e{i}", {"acc": 0.9}, cost_amount=0.005) for i in range(n)],  # 4x cheaper
    )
    report = compute_comparison(a, b)
    text = render_text(report)
    assert "<0.001" in text  # cost regression should be hugely significant
