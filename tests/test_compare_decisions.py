"""Regression tests for the decision logic: each case is a scenario from an issue.

#1 "ship" while a large drop is plausible · #2 gates that pass on thin data ·
#3 errors dropped silently · #4 false "don't ship" growing with scorers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prompteval.compare import compute_comparison, evaluate_gates, parse_gate_spec, render_text
from prompteval.cost import Usage, compute_cost
from prompteval.eval.runner import ExampleResult, RunResult, ScoreOutcome


def _run(
    tag: str,
    scores: dict[str, Any],
    costs: Any,
    errors: set[int] | None = None,
) -> RunResult:
    errors = errors or set()
    examples = []
    for i, tokens in enumerate(costs):
        failed = i in errors
        examples.append(
            ExampleResult(
                id=f"e{i}",
                input="",
                expected={},
                output="",
                scores=[] if failed else [ScoreOutcome(k, float(v[i])) for k, v in scores.items()],
                cost=compute_cost(
                    "gpt-4o-mini",
                    Usage(prompt_tokens=0 if failed else int(tokens), completion_tokens=0),
                ),
                latency_s=1.0,
                error="RateLimitError: slow down" if failed else None,
            )
        )
    return RunResult(
        tag=tag,
        eval_name="e",
        model="gpt-4o-mini",
        prompt_path="p.txt",
        prompt_text="p",
        started_at="",
        finished_at="",
        examples=examples,
    )


def _plausible_drop_case() -> tuple[RunResult, RunResult]:
    """The #1 probe: binary scorer, n=20, true pass rate 80% → 65%, cost -30%."""
    rng = np.random.default_rng(1)
    a = (rng.random(20) < 0.80).astype(float)
    b = (rng.random(20) < 0.65).astype(float)
    cost_a = rng.normal(1000, 50, 20)
    return _run("v1", {"q": a}, cost_a), _run("v2", {"q": b}, cost_a * 0.7)


# ---------------------------------------------------------------- #1


def test_plausible_large_drop_is_inconclusive_not_ship() -> None:
    report = compute_comparison(*_plausible_drop_case())
    assert not report.recommendation.startswith("ship")
    assert report.recommendation.startswith("inconclusive")
    assert "examples would settle it" in report.recommendation or "beyond the margin" in (
        report.recommendation
    )
    assert report.scorer_deltas[0].status == "unclear"


def test_quality_within_margin_and_cheaper_ships() -> None:
    rng = np.random.default_rng(2)
    a = rng.uniform(0.7, 0.9, 200)
    b = np.clip(a + rng.normal(0, 0.01, 200), 0, 1)  # same quality, tiny noise
    cost_a = rng.normal(1000, 50, 200)
    report = compute_comparison(_run("v1", {"q": a}, cost_a), _run("v2", {"q": b}, cost_a * 0.7))
    assert report.scorer_deltas[0].status == "ok"
    assert report.recommendation.startswith("ship v2")


def test_certain_drop_is_dont_ship() -> None:
    a = np.full(30, 0.9)
    b = np.full(30, 0.5)
    report = compute_comparison(_run("v1", {"q": a}, [1000] * 30), _run("v2", {"q": b}, [500] * 30))
    assert report.recommendation.startswith("don't ship v2")


def test_margin_is_configurable() -> None:
    run_a, run_b = _plausible_drop_case()
    loose = compute_comparison(run_a, run_b, margin=0.6)
    assert loose.scorer_deltas[0].status == "ok"
    assert loose.recommendation.startswith("ship v2")


# ---------------------------------------------------------------- #2


def test_bound_gate_fails_when_a_breach_is_plausible() -> None:
    report = compute_comparison(*_plausible_drop_case())
    breaches = evaluate_gates(report, parse_gate_spec("quality-5%"))
    assert len(breaches) == 1
    assert "could be" in breaches[0].detail


def test_bound_gate_catches_a_noisy_cost_rise_on_few_examples() -> None:
    ones = [1.0] * 4
    report = compute_comparison(
        _run("v1", {"q": ones}, [1000, 200, 3000, 500]),
        _run("v2", {"q": ones}, [600, 900, 4500, 500]),
    )
    assert evaluate_gates(report, parse_gate_spec("cost+10%")) != []
    # The v0.1 behaviour is still available and still passes it.
    assert evaluate_gates(report, parse_gate_spec("cost+10%"), mode="significant") == []


def test_bound_gate_passes_a_clear_improvement() -> None:
    ones = [1.0] * 20
    report = compute_comparison(
        _run("v1", {"q": ones}, [1000] * 20), _run("v2", {"q": ones}, [500] * 20)
    )
    assert evaluate_gates(report, parse_gate_spec("cost+10%,quality-5%")) == []


# ---------------------------------------------------------------- #3


def test_candidate_errors_count_against_it() -> None:
    ones = [1.0] * 20
    cost = [1000] * 20
    report = compute_comparison(
        _run("v1", {"q": ones}, cost), _run("v2", {"q": ones}, [700] * 20, errors=set(range(6)))
    )
    assert report.errors_a == 0 and report.errors_b == 6
    assert report.error_delta is not None and report.error_delta.status == "regressed"
    assert report.recommendation.startswith("don't ship v2")
    assert "error rate" in report.recommendation
    text = render_text(report)
    assert "Errors: 0 in v1, 6 in v2" in text
    assert "error rate" in text
    assert evaluate_gates(report, parse_gate_spec("quality-5%")) != []


def test_no_error_row_when_neither_run_errored() -> None:
    ones = [1.0] * 5
    report = compute_comparison(_run("v1", {"q": ones}, [1] * 5), _run("v2", {"q": ones}, [1] * 5))
    assert report.error_delta is None
    assert "Errors:" not in render_text(report)


# ---------------------------------------------------------------- #4


def test_identical_prompts_rarely_get_dont_ship_with_many_scorers() -> None:
    trials, scorers, n = 100, 6, 20
    false_alarms = 0
    for t in range(trials):
        rng = np.random.default_rng(100 + t)
        runs = [
            _run(
                tag,
                {f"s{k}": (rng.random(n) < 0.7).astype(float) for k in range(scorers)},
                [1000] * n,
            )
            for tag in ("a", "b")
        ]
        if compute_comparison(*runs).recommendation.startswith("don't ship"):
            false_alarms += 1
    assert false_alarms / trials <= 0.05  # was 15% before the correction


# ---------------------------------------------------------------- #7


def test_different_evals_are_refused() -> None:
    from dataclasses import replace

    import pytest

    a = _run("v1", {"q": [1.0] * 3}, [1] * 3)
    b = replace(_run("v2", {"q": [1.0] * 3}, [1] * 3), eval_name="other")
    with pytest.raises(ValueError, match="different evals"):
        compute_comparison(a, b)


def test_mismatched_dataset_models_and_ids_are_warned_about() -> None:
    from dataclasses import replace

    a = replace(_run("v1", {"q": [1.0] * 4}, [1] * 4), dataset_sha256="aaa")
    b = replace(_run("v2", {"q": [1.0] * 3}, [1] * 3), dataset_sha256="bbb", model="gpt-4o")
    report = compute_comparison(a, b)
    text = render_text(report)
    assert "different dataset files" in text
    assert "1 example(s) only in 'v1' and 0 only in 'v2'" in text
    assert "the models differ (gpt-4o-mini vs gpt-4o)" in text
    assert "Models: v1=gpt-4o-mini, v2=gpt-4o" in text


def test_matching_runs_carry_no_warnings() -> None:
    from dataclasses import replace

    a = replace(_run("v1", {"q": [1.0] * 3}, [1] * 3), dataset_sha256="same")
    b = replace(_run("v2", {"q": [1.0] * 3}, [1] * 3), dataset_sha256="same")
    assert compute_comparison(a, b).warnings == ()
