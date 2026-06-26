"""Compare two RunResults — paired deltas with bootstrap CIs + significance + verdicts.

This is the **killer report** the wedge depends on. Given two runs that hit the
same dataset (e.g. `baseline` vs `short-prompt`), produce:

- Per-scorer paired comparison: delta = mean(score_b - score_a), with 95% bootstrap CI
  and paired t-test p-value
- Cost paired comparison: total % change, with bootstrap CI + paired t-test
- Latency paired comparison: same shape as cost
- Quality / cost verdicts in plain English
- Ship/don't-ship recommendation

Pairing is by `example.id` — only common ids are used. Errored examples are
excluded from both runs. If <2 paired examples are found, comparison raises
(stats are undefined).

## Why scipy

We use `scipy.stats.ttest_rel` for paired p-values and `scipy.stats.bootstrap`
for mean-delta CIs. Custom bootstrap loop for percent-change CIs (those need
to resample paired indices and re-sum, not just average a 1D sample).

## v1 stats choices

- **Bootstrap for CIs** (percentile method, 10K resamples, default seeded RNG
  for reproducibility) — robust, doesn't assume normality, works on any sample
  shape including 0/1 deterministic scorer outputs.
- **Paired t-test for p-values** — simple, well-understood, "good enough" for
  v1. For binary-output scorers Wilcoxon signed-rank would be more correct;
  we can add it later if real users complain. The decision-driving signal is
  the bootstrap CI; the p-value is a secondary check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats  # type: ignore[import-untyped]

from prompteval.eval.runner import RunResult

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDelta:
    """One metric (a scorer mean, cost, or latency) compared across two runs."""

    name: str
    value_a: float
    value_b: float
    delta: float  # value_b - value_a in original units
    delta_pct: float | None  # percent change relative to value_a; None for [0,1] scores
    ci_low: float  # 95% CI lower bound — same units as `delta` (or `delta_pct` for cost/latency)
    ci_high: float
    p_value: float
    significant: bool  # convenience: p < 0.05


@dataclass(frozen=True)
class ComparisonReport:
    """Result of comparing two RunResults."""

    tag_a: str
    tag_b: str
    n_paired: int  # how many examples were actually compared (post-error filtering)
    scorer_deltas: list[MetricDelta]  # one per common scorer
    cost_delta: MetricDelta
    latency_delta: MetricDelta
    quality_verdict: str  # "no significant regression" / "significant regression on X"
    cost_verdict: str  # "significant 37% reduction (p<0.001)" / similar
    recommendation: str  # "ship tag_b — ..." / "don't ship tag_b — ..." / "inconclusive"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _pair_examples(
    run_a: RunResult,
    run_b: RunResult,
) -> tuple[list[str], list[int], list[int]]:
    """Return `(common_ids, indices_a, indices_b)` for examples successful in BOTH runs.

    Errored examples are excluded — even if both runs share the example id,
    pairing it would compare zero-cost / no-output records and report nonsense.
    """
    by_id_a = {ex.id: i for i, ex in enumerate(run_a.examples) if ex.error is None}
    by_id_b = {ex.id: i for i, ex in enumerate(run_b.examples) if ex.error is None}
    common = sorted(set(by_id_a) & set(by_id_b))
    return common, [by_id_a[i] for i in common], [by_id_b[i] for i in common]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI on the mean of `values`, percentile method.

    Seeded RNG so the same data produces the same CI bounds across runs —
    important for snapshot-style tests + reproducible reports.
    """
    if len(values) < 2:
        # Single-sample CI is the value itself ± nothing. Avoid scipy's error.
        v = float(values[0]) if len(values) == 1 else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    result = scipy.stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=confidence,
        method="percentile",
        n_resamples=n_resamples,
        random_state=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _bootstrap_percent_change_ci(
    a: np.ndarray,
    b: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI on percent change `(total_b - total_a) / total_a * 100`.

    Paired bootstrap: resample example indices, recompute the % change from
    the resampled paired totals. Custom loop because scipy.stats.bootstrap's
    statistic signature wants a single sample, not paired arrays.
    """
    if len(a) != len(b):
        raise ValueError(f"Paired arrays differ in length: {len(a)} vs {len(b)}")
    if len(a) < 2:
        # Degenerate — no resampling possible.
        total_a = float(a.sum())
        if total_a == 0:
            return 0.0, 0.0
        pct = (float(b.sum()) - total_a) / total_a * 100
        return pct, pct

    rng = np.random.default_rng(seed)
    n = len(a)
    deltas_pct: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a_sum = float(a[idx].sum())
        b_sum = float(b[idx].sum())
        if a_sum == 0:
            # Resample that hit only zero-cost examples — skip rather than div-by-zero.
            # Rare in practice; if it dominates the CI we have bigger problems.
            continue
        deltas_pct.append((b_sum - a_sum) / a_sum * 100)
    if not deltas_pct:
        return 0.0, 0.0
    deltas_pct.sort()
    lo_idx = int((1 - confidence) / 2 * len(deltas_pct))
    hi_idx = int((1 + confidence) / 2 * len(deltas_pct)) - 1
    return deltas_pct[lo_idx], deltas_pct[hi_idx]


def _paired_p_value(a: np.ndarray, b: np.ndarray) -> float:
    """Paired t-test p-value (two-sided) for H0: mean(b - a) == 0.

    Constant-diff edge cases handled explicitly because scipy returns NaN
    when the variance of differences is zero:
      - All diffs == 0     → no signal at all → p = 1.0
      - All diffs == k ≠ 0 → perfect signal (every paired example agrees on
                              the same nonzero delta) → p ≈ 0.0
    Real-world data won't hit these (every API call has noise), but synthetic
    test data does — and the report should still be sensible.
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0
    diffs = b - a
    if np.allclose(diffs, diffs[0]):
        return 1.0 if np.allclose(diffs[0], 0) else 0.0
    result = scipy.stats.ttest_rel(b, a)
    p = float(result.pvalue)
    if np.isnan(p):
        return 1.0
    return p


# ---------------------------------------------------------------------------
# Per-metric construction
# ---------------------------------------------------------------------------


def _build_scorer_delta(
    name: str,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> MetricDelta:
    """Paired comparison for one scorer. Scores are typically in [0, 1]; delta
    is reported in original units (NOT percent change, which is meaningless
    when a denominator can be 0)."""
    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())
    diffs = scores_b - scores_a
    delta = float(diffs.mean())
    ci_low, ci_high = _bootstrap_mean_ci(diffs)
    p = _paired_p_value(scores_a, scores_b)
    return MetricDelta(
        name=name,
        value_a=mean_a,
        value_b=mean_b,
        delta=delta,
        delta_pct=None,  # percent change is misleading for [0, 1] scores
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=p < 0.05,
    )


def _build_relative_delta(
    name: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> MetricDelta:
    """Paired comparison for cost or latency. Reports % change (the metric
    users care about) with paired-bootstrap CI on that % change."""
    total_a = float(values_a.sum())
    total_b = float(values_b.sum())
    # Defensive: report 0 change when there's no baseline cost to divide by.
    delta_pct: float | None = (total_b - total_a) / total_a * 100 if total_a > 0 else None
    ci_low, ci_high = _bootstrap_percent_change_ci(values_a, values_b)
    p = _paired_p_value(values_a, values_b)
    # For absolute delta, use means so cost and latency have comparable shapes.
    return MetricDelta(
        name=name,
        value_a=float(values_a.mean()),
        value_b=float(values_b.mean()),
        delta=float(values_b.mean()) - float(values_a.mean()),
        delta_pct=delta_pct,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=p < 0.05,
    )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _format_p(p: float) -> str:
    """Display p-value as the user expects in the killer report — "<0.001"
    when very small, otherwise 2-3 sig figs."""
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _make_verdicts(
    tag_b: str,
    scorer_deltas: list[MetricDelta],
    cost_delta: MetricDelta,
) -> tuple[str, str, str]:
    """Quality + cost verdict + ship/don't-ship recommendation.

    Quality regression = any scorer with delta < 0 AND p < 0.05. We don't
    treat "delta < 0 but not significant" as regression — that's the whole
    point of significance testing.
    """
    regressions = [d for d in scorer_deltas if d.delta < 0 and d.significant]

    if regressions:
        names = ", ".join(r.name for r in regressions)
        quality_verdict = f"significant regression on {names}"
    else:
        quality_verdict = (
            "no significant regression (all quality deltas overlap zero or are positive)"
        )

    if cost_delta.significant and cost_delta.delta_pct is not None:
        magnitude = abs(cost_delta.delta_pct)
        direction = "reduction" if cost_delta.delta_pct < 0 else "increase"
        p_str = _format_p(cost_delta.p_value)
        p_render = f"p{p_str}" if p_str.startswith("<") else f"p={p_str}"
        cost_verdict = f"significant {magnitude:.0f}% {direction} ({p_render})"
    else:
        cost_verdict = "no significant cost change"

    # Recommendation
    if regressions:
        names = ", ".join(r.name for r in regressions)
        recommendation = f"don't ship {tag_b} — significant regression on {names}"
    elif cost_delta.significant and cost_delta.delta_pct is not None and cost_delta.delta_pct < 0:
        recommendation = f"ship {tag_b} — cost savings real, quality holds"
    elif cost_delta.significant and cost_delta.delta_pct is not None and cost_delta.delta_pct > 0:
        recommendation = (
            f"{tag_b} is significantly more expensive — only ship if other benefits justify"
        )
    else:
        recommendation = "inconclusive — increase sample size or compare again"

    return quality_verdict, cost_verdict, recommendation


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_comparison(run_a: RunResult, run_b: RunResult) -> ComparisonReport:
    """Pair two runs by example id and produce a ComparisonReport.

    Raises ValueError if fewer than 2 paired examples are available — bootstrap
    + t-test are undefined below that and silently producing zeros would mislead.
    """
    common_ids, idx_a, idx_b = _pair_examples(run_a, run_b)
    n = len(common_ids)
    if n < 2:
        raise ValueError(
            f"Cannot compare {run_a.tag!r} vs {run_b.tag!r}: only {n} paired examples "
            f"after excluding errors. Need at least 2 for meaningful stats."
        )

    # Build aligned scorer-name → array maps so we can paired-compare.
    common_scorer_names = _common_scorer_names(run_a, run_b)
    scorer_deltas: list[MetricDelta] = []
    for s_name in common_scorer_names:
        scores_a = np.array([_score_for(run_a.examples[i], s_name) for i in idx_a], dtype=float)
        scores_b = np.array([_score_for(run_b.examples[i], s_name) for i in idx_b], dtype=float)
        scorer_deltas.append(_build_scorer_delta(s_name, scores_a, scores_b))

    costs_a = np.array([run_a.examples[i].cost.total_cost for i in idx_a], dtype=float)
    costs_b = np.array([run_b.examples[i].cost.total_cost for i in idx_b], dtype=float)
    cost_delta = _build_relative_delta("total cost", costs_a, costs_b)

    latencies_a = np.array([run_a.examples[i].latency_s for i in idx_a], dtype=float)
    latencies_b = np.array([run_b.examples[i].latency_s for i in idx_b], dtype=float)
    latency_delta = _build_relative_delta("avg latency", latencies_a, latencies_b)

    quality_verdict, cost_verdict, recommendation = _make_verdicts(
        run_b.tag, scorer_deltas, cost_delta
    )

    return ComparisonReport(
        tag_a=run_a.tag,
        tag_b=run_b.tag,
        n_paired=n,
        scorer_deltas=scorer_deltas,
        cost_delta=cost_delta,
        latency_delta=latency_delta,
        quality_verdict=quality_verdict,
        cost_verdict=cost_verdict,
        recommendation=recommendation,
    )


def _common_scorer_names(run_a: RunResult, run_b: RunResult) -> list[str]:
    """Scorers that appear in EVERY successful example of both runs."""

    def _names_in_all(run: RunResult) -> set[str]:
        successful = [ex for ex in run.examples if ex.error is None]
        if not successful:
            return set()
        names = {s.name for s in successful[0].scores}
        for ex in successful[1:]:
            names &= {s.name for s in ex.scores}
        return names

    return sorted(_names_in_all(run_a) & _names_in_all(run_b))


def _score_for(example: object, scorer_name: str) -> float:
    """Look up a single scorer's score on one example. 0.0 if absent (defensive)."""
    for s in example.scores:  # type: ignore[attr-defined]
        if s.name == scorer_name:
            return float(s.score)
    return 0.0


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def render_text(report: ComparisonReport) -> str:
    """Render the ComparisonReport in the format the README anchors on."""
    lines: list[str] = []
    lines.append(f"=== {report.tag_a} vs {report.tag_b} ===")
    lines.append(f"Paired examples: {report.n_paired}")
    lines.append("")

    header = f"{'':<26}{report.tag_a:<14}{report.tag_b:<16}{'Δ':<14}{'95% CI':<22}{'p-value':<10}"
    lines.append(header)
    lines.append("─" * len(header))

    # Scorer rows — values are scores in [0, 1], delta is absolute
    for d in report.scorer_deltas:
        delta_str = _signed(d.delta)
        ci_str = f"[{_signed(d.ci_low)}, {_signed(d.ci_high)}]"
        p_str = _format_p(d.p_value)
        lines.append(
            f"{d.name:<26}"
            f"{d.value_a:<14.2f}"
            f"{d.value_b:<16.2f}"
            f"{delta_str:<14}"
            f"{ci_str:<22}"
            f"{p_str:<10}"
        )

    lines.append("─" * len(header))

    # Cost row — values + percent change
    cost = report.cost_delta
    cost_a_str = f"${cost.value_a * report.n_paired:.4f}"
    cost_b_str = f"${cost.value_b * report.n_paired:.4f}"
    cost_delta_str = (
        f"{cost.delta_pct:+.0f}%" if cost.delta_pct is not None else _signed(cost.delta)
    )
    cost_ci_str = f"[{cost.ci_low:+.0f}%, {cost.ci_high:+.0f}%]"
    cost_p_str = _format_p(cost.p_value)
    lines.append(
        f"{'total cost':<26}"
        f"{cost_a_str:<14}"
        f"{cost_b_str:<16}"
        f"{cost_delta_str:<14}"
        f"{cost_ci_str:<22}"
        f"{cost_p_str:<10}"
    )

    # Latency row — same shape as cost
    lat = report.latency_delta
    lat_a_str = f"{lat.value_a:.2f}s"
    lat_b_str = f"{lat.value_b:.2f}s"
    lat_delta_str = f"{lat.delta_pct:+.0f}%" if lat.delta_pct is not None else _signed(lat.delta)
    lat_ci_str = f"[{lat.ci_low:+.0f}%, {lat.ci_high:+.0f}%]"
    lat_p_str = _format_p(lat.p_value)
    lines.append(
        f"{'avg latency':<26}"
        f"{lat_a_str:<14}"
        f"{lat_b_str:<16}"
        f"{lat_delta_str:<14}"
        f"{lat_ci_str:<22}"
        f"{lat_p_str:<10}"
    )

    lines.append("")
    lines.append(f"Quality verdict:  {report.quality_verdict}")
    lines.append(f"Cost verdict:     {report.cost_verdict}")
    lines.append("")
    lines.append(f"Recommendation:   {report.recommendation}")

    return "\n".join(lines)


def _signed(x: float) -> str:
    """Format a number with explicit sign — README's killer report style."""
    return f"{x:+.3f}" if abs(x) < 1 else f"{x:+.2f}"
