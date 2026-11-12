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

## Stats choices

- **One method throughout: the paired bootstrap** (percentile, 10K resamples,
  seeded for reproducibility). The 95% CI and the p-value shown in the table
  come from the same resampled distribution, so they cannot disagree (#1).
  It assumes nothing about the shape of the scores, including 0/1 scorers.
- **Decisions use non-inferiority against a margin** (#1). "No significant
  drop" is not "quality holds": with 20 examples almost nothing is
  significant. A scorer passes only when the worst plausible drop (the CI's
  lower bound) is within `margin` (default 0.05). If a drop is certain it is
  a regression; otherwise the result is inconclusive, with an estimate of how
  many examples would settle it.
- **Decision CIs are Bonferroni-adjusted** across the quality metrics (each
  scorer plus the error rate), so adding scorers does not add false
  "don't ship" verdicts (#4). The table still shows plain 95% CIs.
- **Errors count** (#3). An example that failed in the candidate but not the
  baseline is a quality loss: the error rate is a quality metric of its own,
  higher being worse.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from prompteval.eval.runner import RunResult

#: Largest drop (absolute, on a [0, 1] scorer) treated as "quality holds".
DEFAULT_MARGIN = 0.05
#: Family-wise error rate for the ship decision.
ALPHA = 0.05
_N_RESAMPLES = 10_000
_SEED = 42

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
    significant: bool  # the 95% CI excludes zero (equivalently p < 0.05)
    # Bounds used for the decision: Bonferroni-adjusted across the quality
    # metrics. None for cost and latency, and for hand-built deltas.
    decision_low: float | None = None
    decision_high: float | None = None
    # Quality metrics only: "ok" (within the margin), "regressed", "unclear".
    status: str = ""


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
    margin: float = DEFAULT_MARGIN
    decision_level: float = 1 - ALPHA  # confidence level of the decision CIs
    # Error rate over the examples both runs attempted; None when neither errored.
    error_delta: MetricDelta | None = None
    errors_a: int = 0
    errors_b: int = 0
    n_attempted: int = 0  # examples present in both runs, errored or not
    model_a: str = ""
    model_b: str = ""
    # Things that make the comparison weaker than it looks (#7).
    warnings: tuple[str, ...] = ()


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


def _bootstrap_mean_dist(
    values: np.ndarray, n_resamples: int = _N_RESAMPLES, seed: int = _SEED
) -> np.ndarray:
    """Bootstrap distribution of the mean of `values` (paired diffs)."""
    out = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // max(1, len(values)))  # bound memory on large datasets
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        out[start : start + size] = values[idx].mean(axis=1)
    return out


def _bootstrap_pct_dist(
    a: np.ndarray, b: np.ndarray, n_resamples: int = _N_RESAMPLES, seed: int = _SEED
) -> np.ndarray:
    """Bootstrap distribution of `(sum(b) - sum(a)) / sum(a) * 100`, paired.

    Resamples whose baseline total is zero are dropped (no percent change exists).
    """
    if len(a) != len(b):
        raise ValueError(f"Paired arrays differ in length: {len(a)} vs {len(b)}")
    out: list[np.ndarray] = []
    chunk = max(1, 2_000_000 // max(1, len(a)))
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, len(a), size=(size, len(a)))
        sa = a[idx].sum(axis=1)
        sb = b[idx].sum(axis=1)
        keep = sa > 0
        out.append((sb[keep] - sa[keep]) / sa[keep] * 100)
    return np.concatenate(out) if out else np.empty(0)


def _small_sample_widen(dist: np.ndarray, n: int, level: float) -> np.ndarray:
    """Widen a bootstrap distribution about its mean for small samples.

    The percentile bootstrap is too narrow at small n (it behaves like a z
    interval, and the plug-in variance is biased by (n-1)/n), so decisions
    made from it fire too often: measured 7% false "don't ship" at n=20 with
    six scorers, against a 5% target (#4). Scaling by sqrt(n/(n-1)) * t/z
    gives it the coverage of the matching t interval. Used for decisions only.
    """
    from scipy.stats import norm, t  # type: ignore[import-untyped]

    tail = 1 - (1 - level) / 2
    factor = math.sqrt(n / (n - 1)) * float(t.ppf(tail, n - 1)) / float(norm.ppf(tail))
    centre = float(dist.mean())
    return centre + (dist - centre) * factor


def _percentile_ci(dist: np.ndarray, level: float) -> tuple[float, float]:
    if dist.size == 0:
        return 0.0, 0.0
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(dist, [tail, 100 - tail])
    return float(lo), float(hi)


def _bootstrap_p(dist: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: the statistic is zero.

    Consistent with the percentile CI: the 95% CI excludes zero exactly when
    p < 0.05. A degenerate distribution at zero gives 1.0; at a nonzero
    constant, 0.0.
    """
    if dist.size == 0:
        return 1.0
    below = float(np.mean(dist <= 0))
    above = float(np.mean(dist >= 0))
    return min(1.0, 2 * min(below, above))


def _bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = _N_RESAMPLES,
    seed: int = _SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`."""
    if len(values) < 2:
        v = float(values[0]) if len(values) == 1 else 0.0
        return v, v
    return _percentile_ci(_bootstrap_mean_dist(values, n_resamples, seed), confidence)


def _bootstrap_percent_change_ci(
    a: np.ndarray,
    b: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = _N_RESAMPLES,
    seed: int = _SEED,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI on the percent change of totals."""
    if len(a) != len(b):
        raise ValueError(f"Paired arrays differ in length: {len(a)} vs {len(b)}")
    if len(a) < 2:
        total_a = float(a.sum())
        if total_a == 0:
            return 0.0, 0.0
        pct = (float(b.sum()) - total_a) / total_a * 100
        return pct, pct
    return _percentile_ci(_bootstrap_pct_dist(a, b, n_resamples, seed), confidence)


# ---------------------------------------------------------------------------
# Per-metric construction
# ---------------------------------------------------------------------------


def _build_mean_delta(
    name: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    decision_level: float,
) -> MetricDelta:
    """Paired comparison of a [0, 1] metric (a scorer, or the error rate).

    The delta is in original units: a percent change is meaningless when the
    baseline can be 0.
    """
    diffs = values_b - values_a
    if len(diffs) >= 2:
        dist = _bootstrap_mean_dist(diffs)
        ci_low, ci_high = _percentile_ci(dist, 0.95)
        dec_low, dec_high = _percentile_ci(
            _small_sample_widen(dist, len(diffs), decision_level), decision_level
        )
        p = _bootstrap_p(dist)
    else:
        ci_low = ci_high = dec_low = dec_high = float(diffs.mean())
        p = 1.0
    return MetricDelta(
        name=name,
        value_a=float(values_a.mean()),
        value_b=float(values_b.mean()),
        delta=float(diffs.mean()),
        delta_pct=None,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=p < ALPHA,
        decision_low=dec_low,
        decision_high=dec_high,
    )


def _build_scorer_delta(
    name: str,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    decision_level: float = 1 - ALPHA,
) -> MetricDelta:
    """Paired comparison for one scorer (scores in [0, 1])."""
    return _build_mean_delta(name, scores_a, scores_b, decision_level)


def _build_relative_delta(
    name: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> MetricDelta:
    """Paired comparison for cost or latency: % change of totals with its CI."""
    total_a = float(values_a.sum())
    total_b = float(values_b.sum())
    delta_pct: float | None = (total_b - total_a) / total_a * 100 if total_a > 0 else None
    if len(values_a) >= 2:
        dist = _bootstrap_pct_dist(values_a, values_b)
        ci_low, ci_high = _percentile_ci(dist, 0.95)
        p = _bootstrap_p(dist)
    else:
        ci_low, ci_high = _bootstrap_percent_change_ci(values_a, values_b)
        p = 1.0
    return MetricDelta(
        name=name,
        value_a=float(values_a.mean()),
        value_b=float(values_b.mean()),
        delta=float(values_b.mean()) - float(values_a.mean()),
        delta_pct=delta_pct,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=p < ALPHA and delta_pct is not None,
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


def _bounds(d: MetricDelta) -> tuple[float, float]:
    lo = d.decision_low if d.decision_low is not None else d.ci_low
    hi = d.decision_high if d.decision_high is not None else d.ci_high
    return lo, hi


def _quality_status(d: MetricDelta, margin: float, lower_is_better: bool = False) -> str:
    """ok: the worst plausible change is within the margin. regressed: a drop is
    certain (and may exceed the margin). unclear: neither."""
    lo, hi = _bounds(d)
    if lower_is_better:  # error rate: an increase is the bad direction
        lo, hi = -hi, -lo
    if lo >= -margin:
        return "ok"
    if hi < 0:
        return "regressed"
    return "unclear"


def _examples_needed(d: MetricDelta, n: int, margin: float, lower_is_better: bool) -> int | None:
    """Rough total sample size at which the decision CI would clear the margin,
    assuming the measured change holds (CI width shrinks as 1/sqrt(n))."""
    lo, hi = _bounds(d)
    delta = -d.delta if lower_is_better else d.delta
    room = delta + margin
    half = (hi - lo) / 2
    if n <= 0 or room <= 0 or half <= 0:
        return None
    return math.ceil(n * (half / room) ** 2)


def _make_verdicts(
    tag_b: str,
    scorer_deltas: list[MetricDelta],
    cost_delta: MetricDelta,
    *,
    margin: float = DEFAULT_MARGIN,
    error_delta: MetricDelta | None = None,
    n: int = 0,
) -> tuple[str, str, str]:
    """Quality verdict, cost verdict and recommendation.

    Quality is decided per metric against `margin` (see the module docstring):
    any regression → don't ship; anything unclear → inconclusive; otherwise
    the cost decides.
    """
    metrics: list[tuple[MetricDelta, bool]] = [(d, False) for d in scorer_deltas]
    if error_delta is not None:
        metrics.append((error_delta, True))
    regressed = [(d, lib) for d, lib in metrics if _quality_status(d, margin, lib) == "regressed"]
    unclear = [(d, lib) for d, lib in metrics if _quality_status(d, margin, lib) == "unclear"]

    if cost_delta.significant and cost_delta.delta_pct is not None:
        magnitude = abs(cost_delta.delta_pct)
        direction = "reduction" if cost_delta.delta_pct < 0 else "increase"
        p_str = _format_p(cost_delta.p_value)
        p_render = f"p{p_str}" if p_str.startswith("<") else f"p={p_str}"
        cost_verdict = f"significant {magnitude:.0f}% {direction} ({p_render})"
    else:
        cost_verdict = "no significant cost change"

    if regressed:
        names = ", ".join(_metric_label(d, lib) for d, lib in regressed)
        quality_verdict = f"significant regression on {names}"
        recommendation = f"don't ship {tag_b} — significant regression on {names}"
        return quality_verdict, cost_verdict, recommendation

    if unclear:
        names = ", ".join(_metric_label(d, lib) for d, lib in unclear)
        worst = max((_bounds(d)[1] if lib else -_bounds(d)[0]) for d, lib in unclear)
        quality_verdict = (
            f"can't rule out a drop larger than {margin:.2f} on {names} "
            f"(could be up to {worst:.2f} worse)"
        )
        needed = [_examples_needed(d, n, margin, lib) for d, lib in unclear]
        if all(x is not None for x in needed):
            target = max(x for x in needed if x is not None)
            advice = f"about {target} examples would settle it (you have {n})"
        else:
            advice = "the measured drop is already beyond the margin; add examples to confirm"
        recommendation = f"inconclusive — {names} could be up to {worst:.2f} worse; {advice}"
        return quality_verdict, cost_verdict, recommendation

    quality_verdict = f"holds — no metric can have dropped by more than {margin:.2f}"
    if cost_delta.significant and cost_delta.delta_pct is not None and cost_delta.delta_pct < 0:
        recommendation = f"ship {tag_b} — cost savings real, quality holds"
    elif cost_delta.significant and cost_delta.delta_pct is not None and cost_delta.delta_pct > 0:
        recommendation = (
            f"{tag_b} is significantly more expensive — only ship if other benefits justify"
        )
    else:
        recommendation = (
            "inconclusive on cost — quality holds within the margin, no significant cost change"
        )
    return quality_verdict, cost_verdict, recommendation


def _metric_label(d: MetricDelta, lower_is_better: bool) -> str:
    return "error rate" if lower_is_better else d.name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_comparison(
    run_a: RunResult,
    run_b: RunResult,
    *,
    margin: float = DEFAULT_MARGIN,
    family_size: int = 1,
) -> ComparisonReport:
    """Pair two runs by example id and produce a ComparisonReport.

    Raises ValueError if fewer than 2 paired examples are available — bootstrap
    + t-test are undefined below that and silently producing zeros would mislead.
    """
    if run_a.eval_name != run_b.eval_name:
        raise ValueError(
            f"Cannot compare {run_a.tag!r} vs {run_b.tag!r}: they come from different evals "
            f"({run_a.eval_name!r} and {run_b.eval_name!r}), so their scores mean different things."
        )
    warnings = _compatibility_warnings(run_a, run_b)

    common_ids, idx_a, idx_b = _pair_examples(run_a, run_b)
    n = len(common_ids)
    if n < 2:
        raise ValueError(
            f"Cannot compare {run_a.tag!r} vs {run_b.tag!r}: only {n} paired examples "
            f"after excluding errors ({_count_errors(run_a)} errored in {run_a.tag!r}, "
            f"{_count_errors(run_b)} in {run_b.tag!r}). Need at least 2 for meaningful stats."
        )

    if not 0 <= margin < 1:
        raise ValueError(f"margin must be in [0, 1), got {margin}")

    # Error rate over the examples both runs attempted (#3).
    attempted = sorted({e.id for e in run_a.examples} & {e.id for e in run_b.examples})
    err_a_by_id = {e.id: e.error is not None for e in run_a.examples}
    err_b_by_id = {e.id: e.error is not None for e in run_b.examples}
    errs_a = np.array([float(err_a_by_id[i]) for i in attempted])
    errs_b = np.array([float(err_b_by_id[i]) for i in attempted])
    any_errors = bool(errs_a.sum() or errs_b.sum())

    common_scorer_names = _common_scorer_names(run_a, run_b)
    # Bonferroni across the quality metrics, so more scorers ≠ more false alarms (#4).
    k = len(common_scorer_names) + (1 if any_errors else 0)
    # `family_size` > 1 when this is one of several comparisons made together
    # (a model sweep), so the family-wise error stays at ALPHA across all of them.
    decision_level = 1 - ALPHA / (max(1, k) * max(1, family_size))

    scorer_deltas: list[MetricDelta] = []
    for s_name in common_scorer_names:
        scores_a = np.array([_score_for(run_a.examples[i], s_name) for i in idx_a], dtype=float)
        scores_b = np.array([_score_for(run_b.examples[i], s_name) for i in idx_b], dtype=float)
        d = _build_scorer_delta(s_name, scores_a, scores_b, decision_level)
        scorer_deltas.append(replace(d, status=_quality_status(d, margin)))

    error_delta: MetricDelta | None = None
    if any_errors:
        e = _build_mean_delta("error rate", errs_a, errs_b, decision_level)
        error_delta = replace(e, status=_quality_status(e, margin, lower_is_better=True))

    costs_a = np.array([run_a.examples[i].cost.total_cost for i in idx_a], dtype=float)
    costs_b = np.array([run_b.examples[i].cost.total_cost for i in idx_b], dtype=float)
    cost_delta = _build_relative_delta("total cost", costs_a, costs_b)

    latencies_a = np.array([run_a.examples[i].latency_s for i in idx_a], dtype=float)
    latencies_b = np.array([run_b.examples[i].latency_s for i in idx_b], dtype=float)
    latency_delta = _build_relative_delta("avg latency", latencies_a, latencies_b)

    quality_verdict, cost_verdict, recommendation = _make_verdicts(
        run_b.tag, scorer_deltas, cost_delta, margin=margin, error_delta=error_delta, n=n
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
        margin=margin,
        decision_level=decision_level,
        error_delta=error_delta,
        errors_a=int(errs_a.sum()),
        errors_b=int(errs_b.sum()),
        n_attempted=len(attempted),
        model_a=run_a.model,
        model_b=run_b.model,
        warnings=warnings,
    )


def _compatibility_warnings(run_a: RunResult, run_b: RunResult) -> tuple[str, ...]:
    out: list[str] = []
    ha, hb = run_a.dataset_sha256, run_b.dataset_sha256
    if ha and hb and ha != hb:
        out.append(
            "the two runs used different dataset files; only examples with matching ids "
            "are compared, and a matching id may not be the same example"
        )
    ids_a = {e.id for e in run_a.examples}
    ids_b = {e.id for e in run_b.examples}
    only_a, only_b = len(ids_a - ids_b), len(ids_b - ids_a)
    if only_a or only_b:
        out.append(
            f"{only_a} example(s) only in {run_a.tag!r} and {only_b} only in {run_b.tag!r} "
            "were left out"
        )
    if run_a.params != run_b.params:
        out.append(
            f"the request settings differ ({run_a.params or 'defaults'} vs "
            f"{run_b.params or 'defaults'}): they are part of what is being compared"
        )
    if run_a.model != run_b.model:
        out.append(
            f"the models differ ({run_a.model} vs {run_b.model}): this compares the model "
            "change and any prompt change together"
        )
    return tuple(out)


def _count_errors(run: RunResult) -> int:
    return sum(1 for e in run.examples if e.error is not None)


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
    if report.model_a or report.model_b:
        lines.append(f"Models: {report.tag_a}={report.model_a}, {report.tag_b}={report.model_b}")
    for w in report.warnings:
        lines.append(f"Warning: {w}")
    if report.error_delta is not None:
        lines.append(
            f"Errors: {report.errors_a} in {report.tag_a}, {report.errors_b} in {report.tag_b} "
            f"(of {report.n_attempted} attempted by both)"
        )
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

    if report.error_delta is not None:
        e = report.error_delta
        lines.append(
            f"{'error rate':<26}"
            f"{e.value_a:<14.0%}"
            f"{e.value_b:<16.0%}"
            f"{_signed(e.delta):<14}"
            f"{f'[{_signed(e.ci_low)}, {_signed(e.ci_high)}]':<22}"
            f"{_format_p(e.p_value):<10}"
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
    lines.append("")
    lines.append(_method_note(report))

    return "\n".join(lines)


def _method_note(report: ComparisonReport) -> str:
    k = len(report.scorer_deltas) + (1 if report.error_delta is not None else 0)
    adjusted = (
        f", adjusted for {k} quality metrics ({report.decision_level:.1%} CIs)" if k > 1 else ""
    )
    return (
        f"Method: paired bootstrap; quality holds when no metric can have dropped by more "
        f"than {report.margin:.2f}{adjusted}."
    )


def _signed(x: float) -> str:
    """Format a number with explicit sign — README's killer report style."""
    return f"{x:+.3f}" if abs(x) < 1 else f"{x:+.2f}"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def render_html(report: ComparisonReport) -> str:
    """Render the ComparisonReport as a single self-contained HTML document.

    No external CSS/JS — inline styles only. Safe to email, attach to a CI
    artifact, or open straight from the filesystem. Same data as `render_text`,
    just with semantic markup and color-coded verdict badges.
    """
    rec_class = _recommendation_class(report.recommendation)
    errors_meta = (
        f" &middot; errors: {report.errors_a} in {_h(report.tag_a)}, "
        f"{report.errors_b} in {_h(report.tag_b)} (of {report.n_attempted})"
        if report.error_delta is not None
        else ""
    )
    models_meta = (
        f" &middot; models: {_h(report.tag_a)}={_h(report.model_a)}, "
        f"{_h(report.tag_b)}={_h(report.model_b)}"
        if report.model_a or report.model_b
        else ""
    )
    warnings_html = "".join(f'<p class="warn">Warning: {_h(w)}</p>' for w in report.warnings)
    error_row = (
        _html_scorer_row(report.error_delta, lower_is_better=True)
        if report.error_delta is not None
        else ""
    )

    scorer_rows = "\n".join(_html_scorer_row(d) for d in report.scorer_deltas)
    cost_row = _html_relative_row(
        "total cost",
        report.cost_delta,
        value_fmt=lambda v: f"${v * report.n_paired:.4f}",
    )
    latency_row = _html_relative_row(
        "avg latency",
        report.latency_delta,
        value_fmt=lambda v: f"{v:.2f}s",
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>prompteval — {_h(report.tag_a)} vs {_h(report.tag_b)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 920px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    color: #1a1a1a;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem 0; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e5e5; }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  td.pos {{ color: #0a7d2a; }}
  td.neg {{ color: #c8202b; }}
  td.sig {{ font-weight: 600; }}
  .verdicts {{
    background: #f8f8f8; padding: 1rem 1.25rem;
    border-radius: 6px; margin-top: 1.5rem;
  }}
  .verdicts dt {{ font-weight: 600; color: #666; font-size: 0.85rem; margin-top: 0.5rem; }}
  .verdicts dd {{ margin: 0.15rem 0 0 0; }}
  .rec {{ display: inline-block; padding: 0.4rem 0.9rem; border-radius: 4px;
          font-weight: 600; margin-top: 1rem; }}
  .rec.ship {{ background: #d8f0de; color: #0a5e1f; }}
  .rec.dont-ship {{ background: #fad9dc; color: #8a131c; }}
  .rec.inconclusive {{ background: #fff3cd; color: #6e5005; }}
  .warn {{ background: #fff3cd; color: #6e5005; padding: 0.5rem 0.75rem; border-radius: 4px; }}
  footer {{ color: #999; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>{_h(report.tag_a)} vs {_h(report.tag_b)}</h1>
<div class="meta">Paired examples: {report.n_paired}{errors_meta}{models_meta}</div>
{warnings_html}

<table>
<thead>
<tr>
  <th>Metric</th>
  <th>{_h(report.tag_a)}</th>
  <th>{_h(report.tag_b)}</th>
  <th>Δ</th>
  <th>95% CI</th>
  <th>p-value</th>
</tr>
</thead>
<tbody>
{scorer_rows}
{error_row}
{cost_row}
{latency_row}
</tbody>
</table>

<div class="verdicts">
<dl>
  <dt>Quality verdict</dt><dd>{_h(report.quality_verdict)}</dd>
  <dt>Cost verdict</dt><dd>{_h(report.cost_verdict)}</dd>
</dl>
<div class="rec {rec_class}">Recommendation: {_h(report.recommendation)}</div>
</div>

<footer>Generated by prompteval &mdash; {_h(_method_note(report))}</footer>
</body>
</html>
"""


def _recommendation_class(recommendation: str) -> str:
    """Pick the CSS class for the recommendation badge based on its leading verb."""
    rec_lower = recommendation.lower()
    if rec_lower.startswith("ship "):
        return "ship"
    if rec_lower.startswith("don't ship") or "more expensive" in rec_lower:
        return "dont-ship"
    return "inconclusive"


def _html_scorer_row(d: MetricDelta, lower_is_better: bool = False) -> str:
    sig_class = " sig" if d.significant else ""
    delta_class = _delta_class(-d.delta if lower_is_better else d.delta, d.significant)
    return (
        "<tr>"
        f"<td>{_h(d.name)}</td>"
        f"<td class='num'>{d.value_a:.2f}</td>"
        f"<td class='num'>{d.value_b:.2f}</td>"
        f"<td class='num {delta_class}{sig_class}'>{_signed(d.delta)}</td>"
        f"<td class='num'>[{_signed(d.ci_low)}, {_signed(d.ci_high)}]</td>"
        f"<td class='num{sig_class}'>{_format_p(d.p_value)}</td>"
        "</tr>"
    )


def _html_relative_row(
    label: str,
    d: MetricDelta,
    value_fmt: Callable[[float], str],
) -> str:
    """Render a row for cost/latency — where negative `delta_pct` is good (cheaper/faster)."""
    sig_class = " sig" if d.significant else ""
    # For cost/latency, negative delta = improvement → green.
    sign_for_color = -d.delta_pct if d.delta_pct is not None else -d.delta
    delta_class = _delta_class(sign_for_color, d.significant)
    delta_str = f"{d.delta_pct:+.0f}%" if d.delta_pct is not None else _signed(d.delta)
    ci_str = f"[{d.ci_low:+.0f}%, {d.ci_high:+.0f}%]"
    return (
        "<tr>"
        f"<td>{_h(label)}</td>"
        f"<td class='num'>{value_fmt(d.value_a)}</td>"
        f"<td class='num'>{value_fmt(d.value_b)}</td>"
        f"<td class='num {delta_class}{sig_class}'>{delta_str}</td>"
        f"<td class='num'>{ci_str}</td>"
        f"<td class='num{sig_class}'>{_format_p(d.p_value)}</td>"
        "</tr>"
    )


def _delta_class(signed_delta: float, significant: bool) -> str:
    """Green if signed_delta is positive (caller flips sign for cost/latency); else red."""
    if not significant:
        return ""
    return "pos" if signed_delta > 0 else "neg"


def _h(s: str) -> str:
    """Tiny HTML-escape — avoids dragging in `html` module for four entities."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
