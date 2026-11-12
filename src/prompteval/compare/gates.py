"""CI gate evaluation — parse `--fail-on` specs, evaluate against a report.

A gate spec is a comma-separated list of clauses. Two clause types are supported
in v1:

- `cost+X%`   — fail if cost increased by more than X% (relative to baseline,
                using `cost_delta.delta_pct`).
- `quality-Y%` — fail if any scorer's mean dropped by more than Y/100 absolute.
                (Scorers are in [0, 1], so "5%" maps to a 0.05 absolute drop —
                what most users actually mean when they say "5% quality loss".)

Two modes:

- **bound** (default): gate on the worst plausible case, the 95% CI. `cost+X%`
  fails when the cost change could be above +X%; `quality-Y%` fails when any
  scorer could have dropped by more than Y points, or the error rate could
  have risen by more than Y points. A thin or noisy dataset fails until it has
  enough examples to rule the breach out (#2): a safety gate must not get
  weaker as the evidence gets thinner.
- **significant** (the v0.1 behaviour): only statistically significant
  breaches of the measured change count.

Whitespace is tolerated. Case-insensitive. Examples:

    cost+10%
    quality-5%
    cost+10%, quality-5%
    COST+10%,QUALITY-2%
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prompteval.compare.core import ComparisonReport

# A single clause: "cost+10%" → ("cost", 10.0), "quality-5%" → ("quality", -5.0)
_CLAUSE_RE = re.compile(r"^\s*(cost|quality)\s*([+-])\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GateClause:
    """One parsed `--fail-on` clause."""

    metric: str  # "cost" or "quality"
    threshold_pct: float  # signed: cost+10% → +10.0; quality-5% → -5.0


@dataclass(frozen=True)
class GateBreach:
    """One failure surfaced during gate evaluation."""

    clause: GateClause
    actual_pct: float  # the observed delta (cost %, or scorer absolute * 100)
    detail: str  # human-readable message ready for CLI output


class GateSpecError(ValueError):
    """Raised when a `--fail-on` spec can't be parsed."""


def parse_gate_spec(spec: str) -> list[GateClause]:
    """Parse `cost+10%,quality-5%` into [GateClause(...), GateClause(...)].

    Raises GateSpecError on malformed input — the message names the offending
    clause so users can fix it directly.
    """
    clauses: list[GateClause] = []
    for raw in spec.split(","):
        if not raw.strip():
            continue
        match = _CLAUSE_RE.match(raw)
        if not match:
            raise GateSpecError(
                f"Invalid --fail-on clause: {raw.strip()!r}. "
                "Expected forms: 'cost+X%' or 'quality-Y%' (e.g. 'cost+10%,quality-5%')."
            )
        metric, sign, magnitude = match.groups()
        signed = float(magnitude) * (1 if sign == "+" else -1)
        clauses.append(GateClause(metric=metric.lower(), threshold_pct=signed))
    if not clauses:
        raise GateSpecError(f"Empty --fail-on spec: {spec!r}.")
    return clauses


GATE_MODES = ("bound", "significant")


def evaluate_gates(
    report: ComparisonReport, clauses: list[GateClause], mode: str = "bound"
) -> list[GateBreach]:
    """Return the list of breached clauses. Empty list = gate passed.

    See the module docstring for the two modes.
    """
    if mode not in GATE_MODES:
        raise GateSpecError(f"Unknown gate mode {mode!r}; expected one of {GATE_MODES}.")
    breaches: list[GateBreach] = []
    for clause in clauses:
        if clause.metric == "cost":
            breach = (
                _bound_cost_clause(clause, report)
                if mode == "bound"
                else _evaluate_cost_clause(clause, report)
            )
        elif clause.metric == "quality":
            breach = (
                _bound_quality_clause(clause, report)
                if mode == "bound"
                else _evaluate_quality_clause(clause, report)
            )
        else:  # pragma: no cover — parse_gate_spec guarantees this
            continue
        if breach is not None:
            breaches.append(breach)
    return breaches


_NARROW = "add examples to narrow the range"


def _bound_cost_clause(clause: GateClause, report: ComparisonReport) -> GateBreach | None:
    """cost+X% fails when the CI's upper bound is above +X%; cost-X% (a required
    reduction) fails when the reduction isn't certain to reach X%."""
    cost = report.cost_delta
    if cost.delta_pct is None:
        return None
    if cost.ci_high <= clause.threshold_pct:
        return None
    measured = (
        f"measured {cost.delta_pct:+.1f}%, 95% CI [{cost.ci_low:+.1f}%, {cost.ci_high:+.1f}%]"
    )
    if clause.threshold_pct >= 0 and cost.delta_pct > clause.threshold_pct:
        detail = (
            f"cost regressed {cost.delta_pct:+.1f}% (95% CI [{cost.ci_low:+.1f}%, "
            f"{cost.ci_high:+.1f}%]); gate was cost+{clause.threshold_pct:g}%"
        )
    elif clause.threshold_pct >= 0:
        detail = (
            f"cost could be up {cost.ci_high:+.1f}% ({measured}); gate was "
            f"cost+{clause.threshold_pct:g}%"
        )
    else:
        detail = (
            f"cost reduction of {abs(clause.threshold_pct):g}% is not certain ({measured}); "
            f"gate was cost{clause.threshold_pct:g}%"
        )
    if cost.delta_pct <= clause.threshold_pct:
        detail += f" — the measured change is within the gate; {_NARROW}"
    return GateBreach(clause=clause, actual_pct=cost.ci_high, detail=detail)


def _bound_quality_clause(clause: GateClause, report: ComparisonReport) -> GateBreach | None:
    """quality-Y% fails when any scorer's CI lower bound is below -Y/100, or the
    error rate's CI upper bound is above +Y/100."""
    if clause.threshold_pct >= 0:
        return None
    limit = abs(clause.threshold_pct) / 100.0
    # (worst plausible drop, label, measured change, ci)
    worst: list[tuple[float, str, float, tuple[float, float]]] = [
        (-d.ci_low, f"scorer {d.name!r}", d.delta, (d.ci_low, d.ci_high))
        for d in report.scorer_deltas
    ]
    if report.error_delta is not None:
        e = report.error_delta
        worst.append((e.ci_high, "error rate", -e.delta, (e.ci_low, e.ci_high)))
    breached = [w for w in worst if w[0] > limit]
    if not breached:
        return None
    drop, label, measured, (lo, hi) = max(breached, key=lambda w: w[0])
    detail = (
        f"{label} could be {drop:.3f} worse (measured {measured:+.3f}, 95% CI "
        f"[{lo:+.3f}, {hi:+.3f}]); gate was quality-{abs(clause.threshold_pct):g}%"
    )
    if -measured <= limit:
        detail += f" — the measured change is within the gate; {_NARROW}"
    return GateBreach(clause=clause, actual_pct=-drop * 100, detail=detail)


def _evaluate_cost_clause(clause: GateClause, report: ComparisonReport) -> GateBreach | None:
    cost = report.cost_delta
    if cost.delta_pct is None or not cost.significant:
        return None
    # cost+10% means "fail if cost increased by more than 10%".
    if clause.threshold_pct > 0 and cost.delta_pct > clause.threshold_pct:
        return GateBreach(
            clause=clause,
            actual_pct=cost.delta_pct,
            detail=(
                f"cost regressed {cost.delta_pct:+.1f}% (significant, p={cost.p_value:.3g}); "
                f"gate was cost+{clause.threshold_pct:.0f}%"
            ),
        )
    # cost-X% means "fail if cost did NOT improve by at least X%" — rare but legal.
    if clause.threshold_pct < 0 and cost.delta_pct > clause.threshold_pct:
        return GateBreach(
            clause=clause,
            actual_pct=cost.delta_pct,
            detail=(
                f"cost change {cost.delta_pct:+.1f}% did not meet required reduction "
                f"of {abs(clause.threshold_pct):.0f}%"
            ),
        )
    return None


def _evaluate_quality_clause(clause: GateClause, report: ComparisonReport) -> GateBreach | None:
    """Quality regression = any scorer mean dropped more than `|threshold|/100` absolute.

    Scorers live in [0, 1] so we interpret "quality-5%" as "5 percentage points"
    of absolute drop — i.e. mean fell by 0.05 or more. Only significant drops count.
    """
    if clause.threshold_pct >= 0:
        # quality+X% would mean "require improvement of at least X%" — not in v1's scope.
        return None
    threshold_abs = abs(clause.threshold_pct) / 100.0
    regressions = [
        d
        for d in report.scorer_deltas
        if d.significant and d.delta < 0 and abs(d.delta) > threshold_abs
    ]
    if not regressions:
        return None
    worst = min(regressions, key=lambda d: d.delta)
    return GateBreach(
        clause=clause,
        actual_pct=worst.delta * 100,
        detail=(
            f"scorer {worst.name!r} dropped {worst.delta:+.3f} "
            f"(significant, p={worst.p_value:.3g}); "
            f"gate was quality-{abs(clause.threshold_pct):.0f}%"
        ),
    )
