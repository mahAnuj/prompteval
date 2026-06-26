"""CI gate evaluation — parse `--fail-on` specs, evaluate against a report.

A gate spec is a comma-separated list of clauses. Two clause types are supported
in v1:

- `cost+X%`   — fail if cost increased by more than X% (relative to baseline,
                using `cost_delta.delta_pct`).
- `quality-Y%` — fail if any scorer's mean dropped by more than Y/100 absolute.
                (Scorers are in [0, 1], so "5%" maps to a 0.05 absolute drop —
                what most users actually mean when they say "5% quality loss".)

Only *statistically significant* breaches count (p < 0.05). A 4% cost rise that
isn't significant won't trip `cost+3%` — that's the whole point of having
significance testing in the report.

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


def evaluate_gates(report: ComparisonReport, clauses: list[GateClause]) -> list[GateBreach]:
    """Return the list of breached clauses. Empty list = gate passed.

    Only counts breaches that are *statistically significant* — a noisy 12%
    cost wobble with p=0.3 doesn't trip `cost+10%`. That's the gate's whole
    job: keep CI green when the signal is real, red when it isn't.
    """
    breaches: list[GateBreach] = []
    for clause in clauses:
        if clause.metric == "cost":
            breach = _evaluate_cost_clause(clause, report)
        elif clause.metric == "quality":
            breach = _evaluate_quality_clause(clause, report)
        else:  # pragma: no cover — parse_gate_spec guarantees this
            continue
        if breach is not None:
            breaches.append(breach)
    return breaches


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
