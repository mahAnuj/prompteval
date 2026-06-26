"""prompteval.compare — paired delta + significance reports for two RunResults.

Public surface:
- `compute_comparison(run_a, run_b)` — pure function: returns a ComparisonReport
- `render_text(report)` — format the report as the README's killer output
- `render_html(report)` — single-file self-contained HTML version of the same
- `parse_gate_spec`, `evaluate_gates`, `GateBreach`, `GateClause`, `GateSpecError`
  — power the `--fail-on` CI flag (also usable directly from Python)
- `ComparisonReport`, `MetricDelta` — data shapes
"""

from prompteval.compare.core import (
    ComparisonReport,
    MetricDelta,
    compute_comparison,
    render_html,
    render_text,
)
from prompteval.compare.gates import (
    GateBreach,
    GateClause,
    GateSpecError,
    evaluate_gates,
    parse_gate_spec,
)

__all__ = [
    "ComparisonReport",
    "GateBreach",
    "GateClause",
    "GateSpecError",
    "MetricDelta",
    "compute_comparison",
    "evaluate_gates",
    "parse_gate_spec",
    "render_html",
    "render_text",
]
