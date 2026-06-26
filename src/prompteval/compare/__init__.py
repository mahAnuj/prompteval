"""prompteval.compare — paired delta + significance reports for two RunResults.

Public surface:
- `compute_comparison(run_a, run_b)` — pure function: returns a ComparisonReport
- `render_text(report)` — format the report as the README's killer output
- `ComparisonReport`, `MetricDelta` — data shapes

Week 6 will add an HTML render alongside `render_text`.
"""

from prompteval.compare.core import (
    ComparisonReport,
    MetricDelta,
    compute_comparison,
    render_text,
)

__all__ = [
    "ComparisonReport",
    "MetricDelta",
    "compute_comparison",
    "render_text",
]
