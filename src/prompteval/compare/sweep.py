"""Model sweep: the same prompt on several models, each compared to a baseline (#9).

Answers "which model is cheapest for my task without losing quality?" in one
table. Every model is compared to the baseline with `compute_comparison`, using
the same non-inferiority rule as `compare`, and the decisions are corrected for
the number of models compared (`family_size`) so a longer list of models does
not produce more false verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompteval.compare.core import DEFAULT_MARGIN, ComparisonReport, compute_comparison
from prompteval.eval.runner import RunResult


@dataclass(frozen=True)
class SweepRow:
    model: str
    run: RunResult
    report: ComparisonReport | None  # None for the baseline row
    holds_quality: bool  # every quality metric within the margin (always True for baseline)
    verdict: str


@dataclass(frozen=True)
class SweepReport:
    baseline: str
    margin: float
    rows: list[SweepRow]
    answer: str  # the one-line conclusion


def compute_sweep(
    runs: dict[str, RunResult], baseline: str, *, margin: float = DEFAULT_MARGIN
) -> SweepReport:
    """`runs` maps model name to its run, in display order; `baseline` is one of them."""
    if baseline not in runs:
        raise ValueError(f"Baseline model {baseline!r} has no run in the sweep.")
    if len(runs) < 2:
        raise ValueError("A sweep needs at least two models.")
    family = len(runs) - 1
    base_run = runs[baseline]
    rows: list[SweepRow] = []
    for model, run in runs.items():
        if model == baseline:
            rows.append(SweepRow(model, run, None, True, "baseline"))
            continue
        report = compute_comparison(base_run, run, margin=margin, family_size=family)
        statuses = [d.status for d in report.scorer_deltas]
        if report.error_delta is not None:
            statuses.append(report.error_delta.status)
        holds = all(s == "ok" for s in statuses)
        rows.append(SweepRow(model, run, report, holds, _verdict(report, holds, statuses)))
    return SweepReport(baseline, margin, rows, _answer(rows, baseline))


def _verdict(report: ComparisonReport, holds: bool, statuses: list[str]) -> str:
    cost = report.cost_delta
    if report.unpriced_tags:
        if "regressed" in statuses:
            return "worse quality"
        return "quality holds, unpriced" if holds else "inconclusive: add examples"
    cheaper = cost.significant and cost.delta_pct is not None and cost.delta_pct < 0
    dearer = cost.significant and cost.delta_pct is not None and cost.delta_pct > 0
    if "regressed" in statuses:
        return "worse quality"
    if not holds:
        return "inconclusive: add examples"
    if cheaper:
        return f"cheaper ({cost.delta_pct:+.0f}%), quality holds"
    if dearer:
        return f"dearer ({cost.delta_pct:+.0f}%), quality holds"
    return "quality holds, cost about the same"


def _answer(rows: list[SweepRow], baseline: str) -> str:
    candidates = [
        r
        for r in rows
        if r.report is not None
        and not r.report.unpriced_tags
        and r.holds_quality
        and r.report.cost_delta.significant
        and r.report.cost_delta.delta_pct is not None
        and r.report.cost_delta.delta_pct < 0
    ]
    if candidates:
        best = min(candidates, key=lambda r: r.run.total_cost)
        pct = best.report.cost_delta.delta_pct if best.report else 0.0
        return f"Cheapest model that holds quality: {best.model} ({pct:+.0f}% cost vs {baseline})."
    if any(r.verdict.startswith("inconclusive") for r in rows):
        return (
            "No cheaper model is proven to hold quality yet; some results are "
            "inconclusive. Add examples and run the sweep again (saved runs are reused)."
        )
    return f"No model is cheaper than {baseline} without losing quality on this dataset."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cells(report: SweepReport) -> tuple[list[str], list[list[str]]]:
    scorer_names = sorted({d.name for r in report.rows if r.report for d in r.report.scorer_deltas})
    header = [
        "model",
        *scorer_names,
        "errors",
        "cost/example",
        "Δ cost [95% CI]",
        "latency",
        "verdict",
    ]
    body: list[list[str]] = []
    for r in report.rows:
        n = len(r.run.examples)
        errors = sum(1 for e in r.run.examples if e.error is not None)
        means = r.run.scorer_means
        if r.report is None:
            scores = [f"{means.get(s, float('nan')):.2f}" for s in scorer_names]
            delta = "—"
        else:
            by = {d.name: d for d in r.report.scorer_deltas}
            scores = [
                f"{by[s].value_b:.2f} ({by[s].delta:+.2f})" if s in by else "n/a"
                for s in scorer_names
            ]
            c = r.report.cost_delta
            delta = (
                f"{c.delta_pct:+.0f}% [{c.ci_low:+.0f}%, {c.ci_high:+.0f}%]"
                if c.delta_pct is not None
                else "n/a"
            )
        ok = n - errors
        body.append(
            [
                r.model + (" (baseline)" if r.report is None else ""),
                *scores,
                f"{errors}/{n}",
                "unpriced"
                if not r.run.priced
                else f"${r.run.total_cost / ok:.6f}"
                if ok
                else "n/a",
                "unpriced" if (r.report and r.report.unpriced_tags) or not r.run.priced else delta,
                f"{r.run.avg_latency_s:.2f}s",
                r.verdict,
            ]
        )
    return header, body


def _note(report: SweepReport) -> str:
    k = len(report.rows) - 1
    return (
        f"Each model is compared to {report.baseline} on the same examples. Quality holds when "
        f"no metric can have dropped by more than {report.margin:.2f}; decisions are corrected "
        f"for {k} comparison{'s' if k != 1 else ''}."
    )


def render_sweep_text(report: SweepReport) -> str:
    header, body = _cells(report)
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(len(header))]

    def line(row: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))

    out = [line(header), "─" * (sum(widths) + 2 * (len(widths) - 1))]
    out += [line(r) for r in body]
    out += ["", report.answer, "", _note(report)]
    return "\n".join(out)


def render_sweep_markdown(report: SweepReport) -> str:
    header, body = _cells(report)
    out = [f"**{report.answer}**", "", "| " + " | ".join(header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    out += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in body]
    out += ["", f"_{_note(report)}_"]
    return "\n".join(out)


def render_sweep_html(report: SweepReport) -> str:
    from html import escape

    header, body = _cells(report)
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    rows = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in body)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>prompteval sweep — baseline {escape(report.baseline)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e5e5;
           font-variant-numeric: tabular-nums; }}
  th {{ background: #f8f8f8; }}
  .answer {{ font-size: 1.1rem; font-weight: 600; margin: 1rem 0; }}
  footer {{ color: #777; font-size: 0.85rem; margin-top: 1.5rem; }}
</style>
</head>
<body>
<h1>Model sweep</h1>
<p class="answer">{escape(report.answer)}</p>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
<footer>{escape(_note(report))}</footer>
</body>
</html>
"""
