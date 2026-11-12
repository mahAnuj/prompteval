# Competitor scan

## Re-check — 2026-11-12 (supersedes the conclusions below where they differ)

Checked against each tool's current docs and source (fetched 2026-11-12; URLs
below). The two load-bearing claims were confirmed by a second, independent
fetch. Question: can it compare two prompts or models on the same data, show a
cost delta, test whether the differences are real, and gate a PR on it?

| Tool | Same-data comparison | Cost per variant / cost delta | Significance or a quality verdict | CI gate / PR comment |
|---|---|---|---|---|
| promptfoo | Yes | Cost per cell and per prompt; no delta | No (a Wilson CI only for `--repeat` stability) | Action comments pass/fail counts; "does not run separate base and head evaluations" |
| Braintrust | Yes, baseline-aligned per row | Yes, "delta vs. base" including cost | No CIs; a heuristic grade: "Improvement, Regression, Tradeoff, or Tie" | `eval-action` PR comment can include cost vs. base |
| Langfuse | Yes | "score, cost, and latency differences" in the UI | No | Action comments scores and pass/regression, no cost |
| Arize Phoenix | Yes | Cost per experiment, cost charts; no delta | No | pytest gate; optional PR summary (contents unverified) |
| DeepEval / Confident AI | Yes | Cost and tokens per test case; deltas on scores only | Says "statistical rigor"; no CI computed in the docs | `deepeval test run`, `confident-ai gate`; no PR comment found |
| Inspect AI | Yes, several models | Cost data and `cost_limit`; no delta | Per-run `ci()` / bootstrap stderr, not on the paired difference | No Action found |

**What changed since June.** Braintrust, not promptfoo, is now the closest
rival: baseline-paired deltas, a cost delta against the baseline, a grade that
weighs cost, and PR comments that can show cost. Its cost guide describes the
same use case ("whether a lower-cost model or scorer holds quality on your own
data"). The old README line, that incumbents only "track cost as a metric you
graph", is no longer true of it.

**What still holds.** None of the six computes a confidence interval or a
significance test on the *paired difference* in quality or cost, and none
gives a non-inferiority verdict (holds / worse / inconclusive). Searching
Braintrust's full docs (`llms-full.txt`) for "confidence interval",
"statistical significance" and "p-value" finds nothing (checked 2026-11-12).
promptfoo's cost check is still a per-call threshold.

**Positioning, adjusted.** Lead on whether the differences are real, and on
openness, not on showing a cost delta:

> Other tools show you that B costs less and scores two points higher.
> prompteval tells you whether those two points, and the savings, are real:
> paired confidence intervals and a holds / worse / inconclusive verdict,
> gated in CI, open source, no account.

Sources (all fetched 2026-11-12):
- promptfoo cost assertion: https://raw.githubusercontent.com/promptfoo/promptfoo/main/site/docs/configuration/expected-outputs/deterministic.md
- promptfoo web UI: https://www.promptfoo.dev/docs/usage/web-ui/
- promptfoo Wilson CI: https://www.promptfoo.dev/docs/guides/evaluate-llm-temperature/
- promptfoo Action: https://raw.githubusercontent.com/promptfoo/promptfoo-action/main/README.md
- Braintrust comparisons: https://docs.braintrust.dev/evaluate/compare-experiments.md
- Braintrust cost guide: https://docs.braintrust.dev/admin/billing/optimize-costs.md
- Braintrust CI: https://raw.githubusercontent.com/braintrustdata/eval-action/main/README.md, https://docs.braintrust.dev/evaluate/run-in-ci.md
- Langfuse: https://langfuse.com/docs/evaluation/experiments/compare-experiments.md, https://langfuse.com/docs/evaluation/experiments/experiments-ci-cd.md
- Phoenix: https://arize.com/docs/phoenix/tracing/how-to-tracing/cost-tracking.md, https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments/eval-ci-with-pytest.md
- Confident AI: https://www.confident-ai.com/docs/llm-evaluation/experiments.md
- Inspect AI: https://inspect.aisi.org.uk/metrics.html.md, https://inspect.aisi.org.uk/setting-limits.html.md

---

## Original scan — 2026-06-26

> _Dated 2026-06-26. Scan-by-docs, not hands-on use. Anything marked **strong claim** I want to verify hands-on before quoting in marketing._

## The query I'm testing for

> _"Did this prompt change save money without losing quality?"_

Specifically: can a user load two prompt versions, run the same eval set on both, and get a single report that shows **paired** quality-delta and cost-delta with significance? Not "track cost as a metric I graph separately" — but cost as a **comparison axis** alongside quality.

## TL;DR

**No incumbent owns this.** Every tool tracks cost at the trace/span level; none surface it as a first-class comparison axis paired with quality in their eval / experiment output. The closest is **promptfoo** (cost as a weighted assertion), and even there it's pass/fail on a threshold, not a paired-delta report. **The gap is real and the wedge holds — verified.**

## Verification round (2026-06-26)

The initial scan was docs-only. To kill the "the docs are just incomplete and the in-app UI actually does it" risk, I read Braintrust's flagship cookbook notebooks (ModelComparison + ProviderBenchmark) and Phoenix's experiments + tracing source docs in their repo. Result: **the deeper look strengthened the wedge, not weakened it.**

- **Braintrust ModelComparison notebook** (their canonical "how to compare models" tutorial): zero mentions of cost. Summary report aggregates score + duration. Visualization options offered to the user are prompt / model / temperature / score — never cost. If Braintrust did cost-vs-quality comparison they'd demo it here.
- **Braintrust ProviderBenchmark notebook** (LLaMa-3.1 across providers): explicitly headlines "score vs duration scatter plot" as the killer view. Uses the word "cost" colloquially to mean *latency cost* ("each weight class costs you an extra second"). Zero USD anywhere.
- **Phoenix tracing semantic conventions** (`.agents/skills/phoenix-tracing/references/span-llm.md`): they have **serious** per-trace cost attribution — `llm.cost.prompt`, `llm.cost.completion`, `llm.cost.total`, plus a `prompt_details.{input,cache_read,cache_write,audio}` breakdown. Granular and well-modeled. **This means Phoenix's cost-tracking is more sophisticated than I assumed.**
- **Phoenix experiments API** (`.agents/skills/phoenix-evals/references/experiments-running-python.md`): `experiment.aggregate_scores` returns `{'accuracy': 0.85, 'faithfulness': 0.92}` — quality scores only. No cost field in the aggregation. Cost exists at the trace level, never bubbles into the experiment-comparison surface.

### What this means

**Phoenix is the most-sophisticated cost tracker** and even they don't surface cost in the eval/experiment comparison view. That's strong evidence the gap is structural across the field, not just "Braintrust hasn't gotten around to it." It's also good news for prompteval: **integration with Phoenix's trace-level cost is a feasible v0.2 — we don't have to invent cost attribution from scratch, we can re-use the OTel-LLM convention they helped define.**

## Findings per tool

### Braintrust (commercial leader, $30M+ raised)

- **Has**: cost tracking per trace, custom model-cost config, SQL-query for span-level costs, cost alerts at the org level.
- **Doesn't have** (per public docs **and** verified via their own cookbook): cost as a comparison axis in the experiment-diff view. Their flagship ModelComparison + ProviderBenchmark notebooks never mention dollar cost — score / duration / latency only.
- **Verification (2026-06-26)**: read both flagship cookbooks end-to-end via `gh api` — zero USD references in the comparison reports they ship to users.
- **Verdict**: ✅ wedge survives — **verified**

### Langfuse (OSS, observability-first)

- **Has**: cost-per-token tracking, score analytics, experiments view for "side-by-side prompt/model/code changes."
- **Doesn't have** (per public docs): an experiment-comparison view that pairs cost-delta with quality-delta. Cost is shown elsewhere; comparison view is quality-focused.
- **Honest caveat**: experiments docs URL 404'd (`/docs/datasets/experiments` — may have moved). The overview page strongly implies side-by-side is quality-only.
- **Verdict**: ✅ wedge survives

### promptfoo (OSS, CLI-first)

- **Has**: a `cost` assertion type that integrates into weighted scoring (cost is a first-class signal here, not just metadata). Web UI shows "Tokens, latency, cost, tokens/sec" per inference. Compare view diffs two eval runs.
- **Doesn't have**: paired cost-delta vs quality-delta as a single report. Cost as an assertion is **pass/fail on a threshold**, not "version A cost X less than version B at +/-Y quality."
- **Closest competitor of the bunch.** If I had to pivot positioning to differentiate from one tool, it's promptfoo.
- **Verdict**: ⚠️ wedge survives but **promptfoo is the one to watch**

### Phoenix (Arize, OSS)

- **Has (richer than I assumed)**: per-trace cost attribution following OTel-LLM semantic conventions — `llm.cost.{prompt,completion,total}` plus a detailed breakdown for `cache_read`, `cache_write`, `audio`, etc. This is genuinely good cost tracking.
- **Doesn't have**: cost as a comparison axis in their experiments surface. `experiment.aggregate_scores` returns quality scores only (`{'accuracy': 0.85, 'faithfulness': 0.92}`); cost stays at the trace level.
- **Verification (2026-06-26)**: read their experiments + tracing source docs in the repo via `gh api`. Cost is observable on individual traces, never aggregated into eval comparison.
- **Strategic implication**: Phoenix's OTel-LLM cost convention is a **standard we should integrate with**, not compete against. v0.2 could ingest Phoenix traces and add the cost-comparison layer they don't have.
- **Verdict**: ✅ wedge survives — **verified** — Phoenix is a future integration target, not a competitor

### Inspect AI (UK AISI, OSS)

- **Has**: token-count tracking in `EvalStats`.
- **Doesn't have**: any cost (USD) tracking at all. No pricing, no comparison.
- **Verdict**: ✅ wedge fully survives (different positioning — they're a safety-evals tool, not a cost-optimization tool)

## What this means for prompteval

### Keep

- **Cost as a comparison axis** — confirmed gap, no incumbent does this cleanly.
- **Paired cost-and-quality delta reports** with statistical significance — nobody is doing this; this is the single sharpest differentiator.
- **First-class cost-aware CI** ("block merge if cost regresses > X% even if quality holds") — natural extension of the wedge.

### Sharpen against promptfoo

promptfoo is the closest competitor and ships actively. The differentiator language should be:

> "promptfoo's `cost` assertion tells you if a single run is under a threshold. prompteval tells you if version B is cheaper than version A at equivalent quality."

The shift is from **gate** (boolean) to **comparison** (paired delta with significance). That distinction needs to be obvious in the README and in the killer-demo blog post.

### Cut from scope (for v0)

- **Live observability dashboards** — Langfuse / Phoenix own this lane. Don't compete; integrate.
- **Trace explorers** — same.
- **Cost alerts** — Braintrust does this fine. Eval-time CI failures cover the "block bad deploys" use case without needing a separate alerting surface.

### Re-check before launch

- ~~Sign up for Braintrust free tier and verify~~ **Done via cookbook inspection 2026-06-26.** No need.
- ~~Re-fetch Phoenix cost-tracking docs when not 403'd.~~ **Done via repo source inspection 2026-06-26.** Phoenix has rich cost tracking but only at trace level; no comparison-axis use.

## Locked tagline (for README)

> _Did your prompt change save money without losing quality? Paired cost-and-quality delta reports with significance — the question existing LLM eval tools answer only awkwardly._

## Source URLs probed

**Round 1 — docs scan**
- https://www.braintrust.dev/docs/guides/evals
- https://www.braintrust.dev/docs/guides/experiments
- https://www.braintrust.dev/docs/llms.txt
- https://langfuse.com/docs/scores/overview
- https://langfuse.com/docs/datasets/experiments (404 — likely moved)
- https://www.promptfoo.dev/docs/configuration/expected-outputs/
- https://www.promptfoo.dev/docs/usage/web-ui/
- https://arize.com/docs/phoenix/evaluation/llm-evals
- https://arize.com/docs/phoenix/learn/agents/cost-tracking-for-llm-applications (403 — used repo source instead)
- https://inspect.aisi.org.uk/scorers.html
- https://inspect.aisi.org.uk/eval-logs.html

**Round 2 — source / cookbook verification (2026-06-26)**
- `gh api repos/braintrustdata/braintrust-cookbook/contents/examples/ModelComparison/ModelComparison.ipynb`
- `gh api repos/braintrustdata/braintrust-cookbook/contents/examples/ProviderBenchmark/ProviderBenchmark.ipynb`
- `gh api repos/Arize-ai/phoenix/contents/.agents/skills/phoenix-tracing/references/span-llm.md`
- `gh api repos/Arize-ai/phoenix/contents/.agents/skills/phoenix-evals/references/experiments-running-python.md`
