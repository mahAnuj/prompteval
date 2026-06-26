# Week 0 — competitor scan

> _Dated 2026-06-26. Scan-by-docs, not hands-on use. Anything marked **strong claim** I want to verify hands-on before quoting in marketing._

## The query I'm testing for

> _"Did this prompt change save money without losing quality?"_

Specifically: can a user load two prompt versions, run the same eval set on both, and get a single report that shows **paired** quality-delta and cost-delta with significance? Not "track cost as a metric I graph separately" — but cost as a **comparison axis** alongside quality.

## TL;DR

**No incumbent owns this.** Every tool tracks cost; none treat it as a first-class comparison axis paired with quality. The closest is **promptfoo** (cost as a weighted assertion), and even there it's pass/fail on a threshold, not a paired-delta report. **The gap is real and the wedge holds.**

## Findings per tool

### Braintrust (commercial leader, $30M+ raised)

- **Has**: cost tracking per trace, custom model-cost config, SQL-query for span-level costs, cost alerts at the org level.
- **Doesn't have** (per public docs): cost as a comparison axis in the experiment-diff view. Experiments compare "which test cases improved or regressed" on quality only. Cost lives in a parallel monitoring surface.
- **Honest caveat**: their `llms.txt` digest is incomplete; the in-app experiment UI may show cost alongside quality even if the docs don't say so. Worth a free-tier signup test if I push the marketing hard.
- **Verdict**: ✅ wedge survives

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

- **Has**: cost tracking for LLM applications (separate docs page exists).
- **Doesn't have** (per visible docs — the cost-tracking page 403'd, so this is inferred): cost as a comparison axis in eval views. Phoenix's positioning is observability-first.
- **Honest caveat**: I couldn't access the dedicated cost-tracking page; should re-check if blocking on Phoenix specifically.
- **Verdict**: ✅ wedge probably survives — re-verify if it matters

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

- Sign up for **Braintrust free tier** and verify the experiment-comparison view doesn't actually show cost alongside quality. If it does, sharpen the differentiator language further (likely toward "OSS + cost-first" vs Braintrust's "commercial + quality-first").
- Re-fetch **Phoenix cost-tracking docs** when not 403'd.

## Locked tagline (for README)

> _Did your prompt change save money without losing quality? Paired cost-and-quality delta reports with significance — the question existing LLM eval tools answer only awkwardly._

## Source URLs probed

- https://www.braintrust.dev/docs/guides/evals
- https://www.braintrust.dev/docs/guides/experiments
- https://www.braintrust.dev/docs/llms.txt
- https://langfuse.com/docs/scores/overview
- https://langfuse.com/docs/datasets/experiments (404 — likely moved)
- https://www.promptfoo.dev/docs/configuration/expected-outputs/
- https://www.promptfoo.dev/docs/usage/web-ui/
- https://arize.com/docs/phoenix/evaluation/llm-evals
- https://arize.com/docs/phoenix/learn/agents/cost-tracking-for-llm-applications (403)
- https://inspect.aisi.org.uk/scorers.html
- https://inspect.aisi.org.uk/eval-logs.html
