# GitHub Actions example workflow

Drop [`prompteval.yml`](./prompteval.yml) into your repo at
`.github/workflows/prompteval.yml` to run prompteval on every PR that touches
`evals/`.

## What you get

- **Paired delta report** on every PR — same shape as `prompteval compare` locally.
- **HTML artifact** uploaded to the PR's Checks tab — shareable without re-running.
- **CI gate** — fails the build if cost regresses more than 10% or any scorer
  drops more than 5 percentage points. Only *statistically significant* changes
  count (p < 0.05), so a noisy 12% wobble on a 5-example eval won't trip it.

## Required secrets

| Name | Purpose |
|---|---|
| `OPENAI_API_KEY` | Used by `prompteval run` to call the OpenAI API. |

Add this via *Settings → Secrets and variables → Actions → New repository secret*.

## Tuning the gate

The `--fail-on cost+10%,quality-5%` line is the policy knob:

| Spec | Meaning |
|---|---|
| `cost+10%` | Fail if cost increased by more than 10% (significant). |
| `quality-5%` | Fail if any scorer's mean dropped by more than 0.05 absolute (significant). |
| `cost+10%,quality-5%` | Either triggers a fail. |

Real production prompts should probably set tighter bounds (e.g. `quality-2%`).
For experimental prompts, you can loosen them or omit `--fail-on` entirely and
just look at the HTML report on each PR.

## Cost note

The workflow runs your eval twice per PR (baseline + candidate). With a
20-example dataset at `gpt-4o-mini` rates (~$0.0002/call), that's roughly
$0.008 per PR — basically free. With larger datasets or pricier models, factor
that into your CI budget.
