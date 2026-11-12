# GitHub Actions example workflow

Drop [`prompteval.yml`](./prompteval.yml) into your repo at
`.github/workflows/prompteval.yml` to run prompteval on every PR that touches
`evals/`.

## What you get

- **Paired delta report** on every PR — same shape as `prompteval compare` locally.
- **HTML artifact** uploaded to the PR's Checks tab — shareable without re-running.
- **CI gate** — fails the build if cost *could* have risen more than 10%, or
  any scorer *could* have dropped more than 5 percentage points, judged on the
  95% confidence interval. A 5-example eval with a noisy 12% wobble fails until
  more examples rule the rise out: a gate should get stricter, not looser, when
  the evidence is thin. `--gate-mode significant` counts only statistically
  significant breaches instead (the v0.1 behaviour).

## Required secrets

| Name | Purpose |
|---|---|
| `OPENAI_API_KEY` | Used by `prompteval run` to call the OpenAI API. |

Add this via *Settings → Secrets and variables → Actions → New repository secret*.

## Tuning the gate

The `--fail-on cost+10%,quality-5%` line is the policy knob:

| Spec | Meaning |
|---|---|
| `cost+10%` | Fail if the cost change's 95% CI reaches above +10%. |
| `quality-5%` | Fail if any scorer's 95% CI reaches below -0.05, or the error rate's reaches above +0.05. |
| `cost+10%,quality-5%` | Either triggers a fail. |

Real production prompts should probably set tighter bounds (e.g. `quality-2%`).
For experimental prompts, you can loosen them or omit `--fail-on` entirely and
just look at the HTML report on each PR.

## Cost note

The workflow runs your eval twice per PR (baseline + candidate). With a
20-example dataset at `gpt-4o-mini` rates (~$0.0002/call), that's roughly
$0.008 per PR — basically free. With larger datasets or pricier models, factor
that into your CI budget.
