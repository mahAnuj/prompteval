# Changelog

## Unreleased

- `prompteval dataset import`: build dataset.jsonl from a CSV or JSONL export of
  real inputs, with column mapping, de-duplication, reproducible sampling and
  e-mail/phone masking, all local. Rows are marked `"source": "real"`, and
  `compare` warns when examples are synthetic (#21).
- `compare --pairwise` / `sweep --pairwise`: a judge model compares the runs'
  answers directly, no scorers needed. Each example is judged in both orders;
  disagreement counts as a tie. Win/tie/loss feeds the usual statistics and
  verdict; judge model, decisiveness and cost are reported (#19).
- An eval may now have no scorers. With no quality measure, `compare` reports
  quality as not evaluated instead of "holds".

## 0.2.2 — 2026-11-12

- Competitor scan re-checked against current docs (2026-11-12); README
  positioning updated: Braintrust now shows cost deltas, so prompteval leads on
  whether differences are real (paired CIs, verdict), not on showing them (#16).
- `prompteval demo`: a full comparison report in a second, with no API key or
  network, from two bundled, clearly labelled illustrative runs (#11).
- `--unpriced` on `run` and `sweep` runs a model with no known price: tokens
  are recorded, cost is reported as not evaluated (token deltas instead), a
  cost gate fails closed, and a sweep never calls an unpriced model cheapest (#17).
- Docs pin the Action to a release tag (`@v0.2.1`). The `@v1` they showed
  was never created: GitHub refuses the workflow token permission to create
  a tag that contains workflow files, so a moving major tag needs a manual
  push. The workflow that tried to move it is removed.

## 0.2.1 — 2026-11-12

- `.env` and `evals/.env` are now read at startup, as `prompteval init` always
  said to do; nothing loaded them before, so a key placed there gave an
  authentication error. A variable set in the shell or CI still wins.
- README: whose API keys are used, and which commands call models.

## 0.2.0 — 2026-11-12

### Verdicts you can trust (breaking changes in behaviour)

- **"Quality holds" now means it.** A scorer holds only when its worst
  plausible drop is within a margin (0.05 by default, `--margin`). A certain
  drop is a regression; anything else is **inconclusive**, with an estimate of
  how many examples would settle it. Before, "no significant drop" counted as
  "holds", so a 20-example eval said *ship* even when the CI allowed a
  45-point drop (#1).
- **`--fail-on` gates on the 95% CI bound**: a breach that is plausible fails.
  Thin or noisy data no longer passes by default. `--gate-mode significant`
  keeps the 0.1 behaviour (#2).
- **Errors count.** A prompt that fails on more examples is worse, not cheaper:
  the error rate is a quality metric with its own row (#3).
- **More scorers no longer mean more false alarms**: decisions are corrected
  for the number of scorers, and for small samples (#4).
- The CI and the p-value now come from the same paired bootstrap, so the table
  and the verdict cannot disagree.

### Fixes

- An unknown model fails before the first paid request (#5).
- Latency measures the model call only, not scoring or judge time (#6).
- `compare` refuses runs from different evals and warns when datasets, models,
  settings, repeats or example ids differ (#7).

### New

- **Price catalog of ~3,900 models** from LiteLLM's price list (MIT) and
  OpenRouter, matched to the provider you call. `prompteval models update`
  refreshes it without upgrading the package, and runs remind you when prices
  are over 30 days old; `models search`, `models price --base-url` (#8).
- **Any OpenAI-compatible provider** (Groq, Together, OpenRouter, Fireworks,
  Ollama, vLLM): `--base-url`, `--api-key-env`, `--price`, `prices.yaml` (#8).
- **`prompteval sweep`**: which model is cheapest for this prompt without
  losing quality, in one table; re-runs reuse saved results (#9).
- **Prompt templates** with `{variables}`, a `user_template`, and request
  settings (`params`, `--param temperature=0`) (#10).
- **GitHub Action** (`uses: mahAnuj/prompteval@v0.2.0`) posting one PR comment,
  updated on every push (#12); `compare --markdown`.
- **Parallel runs** (`--concurrency`, default 4) with more retries (#13).
- **Noise**: `--repeats`, a temperature-0 judge, `prompteval noise` (#14).
- **Judge cost** reported on its own line (#15).
- `docs/writing-scorers.md` (#16).
