# prompteval

**Paired cost-and-quality delta reports for LLM prompts — so you can ship the cheaper version without guessing.**

> _Did your prompt change save money without losing quality?_

prompteval is **pytest for LLM prompts**. You define a small golden test set and one or more scorers. You run it against your v1 prompt, then your v2 prompt, then ask it to compare. You get back a single report:

```
                          v1          v2          Δ        95% CI         p
─────────────────────────────────────────────────────────────────────────────
mentions_required_terms   0.84        0.79        −0.05    [−0.13, +0.03]  0.21
professional_tone         0.81        0.83        +0.02    [−0.05, +0.09]  0.58
─────────────────────────────────────────────────────────────────────────────
total cost                $0.158      $0.099      −37%     [−42%, −32%]   <0.001
avg latency               1.4s        1.1s        −21%     [−28%, −14%]   <0.001

Quality verdict:  holds — no metric can have dropped by more than 0.05
Cost verdict:     significant 37% reduction
Recommendation:   ship v2 — cost savings real, quality holds
```

Other tools can show you that version B costs less and scores two points higher. prompteval tells you **whether those two points, and the savings, are real**: paired confidence intervals on quality and cost, a holds / worse / inconclusive verdict, and a CI gate, open source with no account. See [docs/competitor-scan.md](docs/competitor-scan.md) for what each tool does, with sources (re-checked 2026-11-12).

[![PyPI](https://img.shields.io/pypi/v/prompteval-delta)](https://pypi.org/project/prompteval-delta/)
[![PyPI downloads](https://img.shields.io/pypi/dm/prompteval-delta)](https://pypistats.org/packages/prompteval-delta)
[![GitHub stars](https://img.shields.io/github/stars/mahAnuj/prompteval?style=flat)](https://github.com/mahAnuj/prompteval/stargazers)

> **Alpha — v0.3.0, on PyPI as [`prompteval-delta`](https://pypi.org/project/prompteval-delta/).**
> **Do not `pip install prompteval`** — that name on PyPI belongs to an
> unrelated project. The import and the command are still `prompteval`. The
> public API may still change before 1.0. Stars and issues welcome.

**Guide and write-ups:** [product guide](https://anujmahajan.com/products/prompteval) ·
[Is the cheaper model good enough?](https://anujmahajan.com/blog/is-the-cheaper-model-good-enough)
(a worked comparison) ·
[Let an LLM draft your eval](https://anujmahajan.com/blog/let-an-llm-draft-your-eval)
(drafting the test set and criteria with an LLM).

---

## What prompteval is, and is not

| ✅ It is | ❌ It is not |
|---|---|
| A **pre-release** eval framework (think pytest for prompts): you run it before shipping, not inside your product | A runtime proxy or LLM gateway |
| Run from your CLI or your CI | Sitting in your prod request path |
| Compares two prompt versions on quality **and** cost | Caching prod LLM calls |
| Statistical significance built in | Replacing Helicone / Portkey / LiteLLM |
| Honest about scorer methodology | A black-box "AI judges your AI" |

**It does call your models.** Measuring real quality and real cost means real
requests: `run`, `sweep` and `noise` send each example to the model (times
`--repeats`), plus any `llm_judge` calls in your scorers, on your API key.
`compare`, the CI gate and the `models` commands make no model calls; they read
saved runs and local prices. Re-running an unchanged `sweep` reuses saved runs,
and the GitHub Action makes no calls when the prompt didn't change.

**Whose API keys?** Yours. prompteval has no hosted service and no key of
its own: it calls each provider directly with your key, so the provider bills
your account and nobody else sees the requests.

- **OpenAI** (the default): `OPENAI_API_KEY`.
- **Any other provider**: put its key in an environment variable and name it
  with `--api-key-env GROQ_API_KEY` (or `api_key_env:` in `prices.yaml`). A
  local Ollama server needs none. A sweep across providers needs one key per
  provider.
- **`llm_judge`** uses `OPENAI_API_KEY` by default; to judge with another
  provider, pass it a client: `llm_judge(..., client=OpenAI(base_url=..., api_key=...))`.
- Keys can live in a `.env` file (or `evals/.env`, where `init` puts the
  example), read automatically. A variable already set in your shell, or a CI
  secret, always wins.

**Closest neighbors:** pytest, [Braintrust Experiments](https://braintrust.dev/), [promptfoo](https://promptfoo.dev/), [Inspect AI](https://inspect.aisi.org.uk/).

**Not what we do:** [Helicone](https://helicone.ai/), [Portkey](https://portkey.ai/), [LiteLLM](https://github.com/BerriAI/litellm), [LangChain](https://langchain.com/).

---

## User journey (v1)

Eight steps. ~30 minutes the first time. ~5 minutes every time after.

### 1. Install

```bash
uv add prompteval-delta        # or: pip install prompteval-delta
prompteval --version
```

Want to see a report before setting anything up?

```bash
prompteval demo        # no API key, no network: a full comparison in a second
```

It compares two bundled runs of a support-reply prompt, before and after it was
shortened. The runs are illustrative (generated by `scripts/build_demo_runs.py`,
not recorded from a model), and the command says so.

### 2. Bootstrap an evals folder

```bash
cd ~/my-app
prompteval init
```

Creates:
```
evals/
├── prompts/
│   ├── v1.txt              # paste your current prompt
│   └── v2.txt              # paste the variant you want to test
├── dataset.jsonl           # one example pre-filled
├── eval.py                 # pre-filled with 3 scorer examples
└── .env.example            # OPENAI_API_KEY=...
```

**Or let AI draft the examples and scorers for you to edit:**

```bash
prompteval bootstrap --prompt prompts/v1.txt          # invents ~25 inputs
prompteval bootstrap --prompt prompts/v1.txt --from evals/dataset.jsonl   # uses your real ones
```

One model call (cost shown before it runs, on your key) drafts `dataset.jsonl`
(typical, edge and adversarial cases), `eval.py` and `BOOTSTRAP.md`. Hard rules
your prompt states become plain code checks (required phrases, JSON keys,
length limits); the subjective part becomes your judging rules in
`criteria/<prompt name>.md`, drafted for you to rewrite (see the `criteria/`
folder under step 7).
The AI supplies only data: prompteval writes `eval.py` from a fixed template,
so no generated code runs. Every file starts with a DRAFT line that `run`
warns about until you delete it, invented inputs are marked synthetic (and
`compare` says how many there are), and the generator is named so you can
evaluate with a different model.

### 3. Bring your prompts + a few examples

Paste prompts into `prompts/v1.txt` and `prompts/v2.txt`. Add 10–30 examples to `dataset.jsonl`:

```jsonl
{"id": "refund-1", "input": "My item arrived broken, refund please", "expected": {"must_mention": ["30-day refund"], "tone": "empathetic"}}
{"id": "refund-2", "input": "Refund request 45 days after purchase", "expected": {"must_mention": ["30-day", "outside policy"], "tone": "polite"}}
```

The `expected` field is whatever your scorers want — there's no fixed schema.

**Have real traffic?** Real inputs beat invented ones. Turn a CSV or JSONL
export of logs into the dataset, locally, with personal data masked:

```bash
prompteval dataset import support-export.csv --input-column message \
  --map team=vars.team --sample 40 --redact emails,phones
```

Duplicates are dropped, the sample is reproducible (`--seed`), nothing is sent
anywhere, and each row is marked as real, so reports can say how many examples
are synthetic.

**Prompts with variables.** If your prompt is a template, write it as one:
`Summarise this ticket for {customer}: {input}`. Each `{name}` is filled from
the example's `input` and an optional `vars` object, and the filled prompt is
sent as the user message (a prompt with no placeholders is sent as the system
message, with `input` as the user message). JSON in a prompt is left alone,
and `{{name}}` stays a literal `{name}`. To keep a system prompt and template the user message
instead, set `Eval(..., user_template="Customer {customer} asks: {input}")`.

```jsonl
{"id": "t-1", "input": "Item arrived broken", "vars": {"customer": "Asha"}, "expected": {"must_mention": ["refund"]}}
```

**Request settings** such as `temperature`, `max_tokens` or JSON mode go on the
eval, `Eval(..., params={"temperature": 0, "max_tokens": 300})`, or on a run,
`--param max_tokens=300`. They are saved with the run, and `compare` warns when
two runs used different settings, because `max_tokens` is one of the most
effective cost levers there is and is worth evaluating on its own.

### 4. Define what "good" means for your use case (scorers)

`eval.py` ships with three example scorers. Edit them, write new ones, or copy from templates:

```python
from prompteval import Eval, scorer, llm_judge

@scorer
def mentions_required_terms(output: str, expected: dict) -> float:
    """Hard scorer: did the reply contain required phrases?"""
    must = expected.get("must_mention", [])
    if not must:
        return 1.0
    return sum(1 for t in must if t.lower() in output.lower()) / len(must)

@scorer
def professional_tone(output: str, expected: dict) -> float:
    """LLM-as-judge: rate professionalism 0-1."""
    return llm_judge(
        rubric="Rate the tone of this support reply 0-1. 1=empathetic+professional, 0=rude.",
        text=output,
    )

eval = Eval(
    name="support-assistant",
    dataset="dataset.jsonl",
    scorers=[mentions_required_terms, professional_tone],
)
```

**Judged qualities go in a criteria file.** For what a code check can't
measure (accuracy against your data, tone, what must never be said), write your
rules once in `criteria/<prompt name>.md`, most important first, and decide
which prompt each file judges. The pairwise judge and `llm_judge` both read it.
See [the `criteria/` folder](#7-compare--the-moment-of-truth) in step 7.

Don't know where to start? `prompteval scorer list` shows 9 stock scorers (exact match, contains substring, regex match, not empty, length between, JSON keys present, required terms, LLM-judge tone, LLM-judge factuality). `prompteval scorer copy <name>` drops one into your `eval.py`.

> **Honest caveat that matters:** prompteval doesn't define quality for your business — you do, by writing scorers. Get this part right and the tool earns its keep. Get it wrong and the tool will tell you wrong things confidently. **Every report shows which scorers ran on how many examples — methodology is always visible.** See [docs/writing-scorers.md](docs/writing-scorers.md): deterministic checks first, when a judge is worth it, and why binary scorers need more examples.

### 5. Run baseline (v1)

```bash
prompteval run --prompt prompts/v1.txt --tag baseline
```

```
Running 20 examples × 1 prompt against gpt-4o...
  refund-1   ✓ mentions_required_terms=1.00  professional_tone=0.85
  refund-2   ✓ mentions_required_terms=0.50  professional_tone=0.90
  ...
=== baseline ===
mentions_required_terms:  0.84 (n=20)
professional_tone:         0.81 (n=20)
total cost:                $0.158  (avg $0.0079/call)
avg latency:               1.4s
saved to: .prompteval/runs/baseline-2026-06-26T1240.json
```

### 6. Run the variant (v2)

```bash
prompteval run --prompt prompts/v2.txt --tag short-prompt
```

### 7. Compare — the moment of truth

**No scorers yet?** Add `--pairwise` and a judge model compares the two runs'
answers directly, example by example:

```bash
prompteval compare baseline short-prompt --pairwise
```

Each example is judged twice with the answers swapped; if the two verdicts
disagree it counts as a tie, so a judge that favours whichever answer it reads
first can't manufacture a win. The judge defaults to a model neither run used,
costs two calls per example (on your key, reported in the output), and its
result goes through the same statistics and verdict as any scorer. Scorers
still give sharper, cheaper answers once you know what "good" means; pairwise
is the way to get a first result without writing any. `sweep --pairwise` works
the same way.

**You decide how answers are judged: the `criteria/` folder.** Only you know
what makes one answer better than another for your product, so write it down,
one file per task, most important rule first:

```bash
prompteval criteria new prompt-create-kundli   # writes criteria/prompt-create-kundli.md
```

```markdown
1. Uses only the birth details and chart data given; never guesses a placement.
2. When the birth time is missing, says what that limits instead of filling it in.
3. Makes no fatalistic predictions and does not push paid remedies.
4. Explains any Sanskrit term the first time it appears.
```

**You decide which criteria judge which prompt.** Keep as many files as you
have prompts (and shared ones, like `brand-voice.md`), then pick, from the
most specific to the most automatic; the first that applies wins:

| How | Example |
| --- | --- |
| On the comparison | `prompteval compare v1 v2 --pairwise --criteria prompt-create-kundli` |
| When you run the prompt (saved on the run) | `prompteval run --prompt prompts/v2.txt --tag v2 --criteria prompt-create-kundli` |
| In `eval.py`, for every run of that eval | `Eval(..., criteria="prompt-create-kundli")` |
| Name it after the prompt file | `prompts/prompt-create-kundli.txt` → `criteria/prompt-create-kundli.md` |
| Name it after the eval | `Eval(name="kundli")` → `criteria/kundli.md` |

Combine files in priority order with a comma: `--criteria brand-voice,prompt-create-kundli`.
`prompteval criteria which v1 v2` prints the files a comparison would use and
why, and the report says the same. A name that doesn't exist is an error (listing
the files you have), never a silent fallback, and `run --criteria` checks it
before spending anything. If two runs were made with different criteria,
compare says so and uses the baseline's unless you pass `--criteria`.

For a per-reply score against the same rules, use
`llm_judge(rubric=criteria_rubric("prompt-create-kundli"), text=output)`.
`bootstrap` drafts this file from your prompt for you to rewrite, and
[`examples/criteria/prompt-create-kundli.md`](examples/criteria/prompt-create-kundli.md)
is a full example.

Without a criteria file the judge uses a built-in checklist: follows the
assistant's rules; contradicts nothing and invents no facts, prices, policies
or promises; answers what was asked; complete; clear; stays within its rules
when a user tries to override them. Either way it sees the rules the assistant
was given (by default the baseline run's prompt; point it at your real policy
with `--judge-context policy.md`), it is told that length is not quality, and
the report says whose criteria it applied and what it was shown.


```bash
prompteval compare baseline short-prompt
```

You get the report shown at the top of this README — paired quality and cost deltas with 95% CIs, p-values, and a plain-English recommendation.

**How the recommendation is decided.** "No significant drop" is not the same as
"quality holds": with 20 examples almost nothing is significant. So a scorer
counts as holding only when its worst plausible drop is within a margin (0.05
by default, `--margin` to change it). A certain drop is a regression; anything
else is reported as **inconclusive**, with an estimate of how many examples
would settle it. Examples that fail (timeouts, errors) count against the run
that failed them, and the decision is adjusted for the number of scorers so
adding scorers doesn't add false alarms.

### See it: `prompteval ui`

```bash
prompteval ui      # opens http://127.0.0.1:8765
```

A local page over what the commands above already saved:

- **Compare**: pick a baseline and a candidate and get the verdict and metric
  table, then every example with both answers side by side, each scorer's
  score (better or worse than the other run), and the judge's verdict with its
  one-line reason for both orders. Filter to the examples where the candidate
  lost, where the judge flipped, or where scores differ, and search inputs and
  answers. Both prompts are one click away.
- **Runs**: every run with its model, prompt file, criteria, scores and cost,
  and each run's answers.
- **Criteria**: read and edit your criteria files, and remove the DRAFT line
  once you've reviewed one.

It runs on your machine only (127.0.0.1), needs no account and never calls a
model: re-running or re-judging costs money, so it shows you the command
instead. Its only write is saving a criteria file, inside `criteria/`.
`compare --pairwise` keeps each judgement in `.prompteval/judgements/` for it.

### Any model, any OpenAI-compatible provider

**Prices come from a catalog of ~3,900 models**, merged from two maintained
public sources: [LiteLLM's model price list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
(MIT) and [OpenRouter's model list](https://openrouter.ai/api/v1/models). It
covers OpenAI, Anthropic, Gemini, Mistral, DeepSeek, xAI, Together, Fireworks,
DeepInfra, Cerebras, OpenRouter and more. A dated snapshot ships with each
release; refresh it whenever you like:

```bash
prompteval models update          # fetch current prices into ~/.cache/prompteval (no upgrade needed)
prompteval models search llama    # find a model's name and price
prompteval models price claude-sonnet-4-5
```

You don't need a new release for new prices: `models update` is all it takes,
and `run` and `sweep` remind you when the prices they used are over 30 days
old.

The price follows the endpoint: the same open model costs different amounts on
Groq, Together and OpenRouter, so `--base-url` decides which provider's entry
applies. Every run records the catalog entry and the date its price was
fetched. Prices are what those sources publish; check the provider's own page
before quoting a number.

Anything that speaks the OpenAI API (Groq, Together, OpenRouter, Fireworks, a
local Ollama or vLLM) runs. If a model isn't in the catalog, give its price:

```bash
# Groq
prompteval run --prompt prompts/v1.txt --tag llama \
  --model llama-3.3-70b-versatile \
  --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY \
  --price input=0.59,output=0.79        # USD per 1M tokens, from the provider's pricing page

# A local Ollama model: no key, and free
prompteval run --prompt prompts/v1.txt --tag local \
  --model llama3.2 --base-url http://localhost:11434/v1 --price input=0,output=0
```

Or keep prices in `prices.yaml` next to `eval.py`, read automatically:

```yaml
models:
  - name: llama-3.3-70b-versatile
    input_per_1m: 0.59
    output_per_1m: 0.79
    source: https://groq.com/pricing (checked 2026-11-12)
```

`base_url` and `api_key_env` can also be set on the `Eval` itself. Prices you
supply are recorded with the run, so a report always says where its costs came
from. An unknown model with no price fails before any request is made, unless you
pass `--unpriced`: then it runs, token counts are recorded, and `compare` and
`sweep` say cost was not evaluated (comparing tokens instead). A cost gate
fails rather than passing on $0, and a sweep never names an unpriced model as
the cheapest.

### No API key? Use Claude Code

If you have [Claude Code](https://claude.com/claude-code) installed and signed
in, prompteval can run models through it, on your own login and plan, with no
API key:

```bash
prompteval bootstrap --prompt prompts/v1.txt --base-url claude-code   # Sonnet drafts, Haiku runs
prompteval run --prompt prompts/v1.txt --tag v1 --eval-file evals/eval.py
prompteval compare v1 v2 --pairwise --judge-base-url claude-code
```

Each call is `claude -p` with your prompt as the system prompt and tools off
(measured: $0.00027 for a short reply, against $0.087 with Claude Code's own
setup), run from an empty temporary folder so a project `CLAUDE.md` can't leak
into the answers. Costs are the figures Claude Code reports per call (API list
prices; on a subscription, usage against your plan). Worth knowing:

- Claude models only, and slower than an API (a process per call).
- Each call carries a small fixed overhead (cache writes), so absolute costs
  are higher than the API and a shorter prompt saves proportionally less.
- No temperature setting; use `--repeats` if answers vary.
- A user-level `~/.claude/CLAUDE.md` still applies; keep it out of eval
  machines.

### Which model is cheapest for this prompt?

```bash
prompteval sweep --prompt prompts/v1.txt --models gpt-4o,gpt-4o-mini,gpt-4.1-nano --markdown sweep.md
```

Runs the prompt on every model, compares each to the first (or `--baseline`)
with the same rules as `compare`, corrected for the number of models, and
prints one table ending in a one-line answer such as *"Cheapest model that
holds quality: gpt-4o-mini (−94% cost vs gpt-4o)"*. When nothing is proven,
it says so rather than guessing.

- Mix providers: give a model a `base_url` and `api_key_env` in `prices.yaml`.
- Re-running is free: a saved run with the same prompt, dataset, model and
  settings is reused (`--rerun` to force). Add examples and sweep again.
- `--markdown` and `--html` write the table for a PR, an issue or a post.

**How big a change can your dataset detect?** `prompteval noise --prompt
prompts/v1.txt` runs the same prompt twice; whatever differs is noise, and it
prints, per scorer, the size of change that is indistinguishable from it.
`--repeats K` answers each example K times and averages the scores (costing K
times more), and `llm_judge` samples at temperature 0 by default, so the judge
adds as little noise as possible.

`run` and `sweep` call the model for 4 examples at once (`--concurrency`, lower
it if your provider rate-limits you); rate limits and server errors are retried
with backoff, and results are saved in dataset order either way.

### 8. (Optional but the moat) Lock it into CI

Add the Action, and every pull request that changes the prompt gets the report
as a comment, verdict first:

```yaml
# .github/workflows/prompteval.yml
on:
  pull_request:
    paths: ["evals/**"]
permissions:
  contents: read
  pull-requests: write      # to post the comment
jobs:
  prompteval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: mahAnuj/prompteval@v0.3.0
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        with:
          prompt: evals/prompts/v1.txt
          fail-on: cost+10%,quality-5%
```

Pin a release tag (`@v0.3.0`, or whatever is newest on the releases page);
bump it when you want a newer version.

It runs the PR base's version of the prompt and the head version on the same
eval, posts **one** comment that it updates on every push, and writes the same
report to the job summary. An unchanged prompt costs nothing: no runs are made.
Set `models: gpt-4o,gpt-4o-mini` instead to post a model sweep. Other inputs:
`eval-file`, `margin`, `baseline-ref`, `comment: false`. Pull requests from
forks get no secrets, so the runs fail there by design.

Prefer plain steps? [`examples/github-actions/prompteval.yml`](examples/github-actions/prompteval.yml)
does the same with `prompteval run` and `compare --markdown` (or `--html`).

Any future prompt change gets gated. PR fails if cost *could* have risen more
than 10%, or any scorer *could* have dropped more than 5 percentage points,
judged on the 95% confidence interval. A small or noisy eval therefore fails
until it has enough examples to rule the breach out, rather than passing
because nothing reached significance. (`--gate-mode significant` restores the
v0.1 behaviour of counting only significant breaches.) The `--html` report is uploaded as a build artifact and
linked from the PR's Checks tab.

See [`examples/github-actions/README.md`](examples/github-actions/README.md)
for setup steps, required secrets, and how to tune the gate.

---

## Status, scope, and roadmap

Living source of truth lives in **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** —
week-by-week status table, decisions log, explicit non-goals, and post-v1 roadmap.
That file is updated the same commit any code ships; this README only summarises.

**As of 12 Nov 2026 (v0.3.0).** `init`, `bootstrap`, `run`, `compare`,
`sweep`, `noise`, `criteria`, `dataset`, `ui`, `models` and `scorer` CLIs; a local
UI; AI-drafted
evals; side-by-side judging against your own criteria; Claude Code as a
no-key provider; a price catalog of ~3,900 models; any
OpenAI-compatible provider; paired deltas with bootstrap CIs; verdicts decided
by non-inferiority against a margin; CI gates on the worst plausible case; the
error rate as a quality metric; judge cost reported on its own line; HTML and
Markdown reports; and a GitHub Action that comments on pull requests.
**361/361 tests green.** What changed is in [CHANGELOG.md](CHANGELOG.md).

Still alpha: treat the public API as movable until 1.0.

Original plan was 10 weeks with a 2026-08-16 review point. That date has passed
with the build ahead of where it needed to be, so the project continues; the
week-by-week table in the plan is the live record.

---

## Development

```bash
# Install (creates a venv via uv, pulls Python 3.12)
uv sync

# Full quality gate
uv run ruff check .            # lint
uv run ruff format --check .   # format
uv run mypy src tests          # types (strict)
uv run pytest                  # tests
uv run prompteval --version    # CLI smoke

# One-liner
uv run pytest && uv run mypy src tests && uv run ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
