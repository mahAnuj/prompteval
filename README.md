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

Existing eval tools (Braintrust, Langfuse, promptfoo, Phoenix) track cost as a metric you graph; prompteval treats it as a **comparison axis** alongside quality. See [docs/competitor-scan.md](docs/competitor-scan.md) for verified evidence.

[![PyPI](https://img.shields.io/pypi/v/prompteval-delta)](https://pypi.org/project/prompteval-delta/)

> **Alpha — v0.1.0, on PyPI as [`prompteval-delta`](https://pypi.org/project/prompteval-delta/).**
> **Do not `pip install prompteval`** — that name on PyPI belongs to an
> unrelated project. The import and the command are still `prompteval`. The
> public API may still change before 1.0. Stars and issues welcome.

---

## What prompteval is, and is not

| ✅ It is | ❌ It is not |
|---|---|
| An **offline** eval framework (think pytest for prompts) | A runtime proxy or LLM gateway |
| Run from your CLI or your CI | Sitting in your prod request path |
| Compares two prompt versions on quality **and** cost | Caching prod LLM calls |
| Statistical significance built in | Replacing Helicone / Portkey / LiteLLM |
| Honest about scorer methodology | A black-box "AI judges your AI" |

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

### 3. Bring your prompts + a few examples

Paste prompts into `prompts/v1.txt` and `prompts/v2.txt`. Add 10–30 examples to `dataset.jsonl`:

```jsonl
{"id": "refund-1", "input": "My item arrived broken, refund please", "expected": {"must_mention": ["30-day refund"], "tone": "empathetic"}}
{"id": "refund-2", "input": "Refund request 45 days after purchase", "expected": {"must_mention": ["30-day", "outside policy"], "tone": "polite"}}
```

The `expected` field is whatever your scorers want — there's no fixed schema.

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

Don't know where to start? `prompteval scorer list` shows 6–8 templates (regex match, JSON-schema valid, contains-substring, exact-match, LLM-judge factuality, LLM-judge tone). `prompteval scorer copy <name>` drops one into your `eval.py`.

> **Honest caveat that matters:** prompteval doesn't define quality for your business — you do, by writing scorers. Get this part right and the tool earns its keep. Get it wrong and the tool will tell you wrong things confidently. **Every report shows which scorers ran on how many examples — methodology is always visible.** See [docs/writing-scorers.md](docs/writing-scorers.md) (coming v0.1) for guidance.

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

### Any model, any OpenAI-compatible provider

**Prices come from a catalog of ~3,900 models**, merged from two maintained
public sources: [LiteLLM's model price list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
(MIT) and [OpenRouter's model list](https://openrouter.ai/api/v1/models). It
covers OpenAI, Anthropic, Gemini, Mistral, DeepSeek, xAI, Together, Fireworks,
DeepInfra, Cerebras, OpenRouter and more. A dated snapshot ships with each
release; refresh it whenever you like:

```bash
prompteval models update          # fetch current prices into ~/.cache/prompteval
prompteval models search llama    # find a model's name and price
prompteval models price claude-sonnet-4-5
```

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
from. An unknown model with no price fails before any request is made.

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

### 8. (Optional but the moat) Lock it into CI

A copy-and-paste GitHub Actions workflow lives at [`examples/github-actions/prompteval.yml`](examples/github-actions/prompteval.yml). The core of it:

```yaml
- run: prompteval run --prompt /tmp/baseline.txt --tag baseline
- run: prompteval run --prompt evals/prompts/v1.txt --tag candidate
- run: prompteval compare baseline candidate --html report.html --fail-on cost+10%,quality-5%
```

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

**As of 31 Aug 2026 — the tool works end to end.** Everything needed to run a
comparison has shipped and is tested: `init`, `run`, `compare`, `models` and
`scorer` CLIs; the cost model; 9 stock scorers plus `llm_judge`; paired deltas
with bootstrap CIs and significance; plain-English verdicts; the HTML report;
and the `--fail-on` CI gate with a GitHub Actions template. **274/274 tests
green.**

**What is left is packaging, not capability:** a scorer-writing guide, an API
stability sweep. v0.1.0 is published on PyPI as `prompteval-delta`; treat the
public API as still movable until 1.0.

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
