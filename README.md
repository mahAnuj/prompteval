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

Quality verdict:  no significant regression
Cost verdict:     significant 37% reduction
Recommendation:   ship v2 — cost savings real, quality holds
```

Existing eval tools (Braintrust, Langfuse, promptfoo, Phoenix) track cost as a metric you graph; prompteval treats it as a **comparison axis** alongside quality. See [docs/competitor-scan.md](docs/competitor-scan.md) for verified evidence.

> ⚠️ **Early alpha — pre-v0.1.** API will change weekly until v0.5. Stars and issues welcome to shape it.

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
uv add prompteval        # or pip install prompteval
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

### 8. (Optional but the moat) Lock it into CI

```yaml
# .github/workflows/prompt-quality.yml
- run: prompteval run --prompt prompts/current.txt --tag pr-${{ github.sha }}
- run: prompteval compare main pr-${{ github.sha }} --fail-on cost+15%,quality-5%
```

Any future prompt change gets gated. PR fails if cost regresses >15% or quality drops >5%, with the comparison report inline as a check.

---

## Status, scope, and roadmap

Living source of truth lives in **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** —
week-by-week status table, decisions log, explicit non-goals, and post-v1 roadmap.
That file is updated the same commit any code ships; this README only summarises.

10-week build. Kill date **2026-08-16**. As of 2026-06-26: Week 0 done (bootstrap +
competitor scan + scope lock); Week 1 (`prompteval init`) up next.

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
