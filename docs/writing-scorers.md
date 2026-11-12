# Writing scorers

prompteval does not know what "good" means for your product. Your scorers
define it. A report is only as trustworthy as they are: a scorer that measures
the wrong thing produces a confident, wrong verdict.

## What a scorer is

A function decorated with `@scorer` that looks at one output and returns a
number in `[0, 1]`, where 1 is best:

```python
from prompteval import scorer

@scorer
def mentions_refund_window(output: str, expected: dict) -> float:
    return 1.0 if "30 days" in output else 0.0
```

- It may ask for any of `input`, `output` and `expected`, by naming them as
  parameters. Only what it asks for is passed. `expected` is the example's
  `expected` object from `dataset.jsonl`, with whatever keys you choose.
- It returns a `float` in `[0, 1]`. A value outside that range fails the
  example loudly instead of skewing the average. Return a `ScorerResult(score,
  reasoning=..., metadata=...)` to keep the reason for a score with the run.
- An exception inside a scorer fails that example, and failures count against
  the run in `compare` (the error rate is a quality metric of its own).

## Or skip scorers for a first result

`prompteval compare a b --pairwise` has a judge model pick the better answer for
each example (asked twice with the order swapped, so position bias becomes a
tie). It is the quickest first signal; scorers are cheaper, more precise and
easier to trust once you know what to check.

## Start with the stock scorers

`prompteval scorer list` shows them; `prompteval scorer copy <name>` drops one
into your `eval.py`. The keys each reads from `expected`:

| Scorer | Score | Reads from `expected` |
|---|---|---|
| `exact_match` | 1 if the output equals the target exactly | `expected` |
| `contains_substring` | 1 if the substring appears (case-sensitive) | `substring` |
| `regex_match` | 1 if the regex matches anywhere | `pattern` |
| `not_empty` | 1 if the output isn't blank | — |
| `length_between` | 1 if the length is within bounds | `min_len`, `max_len` |
| `json_schema_valid` | 1 if valid JSON containing every required key | `required_keys` |
| `mentions_required_terms` | share of required terms present (case-insensitive) | `must_mention` |
| `llm_judge_tone` | judge's 0-1 rating of the tone | `tone` |
| `llm_judge_factuality` | judge's 0-1 rating of consistency with a reference | `reference` |

## Deterministic first, a judge only when you must

Prefer a check you can write in code: a phrase that must appear, a regex, valid
JSON with the right keys, a length limit. Such checks are free, instant, and
give the same answer every time, so any difference between two runs is the
prompt's doing.

Reach for `llm_judge` when quality really is a judgement, such as tone,
helpfulness or faithfulness to a source:

```python
from prompteval import llm_judge, scorer

@scorer
def faithful_to_policy(output: str, expected: dict) -> float:
    return llm_judge(
        rubric=(
            "You check a support reply against the company policy below. "
            "Reply with ONLY a number from 0 to 1: 1 = every claim matches the policy, "
            "0 = it contradicts the policy.\n\nPolicy:\n" + expected["policy"]
        ),
        text=output,
    )
```

Things to know about judges:

- **Ask for only a number.** The first number in the reply is used, and one
  outside `[0, 1]` is an error. That catches a rubric that drifted to a 1-10
  scale.
- **They sample at temperature 0 by default**, so the judge adds as little
  noise as it can. Pass `temperature=None` for models that accept no other
  value.
- **They cost money, and the cost is reported.** `run` prints a "judge cost"
  line, and `compare` shows each run's judge cost below the table. It is kept
  out of the cost verdict, which compares only the prompt under test.
- **A judge is a model with its own blind spots.** Check a handful of its
  scores by hand before trusting it, and keep the rubric specific.
- **Keep the rules in one place.** Write your judging rules in
  `criteria/<task>.md` and score with
  `llm_judge(rubric=criteria_rubric("<task>"), text=output)`; `compare
  --pairwise --criteria <task>` then judges with exactly the same rules, so the
  per-reply score and the side-by-side verdict can't drift apart.

## Binary or graded: it decides how many examples you need

`compare` says "quality holds" only when the worst plausible drop is within the
margin (0.05 by default), so the width of the confidence interval matters.

- **A binary scorer** (0 or 1) is noisy per example. As a rough guide: with
  pass rates around 80% and answers that vary independently, the paired
  difference has a standard deviation near 0.57, so ruling out a 0.05 drop
  takes on the order of `(1.96 × 0.57 / 0.05)² ≈ 500` examples. With 20
  examples, expect **inconclusive** unless the change is large. That is the
  honest answer, not a bug.
- **A graded scorer** (a share of terms found, a 0-1 judge rating) usually
  varies less from example to example, so the same dataset can detect smaller
  changes.

You don't have to guess. Run `prompteval noise --prompt prompts/v1.txt`: it
runs the same prompt twice and prints, per scorer, the size of change that is
indistinguishable from noise on your dataset. When `compare` is inconclusive,
it also estimates how many examples would settle it. If answers vary a lot,
`--repeats 3` averages several answers per example, and `--param
temperature=0` makes the prompt's own answers steadier.

## Three worked examples

**Required facts in a support reply** (deterministic, graded):

```jsonl
{"id": "refund-1", "input": "My order arrived broken", "expected": {"must_mention": ["refund", "30 days"]}}
```

```python
eval = Eval(name="support", dataset="dataset.jsonl", scorers=[mentions_required_terms])
```

**Structured extraction** (deterministic, binary): the reply must be JSON with
the fields your code reads.

```jsonl
{"id": "inv-1", "input": "Invoice 4411 from Acme, total $120", "expected": {"required_keys": ["invoice_id", "vendor", "total"]}}
```

```python
eval = Eval(
    name="extract",
    dataset="dataset.jsonl",
    scorers=[json_schema_valid],
    params={"temperature": 0, "response_format": {"type": "json_object"}},
)
```

**Tone, with a length guard** (a judge plus a deterministic check): the judge
rates tone, and `length_between` stops a prompt from "improving" tone by
getting longer and costlier.

```jsonl
{"id": "t-1", "input": "Where is my order?!", "expected": {"tone": "calm and empathetic", "min_len": 40, "max_len": 600}}
```

```python
eval = Eval(name="tone", dataset="dataset.jsonl", scorers=[llm_judge_tone, length_between])
```

## Checklist

- Every scorer returns 1 for a good answer and 0 for a bad one. Test it on one
  of each before a real run.
- Deterministic where possible; a judge only for real judgement calls.
- Enough examples for the change you care about (`prompteval noise` tells you).
- Keep scorers stable between the two runs you compare: changing a scorer
  changes what "quality" means, and the comparison stops being like for like.
