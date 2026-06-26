"""Example prompteval evaluation for a support-assistant prompt.

Workflow:
    prompteval run --prompt prompts/v1.txt --tag baseline
    prompteval run --prompt prompts/v2.txt --tag short-prompt
    prompteval compare baseline short-prompt

Tune `scorers` to encode your definition of "good" for your use case.
prompteval does not define quality for you — you do, by writing scorers.

See https://github.com/mahAnuj/prompteval for the docs and
`prompteval scorer list` for more starter templates.

Note: requires prompteval >= 0.1.0 (runner + scorer surfaces ship in Week 3-4
of v0.1 development — see IMPLEMENTATION_PLAN.md). The template documents the
v0.1 API contract today.
"""

from prompteval import Eval, llm_judge, scorer


@scorer
def mentions_required_terms(output: str, expected: dict) -> float:
    """Hard scorer: did the reply contain the required phrases?

    Returns the fraction of `expected["must_mention"]` terms found in the
    output, case-insensitive. Returns 1.0 if no required terms are set.
    Prefer this kind of deterministic scorer over LLM-as-judge whenever
    ground truth is checkable — it's cheaper, faster, and reproducible.
    """
    must = expected.get("must_mention", [])
    if not must:
        return 1.0
    text = output.lower()
    return sum(1 for term in must if term.lower() in text) / len(must)


@scorer
def professional_tone(output: str, expected: dict) -> float:
    """LLM-as-judge: rate the tone of the support reply on a 0-1 scale.

    Uses gpt-4o-mini as the judge — cheap and good enough for most rubrics.
    Tune the rubric for your domain; rubric quality dominates judge cost.
    """
    return llm_judge(
        rubric=(
            "Rate the tone of this support reply on a 0-1 scale. "
            "1 = empathetic, professional, and appropriate to the customer's emotional context. "
            "0 = rude, robotic, or dismissive. "
            "Return only the number."
        ),
        text=output,
    )


eval = Eval(
    name="support-assistant",
    dataset="dataset.jsonl",
    scorers=[mentions_required_terms, professional_tone],
    # Default model: gpt-4o-mini ($0.15/$0.60 per 1M tokens — under a cent per
    # full eval run). Override per-run with: prompteval run --model gpt-4o
    model="gpt-4o-mini",
)
