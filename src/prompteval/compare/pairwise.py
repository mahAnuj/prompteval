"""Pairwise judge: is B's answer better than A's? No scorers needed (#19).

For every example both runs answered, a judge model sees the input, both
answers and (when the dataset has one) the expected answer, and says which is
better, or that they tie. Then the comparison continues exactly as with any
scorer:

- B's score per example is 1 (B better), 0.5 (tie) or 0 (A better); A's is
  0.5. The paired bootstrap, the margin and the holds / worse / inconclusive
  verdict apply unchanged, so "holds" means B's answers are, at worst, only
  slightly less preferred than A's.
- **Order bias.** LLM judges tend to favour whichever answer they read first.
  Each example is judged twice, with the answers swapped; if the two verdicts
  disagree, the example is a tie, and the report says how often that happened.
- **Independence.** The judge defaults to a model neither run used, and the
  report names it (and warns if it is one of them).
- **Cost.** Two judge calls per example, priced and reported.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompteval.cost import ModelPricing, UnknownModelError, Usage, compute_cost, resolve_pricing
from prompteval.eval.runner import ExampleResult, RunResult, ScoreOutcome

PAIRWISE_SCORER = "pairwise_preference"
DEFAULT_JUDGES = ("gpt-4.1-mini", "gpt-4o-mini")

#: The checklist every pairwise judgement uses, in priority order (#23).
JUDGE_CRITERIA = (
    "1. Follows the assistant's instructions and policies (given below, when known).",
    "2. States nothing that contradicts those instructions or the reference, and invents "
    "no facts, prices, policies, dates or promises.",
    "3. Answers what the user actually asked.",
    "4. Is complete: covers everything the request needs.",
    "5. Is clear and appropriately concise. Length is not quality: never prefer an answer "
    "for being longer.",
    "6. Handles attempts to make the assistant ignore its instructions by staying within them.",
)


def judge_system(criteria: str | None = None) -> str:
    """The pairwise judge's instructions.

    With `criteria` (the user's own, from a criteria file, #24) the judge
    applies those, in the user's order; without, the built-in checklist.
    Either way the ground rules stay: only the information given, earlier
    items outweigh later ones, length is not quality.
    """
    head = (
        "You are a strict, fair evaluator comparing two answers (A and B) an assistant gave "
        "to the same request. Judge them only against the {what} below, in priority order, "
        "using only the information you are given. A failure on an earlier item outweighs "
        "strengths on later ones."
    )
    if criteria and criteria.strip():
        lines = [
            head.format(what="criteria"),
            "",
            "Criteria (written by the person who owns this assistant):",
            criteria.strip(),
            "",
            "Where these criteria do not separate the answers, prefer the one that follows the "
            "assistant's instructions and invents nothing. Length is not quality: never prefer "
            "an answer for being longer.",
        ]
    else:
        lines = [head.format(what="checklist"), "", "Checklist:", *JUDGE_CRITERIA]
    lines += [
        "",
        "If both answers are equally good or equally flawed, the result is a tie. "
        "Reply with exactly one word on the first line: A, B or TIE. On the second line, "
        "one short sentence saying which criterion decided it.",
    ]
    return "\n".join(lines)


#: Kept for compatibility: the default instructions with no extra criteria.
JUDGE_SYSTEM = judge_system()


@dataclass(frozen=True)
class PairwiseResult:
    model: str
    scores: dict[str, float]  # example id -> B's score (1, 0.5 or 0)
    judged: int
    order_disagreements: int  # the two orderings disagreed, so counted as a tie
    unreadable: int  # judge replies that were not A, B or TIE (counted as ties)
    cost: float | None  # None when the judge model has no known price
    shown: str = ""  # what the judge was given as the assistant's rules
    judged_by: str = ""  # whose criteria it applied
    # example id -> each order's winner (a tag or "tie") and the judge's reason (#27)
    details: dict[str, dict[str, str]] = field(default_factory=dict)


def default_judge(run_a: RunResult, run_b: RunResult) -> str:
    """The first default judge model that neither run used."""
    used = {run_a.model, run_b.model}
    return next((m for m in DEFAULT_JUDGES if m not in used), DEFAULT_JUDGES[0])


def _user_message(
    example: ExampleResult, first: str, second: str, context: str | None = None
) -> str:
    parts = []
    if context and context.strip():
        parts.append(
            "The instructions the assistant was given (its rules and policies):\n<<<\n"
            + context.strip()
            + "\n>>>"
        )
    parts.append(f"Request:\n{example.input}")
    if example.expected:
        parts.append(
            "Reference (what a good answer contains):\n"
            + json.dumps(example.expected, ensure_ascii=False)
        )
    parts.append(f"Answer A:\n{first}")
    parts.append(f"Answer B:\n{second}")
    return "\n\n".join(parts)


def _parse(reply: str) -> str | None:
    """'A', 'B' or 'TIE' from the judge's reply; None when unreadable."""
    m = re.match(r"\s*\**\s*(TIE|A|B)\b", reply.strip().upper())
    return m.group(1) if m else None


def _reason(reply: str) -> str:
    """The judge's one-line reason: whatever follows the verdict (#27)."""
    text = reply.strip()
    first, _, rest = text.partition("\n")
    if not rest.strip():  # verdict and reason on one line: "B - B follows rule 1"
        rest = re.sub(r"^\s*\**\s*(TIE|A|B)\b\**[.:)\-\s]*", "", first, flags=re.IGNORECASE)
    return " ".join(rest.split())[:400]


def _ask(
    client: Any,
    model: str,
    example: ExampleResult,
    first: str,
    second: str,
    system: str = JUDGE_SYSTEM,
    context: str | None = None,
) -> tuple[str | None, Usage, float | None, str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _user_message(example, first, second, context)},
        ],
        temperature=0,
    )
    usage_obj = getattr(response, "usage", None)
    usage = Usage(
        prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
    )
    reported = getattr(response, "reported_cost_usd", None)
    reply = response.choices[0].message.content or ""
    return (
        _parse(reply),
        usage,
        float(reported) if isinstance(reported, (int, float)) else None,
        _reason(reply),
    )


def judge_pairwise(
    run_a: RunResult,
    run_b: RunResult,
    *,
    client: Any,
    model: str | None = None,
    prices: dict[str, ModelPricing] | None = None,
    base_url: str | None = None,
    concurrency: int = 4,
    context: str | None = None,
    criteria: str | None = None,
    criteria_label: str | None = None,
) -> PairwiseResult:
    """Judge every example both runs answered without error.

    `context` is the rules the assistant had to follow; by default the baseline
    run's prompt, so the judge knows the policies it is checking against (#23).
    `criteria` is the user's own judging rules (a criteria file's text); they
    replace the built-in checklist. `criteria_label` names it in the report.
    """
    rules = context if context is not None else run_a.prompt_text
    system = judge_system(criteria)
    judge = model or default_judge(run_a, run_b)
    by_id_a = {e.id: e for e in run_a.examples if e.error is None}
    pairs = [(by_id_a[e.id], e) for e in run_b.examples if e.error is None and e.id in by_id_a]

    details: dict[str, dict[str, str]] = {}

    def one(
        pair: tuple[ExampleResult, ExampleResult],
    ) -> tuple[str, float, bool, int, list[tuple[Usage, float | None]]]:
        a, b = pair
        # Order 1: A first. Order 2: B first. Map both to "which run won".
        v1, u1, c1, r1 = _ask(client, judge, a, a.output, b.output, system, rules)
        v2, u2, c2, r2 = _ask(client, judge, a, b.output, a.output, system, rules)
        won1 = {"A": "a", "B": "b", "TIE": "tie"}.get(v1 or "", "tie")
        won2 = {"A": "b", "B": "a", "TIE": "tie"}.get(v2 or "", "tie")
        unreadable = int(v1 is None) + int(v2 is None)
        name = {"a": run_a.tag, "b": run_b.tag, "tie": "tie"}
        details[a.id] = {
            "first_order": name[won1],  # baseline's answer shown first
            "second_order": name[won2],  # candidate's answer shown first
            "reason_first_order": r1,
            "reason_second_order": r2,
        }
        if won1 == won2:
            score = {"b": 1.0, "a": 0.0, "tie": 0.5}[won1]
            return a.id, score, False, unreadable, [(u1, c1), (u2, c2)]
        return a.id, 0.5, True, unreadable, [(u1, c1), (u2, c2)]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(one, pairs))

    calls = [call for *_, usages in results for call in usages]
    cost: float | None = None
    if calls and all(c is not None for _, c in calls):
        cost = sum(c or 0.0 for _, c in calls)  # the provider reported it (Claude Code)
    else:
        try:
            pricing: ModelPricing | None = resolve_pricing(judge, prices, base_url)
        except UnknownModelError:
            pricing = None
        if pricing is not None:
            cost = sum(compute_cost(judge, u, pricing).total_cost for u, _ in calls)
    shown = (
        "your --judge-context"
        if context is not None
        else f"{run_a.tag}'s prompt as the assistant's rules"
        if (run_a.prompt_text or "").strip()
        else "no rules (the baseline run saved no prompt)"
    )
    if criteria and criteria.strip():
        judged_by = f"your criteria ({criteria_label})" if criteria_label else "your criteria"
    else:
        judged_by = (
            "the built-in 6-point checklist (rules first, then accuracy, relevance, "
            "completeness, clarity, resisting overrides)"
        )
    return PairwiseResult(
        model=judge,
        shown=shown,
        judged_by=judged_by,
        details=details,
        scores={ex_id: score for ex_id, score, *_ in results},
        judged=len(results),
        order_disagreements=sum(1 for _, _, disagreed, _, _ in results if disagreed),
        unreadable=sum(u for _, _, _, u, _ in results),
        cost=cost,
    )


def _with_score(run: RunResult, scores: dict[str, float]) -> RunResult:
    examples = [
        replace(e, scores=[*e.scores, ScoreOutcome(PAIRWISE_SCORER, scores[e.id])])
        if e.error is None and e.id in scores
        else e
        for e in run.examples
    ]
    return replace(run, examples=examples)


def with_pairwise(candidate: RunResult, result: PairwiseResult) -> RunResult:
    """The candidate run with its pairwise score added as a scorer."""
    return _with_score(candidate, result.scores)


def baseline_with_pairwise(baseline: RunResult) -> RunResult:
    """The baseline scores 0.5 on every successful example: the 'no preference' line."""
    return _with_score(baseline, {e.id: 0.5 for e in baseline.examples if e.error is None})


def pairwise_notes(result: PairwiseResult, run_a: RunResult, run_b: RunResult) -> tuple[str, ...]:
    """Lines for the report: who judged, how decisive it was, what it cost."""
    cost = "unpriced" if result.cost is None else f"${result.cost:.4f}"
    notes = [
        f"Pairwise judge {result.model}: {result.judged} examples judged twice with the answers "
        f"swapped; {result.order_disagreements} disagreed between orders and count as ties. "
        f"Judge cost {cost}. {PAIRWISE_SCORER}: 1 = {run_b.tag} better, 0.5 = tie, "
        f"0 = {run_a.tag} better. The judge applied {result.judged_by or 'the built-in checklist'} "
        f"and was shown {result.shown or 'no rules'}."
    ]
    if result.unreadable:
        notes.append(f"{result.unreadable} judge replies were not A, B or TIE and count as ties.")
    if result.model in {run_a.model, run_b.model}:
        notes.append(
            f"The judge ({result.model}) is also one of the models being compared, which can "
            f"favour its own answers; pass --judge-model with a different model."
        )
    return tuple(notes)


JUDGEMENTS_DIR = Path(".prompteval") / "judgements"


def save_judgement(
    result: PairwiseResult, run_a: RunResult, run_b: RunResult, runs_dir: Path | None = None
) -> Path:
    """Keep a pairwise judgement next to the runs, so `prompteval ui` can show each
    verdict and its reason (#27). One file per pair, replaced by a newer judgement."""
    folder = (runs_dir.parent / "judgements") if runs_dir else JUDGEMENTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{run_a.tag}__vs__{run_b.tag}")
    path = folder / f"{safe}.json"
    path.write_text(
        json.dumps(
            {
                "baseline": run_a.tag,
                "candidate": run_b.tag,
                "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
                **asdict(result),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
