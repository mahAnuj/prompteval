"""Stock scorers shipped with prompteval — copy-paste starters for common checks.

Each scorer is intentionally short and editable. Users get a scorer's source via:

    prompteval scorer list           # shows all of them
    prompteval scorer copy <name>    # prints source to stdout

Then they paste it into their own `evals/eval.py` and tune for their domain.

## Categories

Deterministic (cheap, reproducible — always prefer these when you can):
  - exact_match
  - contains_substring
  - regex_match
  - not_empty
  - length_between
  - json_schema_valid
  - mentions_required_terms

LLM-as-judge (slower, costs money, opt-in noise but sometimes the only option):
  - llm_judge_tone
  - llm_judge_factuality

## Convention

Scorers return a float in [0, 1] — 1.0 = perfect match, 0.0 = total miss.
Partial-credit scorers (mentions_required_terms) return the fraction matched.
All scorers gracefully handle missing `expected` keys by returning the most
conservative score (usually 0.0 or 1.0, documented per-scorer).
"""

from __future__ import annotations

import json
import re
from typing import Any

from prompteval.eval.judge import llm_judge
from prompteval.eval.scorer import scorer

# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


@scorer
def exact_match(output: str, expected: dict[str, Any]) -> float:
    """1.0 iff `output` equals `expected["expected"]` (exact byte-for-byte match).

    expected: {"expected": "the exact expected string"}
    Returns 0.0 if `expected["expected"]` is missing — strict by design.
    """
    target = expected.get("expected")
    if target is None:
        return 0.0
    return 1.0 if output == target else 0.0


@scorer
def contains_substring(output: str, expected: dict[str, Any]) -> float:
    """1.0 iff `expected["substring"]` appears in `output` (case-sensitive).

    expected: {"substring": "phrase to find"}
    Returns 0.0 if `expected["substring"]` is missing.
    """
    needle = expected.get("substring", "")
    return 1.0 if needle and needle in output else 0.0


@scorer
def regex_match(output: str, expected: dict[str, Any]) -> float:
    """1.0 iff `expected["pattern"]` (a regex) matches anywhere in `output`.

    expected: {"pattern": r"some regex"}
    Returns 0.0 if `expected["pattern"]` is missing or doesn't match.
    """
    pattern = expected.get("pattern")
    if not pattern:
        return 0.0
    return 1.0 if re.search(pattern, output) else 0.0


@scorer
def not_empty(output: str) -> float:
    """1.0 iff `output` is non-empty after stripping whitespace.

    Useful as a cheap sanity floor — if the LLM returned nothing, no other
    scorer is meaningful. No expected payload required.
    """
    return 1.0 if output.strip() else 0.0


@scorer
def length_between(output: str, expected: dict[str, Any]) -> float:
    """1.0 iff `len(output)` is in `[expected["min_len"], expected["max_len"]]`.

    expected: {"min_len": 50, "max_len": 500}
    Missing keys default to 0 / infinity respectively — opt-in bounds.
    """
    min_len = expected.get("min_len", 0)
    max_len = expected.get("max_len", float("inf"))
    return 1.0 if min_len <= len(output) <= max_len else 0.0


@scorer
def json_schema_valid(output: str, expected: dict[str, Any]) -> float:
    """1.0 iff `output` is valid JSON and contains every key in `expected["required_keys"]`.

    expected: {"required_keys": ["name", "email"]}
    Lightweight has-these-keys check — not full JSON Schema validation. For
    nested-shape validation, write a custom scorer using the `jsonschema`
    package (out of scope for this stock scorer).
    """
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    required = expected.get("required_keys", [])
    return 1.0 if all(k in parsed for k in required) else 0.0


@scorer
def mentions_required_terms(output: str, expected: dict[str, Any]) -> float:
    """Fraction of `expected["must_mention"]` terms found in `output` (case-insensitive).

    expected: {"must_mention": ["term1", "term2"]}
    Returns 1.0 if no terms required (vacuously satisfied). Partial credit
    by design — useful when you want "covered 3 of 4 points" feedback rather
    than pass/fail.
    """
    must = expected.get("must_mention", [])
    if not must:
        return 1.0
    text = output.lower()
    return sum(1 for term in must if term.lower() in text) / len(must)


# ---------------------------------------------------------------------------
# LLM-as-judge scorers — opt in to judge cost + noise
# ---------------------------------------------------------------------------


@scorer
def llm_judge_tone(output: str, expected: dict[str, Any]) -> float:
    """LLM rates how well `output`'s tone matches `expected["tone"]`. 0-1.

    expected: {"tone": "empathetic"}  # or "professional", "concise", etc.

    Defaults to "professional" if no tone is specified. Tune the rubric for
    your domain — generic rubrics produce generic scores.
    """
    target_tone = expected.get("tone", "professional")
    rubric = (
        f"Rate how well this text matches a {target_tone} tone on a 0-1 scale. "
        "1 = perfect match. 0 = opposite tone. "
        "Return only the number, nothing else."
    )
    return llm_judge(rubric=rubric, text=output)


@scorer
def llm_judge_factuality(output: str, expected: dict[str, Any]) -> float:
    """LLM rates how factually consistent `output` is with `expected["reference"]`. 0-1.

    expected: {"reference": "the source-of-truth text the output should match"}

    Useful for RAG-style evals where you have a known-correct reference passage
    and want to check the LLM's summary doesn't contradict it. Returns 0.0
    if no reference is given (defensively conservative).
    """
    reference = expected.get("reference")
    if not reference:
        return 0.0
    rubric = (
        "You are a factual-consistency judge. Given a REFERENCE statement and a "
        "CANDIDATE statement, return a score from 0 to 1 representing how factually "
        "consistent CANDIDATE is with REFERENCE. 1 = fully consistent, 0 = contradicts. "
        f"Return only the number, nothing else.\n\nREFERENCE: {reference}\n"
    )
    return llm_judge(rubric=rubric, text=output)
