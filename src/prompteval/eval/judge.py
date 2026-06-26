"""LLM-as-judge helper — `llm_judge(rubric, text)` returns a float in [0, 1].

Used inside scorer functions when quality can't be checked deterministically.
Always prefer a deterministic scorer if you can write one (exact_match, regex,
JSON-schema-validates) — deterministic scorers are cheaper, faster, and free
of judge noise.

## Cost note (v0.1 limitation)

Judge calls cost real money but are NOT aggregated into the eval's main cost
report in v0.1 — they happen inside scorer functions, separate from the
prompt-evaluation calls we're comparing. v0.2 will add judge-cost separation
(report shows "prompt cost: $X, judge overhead: $Y"). For now, treat judge
calls as eval-time overhead, not part of the v1 vs v2 comparison.

## Mocking in tests

The OpenAI client is constructed lazily and can be injected via `client=`.
Tests pass a MagicMock with a canned response to verify parsing without
hitting the API.
"""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

# Single shared client to avoid reopening connections for every judge call.
# Lazy — only constructed when first needed so importing prompteval doesn't
# require OPENAI_API_KEY in the environment.
_DEFAULT_CLIENT: OpenAI | None = None


class LLMJudgeError(ValueError):
    """Raised when llm_judge can't parse a [0, 1] score from the judge response.

    Subclasses ValueError so existing `except ValueError:` blocks catch it,
    while letting precise handlers distinguish judge failures from other
    validation errors.
    """


def llm_judge(
    rubric: str,
    text: str,
    *,
    model: str = "gpt-4o-mini",
    client: Any = None,
) -> float:
    """Ask an LLM to score `text` against `rubric`. Returns a float in [0, 1].

    The rubric should instruct the judge to return ONLY a number. We parse the
    first number found in the response (tolerates a leading "Score: 0.8" or
    similar). Numbers outside [0, 1] raise `LLMJudgeError` — the rubric is
    almost certainly miscalibrated.

    `client` is left untyped (Any) because mypy chokes on MagicMock injection
    in tests otherwise. Real usage passes a real OpenAI client or None.

    `model` defaults to gpt-4o-mini — cheap enough that judge overhead stays
    negligible (~$0.0001 per typical judge call). Override if you want a
    stronger judge for high-stakes rubrics.
    """
    if client is None:
        client = _get_default_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _parse_score(raw)


def _get_default_client() -> OpenAI:
    """Lazily construct the shared OpenAI client. Reads OPENAI_API_KEY from env."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OpenAI()
    return _DEFAULT_CLIENT


def _parse_score(raw: str) -> float:
    """Extract the first float in `raw` and validate it's in [0, 1].

    Tolerates rubric responses like "0.8", "Score: 0.8", "0.8 — reason: ...".
    Negative numbers parse as a number but fail the range check, so a
    response of "-1" produces an LLMJudgeError, not a silent zero.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if match is None:
        raise LLMJudgeError(
            f"Could not parse a number from judge response: {raw!r}. "
            "Check that your rubric instructs the judge to return only a number."
        )
    value = float(match.group())
    if not 0.0 <= value <= 1.0:
        raise LLMJudgeError(
            f"Judge returned {value}, expected a score in [0, 1]. "
            f"Raw response: {raw!r}. Rubric may be miscalibrated."
        )
    return value
