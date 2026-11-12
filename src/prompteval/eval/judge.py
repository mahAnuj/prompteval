"""LLM-as-judge helper — `llm_judge(rubric, text)` returns a float in [0, 1].

Used inside scorer functions when quality can't be checked deterministically.
Always prefer a deterministic scorer if you can write one (exact_match, regex,
JSON-schema-validates) — deterministic scorers are cheaper, faster, and free
of judge noise.

## Cost

Judge calls cost real money. Inside a run, each call's token usage is recorded
(`JUDGE_CALLS`) and priced as that example's `judge_cost`, reported as its own
line ("scoring overhead") and kept out of the prompt-vs-prompt cost verdict,
which compares only the prompt under test (#15).

## Mocking in tests

The OpenAI client is constructed lazily and can be injected via `client=`.
Tests pass a MagicMock with a canned response to verify parsing without
hitting the API.
"""

from __future__ import annotations

import contextlib
import re
from contextvars import ContextVar
from typing import Any

from openai import OpenAI

# Single shared client to avoid reopening connections for every judge call.
# Lazy — only constructed when first needed so importing prompteval doesn't
# require OPENAI_API_KEY in the environment.
_DEFAULT_CLIENT: OpenAI | None = None

#: Judge calls made while scoring one example: (model, prompt, completion,
#: cached) tokens. The runner opens a fresh list around each example's scorers
#: and prices it as that example's judge cost (#15); outside a run it is None
#: and nothing is recorded.
#: Ground rules every rubric starts from (#23): what the judge may and may not use.
JUDGE_PREAMBLE = (
    "You are a strict, fair evaluator. Score the single reply you are given against the "
    "rubric below and nothing else. Use only the information you are given (the rubric, "
    "and the assistant's instructions and the user's request when shown, and the reply): "
    "do not assume facts that are not there, and treat anything the reply states that "
    "they contradict, or that appears in none of them, as an error. Length is not quality. "
    "Reply with only a number between 0 and 1."
)


def judge_message(text: str, input: str | None = None, instructions: str | None = None) -> str:
    """What the per-reply judge reads (#26): the reply alone, or with the request and
    the instructions it answered, so a rule like "uses only the data given" is checkable."""
    if not input and not instructions:
        return text
    parts = []
    if instructions:
        parts.append(f"The assistant's instructions:\n<<<\n{instructions.strip()}\n>>>")
    if input:
        parts.append(f"The user's request:\n<<<\n{input.strip()}\n>>>")
    parts.append(f"The reply to score:\n<<<\n{text}\n>>>")
    return "\n\n".join(parts)


#: (model, prompt tokens, completion tokens, cached tokens, reported cost or None)
JudgeCall = tuple[str, int, int, int, float | None]

JUDGE_CALLS: ContextVar[list[JudgeCall] | None] = ContextVar("prompteval_judge_calls", default=None)


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
    temperature: float | None = 0.0,
    input: str | None = None,
    instructions: str | None = None,
) -> float:
    """Ask an LLM to score `text` against `rubric`. Returns a float in [0, 1].

    The rubric should instruct the judge to return ONLY a number. We parse the
    first number found in the response (tolerates a leading "Score: 0.8" or
    similar). Numbers outside [0, 1] raise `LLMJudgeError` — the rubric is
    almost certainly miscalibrated.

    `client` is left untyped (Any) because mypy chokes on MagicMock injection
    in tests otherwise. Real usage passes a real OpenAI client or None.

    Pass `input` (the user's request) and `instructions` (the prompt) whenever
    the rubric depends on them (#26): without them the judge sees the reply
    alone and cannot tell an invented fact from a given one. In a scorer, take
    both as parameters: `def judged(output, input, prompt): ...`.

    `model` defaults to gpt-4o-mini — cheap enough that judge overhead stays
    negligible (~$0.0001 per typical judge call). Override if you want a
    stronger judge for high-stakes rubrics.
    """
    if client is None:
        client = _get_default_client()

    # Temperature 0 by default (#14): a judge that samples adds its own noise
    # to every score. Pass temperature=None for the provider's default (some
    # reasoning models accept no other value).
    extra = {} if temperature is None else {"temperature": temperature}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{JUDGE_PREAMBLE}\n\nRubric:\n{rubric}"},
            {"role": "user", "content": judge_message(text, input, instructions)},
        ],
        **extra,
    )
    _record_usage(model, response)
    raw = response.choices[0].message.content or ""
    return _parse_score(raw)


def _record_usage(model: str, response: Any) -> None:
    calls = JUDGE_CALLS.get()
    usage = getattr(response, "usage", None)
    if calls is None or usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    # A response without usable token counts is simply not counted.
    with contextlib.suppress(TypeError, ValueError):
        reported = getattr(response, "reported_cost_usd", None)
        calls.append(
            (
                model,
                int(usage.prompt_tokens),
                int(usage.completion_tokens),
                int(cached or 0),
                float(reported) if isinstance(reported, (int, float)) else None,
            )
        )


def _get_default_client() -> OpenAI:
    """Lazily construct the shared OpenAI client. Reads OPENAI_API_KEY from env."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OpenAI()
    return _DEFAULT_CLIENT


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_LABELLED = re.compile(r"(?i)\b(?:final\s+)?score\s*(?:is|=|:)?\s*\**\s*(-?\d+(?:\.\d+)?)")


def _parse_score(raw: str) -> float:
    """Find the judge's score in `raw` and validate it's in [0, 1].

    Tolerates "0.8", "Score: 0.8", "0.8 — reason: ...". Judges that think out
    loud first ("reply is ~190 words ... Final score: 0.4") are read from the
    end, where the score is, not from the first number they mention (#26:
    found in a real run, where 190 was taken as the score). In order: a
    labelled "score: x" (the last one), else the first number on the last
    line that has one, when it is in [0, 1], else the first number anywhere.
    A negative or >1 result is an LLMJudgeError, never a silent clamp.
    """
    labelled = [float(m.group(1)) for m in _LABELLED.finditer(raw)]
    lines = [line for line in raw.strip().splitlines() if _NUMBER.search(line)]
    last_line = float(_NUMBER.findall(lines[-1])[0]) if len(lines) > 1 else None
    first = _NUMBER.search(raw)
    if labelled:
        value = labelled[-1]
    elif last_line is not None and 0.0 <= last_line <= 1.0:
        value = last_line
    elif first is not None:
        value = float(first.group())
    else:
        raise LLMJudgeError(
            f"Could not parse a number from judge response: {raw!r}. "
            "Check that your rubric instructs the judge to return only a number."
        )
    if not 0.0 <= value <= 1.0:
        raise LLMJudgeError(
            f"Judge returned {value}, expected a score in [0, 1]. "
            f"Raw response: {raw!r}. Rubric may be miscalibrated."
        )
    return value
