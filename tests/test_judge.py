"""Tests for llm_judge + score parsing.

Uses a mocked OpenAI client to avoid hitting the API in CI. Real-API
integration tests come in Week 7 dogfood.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from prompteval import LLMJudgeError, llm_judge


def _mock_client(response_text: str) -> Any:
    """Build a MagicMock that returns `response_text` from chat.completions.create."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=response_text))]
    client.chat.completions.create.return_value = completion
    return client


def test_parses_bare_number() -> None:
    client = _mock_client("0.8")
    assert llm_judge("rate it", "some text", client=client) == 0.8


def test_parses_integer() -> None:
    client = _mock_client("1")
    assert llm_judge("rate it", "some text", client=client) == 1.0


def test_parses_zero() -> None:
    client = _mock_client("0")
    assert llm_judge("rate it", "some text", client=client) == 0.0


def test_parses_number_with_leading_text() -> None:
    """Real LLMs often add 'Score: 0.7' even when told to return only a number."""
    client = _mock_client("Score: 0.7")
    assert llm_judge("rate it", "some text", client=client) == 0.7


def test_parses_first_number_when_multiple_present() -> None:
    client = _mock_client("0.5 (reasoning: confidence is 0.8)")
    assert llm_judge("rate it", "some text", client=client) == 0.5


def test_unparseable_response_raises() -> None:
    client = _mock_client("I think it's pretty good")
    with pytest.raises(LLMJudgeError, match="parse"):
        llm_judge("rate it", "some text", client=client)


def test_empty_response_raises() -> None:
    client = _mock_client("")
    with pytest.raises(LLMJudgeError, match="parse"):
        llm_judge("rate it", "some text", client=client)


def test_score_above_one_raises() -> None:
    client = _mock_client("1.5")
    with pytest.raises(LLMJudgeError, match=r"\[0, 1\]"):
        llm_judge("rate it", "some text", client=client)


def test_negative_score_raises() -> None:
    client = _mock_client("-0.3")
    with pytest.raises(LLMJudgeError, match=r"\[0, 1\]"):
        llm_judge("rate it", "some text", client=client)


def test_llm_judge_error_is_value_error() -> None:
    """LLMJudgeError extends ValueError so legacy `except ValueError:` catches it."""
    client = _mock_client("nope")
    with pytest.raises(ValueError):
        llm_judge("rate it", "some text", client=client)


def test_uses_specified_model() -> None:
    """The `model` kwarg should be forwarded to the OpenAI client call."""
    client = _mock_client("0.5")
    llm_judge("rate it", "some text", model="gpt-4o", client=client)
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"


def test_messages_carry_rubric_and_text() -> None:
    """The system message is the standard evaluator ground rules followed by the
    rubric (#23); the text being judged is the user message."""
    from prompteval.eval.judge import JUDGE_PREAMBLE

    client = _mock_client("0.5")
    llm_judge("custom rubric", "candidate text", client=client)
    call_kwargs = client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {
        "role": "system",
        "content": f"{JUDGE_PREAMBLE}\n\nRubric:\ncustom rubric",
    }
    assert "Length is not quality" in JUDGE_PREAMBLE
    assert messages[1] == {"role": "user", "content": "candidate text"}


def test_handles_none_content_gracefully() -> None:
    """OpenAI's SDK sometimes returns None for message.content (refusals, etc.)."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=None))]
    client.chat.completions.create.return_value = completion
    with pytest.raises(LLMJudgeError):
        llm_judge("rate it", "some text", client=client)


def test_judge_can_see_the_request_and_the_instructions() -> None:
    """#26: without them a judge cannot tell an invented fact from a given one."""
    client = _mock_client("0.5")
    llm_judge(
        "rubric",
        "Your Moon is in Taurus.",
        client=client,
        input="Moon sign: Cancer",
        instructions="Use only the chart data given.",
    )
    user = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "The assistant's instructions:\n<<<\nUse only the chart data given.\n>>>" in user
    assert "The user's request:\n<<<\nMoon sign: Cancer\n>>>" in user
    assert user.endswith("The reply to score:\n<<<\nYour Moon is in Taurus.\n>>>")


def test_scorers_can_ask_for_the_prompt(tmp_path: Any) -> None:
    from prompteval.eval.runner import Example, _call_scorer
    from prompteval.eval.scorer import scorer

    seen: dict[str, str] = {}

    @scorer
    def judged(output: str, input: str, prompt: str) -> float:
        seen.update(output=output, input=input, prompt=prompt)
        return 1.0

    ex = Example(id="a", input="Moon sign: Cancer", vars={"lang": "Hindi"})
    _call_scorer(judged, ex, "reply", "Answer in {lang}.")
    assert seen == {"output": "reply", "input": "Moon sign: Cancer", "prompt": "Answer in Hindi."}


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        # Real reply shape (Claude Sonnet, 2026-11-12): counting words first.
        ("Count words: reply is likely ~190 words. Fails criterion 6.\n\nFinal score: 0.4", 0.4),
        ("Reasoning: 3 of 4 rules met, 150 words.\n0.75", 0.75),
        ("Score: 0.7", 0.7),
        ("0.5 (reasoning: confidence is 0.8)", 0.5),
        ("Over by 40 words.\nScore = 1", 1.0),
    ],
)
def test_scores_are_read_from_where_the_judge_puts_them(reply: str, score: float) -> None:
    assert llm_judge("rate it", "text", client=_mock_client(reply)) == score


def test_a_judge_that_thinks_out_loud_and_ends_out_of_range_still_fails() -> None:
    with pytest.raises(LLMJudgeError):
        llm_judge("rate it", "text", client=_mock_client("190 words.\nIt deserves 7"))
