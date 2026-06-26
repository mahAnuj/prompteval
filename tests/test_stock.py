"""Tests for the stock scorers in prompteval.eval.stock.

Deterministic scorers tested directly. LLM-as-judge scorers use a mocked client
(same pattern as test_judge.py) to avoid the API.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompteval.eval import stock
from prompteval.eval.scorer import is_scorer

# ---------------------------------------------------------------------------
# Module-level sanity
# ---------------------------------------------------------------------------


EXPECTED_STOCK_NAMES = {
    "exact_match",
    "contains_substring",
    "regex_match",
    "not_empty",
    "length_between",
    "json_schema_valid",
    "mentions_required_terms",
    "llm_judge_tone",
    "llm_judge_factuality",
}


def test_stock_module_exposes_expected_scorers() -> None:
    """Catches accidental rename / deletion of a stock scorer (which would
    silently break `prompteval scorer copy <name>` for existing users)."""
    actual = {name for name in dir(stock) if is_scorer(getattr(stock, name))}
    assert actual == EXPECTED_STOCK_NAMES


def test_every_stock_scorer_has_docstring() -> None:
    """The CLI's `scorer list` shows the first line of each docstring —
    blank docstrings would print blank descriptions, bad UX."""
    for name in EXPECTED_STOCK_NAMES:
        func = getattr(stock, name)
        assert func.__doc__, f"{name} has no docstring"


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_matches() -> None:
    assert stock.exact_match("yes", {"expected": "yes"}) == 1.0


def test_exact_match_differs() -> None:
    assert stock.exact_match("YES", {"expected": "yes"}) == 0.0


def test_exact_match_missing_expected_is_zero() -> None:
    assert stock.exact_match("anything", {}) == 0.0


# ---------------------------------------------------------------------------
# contains_substring
# ---------------------------------------------------------------------------


def test_contains_substring_found() -> None:
    assert stock.contains_substring("the quick brown fox", {"substring": "quick"}) == 1.0


def test_contains_substring_missing() -> None:
    assert stock.contains_substring("the quick brown fox", {"substring": "slow"}) == 0.0


def test_contains_substring_empty_needle() -> None:
    """Empty needle returns 0 (instead of vacuously True) — defensive."""
    assert stock.contains_substring("anything", {"substring": ""}) == 0.0
    assert stock.contains_substring("anything", {}) == 0.0


# ---------------------------------------------------------------------------
# regex_match
# ---------------------------------------------------------------------------


def test_regex_match_basic() -> None:
    assert stock.regex_match("user@example.com", {"pattern": r"\S+@\S+"}) == 1.0


def test_regex_match_no_match() -> None:
    assert stock.regex_match("no email here", {"pattern": r"\S+@\S+"}) == 0.0


def test_regex_match_missing_pattern() -> None:
    assert stock.regex_match("whatever", {}) == 0.0


# ---------------------------------------------------------------------------
# not_empty
# ---------------------------------------------------------------------------


def test_not_empty_true() -> None:
    assert stock.not_empty("hello") == 1.0


def test_not_empty_false_for_empty() -> None:
    assert stock.not_empty("") == 0.0


def test_not_empty_false_for_whitespace_only() -> None:
    assert stock.not_empty("   \n\t  ") == 0.0


# ---------------------------------------------------------------------------
# length_between
# ---------------------------------------------------------------------------


def test_length_between_in_range() -> None:
    assert stock.length_between("hello", {"min_len": 3, "max_len": 10}) == 1.0


def test_length_between_below_min() -> None:
    assert stock.length_between("hi", {"min_len": 3, "max_len": 10}) == 0.0


def test_length_between_above_max() -> None:
    assert stock.length_between("a" * 100, {"min_len": 3, "max_len": 10}) == 0.0


def test_length_between_unbounded_upper() -> None:
    """Missing max_len => no upper bound."""
    assert stock.length_between("a" * 10_000, {"min_len": 1}) == 1.0


# ---------------------------------------------------------------------------
# json_schema_valid
# ---------------------------------------------------------------------------


def test_json_schema_valid_all_keys_present() -> None:
    output = json.dumps({"name": "Alice", "email": "a@b.com", "age": 30})
    assert stock.json_schema_valid(output, {"required_keys": ["name", "email"]}) == 1.0


def test_json_schema_valid_missing_key() -> None:
    output = json.dumps({"name": "Alice"})
    assert stock.json_schema_valid(output, {"required_keys": ["name", "email"]}) == 0.0


def test_json_schema_valid_invalid_json() -> None:
    assert stock.json_schema_valid("not json {{", {"required_keys": []}) == 0.0


def test_json_schema_valid_array_at_top_level() -> None:
    """A top-level array isn't a dict, so the has-these-keys check returns 0."""
    assert stock.json_schema_valid("[1, 2, 3]", {"required_keys": []}) == 0.0


# ---------------------------------------------------------------------------
# mentions_required_terms
# ---------------------------------------------------------------------------


def test_mentions_required_terms_all_present() -> None:
    score = stock.mentions_required_terms(
        "Yes, our 30-day refund applies to broken items.",
        {"must_mention": ["30-day refund", "broken"]},
    )
    assert score == 1.0


def test_mentions_required_terms_partial() -> None:
    score = stock.mentions_required_terms(
        "We can issue a refund.",
        {"must_mention": ["30-day refund", "broken"]},
    )
    assert score == 0.0


def test_mentions_required_terms_half_credit() -> None:
    score = stock.mentions_required_terms(
        "Our 30-day refund covers many cases.",
        {"must_mention": ["30-day refund", "broken"]},
    )
    assert score == 0.5


def test_mentions_required_terms_empty_list_is_vacuous_pass() -> None:
    """Nothing required => nothing to fail => 1.0."""
    assert stock.mentions_required_terms("anything", {"must_mention": []}) == 1.0
    assert stock.mentions_required_terms("anything", {}) == 1.0


def test_mentions_required_terms_case_insensitive() -> None:
    score = stock.mentions_required_terms(
        "Our 30-DAY REFUND policy covers BROKEN items.",
        {"must_mention": ["30-day refund", "broken"]},
    )
    assert score == 1.0


# ---------------------------------------------------------------------------
# LLM-as-judge scorers (mocked)
# ---------------------------------------------------------------------------


def _mock_client(response_text: str) -> Any:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=response_text))]
    client.chat.completions.create.return_value = completion
    return client


def test_llm_judge_tone_returns_parsed_score() -> None:
    """Patch the underlying llm_judge so we don't construct a default OpenAI client."""
    with patch("prompteval.eval.stock.llm_judge", return_value=0.85) as mock_judge:
        score = stock.llm_judge_tone(
            "I'm so sorry to hear that — let me help.",
            {"tone": "empathetic"},
        )
    assert score == 0.85
    # Confirm the rubric mentions the requested tone
    rubric = mock_judge.call_args.kwargs["rubric"]
    assert "empathetic" in rubric


def test_llm_judge_tone_defaults_to_professional() -> None:
    with patch("prompteval.eval.stock.llm_judge", return_value=0.5) as mock_judge:
        stock.llm_judge_tone("whatever", {})
    assert "professional" in mock_judge.call_args.kwargs["rubric"]


def test_llm_judge_factuality_returns_score() -> None:
    with patch("prompteval.eval.stock.llm_judge", return_value=0.9) as mock_judge:
        score = stock.llm_judge_factuality(
            "The Earth is approximately 4.5 billion years old.",
            {"reference": "Earth is 4.5 billion years old."},
        )
    assert score == 0.9
    rubric = mock_judge.call_args.kwargs["rubric"]
    assert "Earth is 4.5 billion years old" in rubric


def test_llm_judge_factuality_returns_zero_when_no_reference() -> None:
    """Defensively conservative: no reference => can't be factually consistent."""
    with patch("prompteval.eval.stock.llm_judge") as mock_judge:
        score = stock.llm_judge_factuality("anything", {})
    assert score == 0.0
    # And the judge should NOT have been called — saves a real API hit.
    mock_judge.assert_not_called()


@pytest.mark.parametrize(
    "name",
    sorted(EXPECTED_STOCK_NAMES),
)
def test_every_stock_scorer_is_marked_as_scorer(name: str) -> None:
    """Belt-and-braces — every stock fn must be decorated, or the CLI won't find it."""
    func = getattr(stock, name)
    assert is_scorer(func), f"{name} is missing @scorer"
