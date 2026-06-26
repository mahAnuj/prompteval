"""Tests for compute_cost — the function the entire wedge depends on.

If this is wrong, every cost comparison prompteval reports is wrong. Test heavy.
"""

from __future__ import annotations

import pytest

from prompteval.cost import (
    CostBreakdown,
    UnknownModelError,
    Usage,
    compute_cost,
)


def test_known_cost_matches_hand_calculation() -> None:
    """gpt-4o, 1500 prompt (500 cached), 400 output → $0.007125 exactly.

    Hand-calc:
      uncached input: 1000 * $2.50  / 1M = $0.002500
      cached input:    500 * $1.25  / 1M = $0.000625
      output:          400 * $10.00 / 1M = $0.004000
      total                                = $0.007125
    """
    usage = Usage(prompt_tokens=1500, completion_tokens=400, cached_tokens=500)
    result = compute_cost("gpt-4o", usage)

    assert isinstance(result, CostBreakdown)
    assert result.total_cost == pytest.approx(0.007125, abs=1e-9)
    assert result.uncached_input_cost == pytest.approx(0.0025, abs=1e-9)
    assert result.cached_input_cost == pytest.approx(0.000625, abs=1e-9)
    assert result.output_cost == pytest.approx(0.004, abs=1e-9)


def test_zero_tokens_is_zero_cost() -> None:
    """An empty call (e.g. a connection probe) should cost nothing, not crash."""
    result = compute_cost("gpt-4o-mini", Usage(prompt_tokens=0, completion_tokens=0))
    assert result.total_cost == 0.0


def test_no_cache_means_no_cache_cost() -> None:
    """Default `cached_tokens=0` means full input price applies."""
    usage = Usage(prompt_tokens=1000, completion_tokens=200)
    result = compute_cost("gpt-4o", usage)

    assert result.cached_input_tokens == 0
    assert result.cached_input_cost == 0.0
    assert result.uncached_input_tokens == 1000


def test_all_cached_is_cheaper_than_none_cached() -> None:
    """The whole point of caching: identical usage with cache on costs less."""
    full = Usage(prompt_tokens=10_000, completion_tokens=500, cached_tokens=0)
    cached = Usage(prompt_tokens=10_000, completion_tokens=500, cached_tokens=10_000)

    full_cost = compute_cost("gpt-4o", full).total_cost
    cached_cost = compute_cost("gpt-4o", cached).total_cost

    assert cached_cost < full_cost


def test_cache_discount_matches_pricing_table() -> None:
    """For gpt-4o (50% discount), 10000 cached input should cost half what
    10000 uncached input would. Validates the math, not just the relative order."""
    cached = Usage(prompt_tokens=10_000, completion_tokens=0, cached_tokens=10_000)
    uncached = Usage(prompt_tokens=10_000, completion_tokens=0, cached_tokens=0)

    cached_cost = compute_cost("gpt-4o", cached).total_cost
    uncached_cost = compute_cost("gpt-4o", uncached).total_cost

    assert cached_cost == pytest.approx(uncached_cost * 0.50, rel=1e-6)


def test_gpt41_uses_75_percent_cache_discount() -> None:
    """gpt-4.1 has a steeper cache discount than gpt-4o — guards against
    silent pricing-table flips."""
    cached = Usage(prompt_tokens=10_000, completion_tokens=0, cached_tokens=10_000)
    uncached = Usage(prompt_tokens=10_000, completion_tokens=0, cached_tokens=0)

    cached_cost = compute_cost("gpt-4.1", cached).total_cost
    uncached_cost = compute_cost("gpt-4.1", uncached).total_cost

    # 75% off = pays 25% of input price
    assert cached_cost == pytest.approx(uncached_cost * 0.25, rel=1e-6)


def test_cost_monotonic_in_prompt_tokens() -> None:
    """Holding everything else fixed, more input tokens => higher cost."""
    costs = [
        compute_cost("gpt-4o", Usage(prompt_tokens=n, completion_tokens=100)).total_cost
        for n in (0, 100, 500, 1000, 5000)
    ]
    assert costs == sorted(costs), costs


def test_cost_monotonic_in_completion_tokens() -> None:
    costs = [
        compute_cost("gpt-4o", Usage(prompt_tokens=500, completion_tokens=n)).total_cost
        for n in (0, 50, 200, 1000)
    ]
    assert costs == sorted(costs), costs


def test_reasoning_tokens_are_surfaced_but_dont_change_cost() -> None:
    """Reasoning tokens are a subset of completion tokens — they're already
    billed at the output rate. Surfaced for transparency, not for double-billing."""
    no_reasoning = Usage(prompt_tokens=100, completion_tokens=200, reasoning_tokens=0)
    some_reasoning = Usage(prompt_tokens=100, completion_tokens=200, reasoning_tokens=80)

    r1 = compute_cost("gpt-4o", no_reasoning)
    r2 = compute_cost("gpt-4o", some_reasoning)

    assert r1.total_cost == r2.total_cost
    assert r1.reasoning_tokens == 0
    assert r2.reasoning_tokens == 80


def test_negative_prompt_tokens_raises() -> None:
    with pytest.raises(ValueError, match="prompt_tokens"):
        compute_cost("gpt-4o", Usage(prompt_tokens=-1, completion_tokens=0))


def test_negative_completion_tokens_raises() -> None:
    with pytest.raises(ValueError, match="completion_tokens"):
        compute_cost("gpt-4o", Usage(prompt_tokens=0, completion_tokens=-1))


def test_negative_cached_tokens_raises() -> None:
    with pytest.raises(ValueError, match="cached_tokens"):
        compute_cost("gpt-4o", Usage(prompt_tokens=0, completion_tokens=0, cached_tokens=-1))


def test_cached_exceeding_prompt_raises() -> None:
    """An impossible state — guards against an integration bug downstream."""
    with pytest.raises(ValueError, match="cannot exceed"):
        compute_cost("gpt-4o", Usage(prompt_tokens=100, completion_tokens=0, cached_tokens=200))


def test_reasoning_exceeding_completion_raises() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        compute_cost("gpt-4o", Usage(prompt_tokens=0, completion_tokens=100, reasoning_tokens=200))


def test_unknown_model_raises() -> None:
    """Cost compute should bubble UnknownModelError verbatim — no silent default."""
    with pytest.raises(UnknownModelError):
        compute_cost("nonexistent-model", Usage(prompt_tokens=10, completion_tokens=5))


def test_breakdown_token_counts_match_input() -> None:
    """Defensive: the surfaced token counts in the breakdown should match the input."""
    usage = Usage(prompt_tokens=1234, completion_tokens=567, cached_tokens=234, reasoning_tokens=12)
    result = compute_cost("gpt-4o-mini", usage)

    assert result.cached_input_tokens == 234
    assert result.uncached_input_tokens == 1000  # 1234 - 234
    assert result.output_tokens == 567
    assert result.reasoning_tokens == 12
    assert result.model == "gpt-4o-mini"
