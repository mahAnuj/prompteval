"""Compute the USD cost of an LLM call from its usage stats.

Pure function. Inputs are a model name + a `Usage` record (the tokens
breakdown OpenAI returns in `completion.usage`). Output is a `CostBreakdown`
that surfaces every component cost separately so the comparison reporter
can show where the money went.

Cached input is a subset of `prompt_tokens` (not a separate count). We
charge `cached_tokens` at `cached_input_per_1m` and the remainder at
`input_per_1m`. Reasoning tokens (o-series) are a subset of
`completion_tokens` — they're already billed at the output rate; we surface
them as a separate counter for transparency only, not for billing.

Floats throughout. For v0.1 we're computing costs in cents, not aggregating
across millions of calls, so float precision is adequate. If we ever need
exact-cent aggregation at scale, swap to `decimal.Decimal` here.

## v0.2 dispatch refactor

This function currently assumes OpenAI-shaped pricing (one `cached_input_per_1m`
rate). When v0.2 adds Anthropic + open-source providers, the planned refactor is:

    def compute_cost(model: str, usage: Usage) -> CostBreakdown:
        pricing = get_pricing(model)
        match pricing.provider:
            case "openai":
                return _compute_openai(pricing, usage)
            case "anthropic":
                return _compute_anthropic(pricing, usage)  # different Usage shape
            case "openai-compatible":
                return _compute_simple(pricing, usage)     # no caching
            case _:
                raise ValueError(f"Unknown provider {pricing.provider!r}")

`Usage` itself may need to grow Anthropic-specific fields
(`cache_write_5m_input_tokens`, etc.) or split into per-provider classes.
See `cost/models.py` module docstring for the broader v0.2 design.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompteval.cost.models import get_pricing


@dataclass(frozen=True)
class Usage:
    """Token counts from a single LLM call.

    Mirrors OpenAI's usage shape:
      - prompt_tokens: total input
      - cached_tokens: subset of prompt that hit the prompt cache (default 0)
      - completion_tokens: total output (including reasoning, for o-series)
      - reasoning_tokens: subset of completion that was thinking-tokens
        (default 0, informational only)
    """

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class CostBreakdown:
    """Per-component cost breakdown for a single LLM call.

    `total_cost` is the sum of the three cost components. Token counts are
    surfaced so the reporter can show "where the money went" without
    re-doing math.
    """

    model: str
    uncached_input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    uncached_input_cost: float
    cached_input_cost: float
    output_cost: float
    total_cost: float


def compute_cost(model: str, usage: Usage) -> CostBreakdown:
    """Compute the USD cost of an LLM call.

    Pure function. Validates that token counts are non-negative and that
    `cached_tokens` / `reasoning_tokens` don't exceed their parent counts.
    Raises `UnknownModelError` (from `cost.models`) if the model isn't in
    the pricing table.
    """
    _validate_usage(usage)
    pricing = get_pricing(model)

    uncached_input = usage.prompt_tokens - usage.cached_tokens
    uncached_input_cost = uncached_input * pricing.input_per_1m / 1_000_000
    cached_input_cost = usage.cached_tokens * pricing.cached_input_per_1m / 1_000_000
    output_cost = usage.completion_tokens * pricing.output_per_1m / 1_000_000
    total = uncached_input_cost + cached_input_cost + output_cost

    return CostBreakdown(
        model=model,
        uncached_input_tokens=uncached_input,
        cached_input_tokens=usage.cached_tokens,
        output_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        uncached_input_cost=uncached_input_cost,
        cached_input_cost=cached_input_cost,
        output_cost=output_cost,
        total_cost=total,
    )


def _validate_usage(usage: Usage) -> None:
    """Fail loudly on impossible token combinations.

    All four counts must be non-negative. `cached_tokens` is a subset of
    `prompt_tokens`, `reasoning_tokens` a subset of `completion_tokens`.
    """
    if usage.prompt_tokens < 0:
        raise ValueError(f"prompt_tokens must be >= 0, got {usage.prompt_tokens}")
    if usage.completion_tokens < 0:
        raise ValueError(f"completion_tokens must be >= 0, got {usage.completion_tokens}")
    if usage.cached_tokens < 0:
        raise ValueError(f"cached_tokens must be >= 0, got {usage.cached_tokens}")
    if usage.reasoning_tokens < 0:
        raise ValueError(f"reasoning_tokens must be >= 0, got {usage.reasoning_tokens}")
    if usage.cached_tokens > usage.prompt_tokens:
        raise ValueError(
            f"cached_tokens ({usage.cached_tokens}) cannot exceed "
            f"prompt_tokens ({usage.prompt_tokens})"
        )
    if usage.reasoning_tokens > usage.completion_tokens:
        raise ValueError(
            f"reasoning_tokens ({usage.reasoning_tokens}) cannot exceed "
            f"completion_tokens ({usage.completion_tokens})"
        )
