"""Pricing data + lookup helpers for the LLM models prompteval supports.

Pricing itself lives in `pricing.yaml` (data) — this module loads it at
import time and exposes a typed `PRICING` dict + helpers. The split keeps
the data table editable by non-Python contributors and lets us refresh
prices via a one-line YAML diff. When OpenAI announces a price change,
edit `pricing.yaml`, run pytest, ship.

## v1 schema vs v0.2 polymorphic refactor

The current `ModelPricing` shape (single `cached_input_per_1m` field) fits
OpenAI's auto-caching model cleanly. v0.2 will add Anthropic + open-source
providers, and Anthropic's cache pricing doesn't fit this shape — they use
explicit `cache_control` with three separate rates (write-5m, write-1h, read).

The planned v0.2 refactor is **per-provider pricing classes**:

    class Pricing(Protocol):
        name: str
        provider: str
        def cost_for(self, usage: Usage) -> CostBreakdown: ...

    @dataclass(frozen=True)
    class OpenAIPricing:    # input + cached_input + output
        ...

    @dataclass(frozen=True)
    class AnthropicPricing: # input + cache_write_5m + cache_write_1h + cache_read + output
        ...

    @dataclass(frozen=True)
    class SimplePricing:    # input + output (Groq, Together, Fireworks — no caching)
        ...

`compute_cost` becomes a 3-line dispatch on provider type. YAML schema grows
a nested `pricing:` block whose shape varies by provider.

**Don't try to model Anthropic by stuffing values into the current schema.**
Specifically: don't put Anthropic's `cache_read` price into
`cached_input_per_1m` — that silently mis-prices cache writes (which are
*more* expensive than uncached input on Anthropic, not cheaper).

The `provider` field added in v1 is the seam. Every entry declares its
provider today so v0.2 dispatch has something to switch on.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing for a single model, in USD per 1M tokens.

    Field names mirror what OpenAI publishes. The cache discount is encoded
    as an absolute `cached_input_per_1m` rather than a percentage because
    different model families use different discounts (gpt-4o: 50%, gpt-4.1:
    75%) — the absolute number is unambiguous and easier to verify.

    This dataclass is intentionally OpenAI-shaped for v0.1. The `provider`
    field exists today as the dispatch seam for v0.2's polymorphic refactor
    (see module docstring) — every entry must declare its provider even
    though only "openai" is supported right now.
    """

    name: str
    provider: str  # "openai" today; "anthropic" / "groq" / ... in v0.2
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float
    pricing_updated_at: str  # ISO date string; kept as text so YAML round-trips cleanly.
    notes: str | None = None


class UnknownModelError(KeyError):
    """Raised when a model name isn't in the pricing table.

    Extends KeyError so existing `except KeyError:` blocks keep working;
    the typed exception gives callers a precise catch when they want one.
    """


def _load_pricing() -> dict[str, ModelPricing]:
    """Load + validate pricing.yaml at import time.

    Validates shape (top-level `models` key, list-of-dicts) and per-row
    types (every field present, numeric where required). A bad YAML breaks
    package import — that's the right tradeoff: better than silently
    returning incomplete data later in the call stack.
    """
    raw_text = files("prompteval.cost").joinpath("pricing.yaml").read_text()
    data: Any = yaml.safe_load(raw_text)

    if not isinstance(data, dict) or "models" not in data:
        raise RuntimeError("pricing.yaml must have a top-level 'models' key")
    models = data["models"]
    if not isinstance(models, list):
        raise RuntimeError("pricing.yaml 'models' must be a list")

    out: dict[str, ModelPricing] = {}
    for entry in models:
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"pricing.yaml model entry must be a mapping, got {type(entry).__name__}"
            )
        try:
            pricing = ModelPricing(
                name=str(entry["name"]),
                provider=str(entry["provider"]),
                input_per_1m=float(entry["input_per_1m"]),
                cached_input_per_1m=float(entry["cached_input_per_1m"]),
                output_per_1m=float(entry["output_per_1m"]),
                pricing_updated_at=str(entry["pricing_updated_at"]),
                notes=str(entry["notes"]) if entry.get("notes") is not None else None,
            )
        except KeyError as missing:
            raise RuntimeError(
                f"pricing.yaml entry missing required field {missing.args[0]!r}: {entry!r}"
            ) from missing
        except (TypeError, ValueError) as err:
            raise RuntimeError(f"pricing.yaml entry has invalid types: {entry!r} ({err})") from err

        if pricing.name in out:
            raise RuntimeError(f"pricing.yaml has duplicate model name: {pricing.name!r}")
        out[pricing.name] = pricing

    return out


PRICING: dict[str, ModelPricing] = _load_pricing()


def get_pricing(model_name: str) -> ModelPricing:
    """Look up pricing for `model_name`. Exact match only — no fuzzy matching.

    Fuzzy matching would silently use the wrong pricing for a typo, which is
    worse than failing loud. The error message suggests near-matches (by
    alphanumeric comparison — catches the common "gpt4o" / "gpt-4o" typo)
    so fixing typos stays a one-step process.
    """
    if model_name in PRICING:
        return PRICING[model_name]

    needle = _normalize(model_name)
    candidates = sorted(m for m in PRICING if needle in _normalize(m) or _normalize(m) in needle)
    suggestion = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
    available = ", ".join(sorted(PRICING))
    raise UnknownModelError(f"Unknown model: {model_name!r}.{suggestion} Available: {available}.")


def _normalize(name: str) -> str:
    """Lowercase + strip non-alphanumeric chars for typo-tolerant suggestion matching.

    Only used for *suggestions* — the actual lookup is still exact. So `gpt4o`
    matches `gpt-4o` for the "Did you mean?" prompt, but won't silently resolve
    to that model.
    """
    return "".join(c for c in name.lower() if c.isalnum())


def list_models() -> list[ModelPricing]:
    """All known models, sorted by name."""
    return sorted(PRICING.values(), key=lambda m: m.name)
