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

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
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
    raise UnknownModelError(
        f"Unknown model: {model_name!r}.{suggestion} Available: {available}. "
        f"For any other model (Groq, Together, OpenRouter, Ollama, a newer OpenAI model), "
        f"give its price: --price input=X,output=Y (USD per 1M tokens), or add it to "
        f"prices.yaml next to eval.py."
    )


def resolve_pricing(
    model_name: str, custom: Mapping[str, ModelPricing] | None = None
) -> ModelPricing:
    """User-supplied prices first, then the built-in table (#8)."""
    if custom and model_name in custom:
        return custom[model_name]
    return get_pricing(model_name)


def parse_price_spec(model_name: str, spec: str) -> ModelPricing:
    """`input=0.59,output=0.79[,cached=0.30]` (USD per 1M tokens) → ModelPricing.

    `cached` defaults to the input price: no cache discount unless stated.
    """
    values: dict[str, float] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        key, sep, raw = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in {"input", "output", "cached"}:
            raise ValueError(
                f"Invalid price part {part.strip()!r}: expected input=X, output=Y "
                f"and optionally cached=Z (USD per 1M tokens)."
            )
        try:
            value = float(raw)
        except ValueError as err:
            raise ValueError(f"Invalid price {raw.strip()!r} for {key!r}.") from err
        if value < 0:
            raise ValueError(f"Price for {key!r} cannot be negative.")
        values[key] = value
    if "input" not in values or "output" not in values:
        raise ValueError("A price needs both input=X and output=Y (USD per 1M tokens).")
    return ModelPricing(
        name=model_name,
        provider="custom",
        input_per_1m=values["input"],
        cached_input_per_1m=values.get("cached", values["input"]),
        output_per_1m=values["output"],
        pricing_updated_at="user-supplied",
        notes="--price",
    )


@dataclass(frozen=True)
class ModelEndpoint:
    """Where a model lives, from a user prices.yaml (`base_url`, `api_key_env`)."""

    base_url: str | None = None
    api_key_env: str | None = None


def load_model_endpoints(path: str | Path) -> dict[str, ModelEndpoint]:
    """Per-model endpoints from a prices.yaml, for models that declare one:

    models:
      - name: llama-3.3-70b-versatile
        input_per_1m: 0.59
        output_per_1m: 0.79
        base_url: https://api.groq.com/openai/v1
        api_key_env: GROQ_API_KEY
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    entries = raw.get("models") if isinstance(raw, dict) else None
    out: dict[str, ModelEndpoint] = {}
    for entry in entries if isinstance(entries, list) else []:
        if (
            isinstance(entry, dict)
            and "name" in entry
            and (entry.get("base_url") or entry.get("api_key_env"))
        ):
            out[str(entry["name"])] = ModelEndpoint(
                base_url=str(entry["base_url"]) if entry.get("base_url") else None,
                api_key_env=str(entry["api_key_env"]) if entry.get("api_key_env") else None,
            )
    return out


def load_price_file(path: str | Path) -> dict[str, ModelPricing]:
    """Read a user `prices.yaml`:

    models:
      - name: llama-3.3-70b-versatile
        input_per_1m: 0.59
        output_per_1m: 0.79
        cached_input_per_1m: 0.59   # optional, defaults to input
        source: https://groq.com/pricing (2026-11-12)   # optional, shown in reports
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    entries = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level 'models:' list")
    out: dict[str, ModelPricing] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{path}: every model needs a 'name': {entry!r}")
        try:
            inp = float(entry["input_per_1m"])
            outp = float(entry["output_per_1m"])
            cached = float(entry.get("cached_input_per_1m", inp))
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(
                f"{path}: {entry['name']!r} needs numeric input_per_1m and output_per_1m"
            ) from err
        if min(inp, outp, cached) < 0:
            raise ValueError(f"{path}: {entry['name']!r} has a negative price")
        name = str(entry["name"])
        out[name] = ModelPricing(
            name=name,
            provider=str(entry.get("provider", "custom")),
            input_per_1m=inp,
            cached_input_per_1m=cached,
            output_per_1m=outp,
            pricing_updated_at=str(entry.get("updated", "user-supplied")),
            notes=str(entry["source"]) if entry.get("source") else str(path),
        )
    return out


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
