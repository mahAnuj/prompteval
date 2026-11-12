"""Model price catalog: thousands of models, from maintained public sources.

The hand-kept `pricing.yaml` covers five OpenAI models and goes stale. This
catalog merges two sources that are updated continuously:

- **LiteLLM's `model_prices_and_context_window.json`** (MIT licence,
  github.com/BerriAI/litellm): OpenAI, Anthropic, Gemini, Mistral, DeepSeek,
  xAI, Together, Fireworks, DeepInfra, Cerebras, Bedrock, Azure and more,
  keyed as `model` (OpenAI, Anthropic) or `provider/model`.
- **OpenRouter's public model list** (`/api/v1/models`), keyed here as
  `openrouter/<id>`.

A dated snapshot ships with the package (`catalog.json`). `prompteval models
update` fetches both sources again into a user cache, which is used whenever
it is newer than the snapshot, so prices stay current between releases.

Lookup (`catalog_pricing`) is endpoint-aware: the same open model costs
different amounts on Groq, Together and OpenRouter, so the `base_url` decides
which provider's entry applies. Without a base URL (OpenAI) only an exact key
matches. A price from the catalog is recorded on the run with its key, its
source and the date it was fetched.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from prompteval.cost.models import ModelPricing

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
SOURCES = [
    {"name": "LiteLLM model_prices_and_context_window.json", "url": LITELLM_URL, "license": "MIT"},
    {"name": "OpenRouter model list", "url": OPENROUTER_URL},
]

#: API host → the catalog's provider prefix. OpenAI and Anthropic use bare keys.
HOST_PROVIDERS = {
    "api.groq.com": "groq",
    "api.together.xyz": "together_ai",
    "api.fireworks.ai": "fireworks_ai",
    "openrouter.ai": "openrouter",
    "api.deepseek.com": "deepseek",
    "api.mistral.ai": "mistral",
    "generativelanguage.googleapis.com": "gemini",
    "api.x.ai": "xai",
    "api.deepinfra.com": "deepinfra",
    "api.cerebras.ai": "cerebras",
    "api.perplexity.ai": "perplexity",
    "api.anthropic.com": "anthropic",
}


@dataclass(frozen=True)
class Catalog:
    fetched_at: str
    models: dict[str, tuple[float, float, float, str]]  # key → (input, cached, output, provider)
    origin: str  # "bundled" or the cache path


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _per_1m(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v * 1_000_000, 6) if v >= 0 else None


def build_catalog(
    litellm: dict[str, Any], openrouter: dict[str, Any] | None, fetched_at: str
) -> dict[str, Any]:
    """Merge the two raw sources into the catalog format (prices per 1M tokens)."""
    models: dict[str, list[Any]] = {}
    for key, entry in litellm.items():
        if not isinstance(entry, dict) or entry.get("mode") != "chat":
            continue
        inp = _per_1m(entry.get("input_cost_per_token"))
        out = _per_1m(entry.get("output_cost_per_token"))
        if inp is None or out is None:
            continue
        cached = _per_1m(entry.get("cache_read_input_token_cost"))
        models[key] = [inp, inp if cached is None else cached, out, entry.get("litellm_provider")]
    for entry in (openrouter or {}).get("data", []):
        pricing = entry.get("pricing") or {}
        inp = _per_1m(pricing.get("prompt"))
        out = _per_1m(pricing.get("completion"))
        if not entry.get("id") or inp is None or out is None:
            continue
        cached = _per_1m(pricing.get("input_cache_read"))
        models[f"openrouter/{entry['id']}"] = [
            inp,
            inp if cached is None else cached,
            out,
            "openrouter",
        ]
    return {
        "schema": 1,
        "fetched_at": fetched_at,
        "sources": SOURCES,
        "models": dict(sorted(models.items())),
    }


def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "prompteval-price-catalog"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_catalog(timeout: float = 30.0) -> dict[str, Any]:
    """Download both sources and build a catalog. Raises on any failure, so a
    partial download never replaces a good catalog."""
    litellm = _get_json(LITELLM_URL, timeout)
    openrouter = _get_json(OPENROUTER_URL, timeout)
    if not isinstance(litellm, dict) or not isinstance(openrouter, dict):
        raise ValueError("a price source returned an unexpected shape")
    built = build_catalog(litellm, openrouter, datetime.now(UTC).date().isoformat())
    if len(built["models"]) < 100:
        raise ValueError(f"only {len(built['models'])} priced models found; refusing to save")
    return built


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def cache_path() -> Path:
    base = os.environ.get("PROMPTEVAL_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "prompteval"
    )
    return Path(base) / "catalog.json"


def _parse(raw: dict[str, Any], origin: str) -> Catalog:
    models = {
        k: (float(v[0]), float(v[1]), float(v[2]), str(v[3])) for k, v in raw["models"].items()
    }
    return Catalog(fetched_at=str(raw["fetched_at"]), models=models, origin=origin)


_LOADED: Catalog | None = None


def load_catalog(refresh: bool = False) -> Catalog:
    """The user cache when it is newer than the bundled snapshot, else the snapshot."""
    global _LOADED
    if _LOADED is not None and not refresh:
        return _LOADED
    bundled = _parse(
        json.loads(files("prompteval.cost").joinpath("catalog.json").read_text()), "bundled"
    )
    chosen = bundled
    path = cache_path()
    if path.exists():
        try:
            cached = _parse(json.loads(path.read_text()), str(path))
            if cached.fetched_at >= bundled.fetched_at:
                chosen = cached
        except (ValueError, KeyError, TypeError, IndexError):
            pass  # a damaged cache never breaks pricing; the snapshot is used
    _LOADED = chosen
    return chosen


def save_catalog(raw: dict[str, Any], path: Path | None = None) -> Path:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, separators=(",", ":")))
    tmp.replace(path)
    global _LOADED
    _LOADED = None
    return path


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def provider_for(base_url: str | None) -> str | None:
    if not base_url:
        return None
    return HOST_PROVIDERS.get(urlparse(base_url).hostname or "")


def catalog_key(model: str, base_url: str | None, catalog: Catalog | None = None) -> str | None:
    """The catalog entry for `model` called at `base_url`, or None."""
    cat = catalog or load_catalog()
    provider = provider_for(base_url)
    if provider and provider not in ("anthropic",):
        key = f"{provider}/{model}"
        if key in cat.models:
            return key
    if model in cat.models and (provider is None or cat.models[model][3] == provider):
        return model
    if provider:
        # Some catalog keys carry the provider in the model id already.
        for key, entry in cat.models.items():
            if entry[3] == provider and key.split("/", 1)[-1] == model:
                return key
    return None


def catalog_pricing(model: str, base_url: str | None) -> ModelPricing | None:
    cat = load_catalog()
    key = catalog_key(model, base_url, cat)
    if key is None:
        return None
    inp, cached, out, provider = cat.models[key]
    return ModelPricing(
        name=model,
        provider=provider,
        input_per_1m=inp,
        cached_input_per_1m=cached,
        output_per_1m=out,
        pricing_updated_at=cat.fetched_at,
        notes=f"catalog {key!r} (LiteLLM + OpenRouter, fetched {cat.fetched_at})",
    )


def search(
    text: str, provider: str | None = None, limit: int = 50
) -> list[tuple[str, tuple[float, float, float, str]]]:
    cat = load_catalog()
    needle = text.lower()
    hits = [
        (k, v)
        for k, v in cat.models.items()
        if needle in k.lower() and (provider is None or v[3] == provider)
    ]
    # First-party names (no provider prefix) first, then shorter keys.
    hits.sort(key=lambda kv: (kv[0].count("/"), len(kv[0]), kv[0]))
    return hits[:limit]
