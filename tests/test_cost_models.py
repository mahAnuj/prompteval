"""Tests for the pricing table — the data shipped in pricing.yaml.

These guard against:
- YAML failing to load (broken syntax / missing required keys / bad types)
- Pricing fields drifting silently (e.g. someone "fixes" a typo and accidentally
  changes a number)
- get_pricing/list_models contract changes
- Cache discount math being inverted (cached > input would be a real bug)
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from prompteval.cost import PRICING, UnknownModelError, get_pricing, list_models
from prompteval.cost.models import ModelPricing


def test_pricing_yaml_loads_at_import() -> None:
    """If pricing.yaml is broken the package fails to import — this proves it isn't."""
    assert len(PRICING) >= 5, "v0.1 ships at least 5 models"


def test_v01_default_model_present() -> None:
    """gpt-4o-mini is the v0.1 default — losing it would break templates."""
    assert "gpt-4o-mini" in PRICING


def test_each_pricing_entry_is_well_formed() -> None:
    """All numeric rates non-negative; cached <= input; date parses as ISO;
    provider is one of the v1-known providers (the dispatch seam for v0.2)."""
    iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    known_providers = {"openai"}  # v0.2 will add anthropic, openai-compatible

    for m in PRICING.values():
        assert m.provider in known_providers, (
            f"{m.name}: provider {m.provider!r} not in known set {known_providers}. "
            "If you're adding a new provider, see cost/models.py module docstring "
            "for the polymorphic refactor plan."
        )
        assert m.input_per_1m >= 0, m
        assert m.output_per_1m >= 0, m
        assert m.cached_input_per_1m >= 0, m
        # The whole point of caching is to be cheaper — would be a real bug if reversed.
        assert m.cached_input_per_1m <= m.input_per_1m, (
            f"{m.name}: cached_input_per_1m ({m.cached_input_per_1m}) > "
            f"input_per_1m ({m.input_per_1m})"
        )
        assert iso_date.match(m.pricing_updated_at), m.pricing_updated_at
        # date.fromisoformat() raises if the date is impossible (e.g. 2026-02-30).
        date.fromisoformat(m.pricing_updated_at)


def test_pricing_values_match_known_snapshot() -> None:
    """Anchor specific numbers we've verified, so a fat-fingered YAML edit fails loud."""
    # gpt-4o is the most-quoted model; lock it tightly.
    p = PRICING["gpt-4o"]
    assert p.input_per_1m == 2.50
    assert p.cached_input_per_1m == 1.25
    assert p.output_per_1m == 10.00

    # gpt-4o-mini drives the v0.1 default — lock it too.
    p = PRICING["gpt-4o-mini"]
    assert p.input_per_1m == 0.15
    assert p.cached_input_per_1m == 0.075
    assert p.output_per_1m == 0.60


def test_get_pricing_exact_match() -> None:
    p = get_pricing("gpt-4o-mini")
    assert isinstance(p, ModelPricing)
    assert p.name == "gpt-4o-mini"


def test_get_pricing_unknown_raises_with_suggestion() -> None:
    """A typo of `gpt-4o` should still suggest gpt-4o variants."""
    with pytest.raises(UnknownModelError) as ei:
        get_pricing("gpt4o")  # missing the hyphen
    assert "gpt-4o" in str(ei.value)
    assert "Did you mean" in str(ei.value)


def test_get_pricing_unknown_lists_all_available() -> None:
    """A wholly unrelated name should still print the full catalog so user can pick."""
    with pytest.raises(UnknownModelError) as ei:
        get_pricing("some-other-vendor-model")
    msg = str(ei.value)
    for known in PRICING:
        assert known in msg


def test_unknown_model_is_keyerror_too() -> None:
    """UnknownModelError extends KeyError so legacy catches keep working."""
    with pytest.raises(KeyError):
        get_pricing("definitely-not-a-model")


def test_list_models_is_sorted_by_name() -> None:
    catalog = list_models()
    names = [m.name for m in catalog]
    assert names == sorted(names)
    assert len(catalog) == len(PRICING)
