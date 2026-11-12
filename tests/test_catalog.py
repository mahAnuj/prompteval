"""The price catalog: maintained sources, endpoint-aware lookup, refreshable."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.cost import catalog as cat_mod
from prompteval.cost import resolve_pricing

LITELLM: dict[str, Any] = {
    "sample_spec": {"mode": "chat"},  # no prices: skipped
    "gpt-x": {
        "mode": "chat",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 5e-7,
        "litellm_provider": "openai",
    },
    "groq/llama-x": {
        "mode": "chat",
        "input_cost_per_token": 5e-7,
        "output_cost_per_token": 8e-7,
        "litellm_provider": "groq",
    },
    "together_ai/meta/llama-x": {
        "mode": "chat",
        "input_cost_per_token": 9e-7,
        "output_cost_per_token": 9e-7,
        "litellm_provider": "together_ai",
    },
    "embed-x": {
        "mode": "embedding",
        "input_cost_per_token": 1e-8,
        "output_cost_per_token": 0,
        "litellm_provider": "openai",
    },
}
OPENROUTER: dict[str, Any] = {
    "data": [
        {"id": "meta/llama-x", "pricing": {"prompt": "0.0000012", "completion": "0.0000012"}},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[cat_mod.Catalog]:
    monkeypatch.setenv("PROMPTEVAL_CACHE_DIR", str(tmp_path))
    raw = cat_mod.build_catalog(LITELLM, OPENROUTER, "2999-01-01")  # newer than the bundle
    cat_mod.save_catalog(raw)
    loaded = cat_mod.load_catalog(refresh=True)
    yield loaded
    cat_mod._LOADED = None


def test_build_keeps_priced_chat_models_only() -> None:
    raw = cat_mod.build_catalog(LITELLM, OPENROUTER, "2026-11-12")
    assert set(raw["models"]) == {
        "gpt-x",
        "groq/llama-x",
        "together_ai/meta/llama-x",
        "openrouter/meta/llama-x",
    }
    assert raw["models"]["gpt-x"] == [1.0, 0.5, 2.0, "openai"]
    assert raw["models"]["groq/llama-x"][1] == 0.5  # no cache price: cached = input
    assert raw["sources"][0]["license"] == "MIT"


def test_lookup_follows_the_endpoint(catalog: cat_mod.Catalog) -> None:
    assert catalog.origin.endswith("catalog.json")
    assert cat_mod.catalog_key("gpt-x", None) == "gpt-x"
    assert cat_mod.catalog_key("llama-x", "https://api.groq.com/openai/v1") == "groq/llama-x"
    assert (
        cat_mod.catalog_key("meta/llama-x", "https://api.together.xyz/v1")
        == "together_ai/meta/llama-x"
    )
    assert (
        cat_mod.catalog_key("meta/llama-x", "https://openrouter.ai/api/v1")
        == "openrouter/meta/llama-x"
    )
    # The same open model is never priced from another provider's entry.
    assert cat_mod.catalog_key("llama-x", None) is None
    assert cat_mod.catalog_key("llama-x", "http://localhost:11434/v1") is None


def test_resolve_pricing_records_the_catalog_source(catalog: cat_mod.Catalog) -> None:
    p = resolve_pricing("llama-x", None, "https://api.groq.com/openai/v1")
    assert (p.input_per_1m, p.output_per_1m) == (0.5, 0.8)
    assert p.notes is not None and "groq/llama-x" in p.notes and "2999-01-01" in p.notes


def test_a_damaged_cache_falls_back_to_the_bundled_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPTEVAL_CACHE_DIR", str(tmp_path))
    (tmp_path / "catalog.json").write_text("{not json")
    try:
        loaded = cat_mod.load_catalog(refresh=True)
        assert loaded.origin == "bundled"
        assert len(loaded.models) > 1000
    finally:
        cat_mod._LOADED = None


def test_bundled_snapshot_prices_common_models() -> None:
    cat_mod._LOADED = None
    loaded = cat_mod.load_catalog(refresh=True)
    for key in ("gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5", "deepseek/deepseek-chat"):
        assert key in loaded.models, key
    cat_mod._LOADED = None


def test_fetch_refuses_a_suspiciously_small_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cat_mod, "_get_json", lambda url, timeout: LITELLM if "litellm" in url else OPENROUTER
    )
    with pytest.raises(ValueError, match="refusing to save"):
        cat_mod.fetch_catalog()


def test_cli_update_and_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTEVAL_CACHE_DIR", str(tmp_path))
    big = {f"m{i}": dict(LITELLM["gpt-x"]) for i in range(150)} | LITELLM
    monkeypatch.setattr(
        cat_mod, "_get_json", lambda url, timeout: big if "litellm" in url else OPENROUTER
    )
    try:
        result = CliRunner().invoke(main, ["models", "update"])
        assert result.exit_code == 0, result.output
        assert "Updated: 154 models" in result.output
        assert json.loads((tmp_path / "catalog.json").read_text())["models"]["groq/llama-x"]
        found = CliRunner().invoke(main, ["models", "search", "llama-x", "--provider", "groq"])
        assert "groq/llama-x" in found.output and "together_ai" not in found.output
    finally:
        cat_mod._LOADED = None


def test_cli_update_keeps_the_catalog_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPTEVAL_CACHE_DIR", str(tmp_path))

    def boom(url: str, timeout: float) -> Any:
        raise OSError("network down")

    monkeypatch.setattr(cat_mod, "_get_json", boom)
    result = CliRunner().invoke(main, ["models", "update"])
    assert result.exit_code != 0
    assert "current catalog is kept" in result.output
    assert not (tmp_path / "catalog.json").exists()
