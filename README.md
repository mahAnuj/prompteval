# prompteval

**LLM eval framework with first-class token-cost tracking.**

> Did your prompt change save money without losing quality?

Most eval tools (Braintrust, Langfuse, promptfoo, Phoenix) treat token cost as a metric you can graph. prompteval treats it as a **comparison axis** — every eval run reports quality, latency, **and** cost side-by-side, with statistical significance, so you can answer the only question that matters: *did this prompt change save money without losing quality?*

> ⚠️ **Early alpha.** API will change weekly until v0.5. Stars / issues welcome to shape it.

## Status

- ✅ Project skeleton (uv, mypy strict, ruff, pytest)
- ✅ CLI scaffold (`prompteval --version`)
- 🚧 Cost-tracking core — Week 2
- 🚧 Eval primitives + LLM-as-judge — Week 3-4
- 🚧 Statistical comparison reports — Week 5
- 🚧 GitHub Action — Week 6
- 🚧 PyPI release — Week 9

10-week plan tracked publicly on the [project board](https://github.com/mahAnuj/prompteval/projects) (TODO).

## Development

```bash
# Install everything (creates a venv via uv)
uv sync

# The full quality gate
uv run ruff check .         # lint
uv run ruff format --check . # format
uv run mypy src tests        # types (strict)
uv run pytest                # tests
uv run prompteval hello      # CLI smoke

# Or in one shot
uv run pytest && uv run mypy src tests && uv run ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
