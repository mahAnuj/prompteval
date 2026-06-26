# prompteval

**LLM eval framework with first-class token-cost tracking.**

> _Did your prompt change save money without losing quality?_

Paired cost-and-quality delta reports with significance — the question existing LLM eval tools (Braintrust, Langfuse, promptfoo, Phoenix) answer only awkwardly. They track cost as a metric you graph; prompteval treats it as a **comparison axis** alongside quality, so a single report tells you whether v2 of your prompt is genuinely cheaper at equivalent quality, or just *looks* cheaper inside noise.

See [docs/competitor-scan.md](docs/competitor-scan.md) for what the field currently does (and doesn't) ship.

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
