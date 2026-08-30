# Contributing

Thanks for taking a look. Issues and pull requests are welcome.

## Getting set up

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mahAnuj/prompteval.git
cd prompteval
uv sync --all-extras --dev
uv run pytest
```

The test suite runs offline — no API key, no network. If your setup needs
either to get green, something is wrong; please open an issue.

To try the CLI against real models, set `OPENAI_API_KEY` (see `.env.example`).
Only LLM-as-judge scoring calls out to a model.

## Before you open a pull request

Run what CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

All four must pass on Python 3.12 and 3.13. mypy runs in strict mode — prefer
fixing types over adding `# type: ignore`; if an ignore is genuinely needed,
add a comment saying why.

## What makes a change easy to merge

- **One concern per pull request.** Small and focused beats large and sweeping.
- **A test that fails before your change and passes after.** For a bug fix,
  that test is the most valuable part of the diff.
- **Tests stay offline.** Stub the model client rather than making real calls.
- **Say what you observed.** For a bug, the command you ran and what happened
  matters more than a description of the intended behaviour.

## Adding a model

Add it to `src/prompteval/cost/pricing.yaml` with its input and output token
prices. Nothing else hardcodes pricing, so that is usually the whole change.
Include a link to the vendor's pricing page in the pull request so the numbers
can be checked.

## Adding a scorer

Stock scorers live in `src/prompteval/eval/stock.py` and are surfaced by
`prompteval scorer`. A scorer should be deterministic where it can be; if it
needs a model, it belongs alongside the judge in `eval/judge.py` instead.

## Things to know

- `src/prompteval/init/templates/` is deliberately excluded from ruff and mypy.
  It is scaffolding copied verbatim into user projects and references API that
  does not exist yet. Please leave the excludes in place.
- The package ships `py.typed`, so public signatures are a compatibility
  surface. Flag breaking changes in your pull request description.

## Reporting bugs

Include the `prompteval` version (`uv run prompteval --version`), your Python
version, the exact command, and the full output. A failing test case is ideal.
