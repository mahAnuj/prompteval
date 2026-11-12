# AGENTS.md

Guidance for AI coding agents working in this repository. Also useful to humans.

## Commands

The project uses [uv](https://docs.astral.sh/uv/). Every command runs through it.

```bash
uv sync --all-extras --dev   # install deps (creates .venv)
uv run pytest                # 292 tests, no network or API key needed
uv run ruff check .          # lint
uv run ruff format --check . # formatting gate (drop --check to fix)
uv run mypy src tests        # strict mode, must stay clean
uv run prompteval --help     # exercise the CLI
```

Run a single test file or case:

```bash
uv run pytest tests/test_compare_gates.py
uv run pytest tests/test_scorer.py -k threshold
```

CI (`.github/workflows/ci.yml`) runs lint, format check, mypy and pytest on
Python 3.12 and 3.13. **All four must pass** — mypy is `strict = true` with
`warn_unreachable`, and ruff runs `E,F,I,B,UP,SIM,RUF`. Fix violations rather
than adding ignores.

## Architecture

A CLI that runs an eval suite against a prompt, scores the output, and reports
quality *and* token cost side by side — so a prompt change can be judged on
both at once. Entry point is `src/prompteval/cli.py` (Click), exposing seven
commands: `init`, `models`, `run`, `compare`, `sweep`, `noise`, `scorer`.

- **`cli.py`** — argument parsing and output formatting only. Business logic
  lives in the packages below; keep it that way.
- **`eval/runner.py`** — executes an Eval against a prompt, collecting outputs
  plus token usage per case.
- **`eval/scorer.py`** + **`eval/stock.py`** — scoring. `stock.py` holds the
  built-in scorers that `prompteval scorer` lists and copies out.
- **`eval/judge.py`** — LLM-as-judge scoring. **The only module that needs a
  live `OPENAI_API_KEY`.**
- **`eval/persistence.py`** — writes and reads run artifacts, so `compare` can
  work across two separate invocations.
- **`cost/pricing.yaml`** — per-model token prices (5 models: gpt-4o,
  gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano). **This is the file to edit
  when prices change or a model is added** — nothing else hardcodes pricing.
  **`cost/catalog.py` + `cost/catalog.json`** — ~3,900 models from LiteLLM's
  price list (MIT) and OpenRouter, matched to the provider by `base_url`.
  `.github/workflows/refresh-prices.yml` refreshes the bundled snapshot weekly
  and opens a PR; by hand: `prompteval models update`, then copy
  `~/.cache/prompteval/catalog.json` over `src/prompteval/cost/catalog.json`
  (tests never fetch). Lookup order:
  user prices, catalog, then `pricing.yaml`.
  Users price any other model themselves (`--price`, or a `prices.yaml` next to
  their `eval.py`, read by `load_price_file`); `resolve_pricing` checks theirs
  first. Only add a built-in price with its source and date.
- **`cost/compute.py`** + **`cost/models.py`** — turn token usage into cost using
  that table.
- **`compare/core.py`** — the paired delta report between two persisted runs.
- **`compare/sweep.py`** — `sweep`: one prompt on several models, each compared
  to a baseline, with a one-line answer.
- **`action.yml` + `action/run.sh`** — the GitHub Action (base vs head, or a
  sweep; one PR comment found by a hidden marker). `tests/test_action.py` runs
  the script against stub `prompteval` and `gh`.
- **`compare/gates.py`** — pass/fail thresholds, which is what makes the tool
  usable in CI.
- **`init/bootstrap.py`** + **`init/templates/`** — `prompteval init` scaffolds an
  `evals/` folder into the *user's* project.

### Invariants

- **`src/prompteval/init/templates/` is data, not code.** It is excluded from
  both ruff and mypy on purpose: the templates reference public API that does
  not exist yet, so linting them produces false positives. Do not "fix" the
  templates to satisfy the linters, and do not remove the excludes.
- **Tests must not need network or an API key.** All 292 pass offline; keep new
  tests that way by stubbing the model client rather than calling out.
- **Cost and quality are reported together.** A change that reports one without
  the other defeats the point of the tool.
- The package ships `py.typed`, so it is type-checked by consumers. Public
  signatures are part of the contract.

## Conventions

- Line length 100, target Python 3.12.
- Formatting is ruff's, not black's — `uv run ruff format .`.
- `assert` is allowed in tests only (`per-file-ignores`).
