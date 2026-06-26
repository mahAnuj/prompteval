"""Click-based CLI for prompteval.

Surface today: `--version`, `hello`, `init`, `models`, `scorer`. Run + compare
land in Week 4-5 — see IMPLEMENTATION_PLAN.md for the schedule.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path

import click

from prompteval.cost import UnknownModelError, get_pricing, list_models
from prompteval.eval import stock as stock_scorers
from prompteval.eval.scorer import is_scorer
from prompteval.init import bootstrap
from prompteval.version import __version__


@click.group(help="prompteval — LLM evals with first-class token-cost tracking.")
@click.version_option(version=__version__, prog_name="prompteval")
def main() -> None:
    """Entry point for the `prompteval` console script."""


@main.command()
def hello() -> None:
    """Sanity-check command — proves the CLI installed correctly."""
    click.echo("prompteval is alive. Next stop: cost-vs-quality reports.")


@main.command()
@click.option(
    "--dir",
    "directory",
    default="evals",
    show_default=True,
    help="Folder to create in the current directory.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Write into an existing non-empty folder "
        "(existing files are kept; ours overwrite collisions)."
    ),
)
def init(directory: str, force: bool) -> None:
    """Bootstrap an evals/ folder in the current directory.

    Drops a working evals tree (prompts/, dataset.jsonl, eval.py, .env.example)
    so you can edit content instead of layout. Defaults are silent and
    opinionated; override the folder name with --dir.
    """
    target = Path.cwd() / directory
    try:
        result = bootstrap(target, force=force)
    except FileExistsError as err:
        raise click.ClickException(str(err)) from err

    click.echo(f"Created {result.target} ({result.files_written} files).")
    click.echo("")
    click.echo("Next:")
    click.echo(f"  cd {directory}")
    click.echo("  $EDITOR prompts/v1.txt        # paste your current prompt")
    click.echo("  $EDITOR prompts/v2.txt        # paste the variant to test")
    click.echo("  $EDITOR dataset.jsonl         # add your own examples")
    click.echo("  cp .env.example .env && $EDITOR .env   # add OPENAI_API_KEY")
    click.echo("")
    click.echo("Then (once v0.1 ships — see IMPLEMENTATION_PLAN.md):")
    click.echo("  prompteval run --prompt prompts/v1.txt --tag baseline")
    click.echo("  prompteval run --prompt prompts/v2.txt --tag short-prompt")
    click.echo("  prompteval compare baseline short-prompt")


@main.group()
def models() -> None:
    """List and inspect supported LLM models with their pricing."""


@models.command("list")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON instead of a pretty table.",
)
def models_list(as_json: bool) -> None:
    """Show every supported model with a one-line pricing summary."""
    catalog = list_models()
    if as_json:
        click.echo(json.dumps([asdict(m) for m in catalog], indent=2))
        return

    for m in catalog:
        click.echo(
            f"  {m.name:<14} "
            f"input ${m.input_per_1m:>6.3f}/M  "
            f"cached ${m.cached_input_per_1m:>6.3f}/M  "
            f"output ${m.output_per_1m:>6.3f}/M"
        )


@models.command("price")
@click.argument("model")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
def models_price(model: str, as_json: bool) -> None:
    """Show full pricing detail for a single model."""
    try:
        pricing = get_pricing(model)
    except UnknownModelError as err:
        raise click.ClickException(str(err)) from err

    if as_json:
        click.echo(json.dumps(asdict(pricing), indent=2))
        return

    cache_discount_pct = (1 - pricing.cached_input_per_1m / pricing.input_per_1m) * 100
    click.echo(f"Model:         {pricing.name}")
    click.echo(f"Input:         ${pricing.input_per_1m:>7.3f} per 1M tokens")
    click.echo(
        f"Cached input:  ${pricing.cached_input_per_1m:>7.3f} per 1M tokens "
        f"({cache_discount_pct:.0f}% off uncached)"
    )
    click.echo(f"Output:        ${pricing.output_per_1m:>7.3f} per 1M tokens")
    click.echo(f"Verified on:   {pricing.pricing_updated_at}")
    if pricing.notes:
        click.echo(f"Notes:         {pricing.notes}")


@main.group()
def scorer() -> None:
    """List + copy stock scorers shipped with prompteval."""


@scorer.command("list")
def scorer_list() -> None:
    """Show every stock scorer with its one-line docstring summary."""
    for name, doc in _iter_stock_scorers():
        first_line = doc.splitlines()[0] if doc else ""
        click.echo(f"  {name:<26} {first_line}")


@scorer.command("copy")
@click.argument("name")
def scorer_copy(name: str) -> None:
    """Print the source of stock scorer NAME to stdout.

    Pipe to a file or paste into your evals/eval.py — these are templates,
    not bullet-proof utilities; tune them for your use case.
    """
    func = getattr(stock_scorers, name, None)
    if func is None or not is_scorer(func):
        available = ", ".join(n for n, _ in _iter_stock_scorers())
        raise click.ClickException(f"Unknown stock scorer: {name!r}. Available: {available}.")
    click.echo(inspect.getsource(func))


def _iter_stock_scorers() -> list[tuple[str, str]]:
    """Discover (name, docstring) pairs for every @scorer in prompteval.eval.stock."""
    out: list[tuple[str, str]] = []
    for name in sorted(dir(stock_scorers)):
        obj = getattr(stock_scorers, name)
        if is_scorer(obj):
            out.append((name, inspect.getdoc(obj) or ""))
    return out


if __name__ == "__main__":
    main()
