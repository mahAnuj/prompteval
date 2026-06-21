"""Minimal CLI stub.

Exists so `prompteval --version` works the day the package is installable.
We'll grow this into the real eval-runner CLI in week 5-6.
"""

import click

from prompteval.version import __version__


@click.group(help="prompteval — LLM evals with first-class token-cost tracking.")
@click.version_option(version=__version__, prog_name="prompteval")
def main() -> None:
    """Entry point for the `prompteval` console script."""


@main.command()
def hello() -> None:
    """Sanity-check command — proves the CLI installed correctly."""
    click.echo("prompteval is alive. Next stop: cost-vs-quality reports.")


if __name__ == "__main__":
    main()
