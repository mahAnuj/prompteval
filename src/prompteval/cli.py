"""Click-based CLI for prompteval.

Surface today: `--version`, `hello`, `init`, `models`, `scorer`, `run`,
`compare`. HTML report + `--fail-on` for CI land in Week 6.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from prompteval.compare import (
    compute_comparison,
    evaluate_gates,
    parse_gate_spec,
    render_html,
    render_text,
)
from prompteval.compare.gates import GateSpecError
from prompteval.compare.sweep import (
    compute_sweep,
    render_sweep_html,
    render_sweep_markdown,
    render_sweep_text,
)
from prompteval.cost import (
    ModelEndpoint,
    ModelPricing,
    UnknownModelError,
    get_pricing,
    list_models,
    load_model_endpoints,
    load_price_file,
    parse_price_spec,
    resolve_pricing,
)
from prompteval.eval import Eval, load_run, run_eval, save_run
from prompteval.eval import stock as stock_scorers
from prompteval.eval.runner import RunResult
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


@main.command()
@click.option(
    "--prompt",
    "prompt_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the system-prompt file the runner will send.",
)
@click.option(
    "--tag",
    required=True,
    help="Identifier for this run — used as the persisted filename + compare key.",
)
@click.option(
    "--eval-file",
    "eval_file",
    default="evals/eval.py",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the Python file that defines the Eval instance.",
)
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override the model declared in the Eval (default: use eval_def.model).",
)
@click.option(
    "--base-url",
    "base_url",
    default=None,
    help=(
        "Any OpenAI-compatible endpoint, e.g. https://api.groq.com/openai/v1 or "
        "http://localhost:11434/v1 (Ollama). Default: OpenAI."
    ),
)
@click.option(
    "--api-key-env",
    "api_key_env",
    default=None,
    help="Environment variable holding that endpoint's API key, e.g. GROQ_API_KEY.",
)
@click.option(
    "--price",
    "price_spec",
    default=None,
    help=(
        "Price for a model not in the built-in table: input=X,output=Y[,cached=Z], "
        "USD per 1M tokens."
    ),
)
@click.option(
    "--prices-file",
    "prices_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A prices.yaml of your own (default: prices.yaml next to the eval file, if present).",
)
@click.option(
    "--param",
    "param_specs",
    multiple=True,
    help=(
        "Request setting sent with every call, repeatable: --param temperature=0 "
        "--param max_tokens=300. Values are read as JSON when they parse, else as text."
    ),
)
def run(
    prompt_path: Path,
    tag: str,
    eval_file: Path,
    model_override: str | None,
    base_url: str | None,
    api_key_env: str | None,
    price_spec: str | None,
    prices_file: Path | None,
    param_specs: tuple[str, ...],
) -> None:
    """Run an Eval against a prompt and persist the results.

    Discovers the first `Eval` instance in EVAL_FILE, executes it against the
    dataset that Eval declares, and writes results to .prompteval/runs/<tag>.json.
    """
    eval_def = _discover_eval(eval_file)
    model = model_override or eval_def.model
    dataset_path = Path(eval_def.dataset)

    # The multiplication sign in the f-string is intentional — matches the
    # README's killer-output spec so users see consistent run-header phrasing.
    click.echo(
        f"Running {_count_examples(dataset_path)} examples × {model} against {prompt_path}..."  # noqa: RUF001
    )

    def progress(i: int, n: int, ex_id: str, latency: float, error: str | None) -> None:
        marker = "✗" if error else "✓"
        click.echo(f"  [{i}/{n}] {ex_id:<24} {marker} {latency:0.2f}s")

    prices, endpoints = _user_prices(eval_file, prices_file)
    try:
        if price_spec:
            prices[model] = parse_price_spec(model, price_spec)
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    endpoint = endpoints.get(model, ModelEndpoint())

    try:
        result = run_eval(
            eval_def,
            prompt_path=prompt_path,
            tag=tag,
            model=model_override,
            progress=progress,
            prices=prices,
            base_url=base_url or endpoint.base_url,
            api_key_env=api_key_env or endpoint.api_key_env,
            params=_parse_params(param_specs),
        )
    except (UnknownModelError, ValueError) as err:
        raise click.ClickException(str(err)) from err

    saved_path = save_run(result)

    click.echo("")
    click.echo(f"=== {tag} ===")
    n_ok = sum(1 for e in result.examples if e.error is None)
    n_err = len(result.examples) - n_ok
    for name, mean in result.scorer_means.items():
        click.echo(f"{name:<28} {mean:.2f} (n={n_ok})")
    if n_err:
        click.echo(f"errors                      {n_err} of {len(result.examples)} examples failed")
    avg_cost = result.total_cost / max(1, len(result.examples))
    click.echo(f"total cost                  ${result.total_cost:.4f}  (avg ${avg_cost:.4f}/call)")
    click.echo(f"avg latency                 {result.avg_latency_s:.2f}s")
    click.echo(f"saved to: {saved_path}")


def _user_prices(
    eval_file: Path, prices_file: Path | None
) -> tuple[dict[str, ModelPricing], dict[str, ModelEndpoint]]:
    """The user's prices.yaml (given, or next to the eval file): prices and endpoints."""
    candidate = prices_file or eval_file.parent / "prices.yaml"
    if not candidate.exists():
        return {}, {}
    try:
        return load_price_file(candidate), load_model_endpoints(candidate)
    except ValueError as err:
        raise click.ClickException(str(err)) from err


@main.command()
@click.option(
    "--prompt",
    "prompt_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The prompt to run on every model.",
)
@click.option(
    "--models",
    required=True,
    help="Comma-separated models, e.g. gpt-4o,gpt-4o-mini,llama-3.3-70b-versatile.",
)
@click.option(
    "--baseline",
    default=None,
    help="The model the others are compared to (default: the first in --models).",
)
@click.option(
    "--eval-file",
    "eval_file",
    default="evals/eval.py",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--prices-file",
    "prices_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="prices.yaml with prices (and optional base_url / api_key_env) per model.",
)
@click.option("--param", "param_specs", multiple=True, help="Request setting, as in `run`.")
@click.option(
    "--margin",
    type=click.FloatRange(0.0, 1.0, max_open=True),
    default=0.05,
    show_default=True,
    help="Largest scorer drop (0-1 scale) still counted as 'quality holds'.",
)
@click.option("--rerun", is_flag=True, help="Run every model again instead of reusing saved runs.")
@click.option(
    "--html",
    "html_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the table as a single HTML file.",
)
@click.option(
    "--markdown",
    "markdown_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the table as Markdown (for a PR comment or a README).",
)
def sweep(
    prompt_path: Path,
    models: str,
    baseline: str | None,
    eval_file: Path,
    prices_file: Path | None,
    param_specs: tuple[str, ...],
    margin: float,
    rerun: bool,
    html_path: Path | None,
    markdown_path: Path | None,
) -> None:
    """Which model is cheapest for this prompt without losing quality?

    Runs the prompt on every model (reusing a saved run when the prompt, dataset,
    model and settings are unchanged, so re-running is free), compares each to the
    baseline, and prints one table with a one-line answer.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2 or len(set(model_list)) != len(model_list):
        raise click.ClickException("--models needs at least two different models.")
    base = baseline or model_list[0]
    if base not in model_list:
        raise click.ClickException(f"--baseline {base!r} is not in --models.")

    eval_def = _discover_eval(eval_file)
    prices, endpoints = _user_prices(eval_file, prices_file)
    params = {**eval_def.params, **_parse_params(param_specs)}
    prompt_text = prompt_path.read_text()
    dataset_sha = hashlib.sha256(Path(eval_def.dataset).read_bytes()).hexdigest()

    # Check every model can be priced before spending anything.
    for m in model_list:
        try:
            resolve_pricing(m, prices)
        except UnknownModelError as err:
            raise click.ClickException(str(err)) from err

    runs: dict[str, RunResult] = {}
    for m in model_list:
        tag = f"sweep-{prompt_path.stem}-{m}"
        saved = _reusable_run(tag, prompt_text, dataset_sha, m, params) if not rerun else None
        if saved is not None:
            click.echo(f"{m}: reusing saved run ({tag})")
            runs[m] = saved
            continue
        endpoint = endpoints.get(m, ModelEndpoint())
        click.echo(f"{m}: running {_count_examples(Path(eval_def.dataset))} examples...")
        try:
            result = run_eval(
                eval_def,
                prompt_path=prompt_path,
                tag=tag,
                model=m,
                prices=prices,
                base_url=endpoint.base_url,
                api_key_env=endpoint.api_key_env,
                params=params,
            )
        except (UnknownModelError, ValueError) as err:
            raise click.ClickException(f"{m}: {err}") from err
        save_run(result)
        runs[m] = result

    try:
        report = compute_sweep(runs, base, margin=margin)
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    click.echo("")
    click.echo(render_sweep_text(report))
    if html_path is not None:
        html_path.write_text(render_sweep_html(report), encoding="utf-8")
        click.echo(f"HTML report: {html_path}")
    if markdown_path is not None:
        markdown_path.write_text(render_sweep_markdown(report), encoding="utf-8")
        click.echo(f"Markdown report: {markdown_path}")


def _reusable_run(
    tag: str, prompt_text: str, dataset_sha: str, model: str, params: dict[str, Any]
) -> RunResult | None:
    """A saved run made with exactly this prompt, dataset, model and settings."""
    try:
        saved = load_run(tag)
    except FileNotFoundError:
        return None
    same = (
        saved.prompt_text == prompt_text
        and saved.dataset_sha256 == dataset_sha
        and saved.model == model
        and saved.params == params
    )
    return saved if same else None


def _parse_params(specs: tuple[str, ...]) -> dict[str, Any]:
    """`key=value` pairs; values are JSON when they parse (0, 0.2, true, {...}), else text."""
    out: dict[str, Any] = {}
    for spec in specs:
        key, sep, raw = spec.partition("=")
        if not sep or not key.strip():
            raise click.ClickException(f"Invalid --param {spec!r}: expected key=value.")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _discover_eval(eval_file: Path) -> Eval:
    """Import EVAL_FILE and return the first Eval instance defined in it.

    We add the file's directory to sys.path so relative imports inside the
    user's eval.py work the same way they would for `python evals/eval.py`.
    """
    spec = importlib.util.spec_from_file_location("_user_eval", eval_file)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load {eval_file}")
    module = importlib.util.module_from_spec(spec)

    parent = str(eval_file.parent.resolve())
    added_to_path = parent not in sys.path
    if added_to_path:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as err:
        raise click.ClickException(
            f"Failed to load {eval_file}: {type(err).__name__}: {err}"
        ) from err
    finally:
        if added_to_path:
            sys.path.remove(parent)

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Eval):
            return obj
    raise click.ClickException(
        f"No Eval instance found in {eval_file}. "
        f"Add `eval = Eval(name=..., dataset=..., scorers=[...], model=...)` to the file."
    )


def _count_examples(dataset_path: Path) -> int:
    """Cheap line-count for the dataset, used only for the user-facing progress
    header — the runner does its own validating load."""
    if not dataset_path.exists():
        return 0
    return sum(
        1
        for line in dataset_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


@main.command()
@click.argument("tag_a")
@click.argument("tag_b")
@click.option(
    "--runs-dir",
    "runs_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where the run JSON files live (default: .prompteval/runs).",
)
@click.option(
    "--html",
    "html_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write a single-file HTML report to this path.",
)
@click.option(
    "--fail-on",
    "fail_on",
    default=None,
    help=(
        "CI gate spec, e.g. 'cost+10%,quality-5%'. Exits 1 on a breach. By default a "
        "clause fails when the 95% CI says the breach is plausible (see --gate-mode)."
    ),
)
@click.option(
    "--gate-mode",
    "gate_mode",
    type=click.Choice(["bound", "significant"]),
    default="bound",
    show_default=True,
    help=(
        "bound: fail when the worst plausible change breaches the gate. "
        "significant: fail only on a statistically significant breach (v0.1 behaviour)."
    ),
)
@click.option(
    "--margin",
    type=click.FloatRange(0.0, 1.0, max_open=True),
    default=0.05,
    show_default=True,
    help="Largest scorer drop (0-1 scale) still counted as 'quality holds'.",
)
def compare(
    tag_a: str,
    tag_b: str,
    runs_dir: Path | None,
    html_path: Path | None,
    fail_on: str | None,
    gate_mode: str,
    margin: float,
) -> None:
    """Compare two persisted runs and print the paired delta report.

    Both tags must have been produced by a prior `prompteval run --tag <name>`.
    The report shows per-scorer + cost + latency deltas with 95% bootstrap CIs
    and paired t-test p-values, plus a plain-English recommendation.

    Use --html PATH to also write a shareable single-file HTML report.
    Use --fail-on to gate CI on regressions (exits 1 on breach).
    """
    try:
        run_a = load_run(tag_a, runs_dir=runs_dir)
        run_b = load_run(tag_b, runs_dir=runs_dir)
    except FileNotFoundError as err:
        raise click.ClickException(str(err)) from err

    # Parse the gate spec up-front so a typo fails immediately, not after the run.
    clauses = None
    if fail_on is not None:
        try:
            clauses = parse_gate_spec(fail_on)
        except GateSpecError as err:
            raise click.ClickException(str(err)) from err

    try:
        report = compute_comparison(run_a, run_b, margin=margin)
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    click.echo(render_text(report))

    if html_path is not None:
        html_path.write_text(render_html(report), encoding="utf-8")
        click.echo("")
        click.echo(f"HTML report: {html_path}")

    if clauses is not None:
        breaches = evaluate_gates(report, clauses, mode=gate_mode)
        if breaches:
            click.echo("")
            click.echo("GATE FAILED:")
            for b in breaches:
                click.echo(f"  - {b.detail}")
            sys.exit(1)


if __name__ == "__main__":
    main()
