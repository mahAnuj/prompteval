"""Eval class, default runner, and the per-example loop.

User journey through this module:

  1. User writes `eval = Eval(name=..., dataset=..., scorers=[...], model=...)`
     in `evals/eval.py`.
  2. CLI runs `prompteval run --prompt prompts/v1.txt --tag baseline`.
  3. CLI imports user's eval.py, finds the Eval instance, and calls
     `run_eval(eval, prompt_path=..., tag=...)`.
  4. `run_eval` loads the dataset, iterates examples, calls the runner
     (default = one OpenAI chat completion per example), runs each scorer
     with signature-aware dispatch, records cost + latency per example,
     and returns a RunResult.
  5. `prompteval.eval.persistence.save_run` writes the RunResult to
     `.prompteval/runs/<tag>.json`.

## v1.1 extensibility seam

`Eval` already accepts an internal `runner: Runner | None = None`. v1 doesn't
expose this to the README path — the default runner is always used. v1.1 will
flip the visibility so users can pass `runner=my_crewai_func` for multi-agent
evals. The wire shape is `Runner = Callable[[Example, str, str, ClientLike], RunResult]`
and `RunResult.usages: list[Usage]` is a list from day 1 so multi-call runners
don't force a shape change. See IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from prompteval.cost import CostBreakdown, ModelPricing, Usage, compute_cost, resolve_pricing
from prompteval.eval.scorer import ScorerResult, is_scorer, scorer_params

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One row from `dataset.jsonl`. The runner sees this; scorers see
    its fields via signature-aware dispatch."""

    id: str
    input: str
    expected: dict[str, Any] = field(default_factory=dict)
    # Values for `{name}` placeholders in the prompt, besides `{input}` (#10).
    vars: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    """What a runner returns per example.

    `usages` is a list from day 1 (even though v1's default runner always
    returns exactly one) so v1.1's multi-call runners don't force a shape
    change. v1.1 may also tag each usage with its own model — that's a
    breaking change to handle then, not now.
    """

    output: str
    usages: list[Usage]


@dataclass(frozen=True)
class ScoreOutcome:
    """One scorer's result for one example."""

    name: str
    score: float
    reasoning: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExampleResult:
    """Full outcome of running one example end-to-end."""

    id: str
    input: str
    expected: dict[str, Any]
    output: str
    scores: list[ScoreOutcome]
    cost: CostBreakdown
    latency_s: float
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Aggregate result of running an Eval against one prompt + dataset."""

    tag: str
    eval_name: str
    model: str
    prompt_path: str
    prompt_text: str  # snapshotted — compare report shows what was actually run
    started_at: str
    finished_at: str
    examples: list[ExampleResult]
    # sha256 of the dataset file, so `compare` can tell whether two runs saw
    # the same examples (#7). None for runs saved before it was recorded.
    dataset_sha256: str | None = None
    # Where the run went and how it was priced (#8); None for older runs.
    base_url: str | None = None
    price_source: str | None = None
    params: dict[str, Any] = field(default_factory=dict)  # request settings used (#10)

    @property
    def total_cost(self) -> float:
        return sum(e.cost.total_cost for e in self.examples if e.error is None)

    @property
    def avg_latency_s(self) -> float:
        valid = [e.latency_s for e in self.examples if e.error is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def scorer_means(self) -> dict[str, float]:
        """Mean score per scorer, across examples that didn't error."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ex in self.examples:
            if ex.error is not None:
                continue
            for s in ex.scores:
                sums[s.name] = sums.get(s.name, 0.0) + s.score
                counts[s.name] = counts.get(s.name, 0) + 1
        return {name: sums[name] / counts[name] for name in sums if counts[name] > 0}


class Runner(Protocol):
    """Per-example runner. Default is single-OpenAI-call; v1.1 will allow
    user-provided runners for multi-agent / multi-step pipelines."""

    def __call__(
        self,
        example: Example,
        prompt_text: str,
        model: str,
        client: Any,
    ) -> RunnerResult: ...


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Eval:
    """An eval definition. Users write one of these in `evals/eval.py`.

    `runner` exists today as the v1.1 extensibility seam — v1 always uses
    the default single-OpenAI-call runner. Don't pass it explicitly until
    v1.1 documents the contract.
    """

    name: str
    dataset: str | Path
    scorers: list[Callable[..., float | ScorerResult]]
    model: str = "gpt-4o-mini"
    runner: Runner | None = None
    # Any OpenAI-compatible endpoint (#8): Groq, Together, OpenRouter, Fireworks,
    # a local Ollama or vLLM. None = OpenAI.
    base_url: str | None = None
    # Name of the environment variable holding that endpoint's key, e.g.
    # "GROQ_API_KEY". None = OPENAI_API_KEY (the openai client's default).
    api_key_env: str | None = None
    # Extra request settings sent with every call (#10), e.g.
    # {"temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object"}}.
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scorers:
            raise ValueError(f"Eval {self.name!r} has no scorers — at least one is required.")
        for s in self.scorers:
            if not is_scorer(s):
                raise TypeError(
                    f"Eval {self.name!r}: {getattr(s, '__name__', repr(s))!r} is not "
                    f"a @scorer-decorated function. Wrap it with @scorer first."
                )


# ---------------------------------------------------------------------------
# Default runner (single OpenAI chat completion per example)
# ---------------------------------------------------------------------------


_VAR = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_ESCAPED_VAR = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def prompt_variables(prompt_text: str) -> list[str]:
    """Names of the `{name}` placeholders in a prompt, in order of first use.

    Only `{identifier}` counts, so JSON in a prompt (`{"key": 1}`) is left
    alone. `{{name}}` is an escaped, literal `{name}`.
    """
    seen: dict[str, None] = {}
    for m in _VAR.finditer(prompt_text):
        seen.setdefault(m.group(1), None)
    return list(seen)


def fill_prompt(prompt_text: str, values: dict[str, Any]) -> str:
    """Replace each `{name}` with `values[name]`; `{{name}}` becomes `{name}`.

    Raises KeyError naming the first missing variable.
    """
    missing = [v for v in prompt_variables(prompt_text) if v not in values]
    if missing:
        raise KeyError(
            f"prompt variable {missing[0]!r} has no value: add it to this example's "
            f"'vars' (available: {', '.join(sorted(values)) or 'none'})"
        )
    filled = _VAR.sub(lambda m: str(values[m.group(1)]), prompt_text)
    return _ESCAPED_VAR.sub(lambda m: "{" + m.group(1) + "}", filled)


def build_messages(prompt_text: str, example: Example) -> list[dict[str, str]]:
    """The messages for one example.

    A plain prompt is the system message and the example's input the user
    message. A prompt with `{placeholders}` is a template: it is filled from
    `input` and the example's `vars`, and sent as the user message.
    """
    if prompt_variables(prompt_text):
        values: dict[str, Any] = {"input": example.input, **example.vars}
        return [{"role": "user", "content": fill_prompt(prompt_text, values)}]
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": example.input},
    ]


def default_runner(
    example: Example,
    prompt_text: str,
    model: str,
    client: Any,
    params: dict[str, Any] | None = None,
) -> RunnerResult:
    """One chat completion per example (see `build_messages` for the shape).

    Returns the response text + a single Usage. Cached/reasoning token counts
    are pulled from `prompt_tokens_details` / `completion_tokens_details` when
    present — gracefully zero when not.
    """
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(prompt_text, example),
        **(params or {}),
    )

    output = response.choices[0].message.content or ""
    usage_obj = response.usage

    cached = 0
    reasoning = 0
    if usage_obj is not None:
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached = getattr(prompt_details, "cached_tokens", 0) or 0
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0

    usage = Usage(
        prompt_tokens=usage_obj.prompt_tokens if usage_obj else 0,
        completion_tokens=usage_obj.completion_tokens if usage_obj else 0,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )

    return RunnerResult(output=output, usages=[usage])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eval(
    eval_def: Eval,
    prompt_path: str | Path,
    tag: str,
    *,
    model: str | None = None,
    client: Any = None,
    progress: Callable[[int, int, str, float, str | None], None] | None = None,
    prices: Mapping[str, ModelPricing] | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    params: dict[str, Any] | None = None,
) -> RunResult:
    """Execute `eval_def` against the prompt at `prompt_path`.

    Per-example errors are recorded in `ExampleResult.error` (not raised) —
    one bad example shouldn't waste 19 good ones. The aggregate properties
    (`total_cost`, `avg_latency_s`, `scorer_means`) skip errored examples.

    `model` overrides `eval_def.model` if provided — useful for "same prompt,
    different model" comparisons. Future v0.2 multi-provider work will lean
    on this hook too.

    `prices` adds or overrides model prices (user `prices.yaml` / `--price`),
    and `base_url` / `api_key_env` override the Eval's endpoint (#8).

    `progress` is called once per example with `(index, total, example_id,
    latency_s, error)`. Pass a Click `echo`-flavored callback to render
    progress, or None for silent runs (tests use None).
    """
    prompt_text = Path(prompt_path).read_text()
    effective_model = model or eval_def.model
    # Fail before the first (paid) call when the model can't be priced (#5).
    pricing = resolve_pricing(effective_model, prices)
    effective_base_url = base_url or eval_def.base_url
    examples = _load_dataset(Path(eval_def.dataset))
    effective_params = {**eval_def.params, **(params or {})}
    runner: Runner = eval_def.runner or functools.partial(default_runner, params=effective_params)
    eff_client = (
        client
        if client is not None
        else _make_client(effective_base_url, api_key_env or eval_def.api_key_env)
    )

    started_at = _now_iso()
    results: list[ExampleResult] = []

    for i, example in enumerate(examples, start=1):
        t0 = time.perf_counter()
        error: str | None = None
        output = ""
        scores: list[ScoreOutcome] = []
        cost: CostBreakdown
        latency_s: float | None = None
        try:
            runner_result = runner(example, prompt_text, effective_model, eff_client)
            # Latency is the model call(s) only, never scoring or judge time (#6).
            latency_s = time.perf_counter() - t0
            output = runner_result.output

            # v1: one model per eval, so we can sum usages into one Usage and price once.
            # v1.1 (multi-model runners) will need per-call pricing — different shape.
            summed = _sum_usages(runner_result.usages)
            cost = compute_cost(effective_model, summed, pricing)

            scores = [_call_scorer(s, example, output) for s in eval_def.scorers]
        except Exception as err:
            error = f"{type(err).__name__}: {err}"
            # A zero-cost breakdown so aggregations needn't special-case None.
            # Built directly: the error path must never raise itself (#5).
            cost = _zero_cost(effective_model)

        if latency_s is None:  # the runner itself failed: time until the failure
            latency_s = time.perf_counter() - t0
        results.append(
            ExampleResult(
                id=example.id,
                input=example.input,
                expected=example.expected,
                output=output,
                scores=scores,
                cost=cost,
                latency_s=latency_s,
                error=error,
            )
        )
        if progress is not None:
            progress(i, len(examples), example.id, latency_s, error)

    finished_at = _now_iso()

    return RunResult(
        tag=tag,
        eval_name=eval_def.name,
        model=effective_model,
        prompt_path=str(prompt_path),
        prompt_text=prompt_text,
        started_at=started_at,
        finished_at=finished_at,
        examples=results,
        dataset_sha256=hashlib.sha256(Path(eval_def.dataset).read_bytes()).hexdigest(),
        base_url=effective_base_url,
        params=effective_params,
        price_source=(
            f"user-supplied ({pricing.notes or 'custom'})"
            if prices and effective_model in prices
            else f"built-in ({pricing.pricing_updated_at})"
        ),
    )


def _make_client(base_url: str | None, api_key_env: str | None) -> Any:
    """An OpenAI client for OpenAI or any OpenAI-compatible endpoint."""
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Environment variable {api_key_env} is not set (the API key).")
        return OpenAI(base_url=base_url, api_key=key)
    if base_url and not os.environ.get("OPENAI_API_KEY"):
        # Local servers such as Ollama need no key, but the client insists on one.
        return OpenAI(base_url=base_url, api_key="not-needed")
    return OpenAI(base_url=base_url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_scorer(
    s: Callable[..., float | ScorerResult],
    example: Example,
    output: str,
) -> ScoreOutcome:
    """Invoke `s` with the subset of (input, output, expected) it declares."""
    available = {
        "input": example.input,
        "output": output,
        "expected": example.expected,
    }
    declared = scorer_params(s)
    kwargs = {k: v for k, v in available.items() if k in declared}

    raw = s(**kwargs)

    if isinstance(raw, ScorerResult):
        return ScoreOutcome(
            name=s.__name__,
            score=raw.score,
            reasoning=raw.reasoning,
            metadata=raw.metadata,
        )
    # Bare float / int path. Validate range to fail loud on miscalibrated scorers.
    score = float(raw)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Scorer {s.__name__!r} returned {score}, expected a value in [0, 1].")
    return ScoreOutcome(name=s.__name__, score=score)


def _load_dataset(path: Path) -> list[Example]:
    """Load + validate a JSONL dataset.

    Each line must parse to `{id: str, input: str, expected?: dict}`. Missing
    `expected` defaults to {}. Malformed rows raise ValueError with the
    1-based line number — fail loud rather than silently skipping rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    examples: list[Example] = []
    ids: set[str] = set()
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({err})") from err
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected an object, got {type(obj).__name__}")
            if "vars" in obj and not isinstance(obj["vars"], dict):
                raise ValueError(f"{path}:{lineno}: 'vars' must be an object")
            for required in ("id", "input"):
                if required not in obj:
                    raise ValueError(f"{path}:{lineno}: missing required field {required!r}")
            example_id = str(obj["id"])
            if example_id in ids:
                raise ValueError(f"{path}:{lineno}: duplicate id {example_id!r}")
            ids.add(example_id)
            examples.append(
                Example(
                    id=example_id,
                    input=str(obj["input"]),
                    expected=obj.get("expected", {}) or {},
                    vars=obj.get("vars", {}) or {},
                )
            )
    if not examples:
        raise ValueError(f"{path}: dataset is empty — at least one example required")
    return examples


def _sum_usages(usages: list[Usage]) -> Usage:
    """Aggregate multiple Usage records into one (v1 single-model assumption).

    v1.1 multi-model runners can't use this — they need per-call cost. For
    v1's single-model assumption, summing is equivalent and lets us call
    `compute_cost` once per example.
    """
    if len(usages) == 1:
        return usages[0]
    return Usage(
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        cached_tokens=sum(u.cached_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
    )


def _zero_cost(model: str) -> CostBreakdown:
    """An all-zero cost record, for examples that errored."""
    return CostBreakdown(
        model=model,
        uncached_input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        uncached_input_cost=0.0,
        cached_input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0,
    )


def _now_iso() -> str:
    """UTC ISO timestamp, second-precision. Stable for filename + persistence."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
