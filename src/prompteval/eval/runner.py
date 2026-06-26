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

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from prompteval.cost import CostBreakdown, Usage, compute_cost
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


def default_runner(
    example: Example,
    prompt_text: str,
    model: str,
    client: Any,
) -> RunnerResult:
    """Send `prompt_text` as system + `example.input` as user, one OpenAI call.

    Returns the response text + a single Usage. Cached/reasoning token counts
    are pulled from `prompt_tokens_details` / `completion_tokens_details` when
    present — gracefully zero when not.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": example.input},
        ],
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
) -> RunResult:
    """Execute `eval_def` against the prompt at `prompt_path`.

    Per-example errors are recorded in `ExampleResult.error` (not raised) —
    one bad example shouldn't waste 19 good ones. The aggregate properties
    (`total_cost`, `avg_latency_s`, `scorer_means`) skip errored examples.

    `model` overrides `eval_def.model` if provided — useful for "same prompt,
    different model" comparisons. Future v0.2 multi-provider work will lean
    on this hook too.

    `progress` is called once per example with `(index, total, example_id,
    latency_s, error)`. Pass a Click `echo`-flavored callback to render
    progress, or None for silent runs (tests use None).
    """
    prompt_text = Path(prompt_path).read_text()
    effective_model = model or eval_def.model
    examples = _load_dataset(Path(eval_def.dataset))
    runner = eval_def.runner or default_runner
    eff_client = client if client is not None else OpenAI()

    started_at = _now_iso()
    results: list[ExampleResult] = []

    for i, example in enumerate(examples, start=1):
        t0 = time.perf_counter()
        error: str | None = None
        output = ""
        scores: list[ScoreOutcome] = []
        cost: CostBreakdown
        try:
            runner_result = runner(example, prompt_text, effective_model, eff_client)
            output = runner_result.output

            # v1: one model per eval, so we can sum usages into one Usage and price once.
            # v1.1 (multi-model runners) will need per-call pricing — different shape.
            summed = _sum_usages(runner_result.usages)
            cost = compute_cost(effective_model, summed)

            scores = [_call_scorer(s, example, output) for s in eval_def.scorers]
        except Exception as err:
            error = f"{type(err).__name__}: {err}"
            # Build a zero-cost CostBreakdown for the error case so aggregations
            # don't have to special-case None.
            cost = compute_cost(effective_model, Usage(prompt_tokens=0, completion_tokens=0))

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
    )


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


def _now_iso() -> str:
    """UTC ISO timestamp, second-precision. Stable for filename + persistence."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
