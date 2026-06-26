"""@scorer decorator + ScorerResult.

A scorer is a Python function that judges one LLM output. The decorator marks
the function as a scorer and validates the signature uses only known param
names (`input`, `output`, `expected`). The Week-4 runner inspects the
signature and passes only the params each scorer asks for — same pattern as
pytest fixtures.

Two valid return types:
- `float` in [0, 1] — most scorers
- `ScorerResult` — when you want to surface reasoning or metadata to the
  comparison report (LLM-as-judge scorers especially)

Scorers do NOT need to handle their own LLM cost tracking. Judge calls run
inside scorer functions; in v0.1 their cost is operational overhead, not
aggregated into the eval's cost report. v0.2 adds judge-cost separation.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, Union

P = ParamSpec("P")
R = TypeVar("R", bound=Union[float, "ScorerResult"])

#: Param names a scorer is allowed to declare. The runner passes only what each
#: scorer asks for (signature-driven dispatch — keeps simple scorers terse and
#: lets complex ones access more context without forcing all signatures to be wide).
KNOWN_PARAMS = frozenset({"input", "output", "expected"})


@dataclass(frozen=True)
class ScorerResult:
    """Richer return type for scorers that want to surface why they scored as they did.

    Most scorers return a bare float. Use `ScorerResult` when:
    - You want the comparison report to show *why* (e.g. "judge said: tone is
      curt") rather than just a number
    - You're emitting metadata that's useful for debugging or filtering
      (e.g. which sub-rubric the judge applied)
    """

    score: float
    reasoning: str | None = None
    metadata: dict[str, Any] | None = None


def scorer(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047 — ParamSpec needed to preserve scorer signature shape for users; PEP 695 syntax loses the precise dispatch type info we surface to mypy
    """Mark `func` as a scorer.

    Validates that the function's parameter names are a subset of
    `{input, output, expected}`. The Week-4 runner inspects the signature and
    passes only the params each scorer declares — so `def my(output): ...` and
    `def my(input, output, expected): ...` are both valid scorers and the runner
    handles both.

    Raises `TypeError` at *decoration* time (not call time) if the scorer
    declares an unknown parameter — surfacing typos at import is cheaper than
    at runtime mid-eval.
    """
    sig = inspect.signature(func)
    declared = set(sig.parameters)
    unknown = declared - KNOWN_PARAMS
    if unknown:
        raise TypeError(
            f"Scorer {func.__name__!r} has unknown parameter(s): {sorted(unknown)}. "
            f"Scorers may only accept any subset of: {sorted(KNOWN_PARAMS)}."
        )

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    # Attribute is the marker the runner + CLI use to discover scorers.
    # Underscore-prefixed namespace so it doesn't collide with user attributes.
    wrapper.__prompteval_scorer__ = True  # type: ignore[attr-defined]
    wrapper.__prompteval_scorer_params__ = frozenset(declared)  # type: ignore[attr-defined]
    return wrapper


def is_scorer(obj: object) -> bool:
    """True iff `obj` was decorated with `@scorer`."""
    return getattr(obj, "__prompteval_scorer__", False) is True


def scorer_params(func: object) -> frozenset[str]:
    """Return the param names declared by a scorer. Empty set if not a scorer."""
    return getattr(func, "__prompteval_scorer_params__", frozenset())
