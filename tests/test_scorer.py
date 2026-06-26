"""Tests for the @scorer decorator + ScorerResult.

Guards:
- @scorer accepts the valid param subsets (output / output+expected / etc)
- @scorer rejects unknown param names at decoration time (loud-fail at import)
- is_scorer / scorer_params correctly identify decorated functions
- ScorerResult dataclass is frozen + has the expected fields
"""

from __future__ import annotations

import pytest

from prompteval import ScorerResult, scorer
from prompteval.eval.scorer import is_scorer, scorer_params


def test_scorer_with_only_output() -> None:
    @scorer
    def s(output: str) -> float:
        return 1.0 if output else 0.0

    assert is_scorer(s)
    assert scorer_params(s) == frozenset({"output"})
    assert s("hello") == 1.0
    assert s("") == 0.0


def test_scorer_with_output_and_expected() -> None:
    @scorer
    def s(output: str, expected: dict[str, str]) -> float:
        return 1.0 if output == expected.get("target") else 0.0

    assert scorer_params(s) == frozenset({"output", "expected"})
    assert s("hi", {"target": "hi"}) == 1.0
    assert s("hi", {"target": "bye"}) == 0.0


def test_scorer_with_all_three_params() -> None:
    @scorer
    def s(input: str, output: str, expected: dict[str, str]) -> float:
        # Trivial echo check
        return 1.0 if output == input else 0.0

    assert scorer_params(s) == frozenset({"input", "output", "expected"})


def test_scorer_can_return_scorer_result() -> None:
    @scorer
    def s(output: str) -> ScorerResult:
        return ScorerResult(score=0.7, reasoning="halfway there")

    result = s("anything")
    assert isinstance(result, ScorerResult)
    assert result.score == 0.7
    assert result.reasoning == "halfway there"


def test_scorer_rejects_unknown_param_name_at_decoration() -> None:
    """Loud-fail at import, not at runtime — typos surface fast."""
    with pytest.raises(TypeError, match="unknown parameter"):

        @scorer
        def s(outpot: str) -> float:  # typo: outpot
            return 0.0


def test_scorer_rejects_multiple_unknown_params() -> None:
    with pytest.raises(TypeError, match=r"bar.*foo"):

        @scorer
        def s(foo: str, bar: int) -> float:
            return 0.0


def test_is_scorer_false_for_plain_function() -> None:
    def not_a_scorer(x: int) -> int:
        return x * 2

    assert not is_scorer(not_a_scorer)
    assert scorer_params(not_a_scorer) == frozenset()


def test_scorer_result_is_frozen() -> None:
    result = ScorerResult(score=0.5)
    with pytest.raises(AttributeError):
        result.score = 0.6  # type: ignore[misc]


def test_scorer_preserves_function_name_and_doc() -> None:
    @scorer
    def my_scorer(output: str) -> float:
        """My helpful docstring."""
        return 1.0

    # functools.wraps preserves __name__ and __doc__
    assert my_scorer.__name__ == "my_scorer"
    assert my_scorer.__doc__ == "My helpful docstring."
