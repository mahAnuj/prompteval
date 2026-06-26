"""prompteval.eval — scorer primitives + LLM-as-judge helper.

Week 3 surface (this module's public API):

- `scorer` — decorator that marks a function as a scorer
- `ScorerResult` — richer return type for scorers that want to surface reasoning
- `llm_judge(rubric, text)` — ask an LLM to score text against a rubric
- `LLMJudgeError` — raised when llm_judge can't parse a score
- `stock` — submodule with 9 ready-to-use scorers (import or copy via CLI)

Week 4 will add the `Eval` runner that iterates a dataset and calls each
scorer per example.
"""

from prompteval.eval import stock
from prompteval.eval.judge import LLMJudgeError, llm_judge
from prompteval.eval.scorer import ScorerResult, is_scorer, scorer, scorer_params

__all__ = [
    "LLMJudgeError",
    "ScorerResult",
    "is_scorer",
    "llm_judge",
    "scorer",
    "scorer_params",
    "stock",
]
