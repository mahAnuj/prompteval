"""prompteval.eval — eval primitives, LLM-as-judge, and the Eval runner.

Public API surface (as of Weeks 3-4):

Scorers (Week 3):
- `scorer` — decorator that marks a function as a scorer
- `ScorerResult` — richer return type for scorers that want to surface reasoning
- `llm_judge(rubric, text)` — ask an LLM to score text against a rubric
- `LLMJudgeError` — raised when llm_judge can't parse a score
- `stock` — submodule with 9 ready-to-use scorers (import or copy via CLI)

Runner (Week 3-4):
- `Eval` — eval definition (name, dataset, scorers, model)
- `Example`, `RunnerResult`, `ScoreOutcome`, `ExampleResult`, `RunResult` — data shapes
- `run_eval(eval_def, prompt_path, tag)` — orchestration entry point
- `default_runner` — the single-OpenAI-call runner (v1)
- `save_run` / `load_run` — persistence to `.prompteval/runs/<tag>.json`

Week 5 adds `compare(run_a, run_b)` for the paired delta report.
"""

from prompteval.eval import stock
from prompteval.eval.judge import LLMJudgeError, llm_judge
from prompteval.eval.persistence import DEFAULT_RUNS_DIR, load_run, save_run
from prompteval.eval.runner import (
    Eval,
    Example,
    ExampleResult,
    Runner,
    RunnerResult,
    RunResult,
    ScoreOutcome,
    default_runner,
    run_eval,
)
from prompteval.eval.scorer import ScorerResult, is_scorer, scorer, scorer_params

__all__ = [
    "DEFAULT_RUNS_DIR",
    "Eval",
    "Example",
    "ExampleResult",
    "LLMJudgeError",
    "RunResult",
    "Runner",
    "RunnerResult",
    "ScoreOutcome",
    "ScorerResult",
    "default_runner",
    "is_scorer",
    "llm_judge",
    "load_run",
    "run_eval",
    "save_run",
    "scorer",
    "scorer_params",
    "stock",
]
