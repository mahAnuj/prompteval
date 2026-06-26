"""prompteval — LLM eval framework with first-class token-cost tracking.

Public API as of Week 3:

  Cost (Week 2):
    Usage, CostBreakdown, ModelPricing       — data shapes
    compute_cost(model, usage)               — pure cost computation
    get_pricing(name), list_models()         — pricing table accessors
    UnknownModelError                        — typed exception

  Eval scorers (Week 3):
    scorer                                   — decorator marking a function as a scorer
    ScorerResult                             — richer return type with reasoning/metadata
    llm_judge(rubric, text)                  — LLM-as-judge helper
    LLMJudgeError                            — raised when judge response can't be parsed

  Coming Week 4: Eval runner — load dataset, iterate, call LLM, run scorers,
  record cost+latency, persist run.
"""

from prompteval.cost import (
    CostBreakdown,
    ModelPricing,
    UnknownModelError,
    Usage,
    compute_cost,
    get_pricing,
    list_models,
)
from prompteval.eval import (
    Eval,
    Example,
    ExampleResult,
    LLMJudgeError,
    RunResult,
    ScorerResult,
    llm_judge,
    load_run,
    run_eval,
    save_run,
    scorer,
)
from prompteval.version import __version__

__all__ = [
    "CostBreakdown",
    "Eval",
    "Example",
    "ExampleResult",
    "LLMJudgeError",
    "ModelPricing",
    "RunResult",
    "ScorerResult",
    "UnknownModelError",
    "Usage",
    "__version__",
    "compute_cost",
    "get_pricing",
    "list_models",
    "llm_judge",
    "load_run",
    "run_eval",
    "save_run",
    "scorer",
]
