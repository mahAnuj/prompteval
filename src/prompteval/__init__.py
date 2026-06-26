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

from prompteval.compare import (
    ComparisonReport,
    GateBreach,
    GateClause,
    GateSpecError,
    MetricDelta,
    compute_comparison,
    evaluate_gates,
    parse_gate_spec,
    render_html,
    render_text,
)
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
    "ComparisonReport",
    "CostBreakdown",
    "Eval",
    "Example",
    "ExampleResult",
    "GateBreach",
    "GateClause",
    "GateSpecError",
    "LLMJudgeError",
    "MetricDelta",
    "ModelPricing",
    "RunResult",
    "ScorerResult",
    "UnknownModelError",
    "Usage",
    "__version__",
    "compute_comparison",
    "compute_cost",
    "evaluate_gates",
    "get_pricing",
    "list_models",
    "llm_judge",
    "load_run",
    "parse_gate_spec",
    "render_html",
    "render_text",
    "run_eval",
    "save_run",
    "scorer",
]
