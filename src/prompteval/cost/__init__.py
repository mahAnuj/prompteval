"""Cost-tracking core — pure functions for (model, usage) → USD.

Public surface:
- `Usage` — token-counts dataclass mirroring an OpenAI usage record
- `CostBreakdown` — per-component cost result from `compute_cost`
- `ModelPricing` — pricing for one model
- `compute_cost(model, usage)` — the pure cost function
- `get_pricing(model_name)` / `list_models()` — pricing table accessors
- `UnknownModelError` — raised when a model isn't in the pricing table
"""

from prompteval.cost.compute import CostBreakdown, Usage, compute_cost
from prompteval.cost.models import (
    PRICING,
    ModelPricing,
    UnknownModelError,
    get_pricing,
    list_models,
)

__all__ = [
    "PRICING",
    "CostBreakdown",
    "ModelPricing",
    "UnknownModelError",
    "Usage",
    "compute_cost",
    "get_pricing",
    "list_models",
]
