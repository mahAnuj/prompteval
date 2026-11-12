"""Generate the two runs behind `prompteval demo` (#11).

ILLUSTRATIVE DATA, not recorded from a model: no API key was available when
the demo was built, and inventing "recorded" numbers would be dishonest. The
runs are generated here, deterministically (fixed seed), to show what a real
comparison looks like: a support-reply prompt, before and after shortening it.
Every output of `prompteval demo` says so. To replace them with real runs,
record two with `prompteval run` and copy their JSON over the files in
src/prompteval/demo/.

    uv run python scripts/build_demo_runs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prompteval.cost import Usage, compute_cost, resolve_pricing
from prompteval.eval.persistence import save_run
from prompteval.eval.runner import ExampleResult, RunResult, ScoreOutcome

OUT = Path(__file__).resolve().parents[1] / "src" / "prompteval" / "demo"
MODEL = "gpt-4o-mini"
N = 24
TOPICS = [
    "refund for a damaged item",
    "late delivery",
    "wrong size sent",
    "cancel an order",
    "change the delivery address",
    "promo code not working",
    "charged twice",
    "item out of stock",
    "return window question",
    "warranty claim",
]


def build(
    tag: str, prompt: str, in_tokens: int, out_tokens: float, quality_shift: float, seed: int
) -> RunResult:
    rng = np.random.default_rng(seed)
    pricing = resolve_pricing(MODEL)
    base_terms = np.random.default_rng(7).uniform(0.70, 0.98, N)  # shared difficulty per example
    base_tone = np.random.default_rng(8).uniform(0.72, 0.95, N)
    examples = []
    for i in range(N):
        out = max(40, int(rng.normal(out_tokens, out_tokens * 0.15)))
        cost = compute_cost(
            MODEL,
            Usage(prompt_tokens=in_tokens + int(rng.integers(40, 120)), completion_tokens=out),
            pricing,
        )
        terms = float(np.clip(base_terms[i] + quality_shift + rng.normal(0, 0.02), 0, 1))
        tone = float(np.clip(base_tone[i] + rng.normal(0, 0.02), 0, 1))
        examples.append(
            ExampleResult(
                id=f"support-{i + 1:02d}",
                input=f"Customer asks about: {TOPICS[i % len(TOPICS)]}",
                expected={},
                output="(illustrative run: no model output recorded)",
                scores=[
                    ScoreOutcome("mentions_required_terms", round(terms, 3)),
                    ScoreOutcome("professional_tone", round(tone, 3)),
                ],
                cost=cost,
                latency_s=round(float(rng.normal(0.9 * out / 250 + 0.4, 0.08)), 3),
            )
        )
    return RunResult(
        tag=tag,
        eval_name="support-assistant (demo)",
        model=MODEL,
        prompt_path=f"prompts/{tag}.txt",
        prompt_text=prompt,
        started_at="2026-11-12T00:00:00+00:00",
        finished_at="2026-11-12T00:00:30+00:00",
        examples=examples,
        dataset_sha256="demo",
        price_source="illustrative (catalog price for gpt-4o-mini)",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intro = "You are a support assistant for an online store."
    long_prompt = f"{intro} (long version: policies, tone guide, 12 examples)"
    short_prompt = f"{intro} (short version: policies and tone guide only)"
    save_run(
        build("baseline", long_prompt, in_tokens=1450, out_tokens=210, quality_shift=0.0, seed=1),
        OUT,
    )
    save_run(
        build(
            "short-prompt",
            short_prompt,
            in_tokens=620,
            out_tokens=150,
            quality_shift=-0.005,
            seed=2,
        ),
        OUT,
    )
    print(f"wrote {OUT}/baseline.json and {OUT}/short-prompt.json")


if __name__ == "__main__":
    main()
