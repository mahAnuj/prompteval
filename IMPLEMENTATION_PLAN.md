# Implementation plan

> Living source of truth for prompteval's v1 build. Anything in conflict between
> this file and the README — **this file wins.** README is marketing copy; this
> is the engineering plan.
>
> Update this file the same commit you ship code. If you find yourself building
> something not on this list, either add it explicitly (with rationale) or stop
> and reconsider. Scope creep killed the AI Runtime Platform PDF idea; not this one.

**Project start:** 2026-06-21
**Kill date:** 2026-08-16 (10 weeks)
**Repo:** https://github.com/mahAnuj/prompteval
**npm/PyPI status:** not yet published

---

## v1 scope at a glance

The single sentence to remember: **`prompteval init` → write scorers → `prompteval run` → `prompteval compare` → ship/don't-ship recommendation.** Everything else is out of scope for v1.

See [README → User journey (v1)](README.md#user-journey-v1) for the full Priya story.

---

## Week-by-week status

Legend:
- ✅ **shipped** — code merged to main, tests green
- 🚧 **in progress** — actively building this week
- ⏳ **next** — top of the queue
- 📋 **planned** — agreed v1 scope, not started
- ❄️ **deferred** — pushed to a later version, see Roadmap section

| Week | Block | Status | Notes / commits |
|---|---|---|---|
| 0 | Project bootstrap (uv, ruff, mypy --strict, pytest, CI on 3.12+3.13) | ✅ | 2026-06-21 — first commit, CI green on push |
| 0 | Competitor scan + verification (Braintrust cookbook + Phoenix repo source) | ✅ | 2026-06-26 — `docs/competitor-scan.md` |
| 0 | v1 scope locked, README rewritten with user journey + "is/is not" + "not building" lists | ✅ | 2026-06-26 |
| 0 | This implementation plan | ✅ | 2026-06-26 |
| 1 | `prompteval init` CLI — bootstraps `evals/` folder from templates | ✅ | 2026-06-26 — silent + opinionated; `--dir` / `--force` flags; 13 tests |
| 1 | Templates folder (`prompts/v1.txt`, `prompts/v2.txt`, `dataset.jsonl`, `eval.py`, `.env.example`) | ✅ | 2026-06-26 — support-assistant persona matches README; gpt-4o-mini default |
| 2 | Cost model — provider-agnostic `(model, usage) → USD` with correct cached-input pricing (OpenAI first) | ✅ | 2026-06-26 — `cost/{models,compute}.py`; pricing in YAML; 33 tests |
| 2 | `prompteval models` CLI — list supported models + per-token pricing | ✅ | 2026-06-26 — `models list [--json]`, `models price <model> [--json]` |
| 3 | `@scorer` decorator + return-shape contract | ✅ | 2026-06-26 — signature-driven dispatch (subset of {input, output, expected}); ScorerResult dataclass for reasoning+metadata; 9 tests |
| 3 | `llm_judge()` helper (OpenAI SDK, `gpt-4o-mini` by default) | ✅ | 2026-06-26 — parses first number from response; validates [0, 1]; LLMJudgeError for unparseable / out-of-range; 13 tests with mocked client |
| 3 | 6–8 stock scorer templates | ✅ | 2026-06-26 — shipped 9: exact_match, contains_substring, regex_match, not_empty, length_between, json_schema_valid, mentions_required_terms, llm_judge_tone, llm_judge_factuality; 40 tests |
| 3 | `prompteval scorer list` + `prompteval scorer copy <name>` CLI | ✅ | 2026-06-26 — sources copied via `inspect.getsource` so output is always in sync with the live function; 6 tests |
| 3–4 | `Eval` class + runner — load dataset, iterate, call LLM, run scorers, record usage + latency, persist run JSON to `.prompteval/runs/` | ✅ | 2026-06-26 — `eval/runner.py` (Eval, Example, RunnerResult, ExampleResult, RunResult, run_eval, default_runner, _load_dataset, _call_scorer, _sum_usages); `eval/persistence.py` (save_run, load_run); 33 tests for runner + 8 for persistence |
| 4 | `prompteval run --prompt path --tag name` CLI | ✅ | 2026-06-26 — discovers Eval via `importlib.util.spec_from_file_location`; `--tag` / `--eval-file` / `--model` flags; per-example progress; 6 CLI tests with mocked OpenAI |
| 5 | `prompteval compare <tag-a> <tag-b>` — paired deltas, bootstrap CIs, p-values, plain-text table | 📋 | |
| 5 | Plain-English verdict + recommendation lines | 📋 | |
| 6 | HTML report writer (single static file, no server) | 📋 | |
| 6 | `--fail-on cost+X%,quality-Y%` for CI | 📋 | |
| 6 | GitHub Action template + example workflow | 📋 | |
| 7 | Dogfood eval against `mcp-multi-db` — real evals in that repo, real numbers | 📋 | |
| 7 | First blog post draft (working title: *"How I cut my mcp-multi-db's token cost 38% without breaking tool accuracy"*) | 📋 | |
| 8 | `docs/writing-scorers.md` guide | 📋 | |
| 8 | API stability sweep — freeze public surface; mark internal modules with leading `_` | 📋 | |
| 9 | PyPI publish (test-PyPI first, then real) | 📋 | |
| 9 | GitHub release v0.1.0 + release notes | 📋 | |
| 10 | Launch — LinkedIn, X thread, awesome-llm-evals PR, second blog post | 📋 | |

---

## v1 explicit non-goals

These are valid feature requests we'll **decline** for v1 to keep scope honest:

| Non-goal | Why deferred |
|---|---|
| Web UI / dashboard server | Static HTML report covers v1 reporting need. Server = ops + auth + state. ❄️ v1.x |
| Multi-user / auth / cloud | Single-developer tool by design for v1. ❄️ never |
| Anthropic cost model | OpenAI only for v1 to bound scope (see decisions log 2026-06-26). ❄️ v0.2 |
| Tracing / observability | Phoenix and Langfuse already own this lane. Integrate, don't compete. ❄️ v0.2 |
| Runtime proxy / caching | Helicone / Portkey / LiteLLM exist. Different category. ❄️ never |
| AI assistant that writes scorers from a use-case description | Genuinely great idea, but a 3–4 week sub-project. Templates cover 80% of the pain for 5% of the effort. ❄️ v0.3 |
| Compare >2 prompts in one report | v1 ships pairwise. Multi-way ANOVA-style comparison needs different stats + UX. ❄️ v1.1 |
| Fine-tuning / training loop | Not what an eval framework does. ❄️ never |

---

## Roadmap beyond v1

| Version | Theme | Status |
|---|---|---|
| v1.1 | **Callable-runner hook for multi-agent / multi-step evals** — `Eval(runner=my_func)` accepts any `Callable[[Example], RunResult]` so users can wrap CrewAI / AutoGen / LangGraph / custom pipelines. v1's internal `Eval` already takes a `runner` (default = single-prompt single-call); v1.1 exposes the override. Cost = sum of all LLM calls per example, no `compute_cost` math changes. Deferred from v1 because (a) splits the "edit prompts/v1.txt, run, compare" README on-ramp, (b) needs its own test suite for multi-call aggregation + error semantics, (c) we don't have a real multi-agent workload to validate the abstraction against. | ❄️ planned post-v1 |
| v0.2 | Phoenix OTel integration — ingest existing traces, add cost-comparison layer on top | ❄️ planned post-v1 |
| v0.2 | Anthropic cost model + tokenizer | ❄️ planned post-v1 |
| v0.2 | **Multi-provider model registry** — OpenAI + Anthropic + OpenAI-compatible endpoints (Ollama, vLLM, Together, Groq, Fireworks). Unlocks one-command **frontier-vs-open-source comparison** (e.g. `GPT-4o vs Llama-3.3-70B on your task`). Note: v1 already supports this via two manual runs + `compare`; v0.2 makes it ergonomic and ships a launch post — *"Is Llama 3.3 actually cheaper than GPT-4o for [your task]? Here's how to know in 10 minutes."* | ❄️ planned post-v1 |
| v0.3 | `prompteval scorer init` — interactive AI assistant generates scorers from use-case description | ❄️ planned post-v1, train on v0.1 user feedback |
| v1.1 | Compare >2 prompts in a single report (multi-way) | ❄️ later |
| v1.x | Web dashboard (local-only, not a daemon) | ❄️ later |

---

## Architecture sketch (subject to refactor as code lands)

```
src/prompteval/
├── __init__.py         — public exports (Eval, scorer, llm_judge, run, compare)
├── version.py          — single source of truth for __version__
├── cli.py              — click entry point + subcommands (init, run, compare, scorer, models)
├── cost/
│   ├── __init__.py
│   ├── models.py       — pricing table per (provider, model)
│   └── compute.py      — (usage, model) → USD, including cache-hit math
├── eval/
│   ├── __init__.py
│   ├── scorer.py       — @scorer decorator, ScorerResult shape
│   ├── llm_judge.py    — llm_judge() helper (calls Claude)
│   ├── runner.py       — Eval class, dataset iteration, persistence
│   └── templates/      — stock scorers users copy
├── compare/
│   ├── __init__.py
│   ├── stats.py        — paired deltas, bootstrap CI, p-values
│   └── report.py       — text + HTML output
├── init/
│   └── templates/      — files copied by `prompteval init`
└── _internal/          — anything not exported; future-proofs the public API
tests/
├── test_version.py
├── test_cost_*.py
├── test_scorer_*.py
├── test_runner_*.py
└── test_compare_*.py
docs/
├── competitor-scan.md       — ✅ shipped
├── writing-scorers.md       — 📋 Week 8
└── architecture.md          — 📋 Week 8
```

---

## Decisions log

Decisions made + the rationale, in commit order. Append; don't rewrite.

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-21 | Python over TypeScript | Eval-tooling ecosystem is Python-native (Braintrust, Phoenix, Langfuse, promptfoo, Inspect AI all Python). Stretches founder language coverage after shipping mcp-multi-db in TS. |
| 2026-06-21 | uv as package manager | Modern Python tooling. Faster than pip/poetry. Single tool for venv + install + lock. |
| 2026-06-21 | mypy `--strict` from day 1 | Catches bugs runtime won't. Cheap insurance. |
| 2026-06-21 | MIT license | Convention in Python OSS world; less friction than ISC for downstream users. |
| 2026-06-26 | Anthropic-only cost model for v1 | OpenAI added in v0.2. Scope discipline > universal day-1 support. |
| 2026-06-26 | **REVISED: OpenAI-only for v1, Anthropic moves to v0.2** | Original decision assumed founder had Anthropic API access (because they use Claude Code). Wrong assumption — Claude Code subscription ≠ API access. Founder has an existing OpenAI account. Switching saves a net-new vendor signup, doesn't compromise the wedge (cost-vs-quality demonstrable on OpenAI just as well), and actually *improves* the v0.2 multi-provider launch post (GPT-4o is more mainstream-recognizable than Sonnet for a "vs Llama 3.3" headline). Meta-lesson: validate provider access before locking. |
| 2026-06-26 | `init` defaults: silent + opinionated (no interactive prompts), `evals/` as folder name, `gpt-4o-mini` as template default model | Fastest path to first run. `--dir` / `--force` flags cover the edge cases. Interactive prompts can be added later only if real users ask. |
| 2026-06-26 | Templates live in `prompteval.init.templates` package, read via `importlib.resources` | Standard Python idiom. Works from wheel + source + zipped installs without `__file__` hacks. `__pycache__` gets ignored in `_walk` (caught by smoke test, not theory). |
| 2026-06-26 | One source→target rename: `env.example` → `.env.example` | Dotfiles can be excluded by some packaging tools; storing the source un-dotted keeps the package contents predictable. |
| 2026-06-26 | `templates/` excluded from ruff + mypy | Templates reference v0.1 public API that doesn't exist yet (Week 3-4 work). Treated as data files, not code. Linters off, contents preserved verbatim. |
| 2026-06-26 | Pricing data in YAML (`src/prompteval/cost/pricing.yaml`), not Python | First instinct was Python (type-checked, no deps). User pushback: YAML is easier for non-coders / future contributors to read + edit, separates data from code. Refactored. pyyaml is a 100KB well-maintained dep, validation runs at import time so a bad YAML fails the test suite, not user code. Lesson: bias toward the schema that minimises friction for the *next* change, not the convenience of the *current* author. |
| 2026-06-26 | Pricing schema stays OpenAI-shaped in v0.1, polymorphic refactor deferred to v0.2 | The right long-term design is per-provider pricing classes (`OpenAIPricing`, `AnthropicPricing`, `SimplePricing`) with `compute_cost` dispatching on `pricing.provider`. Building that abstraction today, without ever having implemented Anthropic, risks the wrong abstraction. We add the `provider: str` field as the dispatch seam (every YAML entry must declare it), document the refactor plan in both `models.py` and `compute.py` module docstrings, and ship one provider correctly. v0.2 introduces the polymorphic shape with Anthropic as the second-provider ground truth. "Design for one, ship for one, refactor for two." |
| 2026-06-26 | Cost is a **comparison axis**, not a tracked metric | Verified gap across Braintrust, Langfuse, promptfoo, Phoenix, Inspect AI. The wedge. |
| 2026-06-26 | Phoenix is a future **integration target**, not a competitor | They have rich per-trace cost (OTel-LLM convention) but no comparison-axis use. v0.2 ingests their traces. |
| 2026-06-26 | "AI assistant writes scorers" deferred to v0.3 | Genuinely great idea. Scope creep risk in v1. Templates ship in v0.1 for 80% of the pain. |
| 2026-06-26 | Compare exactly 2 prompts in v1 | Multi-way comparison needs different stats + UX. v1.1. |
| 2026-06-26 | Pairwise statistical comparison via bootstrap CIs + paired t-test / Wilcoxon | Robust; doesn't assume normality; explains itself in plain English. Falls back to McNemar's for binary pass/fail scorers. |
| 2026-06-26 | Run persistence as JSON files in `.prompteval/runs/` (no DB) | Single-developer tool; git-friendly; zero setup. SQLite if v1.x needs query performance. |
| 2026-06-26 | Multi-agent / multi-step evals (CrewAI, AutoGen, LangGraph, custom pipelines) deferred to v1.1 | v1 ships single-LLM-call-per-example. The architectural fix is a `runner: Callable[[Example], RunResult]` hook on `Eval`; v1's internal design includes this seam (default runner = single OpenAI call). v1.1 exposes the user-facing override. Reasons to defer: (a) the README on-ramp splits when there are two valid run-shapes, (b) multi-call aggregation needs its own test suite, (c) we don't have a real multi-agent workload to validate the abstraction against. |
| 2026-06-26 | Scorers use signature-driven dispatch over fixed positional args | Three valid param names (`input`, `output`, `expected`), scorer declares any subset, runner inspects the signature and passes only what's asked. Same pattern as pytest fixtures. Cleaner than `(input, output, expected) -> float` everywhere — simple scorers can be `def my(output)` without the unused-args noise. Unknown param names raise `TypeError` at decoration time, not call time, so typos fail at import. |
| 2026-06-26 | LLM-judge costs are NOT aggregated into v1 eval cost report | Judge calls happen inside scorer functions, separate from the prompt-vs-prompt comparison we're measuring. Aggregating them would muddy "v1 prompt vs v2 prompt cost" (the wedge). v0.2 adds judge-cost separation — report shows "prompt cost: $X, judge overhead: $Y." For now: documented in `llm_judge` docstring as "operational overhead, not part of the v1 vs v2 cost comparison." |
| 2026-06-26 | `scorer copy` outputs source via `inspect.getsource`, not via a templates folder | Stock scorers are real `@scorer` functions in `eval/stock.py`. `inspect.getsource` reads the source directly from the installed wheel. No template-file duplication; what runs and what gets copied are guaranteed identical. |
| 2026-06-26 | `RunnerResult.usages: list[Usage]` from day 1 (v1 always returns `[single]`) | Future-proofs the wire shape so v1.1's multi-agent runners (CrewAI, AutoGen, LangGraph) don't force a breaking change to the runner protocol. v1 always returns a single-element list; `_sum_usages` collapses it for the single-model cost path. v1.1 may add per-call model tags — that's a different shape change handled then. |
| 2026-06-26 | Per-example errors recorded, not raised | If one of 20 examples crashes, the run continues; that example records `error` and `cost=0`; aggregates skip errored examples. Failing the whole batch on one bad example is bad UX for a tool people will run interactively. Errors are still visible in the persisted JSON for post-mortem. |
| 2026-06-26 | Run persistence as `<tag>.json`, latest-wins-on-rerun | Simple identity. Re-running with the same tag overwrites. If you want history, use unique tags (`baseline-v1`, `baseline-v2`) and rely on git. Filename sanitisation strips path separators — defence in depth against malicious tags trying to escape the runs dir. |
| 2026-06-26 | Eval discovery via `importlib.util.spec_from_file_location` + first-`Eval`-instance | Standard Python pattern. We add the eval file's parent dir to sys.path so relative imports inside the user's eval.py work — matching how `python evals/eval.py` would behave. Import errors and missing-Eval-instance both surface as friendly `ClickException`s, not stack traces. |

---

## How to use this file

- **Before a coding session:** open this file, find the next `⏳` or earliest `📋` row, that's what you're building.
- **During the session:** when you ship something, change its status row to `✅` and add the commit hash to "Notes."
- **When you make a scope decision** (build / defer / change approach): append a row to **Decisions log** with date + rationale. Even if it feels small.
- **When you finish a week:** sanity check — is the row above the next `⏳` actually shipped, or are you carrying tech debt forward?
- **If the plan no longer matches reality:** update the plan first, then the code. Resist the urge to "just keep coding."

The kill date is 2026-08-16. **9 working weeks remaining as of 2026-06-26.** Status colors should march top-to-bottom; if a 📋 below the line you're working on starts looking impossible, defer it explicitly rather than letting it slip silently.
