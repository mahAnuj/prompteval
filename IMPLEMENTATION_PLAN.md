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
| 1 | `prompteval init` CLI — bootstraps `evals/` folder from templates | ⏳ | |
| 1 | Templates folder (`prompts/v1.txt`, `prompts/v2.txt`, `dataset.jsonl`, `eval.py`, `.env.example`) | ⏳ | |
| 2 | Cost model — provider-agnostic `(model, usage) → USD` with correct cache-hit pricing (Anthropic first) | 📋 | |
| 2 | `prompteval models` CLI — list supported models + per-token pricing | 📋 | |
| 3 | `@scorer` decorator + return-shape contract | 📋 | |
| 3 | `llm_judge()` helper (Anthropic SDK, Sonnet by default) | 📋 | |
| 3 | 6–8 stock scorer templates (`exact_match`, `regex`, `contains`, `json_schema_valid`, `llm_judge_factuality`, `llm_judge_tone`, `length_between`, `not_empty`) | 📋 | |
| 3 | `prompteval scorer list` + `prompteval scorer copy <name>` CLI | 📋 | |
| 3–4 | `Eval` class + runner — load dataset, iterate, call LLM, run scorers, record usage + latency, persist run JSON to `.prompteval/runs/` | 📋 | |
| 4 | `prompteval run --prompt path --tag name` CLI | 📋 | |
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
| OpenAI cost model | Anthropic only for v1 to bound scope. ❄️ v0.2 |
| Tracing / observability | Phoenix and Langfuse already own this lane. Integrate, don't compete. ❄️ v0.2 |
| Runtime proxy / caching | Helicone / Portkey / LiteLLM exist. Different category. ❄️ never |
| AI assistant that writes scorers from a use-case description | Genuinely great idea, but a 3–4 week sub-project. Templates cover 80% of the pain for 5% of the effort. ❄️ v0.3 |
| Compare >2 prompts in one report | v1 ships pairwise. Multi-way ANOVA-style comparison needs different stats + UX. ❄️ v1.1 |
| Fine-tuning / training loop | Not what an eval framework does. ❄️ never |

---

## Roadmap beyond v1

| Version | Theme | Status |
|---|---|---|
| v0.2 | Phoenix OTel integration — ingest existing traces, add cost-comparison layer on top | ❄️ planned post-v1 |
| v0.2 | OpenAI cost model + tokenizer | ❄️ planned post-v1 |
| v0.2 | **Multi-provider model registry** — Anthropic + OpenAI + OpenAI-compatible endpoints (Ollama, vLLM, Together, Groq, Fireworks). Unlocks one-command **frontier-vs-open-source comparison** (e.g. `Sonnet vs Llama-3.3-70B on your task`). Note: v1 already supports this via two manual runs + `compare`; v0.2 makes it ergonomic and ships a launch post — *"Is Llama 3.3 actually cheaper than Sonnet for [your task]? Here's how to know in 10 minutes."* | ❄️ planned post-v1 |
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
| 2026-06-26 | Cost is a **comparison axis**, not a tracked metric | Verified gap across Braintrust, Langfuse, promptfoo, Phoenix, Inspect AI. The wedge. |
| 2026-06-26 | Phoenix is a future **integration target**, not a competitor | They have rich per-trace cost (OTel-LLM convention) but no comparison-axis use. v0.2 ingests their traces. |
| 2026-06-26 | "AI assistant writes scorers" deferred to v0.3 | Genuinely great idea. Scope creep risk in v1. Templates ship in v0.1 for 80% of the pain. |
| 2026-06-26 | Compare exactly 2 prompts in v1 | Multi-way comparison needs different stats + UX. v1.1. |
| 2026-06-26 | Pairwise statistical comparison via bootstrap CIs + paired t-test / Wilcoxon | Robust; doesn't assume normality; explains itself in plain English. Falls back to McNemar's for binary pass/fail scorers. |
| 2026-06-26 | Run persistence as JSON files in `.prompteval/runs/` (no DB) | Single-developer tool; git-friendly; zero setup. SQLite if v1.x needs query performance. |

---

## How to use this file

- **Before a coding session:** open this file, find the next `⏳` or earliest `📋` row, that's what you're building.
- **During the session:** when you ship something, change its status row to `✅` and add the commit hash to "Notes."
- **When you make a scope decision** (build / defer / change approach): append a row to **Decisions log** with date + rationale. Even if it feels small.
- **When you finish a week:** sanity check — is the row above the next `⏳` actually shipped, or are you carrying tech debt forward?
- **If the plan no longer matches reality:** update the plan first, then the code. Resist the urge to "just keep coding."

The kill date is 2026-08-16. **9 working weeks remaining as of 2026-06-26.** Status colors should march top-to-bottom; if a 📋 below the line you're working on starts looking impossible, defer it explicitly rather than letting it slip silently.
