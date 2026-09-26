# Week 7

> **Days 43–49**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Serving, load, editor selection, and the correlated latency budget
> Ship one serving endpoint, measure the correlated latency budget, and make one evidence-driven optimization.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 43** | Serving contract and WebSocket message schema | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 43](days/day_43.md) |
| **Day 44** | Async streaming service | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 44](days/day_44.md) |
| **Day 45** | Load test to saturation and failure recovery | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 45](days/day_45.md) |
| **Day 46** | LLM stage measurement and consolidation | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 46](days/day_46.md) |
| **Day 47** | Per-stage correlated latency budget | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 47](days/day_47.md) |
| **Day 48** | End-to-end optimization round informed by the budget | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 48](days/day_48.md) |
| **Day 49** | Progress checkpoint against the plan (revised) | `Local CPU` | CORE | [Open Day 49](days/day_49.md) |

---

## Daily Detailed Operating Plans

### DAY 43: Serving contract and WebSocket message schema
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_43.md`](days/day_43.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 26](days/day_26.md), [Day 33](days/day_33.md)
> **Effort:** 2–3 focused hours.

#### Learn
- WebSocket message schemas, per-stream isolation, cancellation and backpressure semantics.

#### Build in MendSpeech
- Define schema in src/serve/schema.py: audio chunks in; partial/final transcripts, confidence, stage events and latency fields out.
- Specify timeout, disconnect, cancellation and bounded-queue semantics; write contract tests in tests/test_serve_schema.py.

#### Experiment and Measure
- Verify the schema round-trips a recorded session.
- Ensure latency fields match the contract definitions and never expose unmeasured values.

#### Required Output Artifacts
- `src/serve/schema.py`
- `tests/test_serve_schema.py`
- `docs/day43_serving_contract.md`

#### Completion Check
> A testable WebSocket contract with explicit latency, confidence, failure and backpressure semantics.

---

### DAY 44: Async streaming service
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_44.md`](days/day_44.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 33](days/day_33.md), [Day 43](days/day_43.md)
> **Effort:** 3–5 focused hours.

#### Learn
- FastAPI/async WebSocket handling, per-stream state isolation and clean cancellation.

#### Build in MendSpeech
- Implement the service in src/serve/app.py around the Day33 candidate, pinned one L4, serialized queue and bounded state.
- Containerize reproducibly under infra/serve/.

#### Experiment and Measure
- Verify concurrent streams do not share or corrupt cache/session state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work; report cold start separately from warm latency.

#### Required Output Artifacts
- `src/serve/app.py`
- `tests/test_serve_isolation.py`
- `infra/serve/Dockerfile`
- `infra/serve/README.md`

#### Completion Check
> Concurrent streams are isolated, disconnects are clean, and cold start is measured separately from warm latency.

---

### DAY 45: Load test to saturation and failure recovery
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_45.md`](days/day_45.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 44](days/day_44.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Load methodology, saturation/queue growth, and throughput-at-saturation versus user experience.

#### Build in MendSpeech
- Build a load harness in src/serve/loadtest.py (configurable concurrency, fixed input, bounded budgets).
- Reproduce overload, disconnect and recovery scenarios.

#### Experiment and Measure
- Sweep concurrency within the authorized limit; report the knee, offered vs achieved rates, per-stream p50/p95/p99, queue wait and rejected/failed work.
- Show that rejection does not masquerade as capacity; document recovery in docs/day45_failure_recovery.md.

#### Required Output Artifacts
- `src/serve/loadtest.py`
- `results/day45_load_curve.csv`
- `docs/day45_failure_recovery.md`
- `reports/day45_serving.md`

#### Completion Check
> A load curve to saturation, a named concurrency knee, and a reproduced failure-and-recovery case within budget.

---

### DAY 46: LLM stage measurement and consolidation
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_46.md`](days/day_46.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 15](days/day_15.md), [Day 32](days/day_32.md), [Day 40](days/day_40.md), [Day 45](days/day_45.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Server TTFT vs completion vs client-observed latency; prefix-cache support detection.

#### Build in MendSpeech
- Measure the Day15 editor (and Day40 RL editor if available) for TTFT/completion, quality, and prefix-cache hit/miss if the runtime exposes it.
- Consolidate prompt-only/SFT/RL editor results on the same frozen editor-test set with the Day13 quality metrics.

#### Experiment and Measure
- Compare editor variants at fixed workload; report misses, fallback rate, and quality.
- If co-residency/throughput prevents joint measurement, record the blocker; do not present isolated numbers as end-to-end.

#### Required Output Artifacts
- `configs/llm.yaml`
- `src/llm/polish.py`
- `results/day46_editor_variants.csv`
- `docs/day46_editor_selection.md`

#### Completion Check
> The editor variant used in serving is selected on measured quality/latency, and the choice is re-checked under load.

---

### DAY 47: Per-stage correlated latency budget
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_47.md`](days/day_47.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 44](days/day_44.md), [Day 45](days/day_45.md), [Day 46](days/day_46.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Critical-path latency attribution; tail ownership by request ID, not percentile sum.

#### Build in MendSpeech
- Instrument the full path in src/bench/budget.py with trace IDs, clock sync notes, stage start/end, queue/prefill/decode/generation/guard/network.
- Produce correlated per-request critical paths for the shipped configuration.

#### Experiment and Measure
- Report per-stage p50/p95/p99, counts, cold/warm and length slices.
- Identify the p99 owner by inspecting the same slow requests; do not sum percentiles or claim a sub-500ms end-to-end guarantee.

#### Required Output Artifacts
- `src/bench/budget.py`
- `results/day47_latency_budget.csv`
- `results/day47_latency_budget.png`
- `docs/day47_latency_budget.md`

#### Completion Check
> A reproducible, request-level decomposition that names the tail owner and the largest optimization target, with no unsupported end-to-end claim.

---

### DAY 48: End-to-end optimization round informed by the budget
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_48.md`](days/day_48.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 47](days/day_47.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Choosing one change from measured evidence and separating it from drift.

#### Build in MendSpeech
- Apply the change the Day47 budget identifies as the largest target in src/.
- Re-run the full Day47 decomposition and the Day32 scorecard after the change.

#### Experiment and Measure
- Report before/after p50/p95/p99 and quality with enough repetitions to separate a real gain from noise.
- A change that does not help, or that breaks a guard, is recorded as a negative result.

#### Required Output Artifacts
- `results/day48_e2e_optimization.csv`
- `docs/day48_optimization_outcome.md`
- `app/audio_lab.py`

#### Completion Check
> A measured end-to-end before/after tied to the budget, or a documented negative outcome with evidence.

---

### DAY 49: Progress checkpoint against the plan (revised)
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_49.md`](days/day_49.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 48](days/day_48.md)
> **Effort:** 0–1 focused hours.

#### Learn
- Comparing measured results to plan claims; re-planning from actual throughput.

#### Build in MendSpeech
- No build. Re-read the plan gate table and update the day-10-onward effort forecast from observed sessions.

#### Experiment and Measure
- Record which gates have evidence and which remain open.
- Adjust remaining estimates; do not compress evidence to preserve a date.

#### Required Output Artifacts
- `docs/day49_progress_review.md`

#### Completion Check
> A short, honest checkpoint that updates the forecast from observed work without claiming completion.

---
