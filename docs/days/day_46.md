# Day 46: LLM stage measurement and consolidation

> **Week 7 • Day 4 of 7**
> **Navigation:** [← Day 45](day_45.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 47 →](day_47.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 15](day_15.md), [Day 32](day_32.md), [Day 40](day_40.md), [Day 45](day_45.md)
> **Effort:** 2–4 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Server TTFT vs completion vs client-observed latency; prefix-cache support detection.

---

### 2. Build in MendSpeech
- Measure the Day15 editor (and Day40 RL editor if available) for TTFT/completion, quality, and prefix-cache hit/miss if the runtime exposes it.
- Consolidate prompt-only/SFT/RL editor results on the same frozen editor-test set with the Day13 quality metrics.

---

### 3. Experiment and Measure
- Compare editor variants at fixed workload; report misses, fallback rate, and quality.
- If co-residency/throughput prevents joint measurement, record the blocker; do not present isolated numbers as end-to-end.

---

### 4. Required Output Artifacts
- `configs/llm.yaml`
- `src/llm/polish.py`
- `results/day46_editor_variants.csv`
- `docs/day46_editor_selection.md`

---

### 5. Completion Check
> **Definition of Done for Day 46:**
> The editor variant used in serving is selected on measured quality/latency, and the choice is re-checked under load.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.
