# Day 15: Early ASR-to-editor baseline and resource pilot

> **Week 3 • Day 1 of 7**
> **Navigation:** [← Day 14](day_14.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 16 →](day_16.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 13](day_13.md), [Day 14](day_14.md)
> **Effort:** 3–5 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Causal-LM generation, prompt/output token limits and KV memory.
- Server/client TTFT versus completion and validated delivery.

---

### 2. Build in MendSpeech
- Select one small causal editor candidate via docs/EDITOR_AND_RL_CONTRACT.md, pin permitted weights/tokenizer/template and isolated optional environment; do not add unverified packages to core dependencies.
- Build src/llm/polish.py and an app/audio_lab.py file/replayed-audio path using the existing ASR, deterministic/identity/LLM choices, guard and bypass. Label it offline/replayed, not completed live streaming.
- Pilot one-L4 ASR+editor co-residency with fixed token caps, serialized work and combined peak memory. ASR text is data, not instructions; no tools or agents.

---

### 3. Experiment and Measure
- Measure validation quality of raw/deterministic/prompt-only outputs, fallback, names/numbers/negation and context-free correction risks.
- Capture correlated stage events, server TTFT/completion, startup and combined memory. Record CPU threads/topology; a failed co-residency test requests scope review, not a hidden second GPU.

---

### 4. Required Output Artifacts
- `src/llm/polish.py`
- `tests/test_llm_polish.py`
- `configs/llm.yaml`
- `infra/editor/requirements.txt`
- `docs/editor_model_card.md`
- `app/audio_lab.py`
- `results/day15_e2e_baseline.csv`
- `results/day15_resource_pilot.csv`

---

### 5. Completion Check
> **Definition of Done for Day 15:**
> One real ASR→guarded-editor baseline runs before optimization; quality, TTFT/completion, memory and topology evidence are recorded without a sub-500ms promise.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.
