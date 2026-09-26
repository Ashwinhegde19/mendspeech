# Week 7: TTS, Speaker Preservation, and Boundary Matched Reconstruction

> **Days 43 to 49**  
> **Navigation:** [← Week 6](Week_6_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 8 →](Week_8_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Build MendSpeech V1 as a cascaded selective repair baseline with explicit seam diagnostics.
>
> **v2 evidence gate:** Gate 6 requires a measured predicted-text repair path,
> tested abstention, and honest seam/feasibility outcomes in one `app/audio_lab.py`.
> Use one permitted TTS stack for two verified languages including one Indian
> language, repair prosody, and short-span latency. Adaptation and native
> streaming have separate capability gates; no promised training session or
> compute budget. No second stack, mobile port, emotion subsystem, or voice agent.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 43** | Serving contract and message schema | `Modal L4` | CORE | [Open Day 43](days/day_43.md) |
| **Day 44** | Async streaming service | `Modal L4` | CORE | [Open Day 44](days/day_44.md) |
| **Day 45** | Load test to saturation | `Modal L4` | CORE | [Open Day 45](days/day_45.md) |
| **Day 46** | LLM post-processing stage | `Modal L4` | CORE | [Open Day 46](days/day_46.md) |
| **Day 47** | Per-stage latency budget decomposition | `Modal L4` | CORE | [Open Day 47](days/day_47.md) |
| **Day 48** | End-to-end latency optimization round | `Modal L4` | CORE | [Open Day 48](days/day_48.md) |
| **Day 49** | Interim review | `Local CPU` | DROPPED | [Open Day 49](days/day_49.md) |

---

## Phase Focus

Serving, the LLM stage, and the latency budget

---

## Daily Detailed Operating Plans
### DAY 43: Serving contract and message schema
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_43.md`](days/day_43.md)

> **v3 STATUS: CORE** Phase P6 begins. Design the contract before implementing, or latency semantics get baked in wrong.
#### Learn
- WebSocket message schemas for streaming audio and incremental transcripts.
- What belongs in a partial result versus a final result.
- Backpressure semantics at the protocol level.
#### Build in MendSpeech
- Define the WebSocket message schema in `src/serve/schema.py`: audio chunks in, partial and final transcripts with confidence and latency out.
- Define timeout, disconnect, and cancellation behaviour in `src/serve/schema.py`.
#### Experiment and Measure
- Write the contract as a testable specification in `docs/day43_serving_contract.md`.
- Verify the schema round-trips in `tests/test_serve_schema.py`.
#### Required Output
['- `src/serve/schema.py`', '- `tests/test_serve_schema.py`', '- `docs/day43_serving_contract.md`']
#### Completion Check
> The message contract is explicit about partial versus final results, latency fields, and failure semantics.

---

### DAY 44: Async streaming service
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_44.md`](days/day_44.md)

> **v3 STATUS: CORE** One provider, one endpoint. The service must be measurable, not merely working.
#### Learn
- FastAPI and async WebSocket handling.
- Per-stream state isolation.
- Correct cancellation when a client disconnects mid-utterance.
#### Build in MendSpeech
- Implement the service in `src/serve/app.py` around the optimized Day 32 configuration.
- Containerize reproducibly in `infra/serve/`.
#### Experiment and Measure
- Verify concurrent streams do not share or corrupt cache state.
- Confirm a mid-utterance disconnect leaves no orphaned GPU work.
- Report cold start separately from warm latency.
#### Required Output
['- `src/serve/app.py`', '- `tests/test_serve_isolation.py`', '- `infra/serve/Dockerfile`', '- `infra/serve/README.md`']
#### Completion Check
> Concurrent streams are isolated, disconnects are clean, and cold start is reported separately from warm latency.

---

### DAY 45: Load test to saturation
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_45.md`](days/day_45.md)

> **v3 STATUS: CORE** The knee, not the maximum, is the number that matters.
#### Learn
- Load testing methodology: fixed hardware, fixed input, fixed configuration.
- Saturation behaviour and queue growth.
- Why throughput at saturation is not a user experience.
#### Build in MendSpeech
- Implement the load harness in `src/serve/loadtest.py` with configurable concurrency and fixed input.
#### Experiment and Measure
- Sweep concurrency until latency degrades; report the knee in `results/day45_load_curve.csv`.
- Report per-stream p50/p95/p99, queue depth, and dropped or delayed chunks at each level.
- Reproduce one controlled overload failure and one recovery in `docs/day45_failure_recovery.md`.
#### Required Output
['- `src/serve/loadtest.py`', '- `results/day45_load_curve.csv`', '- `docs/day45_failure_recovery.md`', '- `reports/day45_serving.md`']
#### Completion Check
> You can name the concurrency knee with measured evidence and show a reproduced failure and recovery.

---

### DAY 46: LLM post-processing stage
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_46.md`](days/day_46.md)

> **v3 STATUS: CORE** One small pinned model, behind an adapter, so the ASR result stays reproducible without it.
#### Learn
- Time-to-first-token versus full response.
- Streaming versus batched generation.
- Prefix caching and why repeated system context should be free.
#### Build in MendSpeech
- Add one small pinned LLM post-processing adapter in `src/llm/polish.py`; the core ASR path must run without it.
- Pin model, revision, quantization, and prompt template in `configs/llm.yaml`.
#### Experiment and Measure
- Measure TTFT and full-response latency separately.
- Measure prefix-cache hit rate across repeated requests and its effect on TTFT.
- Report quality change on the polished output, not only latency.
#### Required Output
['- `src/llm/polish.py`', '- `configs/llm.yaml`', '- `tests/test_llm_polish.py`', '- `results/day46_llm_latency.csv`']
#### Completion Check
> The LLM stage is measured for TTFT and full response, and the ASR result is still reproducible with it disabled.

---

### DAY 47: Per-stage latency budget decomposition
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_47.md`](days/day_47.md)

> **v3 STATUS: CORE** The headline artifact of the whole project. The question is where the time actually goes, not what feels slow.
#### Learn
- Separating queueing, model, decoding, network, and serialization time.
- Why a blended average hides the tail that users feel.
#### Build in MendSpeech
- Instrument every stage in `src/bench/budget.py` using the Day 26 harness conventions.
#### Experiment and Measure
- Decompose waveform-to-polished-text into VAD/endpointing, ASR, decode, LLM TTFT, LLM full response, and network/serialization.
- Report p50/p95/p99 per stage in `results/day47_latency_budget.csv`.
- Name the single stage that owns the p99 and state the largest available optimization target in `docs/day47_latency_budget.md`.
#### Required Output
['- `src/bench/budget.py`', '- `results/day47_latency_budget.csv`', '- `docs/day47_latency_budget.md`', '- `results/day47_latency_budget.png`']
#### Completion Check
> You can point at the stage that owns the tail with per-stage percentiles, and the claim is reproducible from one command.

---

### DAY 48: End-to-end latency optimization round
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_48.md`](days/day_48.md)

> **v3 STATUS: CORE** One targeted change, chosen by the Day 47 budget, then measured.
#### Learn
- Choosing one optimization from measured evidence rather than preference.
- Verifying that an end-to-end gain is real and not measurement drift.
#### Build in MendSpeech
- Apply the change the Day 47 budget identified as the largest target in `src/`.
- Re-run the full Day 47 decomposition after the change.
#### Experiment and Measure
- Report before/after p50/p95/p99 for the whole pipeline in `results/day48_e2e_optimization.csv`.
- Re-run enough repetitions to separate a real gain from noise.
- If the change did not help, say so and record the negative result.
#### Required Output
['- `results/day48_e2e_optimization.csv`', '- `docs/day48_optimization_outcome.md`', '- `app/audio_lab.py`']
#### Completion Check
> You have a measured end-to-end before/after, or a documented negative result with evidence.

---

### DAY 49: Interim review
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_49.md`](days/day_49.md)

> **v3 STATUS: DROPPED** in v3. Content absorbed into Days 47 and 48; the latency budget and the optimization outcome now serve as the review.
#### Learn
- No new material; this slot was the old repair milestone.
#### Build in MendSpeech
- No build.
#### Experiment and Measure
- No measurement.
#### Required Output
['None.']
#### Completion Check
> This slot is intentionally unused; the review content lives in Days 47 and 48.

---
