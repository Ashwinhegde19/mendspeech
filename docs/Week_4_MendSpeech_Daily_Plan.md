# Week 4: FastConformer and Efficient Encoder Behavior

> **Days 22 to 28**  
> **Navigation:** [← Week 3](Week_3_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 5 →](Week_5_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Measure why FastConformer is efficient and freeze a reproducible baseline.
>
> **v2 gate evidence:** Follow Gate 3 in [the execution plan](REVISED_EXECUTION_PLAN.md), shared with Week 3. This week has **5 build sessions: Days 23, 24, 25, 26, and 28**; the combined encoder block has **9 build sessions**, with completion based on evidence rather than a calendar target.
> These are base specification slots. The new decoder/LM experiment adds work
> within Days 24/26/28; estimate it after compatibility checks, not by assuming
> it fits the old session count. An incomplete decoder comparison needs a scope
> review before Gate 3 closes, even if independent streaming work proceeds.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 22** | Why FastConformer exists | `Local CPU` | LEARN-ONLY | [Open Day 22](days/day_22.md) |
| **Day 23** | Temporal subsampling experiment | `Modal L4 useful` | CORE | [Open Day 23](days/day_23.md) |
| **Day 24** | Pretrained streaming ASR baseline and capability check | `Modal L4` | CORE | [Open Day 24](days/day_24.md) |
| **Day 25** | Context and lookahead cost | `Modal L4` | CORE | [Open Day 25](days/day_25.md) |
| **Day 26** | Efficiency benchmark harness | `Modal L4` | CORE | [Open Day 26](days/day_26.md) |
| **Day 27** | Profiling the streaming model | `Modal L4` | CORE | [Open Day 27](days/day_27.md) |
| **Day 28** | torch.compile and graph capture | `Modal L4` | CORE | [Open Day 28](days/day_28.md) |

---

## Phase Focus

FastConformer, capability record, and the benchmark harness

---

## Daily Detailed Operating Plans
### DAY 22: Why FastConformer exists
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_22.md`](days/day_22.md)

> **v3 STATUS: LEARN-ONLY** Merged into Day 23; paper notes and compute estimates only.
#### Learn
- Sequence length as an attention cost driver.
- Subsampling before expensive encoder blocks.
- Depthwise separable convolution.
- Local and limited context attention.
#### Build in MendSpeech
- No standalone build. Day 23 incorporates the comparison checklist and diagram.
#### Experiment and Measure
- Day 23 incorporates attention-matrix estimates before and after temporal subsampling.
#### Required Output
['None for this learn-only session; retained notes and estimate paths are produced within Day 23.']
#### Completion Check
> You can explain FastConformer as a set of concrete efficiency choices, not just a faster model name.

---

### DAY 23: Temporal subsampling experiment
- **Compute:** `Modal L4 useful`
- **Dedicated Daily File:** [`docs/days/day_23.md`](days/day_23.md)

> **v3 STATUS: CORE** Absorbs Day 22: quantify how much sequence length subsampling removes, and what that saves.
#### Learn
- Convolutional subsampling.
- Temporal resolution.
- Information loss versus compute reduction.
#### Build in MendSpeech
- Implement a small subsampling front end in `src/models/subsampling.py`.
- Track frames per second before and after each stage.
- Incorporate Day 22's comparison checklist into `docs/day22_fastconformer_notes.md`.
#### Experiment and Measure
- Compare 2x, 4x, and 8x temporal reduction on tensor length, runtime, and rough output behavior.
- Estimate attention-matrix size before and after subsampling.
#### Required Output
['- `src/models/subsampling.py`', '- `results/day23_subsampling.csv`', '- `docs/day22_fastconformer_notes.md`', '- `results/day22_compute_estimates.csv`']
#### Completion Check
> You can quantify how subsampling changes sequence length and downstream attention cost.

---

### DAY 24: Pretrained streaming ASR baseline and capability check
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_24.md`](days/day_24.md)

> **v3 STATUS: CORE** Record what the selected checkpoint actually supports before later phases depend on it.
#### Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Cache-aware inference, right-context controls, export support, and tokenizer language coverage.
#### Build in MendSpeech
- Run a current streaming-capable ASR checkpoint on clean and damaged sets in `src/asr/streaming_runner.py`.
- Record model revision and all inference settings.
- Record capability status for cache-aware inference, supported right-context values and units, runtime context switching, intended export path, and language coverage in `configs/model_baseline.yaml`.
#### Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Use minimal supported smoke checks where feasible and record failures early.
- Estimate later-phase effort from the capability record; do not start a model hunt to fill a gap.
#### Required Output
['- `src/asr/streaming_runner.py`', '- `results/day24_baseline.csv`', '- `configs/model_baseline.yaml`']
#### Completion Check
> You have a reproducible baseline with model, data, hardware, and settings fixed, plus an evidence-backed capability record.

---

### DAY 25: Context and lookahead cost
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_25.md`](days/day_25.md)

> **v3 STATUS: CORE** Lookahead is a latency knob; this session measures what it costs and buys.
#### Learn
- Right context versus left context.
- Algorithmic latency versus accuracy.
- Why future context cannot be free.
#### Build in MendSpeech
- Implement fixed left/right context settings in `src/streaming/context.py`.
- Log the context configuration with every result.
#### Experiment and Measure
- Compare at least two supported context settings on the same subset.
- Plot WER against measured algorithmic latency and mark the Pareto-efficient points.
- Report where errors cluster near chunk boundaries.
#### Required Output
['- `src/streaming/context.py`', '- `tests/test_context.py`', '- `results/day25_context_tradeoff.csv`']
#### Completion Check
> You can explain exactly why future context creates latency, with a measured curve rather than an assertion.

---

### DAY 26: Efficiency benchmark harness
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_26.md`](days/day_26.md)

> **v3 STATUS: CORE** One harness serves every later measurement. A benchmark built twice is a benchmark you cannot trust.
#### Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.
#### Build in MendSpeech
- Create one benchmark function used by every later experiment in `src/bench/benchmark_asr.py`.
- Log environment, model, batch, and precision metadata automatically in `src/bench/environment.py`.
#### Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons, reporting what was discarded and why.
- Record the timing boundary explicitly: where measurement starts and ends.
#### Required Output
['- `src/bench/benchmark_asr.py`', '- `src/bench/environment.py`', '- `results/day26_repeatability.csv`']
#### Completion Check
> Repeated runs produce stable enough numbers to support comparisons, and the timing boundary is documented.

---

### DAY 27: Profiling the streaming model
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_27.md`](days/day_27.md)

> **v3 STATUS: CORE** Phase P4 begins. Measure before optimizing, or you optimize the wrong thing.
#### Learn
- Where time actually goes in a streaming forward pass.
- CPU launch overhead versus GPU compute time.
- Kernel-level versus end-to-end timing.
#### Build in MendSpeech
- Add per-operator profiling to the Day 26 harness in `src/bench/profile_ops.py`.
- Produce a ranked operator table for one fixed configuration.
#### Experiment and Measure
- Rank operators by measured time and separate launch overhead from compute.
- Identify the top three candidates for optimization and state the expected ceiling for each.
- Write the baseline row into `results/day27_operator_profile.csv`.
#### Required Output
['- `src/bench/profile_ops.py`', '- `results/day27_operator_profile.csv`', '- `docs/day27_optimization_targets.md`']
#### Completion Check
> You can name the top three time consumers with measured evidence and an expected gain for each.

---

### DAY 28: torch.compile and graph capture
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_28.md`](days/day_28.md)

> **v3 STATUS: CORE** First optimization technique, measured against the Day 27 profile rather than assumed.
#### Learn
- torch.compile: graph capture, fusion, and recompilation triggers.
- Dynamic shapes and why recompilation is silent and expensive.
- CUDA graphs for static-shape workloads.
#### Build in MendSpeech
- Apply torch.compile to the hot path in `src/asr/optimized_runner.py`, pinning shapes to avoid recompilation.
- Add a CUDA-graph fast path only for static-shape inputs in `src/asr/optimized_runner.py`.
#### Experiment and Measure
- Measure WER, p50/p95/p99 latency, RTF, and memory against the Day 27 baseline on identical inputs.
- Record compile time and warmup separately from steady-state latency.
- Verify output parity against the unoptimized path; a speedup with different transcripts is not a speedup.
#### Required Output
['- `src/asr/optimized_runner.py`', '- `tests/test_optimized_parity.py`', '- `results/day28_compile_speedup.csv`']
#### Completion Check
> You have a measured before/after for compilation with verified output parity, or the failure and its cause documented.

---
