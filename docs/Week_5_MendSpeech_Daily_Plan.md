# Week 5: Streaming, Cache Aware Inference, and Adaptive Context

> **Days 29 to 35**  
> **Navigation:** [← Week 4](Week_4_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 6 →](Week_6_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Turn the recognizer into a real time system and test uncertainty-guided context spending only within supported checkpoint capabilities.
>
> **v2 gate evidence:** Follow Gate 4 in [the execution plan](REVISED_EXECUTION_PLAN.md), not a calendar target. Days 29–35 retain streaming, cache, and VAD/endpointing evidence; **Add-on B** async serving and load behavior remains required. Day 32 uses supported fixed contexts; Day 34 is a bounded live/simulated/unavailable comparison, not a custom serving project.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 29** | Batching and throughput | `Modal L4` | CORE | [Open Day 29](days/day_29.md) |
| **Day 30** | Quantization: INT8 and FP16 | `Modal L4` | CORE | [Open Day 30](days/day_30.md) |
| **Day 31** | Streaming fast path | `Modal L4` | CORE | [Open Day 31](days/day_31.md) |
| **Day 32** | Optimization scorecard | `Modal L4` | CORE | [Open Day 32](days/day_32.md) |
| **Day 33** | Break the cache on purpose | `Modal L4` | CORE | [Open Day 33](days/day_33.md) |
| **Day 34** | Bounded adaptive-context comparison | `Modal L4` | CORE | [Open Day 34](days/day_34.md) |
| **Day 35** | Endpointing and the VAD baseline | `Local CPU plus Modal L4` | CORE | [Open Day 35](days/day_35.md) |

---

## Phase Focus

Profiling and the inference optimization techniques

---

## Daily Detailed Operating Plans
### DAY 29: Batching and throughput
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_29.md`](days/day_29.md)

> **v3 STATUS: CORE** Throughput and latency are different axes; this session keeps them separate.
#### Learn
- Static versus dynamic batching.
- Queueing delay versus service time.
- Why throughput gains can hurt single-stream latency.
#### Build in MendSpeech
- Implement batched inference in `src/serve/batching.py` with a configurable batch policy.
- Expose batch size and queue wait as separately logged quantities.
#### Experiment and Measure
- Sweep batch size and report throughput and per-request latency separately.
- Find the batch size where queueing delay starts to dominate.
- Report RTF at the best throughput point and the latency at the lowest-concurrency point.
#### Required Output
['- `src/serve/batching.py`', '- `results/day29_batch_sweep.csv`', '- `docs/day29_queueing.md`']
#### Completion Check
> You can state the throughput/latency knee with measured evidence and explain what happens past it.

---

### DAY 30: Quantization: INT8 and FP16
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_30.md`](days/day_30.md)

> **v3 STATUS: CORE** Quantization is the technique most likely to be misreported, so parity comes before speed.
#### Learn
- Dynamic versus static INT8.
- Why static quantization needs a calibration set.
- What quantization can and cannot preserve in a speech model.
#### Build in MendSpeech
- Smoke-check export and precision support on one compatible backend; pin revisions in `docs/day30_quant_notes.md`.
- Verify original-versus-exported parity at equal precision on validation clips before quantizing.
- Apply supported INT8/FP16 variants through `src/asr/quantized_runner.py`.
#### Experiment and Measure
- Measure WER/CER, p50/p95/p99, RTF, and peak memory for baseline and each supported precision.
- Record confidence and logit shifts caused by quantization.
- State explicitly whether a variant is faster. A slower variant is a valid finding.
#### Required Output
['- `src/asr/quantized_runner.py`', '- `docs/day30_quant_notes.md`', '- `results/day30_quantization_tradeoffs.csv`', '- `app/audio_lab.py`']
#### Completion Check
> You have measured accuracy, latency, and memory for every supported precision, or a documented compatibility blocker. No speedup is claimed without a measurement.

---

### DAY 31: Streaming fast path
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_31.md`](days/day_31.md)

> **v3 STATUS: CORE** Optimized offline inference does not automatically make streaming fast; this session checks.
#### Learn
- State reuse versus recomputation across chunks.
- Where redundant computation remains in a cache-aware encoder.
#### Build in MendSpeech
- Add a streaming-specific fast path in `src/streaming/fast_path.py` reusing Day 30's best variant.
- Assert cached and uncached streaming produce equivalent transcripts.
#### Experiment and Measure
- Measure steady-state per-chunk latency after warmup, separately from the first chunk.
- Report the speedup of the fast path against the Day 30 baseline, or state that it did not help.
#### Required Output
['- `src/streaming/fast_path.py`', '- `tests/test_fast_path_parity.py`', '- `results/day31_fast_path.csv`']
#### Completion Check
> You can state measured steady-state streaming latency and whether the fast path earned its complexity.

---

### DAY 32: Optimization scorecard
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_32.md`](days/day_32.md)

> **v3 STATUS: CORE** One table decides what ships. Techniques that did not help are reported with the same prominence.
#### Learn
- Presenting negative results without overclaiming.
- Why an optimization table is a design document, not a log.
#### Build in MendSpeech
- Build the scorecard generator in `src/bench/scorecard.py`.
- Produce the scorecard in `results/day32_optimization_scorecard.csv`.
#### Experiment and Measure
- Tabulate WER, p50/p95/p99, RTF, and memory for baseline and every technique tried.
- Add a `what_did_not_help` section to `docs/day32_optimization_report.md`.
- Select the shipping configuration from the scorecard and justify it on measured grounds.
#### Required Output
['- `src/bench/scorecard.py`', '- `results/day32_optimization_scorecard.csv`', '- `docs/day32_optimization_report.md`']
#### Completion Check
> You can defend the shipping configuration from a table, including the techniques that failed.

---

### DAY 33: Break the cache on purpose
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_33.md`](days/day_33.md)

> **v3 STATUS: CORE** A streaming system that silently mishandles state is worse than a slow correct one.
#### Learn
- State continuity.
- Chunk boundary dependencies.
- Cache reset and truncation.
#### Build in MendSpeech
- Add controlled experiments that reset or shorten the cache at chosen boundaries in `src/streaming/cache_stress.py`.
#### Experiment and Measure
- Measure WER changes around the reset point.
- Determine whether errors cluster at boundaries or propagate, and write it up in `results/day33_cache_failures.md`.
#### Required Output
['- `src/streaming/cache_stress.py`', '- `tests/test_cache_reset.py`', '- `results/day33_cache_failures.md`']
#### Completion Check
> You can explain a concrete failure caused by incorrect state handling and where it appears.

---

### DAY 34: Bounded adaptive-context comparison
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_34.md`](days/day_34.md)

> **v3 STATUS: CORE** , capability-bounded. Live switching only if the checkpoint supports it; otherwise report the deferral honestly.
#### Learn
- Policy-driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.
#### Build in MendSpeech
- Implement a capability-guarded policy in `src/controller/adaptive_context.py` that classifies chunks as easy or uncertain.
- Compare at most two supported right-context settings from Day 25.
- Return an explicit unavailable status when fewer than two settings are supported.
#### Experiment and Measure
- Compare fixed-fast, fixed-accurate, and the bounded adaptive policy with an explicit `live`, `simulated`, or `unavailable` status.
- Keep simulated estimates separate from measured latency; do not count cached reuse as a runtime gain.
#### Required Output
['- `src/controller/adaptive_context.py`', '- `results/day34_adaptive_context.csv`']
#### Completion Check
> You have a measured live comparison, a clearly limited simulated comparison, or an evidence-backed unavailable result.

---

### DAY 35: Endpointing and the VAD baseline
- **Compute:** `Local CPU plus Modal L4`
- **Dedicated Daily File:** [`docs/days/day_35.md`](days/day_35.md)

> **v3 STATUS: CORE** Add-on A, absorbed here. Endpointing decides when a final answer is sent, so its errors are latency errors.
#### Learn
- Energy versus spectral VAD.
- Onset and offset error in milliseconds.
- False alarms versus missed speech in a streaming setting.
#### Build in MendSpeech
- Implement a deterministic frame-level VAD in `src/vad/baseline.py` with framing, timestamp, and silence tests.
- Compare it with one local reference VAD on identical clean and damaged inputs.
#### Experiment and Measure
- Measure precision, recall, F1, false alarms, missed speech, and onset/offset error in ms.
- Report CPU RTF separately from GPU measurements.
- Carry the measured choice into Day 41 and record it in `results/day35_vad_benchmark.csv`.
#### Required Output
['- `src/vad/baseline.py`', '- `tests/test_vad.py`', '- `results/day35_vad_benchmark.csv`', '- `docs/day35_vad_notes.md`']
#### Completion Check
> Endpointing error is quantified in milliseconds and the chosen detector's failure modes are documented.

---
