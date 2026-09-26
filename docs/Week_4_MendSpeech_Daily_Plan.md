# Week 4

> **Days 22–28**
> **Navigation:** [← Index](INDEX.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Executive Plan](REVISED_EXECUTION_PLAN.md)

---

> [!IMPORTANT]
> **Week theme:** Context trade-off, calibration, triage, and the end-to-end latency baseline
> Measure fixed context and produce a calibrated, end-to-end latency baseline.

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 22** | Temporal subsampling and context reasoning | `Local CPU` | LEARN-ONLY | [Open Day 22](days/day_22.md) |
| **Day 23** | VAD, endpointing and finalization state machine | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 23](days/day_23.md) |
| **Day 24** | Fixed-context quality and latency frontier | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 24](days/day_24.md) |
| **Day 25** | Fit and validate calibrated confidence and triage | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 25](days/day_25.md) |
| **Day 26** | Early streaming-to-editor integration and budget | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 26](days/day_26.md) |
| **Day 27** | Profile the current end-to-end path | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 27](days/day_27.md) |
| **Day 28** | torch.compile and graph capture experiment | `Modal L4 for measured GPU work; local CPU for checks` | CORE | [Open Day 28](days/day_28.md) |

---

## Daily Detailed Operating Plans

### DAY 22: Temporal subsampling and context reasoning
- **Compute:** Local CPU
- **Dedicated Daily File:** [`docs/days/day_22.md`](days/day_22.md)

> **STATUS: LEARN-ONLY**
> **Prerequisites:** [Day 18](days/day_18.md)
> **Effort:** 0–0 focused hours.

#### Learn
- Subsampling changes frame counts; buffering/lookahead creates latency.

#### Build in MendSpeech
- No separate module; study the selected model metadata for Day24.

#### Experiment and Measure
- No standalone timing sweep.

#### Required Output Artifacts
- None.

#### Completion Check
> Theory informs context measurements without a second encoder project.

---

### DAY 23: VAD, endpointing and finalization state machine
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_23.md`](days/day_23.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 21](days/day_21.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Frame energy/spectral VAD, hangover, pause/endpoint trade-offs and speech-end labels.

#### Build in MendSpeech
- Implement deterministic framing/VAD with one compatible local reference detector; tune thresholds only on validation.
- Integrate start/end/hangover state with streaming/session.py. Preserve too-quiet/short/silence failure cases; no diarization.

#### Experiment and Measure
- Measure precision/recall, onset/offset ms error and CPU RTF on a separate annotated validation slice; untouched core-test membership.
- Test mid-sentence pauses, trailing silence, short clips and delayed chunks; record endpoint-to-final and annotated-speech-end-to-final separately.

#### Required Output Artifacts
- `src/vad/baseline.py`
- `src/streaming/endpoint.py`
- `tests/test_vad.py`
- `tests/test_endpoint.py`
- `results/day23_endpointing.csv`

#### Completion Check
> Live finalization is tested and endpoint delay is measured, not postponed until after system profiling.

---

### DAY 24: Fixed-context quality and latency frontier
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_24.md`](days/day_24.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 21](days/day_21.md), [Day 22](days/day_22.md), [Day 23](days/day_23.md)
> **Effort:** 2–4 focused hours.

#### Learn
- Algorithmic lookahead, wall-clock finalization and WER versus responsiveness.

#### Build in MendSpeech
- Add only documented fixed context configurations and preserve model/head/cache compatibility.
- Use the shared replay/harness with trace IDs, warm/cold and settings recorded.

#### Experiment and Measure
- Compare supported context settings on validation at fixed decoder and batch; report time-to-first-partial and post-utterance finalization plus WER/CER.
- Fewer than two supported settings yields a single-setting baseline and explicit limit, not invented adaptive gains.

#### Required Output Artifacts
- `src/streaming/context.py`
- `tests/test_context.py`
- `results/day24_context_tradeoff.csv`

#### Completion Check
> Context costs are measured with a working chunk loop and endpoint detector; scope of supported settings is honest.

---

### DAY 25: Fit and validate calibrated confidence and triage
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_25.md`](days/day_25.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 11](days/day_11.md), [Day 14](days/day_14.md), [Day 18](days/day_18.md), [Day 24](days/day_24.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Temperature scaling, word/utterance correctness labels, reliability, Brier/ECE, risk-coverage.

#### Build in MendSpeech
- Implement src/asr/calibration.py with an explicit score-to-correctness unit, calibration-only fitting and separate validation thresholds.
- Map scores to actual decoded tokens/words; if beam scores lack valid aligned features, retain verified greedy triage rather than calling fused scores probabilities.
- Bind calibration to checkpoint/head/tokenizer/decoder/precision and context mode; build accept/uncertain/reject policy that preserves raw text and bypasses editor on uncertainty.

#### Experiment and Measure
- Compare raw/calibrated reliability and risk-coverage on disjoint validation, with bins/counts and corruption slices.
- Test empty/blank/all-correct/all-wrong and serialization/refit; non-improvement is valid but unsupported calibrated claims are not.

#### Required Output Artifacts
- `src/asr/calibration.py`
- `src/controller/triage.py`
- `tests/test_calibration.py`
- `tests/test_triage.py`
- `configs/calibration.yaml`
- `configs/triage_thresholds.yaml`
- `results/day25_reliability.csv`
- `results/day25_reliability.png`
- `results/day25_risk_coverage.csv`

#### Completion Check
> Calibration is actually fitted and independently evaluated, distinct from threshold selection, for the exact shipping-candidate configuration.

---

### DAY 26: Early streaming-to-editor integration and budget
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_26.md`](days/day_26.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 15](days/day_15.md), [Day 21](days/day_21.md), [Day 23](days/day_23.md), [Day 24](days/day_24.md), [Day 25](days/day_25.md)
> **Effort:** 2–4 focused hours.

#### Learn
- End-of-speech finalization, editor trigger semantics and guard-before-delivery.

#### Build in MendSpeech
- Extend one app/audio_lab.py with replay/microphone streaming, final-ASR→editor request, raw bypass and visibly provisional output.
- Instrument all clocks in the latency contract; execute one-L4 joint-memory/interference test, not isolated-model timing only.

#### Experiment and Measure
- Collect raw/deterministic/LLM quality and correlated request-level stage intervals on validation.
- Record first partial, post-utterance ASR finalization, server/client TTFT and final usable text. State microphone versus paced replay and model residence/cold state.

#### Required Output Artifacts
- `app/audio_lab.py`
- `tests/test_pipeline_contract.py`
- `results/day26_e2e_baseline.csv`
- `docs/day26_latency_baseline.md`

#### Completion Check
> The full baseline, including streaming, endpointing, guarded editing and combined resource use, exists before optimization.

---

### DAY 27: Profile the current end-to-end path
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_27.md`](days/day_27.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 14](days/day_14.md), [Day 26](days/day_26.md)
> **Effort:** 2–3 focused hours.

#### Learn
- Kernel versus wall time, CPU launch overhead, queue/prefill/decode/generation and cold-versus-warm.

#### Build in MendSpeech
- Add per-stage instrumentation to src/bench/profile_ops.py on the Day26 pipeline; no optimization technique is applied yet.
- Separate start/compile/load from steady state; report per-request critical paths, not just aggregated percentiles.

#### Experiment and Measure
- Rank measured time contributors and note overlapping/serialized stages. Identify where p99 requests diverge from median, using the same request IDs.
- State expected ceiling for each candidate; unsupported profiling granularity is recorded, not guessed.

#### Required Output Artifacts
- `src/bench/profile_ops.py`
- `results/day27_profile.csv`
- `docs/day27_optimization_targets.md`

#### Completion Check
> The top measured contributors and the p99 divergence are named with request-level evidence, not assumption.

---

### DAY 28: torch.compile and graph capture experiment
- **Compute:** Modal L4 for measured GPU work; local CPU for checks
- **Dedicated Daily File:** [`docs/days/day_28.md`](days/day_28.md)

> **STATUS: CORE**
> **Prerequisites:** [Day 27](days/day_27.md)
> **Effort:** 3–5 focused hours.

#### Learn
- Graph capture, fusion, recompilation on new shapes/frames and static versus dynamic cost.

#### Build in MendSpeech
- Apply torch.compile to a pinned steady-state configuration and experiment with CUDA graphs in src/asr/optimized_runner.py. Pin static shapes; treat chunked streaming as dynamic.
- Test parity on same input/precision; separate warmup/compile time from p50/p95/p99.

#### Experiment and Measure
- Measure WER/latency/RTF/memory against Day27 including joint editor; count recompiles and reject measures that slow p99 or break parity.
- Negative/zero gains are documented with numbers; do not chase a checklist.

#### Required Output Artifacts
- `src/asr/optimized_runner.py`
- `tests/test_optimized_parity.py`
- `results/day28_compile.csv`

#### Completion Check
> A measured, parity-checked before/after for compilation/graph capture, or an honest zero-gain result.

---
