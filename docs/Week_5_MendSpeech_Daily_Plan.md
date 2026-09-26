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

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 29** | Offline versus streaming ASR | You can explain why naive chunking creates boundary errors and redundant compute. | `Modal L4` | [Open Day 29](days/day_29.md) |
| **Day 30** | Buffered streaming | You can quantify the compute waste caused by overlapping history. | `Modal L4` | [Open Day 30](days/day_30.md) |
| **Day 31** | Cache aware streaming internals | You can explain what is cached, what is recomputed, and why cache aware inference can be more efficient. | `Modal L4` | [Open Day 31](days/day_31.md) |
| **Day 32** | Supported fixed-context lookahead ablation | Measured supported operating points, or an explicit unavailable comparison; no invented context configurations. | `Modal L4` | [Open Day 32](days/day_32.md) |
| **Day 33** | Break the cache on purpose | You can explain a concrete failure caused by incorrect state handling. | `Modal L4` | [Open Day 33](days/day_33.md) |
| **Day 34** | Bounded adaptive-context comparison | Retained CSV with explicit live/simulated/unavailable status; unsupported adaptation defers only the adaptive claim. | `Modal L4` | [Open Day 34](days/day_34.md) |
| **Day 35** | Shared audio lab streaming milestone | Incremental text, VAD/endpointing, cache and serving evidence; context mode only where available. | `Modal L4` | [Open Day 35](days/day_35.md) |

---

## v2 Scope Map (Gate Evidence)

| Day | v2 Status | Note |
| :--- | :--- | :--- |
| **Day 29** | CORE | Offline versus naive streaming comparison |
| **Day 30** | CORE | Buffered streaming and measured recomputation |
| **Day 31** | CORE | Cache-aware inference and comparison remain required; use Day 24's capability record |
| **Day 32** | CORE | Separate runs at supported fixed contexts only |
| **Day 33** | CORE | Controlled cache-state failure evidence remains required |
| **Day 34** | CORE — capability-bounded | At most two supported settings; report live, simulated, or unavailable without fabricated adaptive latency |
| **Day 35** | CORE | Extend `app/audio_lab.py`; VAD/endpointing, cache handling, and Add-on B serving/load evidence remain required |

---

## Reference Spine
- Stateful or cache aware Conformer primary material\nNVIDIA NeMo streaming ASR documentation and examples

---

## Daily Detailed Operating Plans

### DAY 29: Offline versus streaming ASR
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_29.md`](days/day_29.md)

#### Learn
- Audio chunks.
- Algorithmic latency.
- Partial hypotheses.
- Endpointing and finalization.

#### Build in MendSpeech
- Create a chunk simulator that feeds audio incrementally.
- Log when each chunk becomes available and when text changes.

#### Experiment and Measure
- Compare offline transcript with naive chunk by chunk transcription.

#### Required Output
- `src/streaming/chunker.py`
- `results/day29_offline_vs_naive.csv`

#### Completion Check
> You can explain why naive chunking creates boundary errors and redundant compute.

---

### DAY 30: Buffered streaming
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_30.md`](days/day_30.md)

#### Learn
- Overlapping windows.
- Buffer size.
- Stride.
- Repeated computation.

#### Build in MendSpeech
- Implement or run buffered streaming with configurable overlap.
- Measure how much audio is recomputed.

#### Experiment and Measure
- Sweep buffer and stride settings.
- Measure WER and latency tradeoffs.

#### Required Output
- `src/streaming/buffered.py`
- `results/day30_buffered_sweep.csv`

#### Completion Check
> You can quantify the compute waste caused by overlapping history.

---

### DAY 31: Cache aware streaming internals
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_31.md`](days/day_31.md)

#### Learn
- Cached activations.
- Past context state.
- Streaming masks.
- Right context and lookahead.

#### Build in MendSpeech
- Use NeMo cache aware streaming inference on a supported FastConformer checkpoint.
- Log cache related configuration and chunk boundaries.

#### Experiment and Measure
- Compare buffered and cache aware inference on the same audio and same hardware.

#### Required Output
- `src/streaming/cache_aware_runner.py`
- `results/day31_buffered_vs_cache.csv`

#### Completion Check
> You can explain what is cached, what is recomputed, and why cache aware inference
can be more efficient.

---

### DAY 32: Supported fixed-context lookahead ablation
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_32.md`](days/day_32.md)

> **v2 STATUS: CORE — fixed context only.** Use only configurations verified for the Day 24 checkpoint. This session does not require live context switching.

#### Learn
- Right context.
- Lookahead.
- Commit delay.
- WER and latency as competing objectives.

#### Build in MendSpeech
- Run the supported fixed-context configurations recorded in Day 24, using a separate run per setting and holding model, audio, hardware, batching, and decoding fixed.
- Store per-utterance and aggregate metrics, exact context values and units, and capability status. Do not coerce unsupported values or build a new streaming path.

#### Experiment and Measure
- Plot measured WER versus measured L4 latency and identify dominated operating points only when multiple supported settings exist.
- If only one setting is supported, retain its measured point and label the comparison unavailable; if none is runnable, record the reason and leave metrics missing. Keep the CSV and figure paths, with an explicitly annotated unavailable comparison rather than fabricated points or latency.

#### Required Output
- `experiments/lookahead_ablation.py`
- `results/day32_lookahead.csv`
- `results/day32_pareto.png`

#### Completion Check
> You can defend a supported fixed operating point using measured data, or show
> why the comparison is unavailable. A single point is not a Pareto frontier;
> unsupported context variation defers that claim, not the rest of Gate 4.

---

### DAY 33: Break the cache on purpose
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_33.md`](days/day_33.md)

#### Learn
- State continuity.
- Chunk boundary dependencies.
- Cache reset and truncation.

#### Build in MendSpeech
- Add controlled experiments that reset or shorten cache at selected boundaries.

#### Experiment and Measure
- Measure WER changes around the reset point.
- Inspect whether errors cluster near boundaries or propagate later.

#### Required Output
- `experiments/cache_break_test.py`
- `results/day33_cache_failures.md`

#### Completion Check
> You can explain a concrete failure caused by incorrect state handling.

---

### DAY 34: Bounded adaptive-context comparison
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_34.md`](days/day_34.md)

> **v2 STATUS: CORE — capability-bounded comparison.** Use only Day 24/32 supported settings. Live adaptation is conditional; unavailable support defers the adaptive claim, not the remaining streaming, endpointing, cache, or serving requirements.

#### Learn
- Policy driven context selection.
- Confidence smoothing.
- Latency budget.
- Stability versus oscillation.

#### Build in MendSpeech
- Implement a small capability-guarded policy that classifies chunks as easy or uncertain using the existing uncertainty signal and fixed decision rules.
- Compare at most two supported right-context settings from Day 32. Use live switching only when the checkpoint supports it; otherwise simulate policy choices from separate fixed-setting runs on the same controlled subset.
- If fewer than two settings are supported, return an explicit unavailable status and reason. No custom serving infrastructure, new model, or architecture change to force adaptation.

#### Experiment and Measure
- Compare the two fixed policies and the bounded adaptive policy where supported, with an explicit `live`, `simulated`, or `unavailable` status for the adaptive comparison.
- A simulated comparison is offline policy evidence, not live adaptive latency. Keep measured fixed-run timings separate; leave adaptive latency missing unless measured on a real live switching run. Do not fabricate measurements or claim a benefit from a simulation alone.

#### Required Output
- `src/controller/adaptive_context.py` — bounded policy and capability guard; reports unavailable when unsupported
- `results/day34_adaptive_context.csv` — retain this path even when unavailable; include status, supported settings, evidence/reason, and missing values for unmeasured metrics

#### Completion Check
> You have a measured live comparison, an explicitly limited simulated policy
> comparison, or an evidence-backed unavailable result. Unsupported behavior
> honestly defers the adaptive claim without waiving the rest of Gate 4.

---

### DAY 35: Week 5 live streaming milestone
- **Compute:** `Modal L4`
- **Dedicated Daily File:** [`docs/days/day_35.md`](days/day_35.md)

> **v2 STATUS: CORE — shared audio lab streaming milestone.** Streaming, VAD/endpointing, cache-state handling, and Gate 4 Add-on B serving/load evidence remain required. Adaptive context is displayed only where supported.

#### Learn
- Review buffered streaming, cache aware inference, lookahead, cache failures,
  adaptive context, VAD-driven endpointing, and partial-versus-final latency.

#### Build in MendSpeech
- Extend the existing `app/audio_lab.py` entrypoint to connect microphone or simulated live audio to the streaming recognizer; do not create another app.
- Reuse Add-on A VAD for endpointing and log speech start, speech end, and
  finalization timestamps.
- Show partial text, confidence timeline, VAD/endpointing state, cache state, queue depth, and measured latency. Show current context mode only when the runner exposes it; distinguish fixed from live adaptive behavior and never present a simulated policy as live.
- Keep Gate 4 Add-on B async serving, per-stream isolation, backpressure, and load/failure evidence required. Link those artifacts rather than building context-specific serving infrastructure.

#### Experiment and Measure
- Record a short demo with clean and damaged speech.
- Measure time to first partial transcript, endpoint delay, false starts, and
  missed endpoints on the same cases.
- Document remaining technical limitations honestly.
- Link Day 31/33 cache evidence, Day 34's live/simulated/unavailable status, and Add-on B serving/load results. Unsupported adaptive context does not waive endpointing, cache correctness, or serving checks.

#### Required Output
- `app/audio_lab.py`
- `demos/week5_streaming_demo.mp4`
- `reports/week5_streaming.md`

#### Completion Check
> A person can speak and watch MendSpeech transcribe incrementally while exposing
> VAD, endpointing, cache, and uncertainty state, with context mode shown only
> where available. The report links required serving/load and cache evidence;
> any adaptive deferral is explicit and does not substitute for those checks.

---

## Gate 4 Add-on B — Async Serving and Load Behavior

### Build

- Wrap the streaming recognizer in an async FastAPI/WebSocket service.
- Dockerize it and deploy on Modal using the same L4 for every comparison.
- Make per-stream queues, maximum queue depth, backpressure, timeout,
  disconnect, retry, and fallback behavior explicit.
- Propagate VAD start/end events and request/run identifiers through the
  service.

### Experiment and Measure

- Run 1, 4, and 8 concurrent streams with fixed audio, hardware, batching, and
  model configuration.
- Record time to first partial transcript, endpoint/finalization delay,
  end-to-end p50/p95/p99, RTF, cold start, CPU/GPU utilization, peak GPU memory,
  queue depth, and dropped or delayed chunks.
- Force one queue-overload or disconnect case and verify bounded, documented
  recovery behavior.

### Required Output

- `infra/serve/`
- `results/addon_b_serving.csv`
- `reports/addon_b_serving.md`

### Completion Check

> You can locate the measured bottleneck, explain the backpressure policy, and
> reproduce one controlled failure and recovery without silently losing audio.

---
