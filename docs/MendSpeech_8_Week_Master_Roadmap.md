# MendSpeech Complete Learning and Research Roadmap

> **A Systems & Research Roadmap for a Real-Time Voice Interface: Streaming ASR, Latency Budget, Inference Optimization, and Personalization.**

---

> [!IMPORTANT]
> **Timeline & Workload Realism:**  
> Eight weeks describes the original content grouping, not a promised delivery date.
> Sessions can span multiple sittings; training/data preparation and debugging
> require explicit estimates. Do not sacrifice evidence to preserve a calendar.
>
> **v3 scope:** [REVISED_EXECUTION_PLAN.md](REVISED_EXECUTION_PLAN.md) governs
> phases and gates. Roughly 39 build sessions remain from Day 10. Cadence is a
> target, not a promise; estimate from observed throughput and re-plan when it
> misses.

---

## 1. Final Project Architecture & Flow

```
              Audio In / SpeechDamageBench conditions
                               │
              VAD + endpointing (src/streaming, src/vad)
                               │
        Streaming ASR + cache (src/asr, src/streaming)
                               │
        Decode: greedy / beam / beam+LM (src/asr)
                               │
     Calibrated confidence + triage (src/asr, src/controller)
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
    [ low confidence ]                  [ accept ]
    return raw transcript,              raw ASR text
    editor bypassed                            │
              │                        Conservative editor
              │                        (src/llm, guard)
              │                                │
              └────────────┬───────────────────┘
                           ▼
                   Guarded final text
              (validated, or raw text + reason)
                           │
                     Latency budget
        (src/bench: correlated per-stage p50/p95/p99)
```

Three invariant rules. Raw ASR text is always available. A rejected, malformed,
timed-out, or low-confidence edit returns the original text with a reason.
Formatting is never permitted to change names, numbers, negation, or units.

## 2. What the Revised Plan Changes
- **Explicit Real-Time Baseline:** The streaming ASR $\rightarrow$ conservative editor path is treated as a systems baseline measured against raw transcripts, deterministic formatting, and a disabled-editor control.
- **Conservative Editing Layer:** the editor may format only, and every delivered edit passes a guard that preserves names, numbers, negation, and units.
- **Standalone `SpeechDamageBench`:** Packaged as an independent, deterministic, versioned Python library with seed-controlled degradations.
- **Bounded Restoration Comparison:** Week 2 checks one candidate; Week 8
  compares its verified behavior or documents why that comparison is unavailable.
- **VAD and Production Serving:** Gate 2 adds a small deterministic VAD baseline
  and reference comparison; Gate 4 carries it into endpointing, async WebSocket serving,
  backpressure, utilization, and tail-latency measurement.
- **Optional Speech-ML Systems Drills:** short fundamentals exercises reinforce
  gradient descent, transformer linear algebra, chunked ASR, profiling, and
  representation factorization without becoming release gates or a quota.
- **Indic and Code-Mixed Extension:** prepare a separate verified manifest after
  Gate 2; complete measurements as the relevant pipeline exists. Reuse the single
  ASR adaptation experiment only where language and data support permit.
- **One TTS Stack:** prioritize controlled adaptation, speaker conditioning,
  two supported languages including one Indian language, repair-focused prosody,
  and short-span latency over multiple architecture installations. Native
  streaming is verified only if the same backend supports it.
- **Controlled Decoding:** greedy, beam-only, and one n-gram LM share a compatible
  acoustic model and the Day 26 harness. Separate offline decoder cost from
  fresh end-to-end latency; keep LM scores distinct from calibrated confidence.
- **Focused Delivery:** no voice-agent loop, encoder-inspector UI, tiny-encoder
  depth sweep, or parallel milestone app implementations.

---

## 3. Eight-Week Progression

| Week | Focus | Milestone | Weekly Plan Link |
| :--- | :--- | :--- | :--- |
| **Week 1** | Audio, Degradation, & Measurement Foundations | Build the audio laboratory and release `SpeechDamageBench` v0 as a standalone package. | [Week 1 Plan](Week_1_MendSpeech_Daily_Plan.md) |
| **Week 2** | ASR, CTC, Confidence, & Repair Localization | Build the recognition and uncertainty layer, plus a reusable Modal cloud execution pipeline. | [Week 2 Plan](Week_2_MendSpeech_Daily_Plan.md) |
| **Week 3** | Conformer From First Principles | Implement Conformer attention, convolutions, and Macaron feed-forwards from scratch in PyTorch. | [Week 3 Plan](Week_3_MendSpeech_Daily_Plan.md) |
| **Week 4** | FastConformer & Efficient Encoder Behavior | Reproducible baseline, efficiency harness, and controlled greedy/beam/LM comparison. | [Week 4 Plan](Week_4_MendSpeech_Daily_Plan.md) |
| **Week 5** | Streaming, Cache-Aware Inference, & Adaptive Context | Implement cache-aware streaming ASR and evaluate uncertainty-guided adaptive context spending. | [Week 5 Plan](Week_5_MendSpeech_Daily_Plan.md) |
| **Week 6** | Robustness, Fine-Tuning, RNN-T, & Calibration | Adapt the recognizer to damaged speech, explore RNN-T, and calibrate confidence scores. | [Week 6 Plan](Week_6_MendSpeech_Daily_Plan.md) |
| **Week 7** | TTS, Speaker Preservation, & Boundary-Matched Reconstruction | One-stack two-language evidence, bounded adaptation, repair prosody, short-span latency, and selective repair. | [Week 7 Plan](Week_7_MendSpeech_Daily_Plan.md) |
| **Week 8** | Research Capstone: Controlled Repair Comparisons | Freeze benchmarks, run ablations, report the external comparator outcome, and reproduce the release. | [Week 8 Plan](Week_8_MendSpeech_Daily_Plan.md) |

---

## 4. Workload Risks

| Phase | Main risk | Bound |
| :--- | :--- | :--- |
| **Weeks 1–3** | Data quality, source leakage, tensor/mask correctness | Frozen corpus, fast tests, one scratch block rather than a second recognizer |
| **Weeks 4–5** | Decoder/head compatibility, LM text leakage, stateful inference, latency | One small LM and acoustic model; validate offline decoding before any streaming integration; fixed-context streaming first |
| **Week 6** | Fine-tuning stability and export/precision support | One adaptation experiment, clean regression, held-out calibration and backend-specific checks |
| **Weeks 7–8** | Serving, editor training budget, correlated latency | One endpoint; separate required latency evidence from conditional post-training; retain report and reproduction |

Session counts are in the execution plan. The removed three nominal sessions
do not guarantee equivalent capacity for TTS data preparation or training.
Decoder integration, two-language review/listening and latency checks add work;
no fixed extra-session or compute estimate is promised before the capability checks.

---

## 5. Daily Operating Protocol (2-Hour Core Session)

| Time | Activity | Rule |
| :--- | :--- | :--- |
| **25 min** | **Theory & Reading** | Read only the specific paper section or framework doc needed for today's task. |
| **65 min** | **Build & Experiment** | Write code, run one controlled experiment, change one variable at a time, save output. |
| **20 min** | **Research Notebook** | Document: *Question, Hypothesis, Method, Result, Surprise, Limitation, Next Step*. |
| **10 min** | **Commit & Explain** | Save code/artifacts, name outputs cleanly, explain what you learned aloud without notes. |

> [!TIP]
> **Handling Incomplete Tasks:** If debugging takes longer than 65 minutes, continue the exact same task in the next session rather than pretending the day is finished.
>
> **Timebox rule:** any day may consume at most two extra sessions before a
> scope review. Required failed checks remain incomplete. Only explicitly
> conditional experiments may be deferred with a blocker and narrowed claims;
> a deadline does not turn unfinished work into a completed gate.

---

## 6. Hardware Strategy
- **Week 1:** Local CPU only.
- **Week 2:** Local CPU for theory days; Modal L4 optional from Day 08 for ASR and the bounded restoration smoke test.
- **Week 3:** Local CPU. GPU only for optional scaling checks.
- **Weeks 4 to 8:** Default to **Modal L4 (24GB VRAM)** for reproducible inference, streaming, fine-tuning, and benchmarks.
- **Hardware Consistency:** Keep hardware strictly fixed across any latency, RTF, or memory comparison.
- **Budget Reality:** Keep the approximate **$15–30** project envelope visible.
  Verify current rates and remaining budget before scheduling compute; new TTS
  adaptation is not assumed to fit. Cache outputs, separate cold/warm runs, and
  never substitute another GPU tier in latency, RTF, or memory comparisons.

---

## 7. Core Research Questions
1. *Can selective semantic repair improve intelligibility while retaining more original speech than full resynthesis?*
2. *Can calibrated ASR uncertainty guide streaming context spending so extra latency is consumed only when speech is degraded?*
3. *When does conservative editing improve a transcript, and when does it introduce an error that raw ASR did not have?*
4. *How should the post-training budget be split between supervised editing and RL, and what does each buy?* A null result is reported as evidence.
5. *Does an external language model improve recognition without making plausible but incorrect transcripts more likely?* Keep decoder and calibrated-triage changes distinguishable.
6. *How should a bounded post-training budget be split between supervised editing and RL, and what does each buy on protected-content violations and identity behavior?* Null results are reported.

---

## 8. Primary Resource Spine
- **Audio DSP:** PyTorch & TorchAudio documentation, Oppenheim/Schafer signal processing fundamentals.
- **ASR & CTC:** Graves et al. Connectionist Temporal Classification, NeMo CTC decoders.
- **Conformer:** Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*.
- **FastConformer:** Rekesh et al., *FastConformer with Linearly Scalable Attention for Efficient Speech Recognition*.
- **Streaming ASR:** NVIDIA Stateful Conformer with Cache-Based Streaming Inference.
- **Transducer:** Graves RNN-T papers and NeMo RNN-T decoders.
- **Post-training:** TRL SFT and GRPO documentation, LoRA reference, and reward-specification literature.
- **Serving and timing:** vLLM metric definitions as a terminology reference, and FastAPI WebSocket documentation.
