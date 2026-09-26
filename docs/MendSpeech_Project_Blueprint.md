# MendSpeech Project Blueprint

> **Selective Semantic Speech Restoration Under Real-Time Constraints, with Controlled Repair Baselines.**
>
> **v2 scope:** One recognition pipeline, one selected TTS stack, one evolving
> application, and one evaluation suite. The [execution plan](REVISED_EXECUTION_PLAN.md)
> governs bounded adaptation and external-comparator feasibility. These are
> target behaviors, not claims that the current implementation is complete.

---

### Core Principle
> *Preserve trustworthy original speech, spend compute only where uncertainty justifies it, and never hide architectural limitations.*

---

## 1. Product Behavior
- Accept microphone audio or an uploaded recording.
- Optionally generate controlled damage through `SpeechDamageBench`.
- Transcribe incrementally with cache-aware FastConformer inference.
- Align confidence and uncertainty to time.
- Classify intervals as **Preserve**, **Inspect**, **Repair**, or **Abstain**.
- **For MendSpeech V1:** Reconstruct only repair intervals with speaker-conditioned TTS, then apply acoustic boundary matching before stitching.
- **For the Research Comparison:** Evaluate one verified pretrained audio-restoration comparator on the same damaged cases. Use masked-inpainting terminology only if its interface and experiment actually support masks; explicitly defer an unavailable comparison.
- Show exactly which milliseconds were preserved, reconstructed, or left unrepaired.

---

## 2. Repair Architecture and External Comparator

| Dimension | MendSpeech V1: Cascaded Baseline | Verified Audio-Restoration Comparator |
| :--- | :--- | :--- |
| **Pipeline** | Streaming ASR $\rightarrow$ Calibrated Uncertainty $\rightarrow$ Repair Policy $\rightarrow$ Speaker-Conditioned TTS $\rightarrow$ Duration Alignment $\rightarrow$ Boundary Matching $\rightarrow$ Waveform Stitching | Damaged Audio $\rightarrow$ Verified Pretrained Restoration Model $\rightarrow$ Restored Audio |
| **Hypothesis to test** | Selective reconstruction can retain more reliable original audio than full resynthesis at a useful latency. | Avoiding the text bottleneck may preserve acoustic information; measure rather than assume this advantage. |
| **Limitations to measure** | ASR errors can become false repairs; text loses acoustic detail and cannot justify inventing unrecoverable content. | Mask support, locality, language coverage, compute, and acoustic preservation depend on the selected model; full-utterance restoration is not selective inpainting. |

Normal reconstruction uses predicted text. Gold-transcript repair is a separately
labeled oracle control. Keep raw damaged audio, full resynthesis, naive selective
stitching, and boundary-matched selective repair even if the external comparator
is unavailable. Never claim superiority over a model that was not evaluated.

---

## 3. Boundary Matching Layer
- Select a repair window with context padding.
- Match generated duration to the target interval.
- Compare short-time energy around both boundaries.
- Match local loudness before mixing.
- Estimate simple spectral mismatch and room tone difference.
- Compare linear and equal-power crossfades.
- Record seam diagnostics so subjective listening is not the only evidence.

---

## 4. SpeechDamageBench
`SpeechDamageBench` is designed as a standalone, versioned package, not a private utility inside MendSpeech.

| Damage Family | Controlled Variables | Purpose |
| :--- | :--- | :--- |
| **Additive Noise** | SNR, noise type, random seed | Test masking robustness. |
| **Clipping** | Threshold, severity | Test lost peaks and saturation. |
| **Bandwidth Limits** | Sample rate, filter settings | Simulate narrow channels (telephony, codecs). |
| **Dropouts** | Span length, frequency, random seed | Simulate missing speech and packet loss. |
| **Reverberation** | Impulse response / room severity | Test temporal smearing. |

> [!NOTE]
> Every generated sample records corruption name, severity, random seed, clean source ID, parameter values, and package version.

---

## 5. Research Console Metrics

| Metric | Why It Matters |
| :--- | :--- |
| **WER & CER** | Recognition correctness. |
| **Latency Percentiles (p50, p90, p99)** | Responsiveness and tail behavior under real-time constraints. |
| **Real-Time Factor (RTF)** | Measures whether processing keeps up with live speech. |
| **GPU Memory** | Captures deployment cost and memory pressure. |
| **Repair Percentage** | Measures how much of the audio was regenerated. |
| **Original Retained Percentage** | Measures original audio preservation directly. |
| **Speaker Similarity Proxy** | Imperfect signal for identity consistency across repairs. |
| **Calibration (ECE / Brier)** | Tests whether confidence values support reliable policy decisions. |
| **Boundary Energy Discontinuity** | Quantitative signal for stitching and seam quality. |
| **Abstention Outcomes** | Measures whether the system avoids hallucinating unrecoverable content. |

---

## 6. Required Ablations
- **Context Policy:** Supported fixed low/high lookahead settings, plus a bounded adaptive experiment. Label live, simulated, and unavailable modes; simulation cannot establish live latency savings.
- **Uncertainty:** Raw confidence vs. calibrated confidence (temperature scaling).
- **Threshold Policies:** Preserve, Balanced, and Rescue repair thresholds.
- **Granularity:** Selective span repair vs. full utterance resynthesis.
- **Stitching Quality:** Naive waveform stitching vs. boundary-matched stitching.
- **ASR Robustness:** Base ASR vs. robustness-adapted (fine-tuned) ASR.
- **Architecture:** Cascaded V1 vs. one verified restoration comparator; include a masked-inpainting claim only if the tested capability supports it. Record blocked comparisons as deferred limitations.
- **TTS Adaptation:** Base vs. adapted selected stack when permitted data, checkpoint, and L4 budget pass the feasibility check; otherwise explicitly defer adaptation, not the repair controls.
- **Clean Speech Regression:** Ensuring already clean speech is not degraded by the pipeline.

---

## 7. Repository Target Structure

```text
mendspeech/
├── src/
│   ├── audio/          # Waveform loaders, STFT, log-Mel, normalization
│   ├── asr/            # FastConformer, CTC, transducer, confidence, calibration
│   ├── streaming/      # Cache-aware runners and lookahead controllers
│   ├── controller/     # Repair policies, adaptive context, abstention logic
│   ├── tts/            # Synthesis and speaker conditioning
│   ├── repair/         # Timing alignment, boundary matching, crossfade stitching
│   ├── baselines/      # One verified restoration adapter with capability metadata
│   ├── metrics/        # WER, CER, RTF, seam discontinuity, speaker similarity
│   └── bench/          # Benchmark harnesses and runners
├── speechdamagebench/  # Standalone benchmark package
│   ├── speechdamagebench/
│   ├── tests/
│   ├── presets/
│   ├── pyproject.toml
│   └── README.md
├── infra/              # Modal cloud execution scripts and container definitions
├── app/                # audio_lab.py is the evolving demo; shared UI components
├── experiments/        # Frozen experiment configs
├── results/            # Measured tables and figures; audio/checkpoints/run logs ignored
└── reports/            # Research report, figures, and failure casebooks
```

---

## 8. Definition of Done
1. A new user can reproduce benchmark results with documented, single-command sequences.
2. The interactive demo visually highlights preserved, repaired, and abstained millisecond spans.
3. Boundary matching is quantitatively measured (discontinuity scores), not just evaluated by ear.
4. The final research report includes at least one surprising result and one limitation that materially constrains claims.
5. The system explicitly abstains from hallucinating speech when audio is too damaged.
6. The external comparator is evaluated fairly with verified capability labels, or its blocked feasibility check and untested claims are explicitly documented. A negative or unavailable result must not be recast as an architectural advantage.
7. You can explain every major model and systems component from first principles without relying on library names as explanations.
8. Benchmark results are reported at a fixed, documented scale (≥30 utterances, ≥5 speakers, reference transcripts, speaker-separated splits) with the statistical caveat stated in the report.
9. One external restoration candidate receives a bounded Week 2 feasibility check. Record the pinned interface, mask support, formats, license, smoke-test outcome, and blocker if any. No open-ended model search or scratch inpainting fallback.
10. ASR adaptation, calibration, cache/endpointing correctness, clean regression,
    and serving failure/recovery have reproducible evidence. The single TTS
    stack has an adaptation result or an explicit feasibility deferral; inference
    alone must not be described as model training.
11. The application exposes measured capabilities only. Release evidence and
    limitations are independent of optional learning drills or extra UI pages.
