# Day 45: Vocoder realism and acoustic boundary diagnostics

> **Week 7 • Day 3 of 7**  
> **Navigation:** [← Day 44](day_44.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 46 →](day_46.md)

> **v2 STATUS: CORE — short-span latency, boundary diagnostics, and the selected stack's vocoder.** Native streaming is conditional on verified support; full-waveform latency is required. No extra stack or vocoder training.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Mel to waveform generation.
- HiFi GAN style generator and discriminator intuition.
- Phase, bandwidth, and vocoder artifacts.
- Short time energy, local loudness, spectral balance, and room tone as boundary signals.
- First playable audio versus completed synthesis, input/output streaming
  distinctions, buffering, and chunk seams in short repair spans.

---

### 2. Build in MendSpeech
- Reuse only Day 43's selected stack's matched pretrained vocoder; keep
  weights frozen. GAN anatomy is theory, not a separate training experiment.
- Add boundary diagnostics that measure short time energy and simple spectral statistics before and after a candidate repair span.
- Save a local room tone estimate where possible.
- Extend the existing benchmark approach with `src/bench/benchmark_tts.py`
  and `tests/test_tts_latency.py`, using Day 43's pinned stack and held-out
  languages. Define the timing start (normalized text/reference ready), minimum
  playable audio buffer, synchronization and termination before measuring.
  Record text length, generated duration, seed, native sample rate, batch size,
  precision, CPU host/workers, L4 environment and warm-up/repeat counts.
- Label actual generation behavior `native_streaming`, `phrase_chunked`, or
  `full_waveform_delivery`. Native streaming must emit usable audio before full
  synthesis completes; state whether it needs full text up front. Delivery of
  an already generated waveform in chunks is full-waveform generation, not
  evidence of streaming synthesis. Phrase chunking is labeled only if actually
  implemented; no separate phrase-chunking feature is required here.
- If Day 43 verifies a native API, test chunk ordering, finalization, sample
  coverage without duplication or loss, short/silent outputs, and boundary
  discontinuities with deterministic fixtures. Otherwise record the unsupported
  branch and benchmark real full-waveform short-span output. Do not install a
  second model to obtain streaming. Keep network playback outside this timing.

---

### 3. Experiment and Measure
- Measure inference speed and real time factor on L4 with fixed batch size,
  warm-up, and sample rate. Distinguish isolated vocoder from end-to-end time;
  mark isolated timing unavailable if the interface does not expose it.
- Create intentionally mismatched generated spans and verify that the boundary diagnostics flag obvious loudness or spectral discontinuities.
- Include unchanged/identity stitch controls and tests for sample-count and
  outside-span preservation. Record both flagged and missed seam artifacts.
- Measure time to first playable audio, total synthesis time, RTF and peak
  memory in `results/day45_tts_latency.csv`; separate cold and warm runs and
  language/length/mode slices. Report p50/p95 with repeat counts and small-sample
  caveats. For full-waveform generation, first playable cannot predate complete
  generation. Mark unavailable modes/isolated stages missing, not zero.
- For supported native streaming, compare the same language/text/reference
  cases to full-waveform output where exposed, measuring chunk quality as well
  as speed. Do not claim end-to-end repair latency from synthesis-only numbers:
  context waiting, validation, stitching and serving are measured at Day 49.

---

### 4. Required Output Artifacts
- `results/day45_vocoder_benchmark.csv`
- `src/repair/boundary_metrics.py`
- `docs/vocoder_and_boundary_notes.md`
- `src/bench/benchmark_tts.py`
- `tests/test_tts_latency.py`
- `results/day45_tts_latency.csv`
- Update `docs/tts_pipeline.md` with verified delivery modes and limitations.

---

### 5. Completion Check
> **Definition of Done for Day 45:**  
> You can separate acoustic model errors from vocoder artifacts and quantify at least
two causes of an audible seam.
> Short-span latency has reproducible per-language evidence and explicit timing
> boundaries. Native streaming is tested only when supported, otherwise clearly
> unsupported; completed-waveform delivery is never labeled streaming generation.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastSpeech 2 paper
- HiFi GAN paper
- VITS paper
- DSP references for energy matching and equal power crossfades
