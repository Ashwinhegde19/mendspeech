# Optional Speech-ML Systems Drills

> **Purpose:** reinforce first-principles explanations and speech-system
> reasoning alongside MendSpeech. This is optional study material, not another
> project, a release dependency, or a substitute for a day file's completion check.

---

## Operating Rules

- Choose a topic only when it helps the current Learn/Build/Measure session.
  There is no weekly quota, stopwatch requirement, mandatory rebuild or gate.
- Use existing code, paper, a notebook or an editor as appropriate. Re-derive
  a small example when useful; no tool restrictions are required.
- Note where the explanation or implementation becomes uncertain and revisit
  the relevant theory or measured evidence. Skip drills before core work.
- Use project evidence in answers; do not memorize product or library names as
  explanations.

---

## Topic Menu

### A. Implementation reasoning

- Implement naive gradient descent for linear regression with NumPy; derive
  the gradient, test convergence, and explain learning-rate failure.
- Implement waveform framing and timestamps without a helper library.
- Implement a minimal energy or spectral VAD, then diagnose false alarms and
  missed speech.
- Trace or implement CTC collapse and WER dynamic programming on a small example.

### B. Mathematics and tensor reasoning

- Trace matrix shapes through Q/K/V projections, attention scores, softmax,
  value mixing, output projection, and residual addition.
- Explain dot products as similarity, projections as learned coordinate
  systems, rank as an information constraint, and why attention cost grows
  quadratically with sequence length.
- Derive the gradient of mean-squared error and connect the chain rule to
  backpropagation through a small network.
- Estimate memory and compute before running an attention or convolution
  experiment; compare the estimate with the profiler.

### C. Speech architecture reasoning

- Draw waveform → features → encoder → CTC/RNN-T decoder → timestamps and
  name the shape at every boundary.
- Explain how Whisper-style fixed windows or chunks differ from cache-aware
  streaming and where boundary errors, recomputation, and latency arise.
- Compare encoder-only, encoder-decoder, and decoder-only architectures; state
  why an ASR or TTS system may choose each structure.
- Explain CTC conditional independence, RNN-T's prediction network, and the
  practical latency/accuracy trade-off.

### D. Production and evaluation judgment

- Given a latency trace, separate capture time, buffering, VAD endpointing,
  model inference, decoding, network, and finalization delay.
- Explain p50, p95, and p99; show why mean latency can hide a bad interactive
  experience.
- Design a stress test for 1, 4, and 8 streams with fixed hardware, input,
  batching, and model configuration.
- Diagnose rising queue depth and propose bounded backpressure, timeout, retry,
  and fallback behavior.
- Explain why lower WER can still produce worse product behavior through late
  endpoints, wrong named entities, or confident semantic errors.

### E. Speech generation representations

- Explain where linguistic content, speaker identity, prosody, and style enter
  a modern TTS system.
- Contrast acoustic features, latent variables, and codec tokens.
- Explain why factorized content/speaker/style representations can be useful
  without proving perfect disentanglement.
- Compare autoregressive, non-autoregressive, flow-matching, diffusion, and
  VITS-style generation at the level of training signal, inference behavior,
  controllability, and latency.

### F. General ML literacy

- Define cross-entropy and perplexity, and state when neither evaluates a
  speech system adequately.
- Explain train/validation/test separation, speaker leakage, calibration, and
  why thresholds are selected on validation data.
- Defend one negative result from MendSpeech: hypothesis, controls, measured
  outcome, limitation, and next experiment.

---

## Optional Review Prompts by Project Stage

These prompts do not add completion checks to the execution-plan gates. The
required Add-on A baseline/reference VAD measurements remain in the Week 2 plan;
they do not require a timed reconstruction here.

### After Gate 2

- Explain the measured baseline/reference VAD errors and onset/offset behavior.
- Explain waveform framing, CTC, WER, confidence, and timestamp alignment.
- Derive a gradient-descent update on a small numerical example.

### After Gate 4

- Draw cache-aware streaming state and explain what is reused.
- Defend a p95 latency measurement and a backpressure policy.
- Explain attention tensor shapes and sequence-length cost.

### After Gate 6

- Explain CTC versus RNN-T and base versus quantized inference behavior.
- Trace content, speaker, style, duration, vocoder, and stitching through the
  cascaded repair path.
- Diagnose one real failure without blaming a library name.

### After Gate 7

- Trace the system from waveform to final benchmark result using its artifacts.
- Revisit an uncertain implementation detail if that would clarify the result.
- Explain the strongest result, the most significant limitation, and an appropriate
  next experiment without adding it to the release scope.

---

## Optional Notes

If useful, record technical explanations outside this file in a notebook or
untracked practice log. No drill artifacts are required. Reusable explanations,
tests or measured artifacts must follow the repository's quality and privacy rules.
