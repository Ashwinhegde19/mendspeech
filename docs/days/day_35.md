# Day 35: Week 5 live streaming milestone

> **Week 5 • Day 7 of 7**  
> **Navigation:** [← Day 34](day_34.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 36 →](day_36.md)

> **v2 STATUS: CORE — shared audio lab streaming milestone.** Streaming, VAD/endpointing, cache-state handling, and Gate 4 Add-on B serving/load evidence remain required. Adaptive context is displayed only where supported.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Review buffered streaming, cache aware inference, lookahead, cache failures,
  adaptive context, VAD-driven endpointing, and partial-versus-final latency.

---

### 2. Build in MendSpeech
- Extend the existing `app/audio_lab.py` entrypoint to connect microphone or simulated live audio to the streaming recognizer; do not create another app.
- Reuse Add-on A VAD for endpointing and log speech start, speech end, and
  finalization timestamps.
- Show partial text, confidence timeline, VAD/endpointing state, cache state, queue depth, and measured latency. Show current context mode only when the runner exposes it; distinguish fixed from live adaptive behavior and never present a simulated policy as live.
- Keep Gate 4 Add-on B async serving, per-stream isolation, backpressure, and load/failure evidence required. Link those artifacts rather than building context-specific serving infrastructure.

---

### 3. Experiment and Measure
- Record a short demo with clean and damaged speech.
- Measure time to first partial transcript, endpoint delay, false starts, and
  missed endpoints on the same cases.
- Document remaining technical limitations honestly.
- Link Day 31/33 cache evidence, Day 34's live/simulated/unavailable status, and Add-on B serving/load results. Unsupported adaptive context does not waive endpointing, cache correctness, or serving checks.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `demos/week5_streaming_demo.mp4`
- `reports/week5_streaming.md`

---

### 5. Completion Check
> **Definition of Done for Day 35:**  
> A person can speak and watch MendSpeech transcribe incrementally while exposing
> VAD, endpointing, cache, and uncertainty state, with context mode shown only
> where available. The report links required serving/load and cache evidence;
> any adaptive deferral is explicit and does not substitute for those checks.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Stateful or cache aware Conformer primary material
- NVIDIA NeMo streaming ASR documentation and examples
