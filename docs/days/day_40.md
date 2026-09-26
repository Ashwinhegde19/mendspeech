# Day 40: RNN-T concepts and quantization lab

> **Week 6 • Day 5 of 7**  
> **Navigation:** [← Day 39](day_39.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 41 →](day_41.md)

> **v2 STATUS: CORE.** Retain the quantization lab behind a compatibility check.
> Unsupported precision is a documented blocker, not a completed optimization.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Encoder.
- Prediction network.
- Joint network.
- Blank handling.
- Streaming emission behavior.
- Difference from CTC independence.
- Post-training quantization: dynamic vs static INT8, and why static needs a calibration set.
- What quantization can and cannot preserve in an ASR model (logit sharpness, confidence behavior).

---

### 2. Build in MendSpeech
- Smoke-check the selected ASR model's export and precision support on one
  compatible L4 backend. Pin model/backend revisions and verify supported
  operators, dynamic lengths, decoding and actual device/kernel placement.
  Do not add a provider sweep or silently compare CPU INT8 with GPU inference.
- Export the Day 38 checkpoint only through the supported path. Check the
  original model versus exported model at the same precision on validation
  clips before quantization: output/logit tolerances where exposed, decoded
  text, lengths and decoding settings. Record and resolve parity failures first.
- Compare FP16 and INT8 only where the same backend supports their intended
  execution on L4. For static INT8, select and record a representative calibration
  slice from validation, disjoint by source/speaker from test; never calibrate
  using the frozen test benchmark. State if the backend does not need calibration.
- If export, parity or precision support blocks measurement, retain working
  inference and record the exact failed check, error and unsupported precision.
  Do not change models/backends repeatedly to manufacture an INT8 result.

---

### 3. Experiment and Measure
- After parity passes, measure original/exported baseline and supported FP16/INT8
  variants on identical frozen cases, decoder, batch size, timing boundaries,
  warm-up and L4 hardware. Report actual WER/CER, latency/RTF and peak memory;
  INT8 may be slower or less accurate and need not be selected for deployment.
- Record logit/confidence shifts for Day 41 validation-based calibration. Report
  precision coverage and CPU fallbacks explicitly; a mixed-device run is not
  a controlled GPU speed comparison.
- Give each variant `measured` or `blocked` status with reasons and blank
  unavailable metrics. Gate 5 can carry an explicit blocked optimization status,
  but cannot claim successful quantization or a speedup without measurements.

---

### 4. Required Output Artifacts
- `docs/rnnt_walkthrough.md` (theory summary from the Learn block)
- `docs/day40_quantization_notes.md` (compatibility, parity, calibration source,
  backend/precision settings and blockers)
- `results/day40_quantization_tradeoffs.csv` (actual metrics or blocked rows)

---

### 5. Completion Check
> **Definition of Done for Day 40:**  
> You can explain RNN-T streaming behavior and demonstrate original/export
parity plus measured supported-precision tradeoffs on L4, or identify the exact
compatibility/parity blocker. A blocked branch stays explicitly unimplemented;
no unsupported INT8, speedup or calibration claim is presented as complete.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR training documentation
- RNN-T primary references
- ONNX Runtime quantization documentation or torch.ao quantization overview
