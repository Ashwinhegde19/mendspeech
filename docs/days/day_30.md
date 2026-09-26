# Day 30: Quantization: INT8 and FP16

> **Week 5 • Day 2 of 7**  
> **Navigation:** [← Day 29](day_29.md) | [Week 5 Plan](../Week_5_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 31 →](day_31.md)

> **v3 STATUS: CORE** Quantization is the technique most likely to be misreported, so parity comes before speed.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Dynamic versus static INT8.
- Why static quantization needs a calibration set.
- What quantization can and cannot preserve in a speech model.

---

### 2. Build in MendSpeech
- Smoke-check export and precision support on one compatible backend; pin revisions in `docs/day30_quant_notes.md`.
- Verify original-versus-exported parity at equal precision on validation clips before quantizing.
- Apply supported INT8/FP16 variants through `src/asr/quantized_runner.py`.

---

### 3. Experiment and Measure
- Measure WER/CER, p50/p95/p99, RTF, and peak memory for baseline and each supported precision.
- Record confidence and logit shifts caused by quantization.
- State explicitly whether a variant is faster. A slower variant is a valid finding.

---

### 4. Required Output Artifacts
['- `src/asr/quantized_runner.py`', '- `docs/day30_quant_notes.md`', '- `results/day30_quantization_tradeoffs.csv`', '- `app/audio_lab.py`']

---

### 5. Completion Check
> **Definition of Done for Day 30:**  
> You have measured accuracy, latency, and memory for every supported precision, or a documented compatibility blocker. No speedup is claimed without a measurement.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- ONNX Runtime quantization documentation
- torch.ao quantization overview
