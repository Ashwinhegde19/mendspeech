# Day 24: Pretrained streaming ASR baseline and capability check

> **Week 4 • Day 3 of 7**  
> **Navigation:** [← Day 23](day_23.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 25 →](day_25.md)

> **v3 STATUS: CORE** Record what the selected checkpoint actually supports before later phases depend on it.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Cache-aware inference, right-context controls, export support, and tokenizer language coverage.

---

### 2. Build in MendSpeech
- Run a current streaming-capable ASR checkpoint on clean and damaged sets in `src/asr/streaming_runner.py`.
- Record model revision and all inference settings.
- Record capability status for cache-aware inference, supported right-context values and units, runtime context switching, intended export path, and language coverage in `configs/model_baseline.yaml`.

---

### 3. Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Use minimal supported smoke checks where feasible and record failures early.
- Estimate later-phase effort from the capability record; do not start a model hunt to fill a gap.

---

### 4. Required Output Artifacts
['- `src/asr/streaming_runner.py`', '- `results/day24_baseline.csv`', '- `configs/model_baseline.yaml`']

---

### 5. Completion Check
> **Definition of Done for Day 24:**  
> You have a reproducible baseline with model, data, hardware, and settings fixed, plus an evidence-backed capability record.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastConformer primary paper
- NVIDIA NeMo streaming ASR documentation
