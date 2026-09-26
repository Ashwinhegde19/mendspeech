# Day 24: Pretrained FastConformer baseline

> **Week 4 • Day 3 of 7**  
> **Navigation:** [← Day 23](day_23.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 25 →](day_25.md)

> **v2 STATUS: CORE — baseline and early capability check.** Verify decoding/LM, streaming, and export support on the selected checkpoint; unsupported capabilities are documented, not replaced by a model search or new architecture.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Model checkpoint loading.
- Tokenizer and decoder configuration.
- Batch versus single utterance inference.
- Checkpoint-specific cache-aware inference, right-context controls, export support, and tokenizer language coverage.
- CTC versus transducer search, token/blank mapping, and external LM fusion.

---

### 2. Build in MendSpeech
- Run a current NeMo FastConformer checkpoint on your clean and damaged sets.
- Record model revision and all inference settings.
- Pin the acoustic head and tokenizer expected by Day 26's greedy/beam/LM
  experiment. Verify a supported beam backend, optional dependencies and
  small n-gram LM format against the pinned framework; do not assume these
  extras are installed or that a CTC recipe works with a transducer head.
  Record offline versus incremental support separately. Use this same acoustic
  checkpoint, not another model just to obtain a decoder comparison.
- In `configs/model_baseline.yaml`, record capability status and evidence for cache-aware inference, supported right-context values and units, runtime context switching, the intended export path, and tokenizer language support for the planned evaluation languages. Distinguish verified, unsupported, and unverified behavior.
- Check the pinned model/framework documentation and use minimal supported smoke checks where feasible. Record failures and limitations early; do not start a model hunt, retrain an encoder, or add custom infrastructure to manufacture support.

---

### 3. Experiment and Measure
- Benchmark WER, latency, and GPU memory by damage type.
- Carry the capability record into Days 25, 31–35 and later export work. Unsupported adaptive switching defers the adaptive claim, not unrelated gate evidence; unsupported cache-aware inference or export remains an explicit dependency issue, not a completed requirement.
- Include decoder compatibility smoke evidence and estimate Day 26's text
  preparation/integration effort. A blocked required comparison remains
  incomplete pending scope review; no scratch backend or open-ended model hunt.

---

### 4. Required Output Artifacts
- `src/asr/fastconformer_runner.py`
- `results/day24_fastconformer_baseline.csv`
- `configs/model_baseline.yaml`

---

### 5. Completion Check
> **Definition of Done for Day 24:**  
> You have a reproducible baseline with model, data, hardware, and settings fixed,
> plus an evidence-backed capability record covering cache-aware inference,
> right context, export, tokenizer language support, and decoder/head/LM backend
> compatibility. Unsupported or unverified
> capabilities and their downstream implications are explicit.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- FastConformer primary paper
- NVIDIA NeMo FastConformer model documentation
