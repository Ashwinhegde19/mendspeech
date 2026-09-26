# Day 38: Fine tune for damaged speech robustness

> **Week 6 • Day 3 of 7**  
> **Navigation:** [← Day 37](day_37.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 39 →](day_39.md)

> **v2 STATUS: CORE.** One bounded ASR adaptation experiment with a clean-speech
> regression control; negative results count as evidence, failed runs do not.

---

### Compute Target
`Modal L4; use a manageable training configuration within budget`

---

### 1. Learn
- Transfer learning.
- Frozen versus trainable layers.
- Mixed precision.
- Gradient accumulation.

---

### 2. Build in MendSpeech
- Adapt the existing compatible ASR checkpoint once using Day 37's audited data.
  Fix trainable layers, learning rate, steps, seed, clean/damaged sampling,
  precision, gradient accumulation and budget before running; choose the best
  checkpoint using validation only, never frozen test outcomes.
- If the optional Indic data passes Day 37's support and verification checks,
  include it in this same experiment and report its slice separately. Otherwise
  retain evaluation-only coverage; do not install/train a second recognizer.
- Save the reproducible configuration and provenance: base code/weight revisions,
  manifest hashes, source permissions, software/hardware and selection rule.
  Keep checkpoints and raw training logs local/gitignored. Day 39's augmentation
  ablation reuses this recipe; it is not a separate adaptation track.

---

### 3. Experiment and Measure
- Compare base and adapted checkpoints on identical frozen clean/damaged cases
  with the same decoder and normalization. Report WER/CER by condition, clean
  regression, confidence shifts and failures, not only aggregate improvement.
- Report any extension results separately from the frozen core benchmark. All
  comparable latency/RTF/peak-memory measurements use the same L4 configuration.
  Record actual cost and a failed run honestly instead of claiming adaptation.

---

### 4. Required Output Artifacts
- `training/finetune.py`
- `configs/finetune.yaml`
- `checkpoints/week6_best/` (local/gitignored weights and raw logs)
- `results/day38_base_vs_adapted.csv`
- `reports/day38_adaptation.md` (provenance, selection rule, cost and limitations)

---

### 5. Completion Check
> **Definition of Done for Day 38:**  
> The single base-versus-adapted experiment is reproducible and states what
improved, what did not and whether clean speech regressed. Test data never
selects the checkpoint, and optional language coverage is labeled separately.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR training documentation
- RNNT primary references
- Calibration and reliability diagram references
