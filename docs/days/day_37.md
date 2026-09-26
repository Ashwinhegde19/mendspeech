# Day 37: Build a robust fine tuning dataset

> **Week 6 • Day 2 of 7**  
> **Navigation:** [← Day 36](day_36.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 38 →](day_38.md)

> **v2 STATUS: CORE.** Prepare one ASR adaptation dataset. The frozen core
> evaluation set is immutable; any Indic extension stays separate.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Train, validation, test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.

---

### 2. Build in MendSpeech
- Prepare train/validation manifests pairing verified transcripts with clean
  and corrupted audio for one adaptation experiment. Record source/speaker IDs,
  language, license/consent, transcript verification and normalization, corruption,
  severity, seed, parameters and package version; retain clean examples.
- Preserve the already frozen test membership and speaker-separated splits.
  All clean/corrupted copies of a source stay in one split. A test manifest is
  a reference to the frozen evaluation set, never a new sample or rewritten set.
- Optionally reuse the separate `data/indic_codemix_manifest.csv` prepared after
  Gate 2 for the same adaptation experiment only if the checkpoint/tokenizer
  supports the language and a competent transcript verifier is available.
  Keep extension train/validation/test speakers disjoint from each other and
  the core evaluation speakers. Otherwise retain it as evaluation-only and
  record the limitation; do not add a second training track.

---

### 3. Experiment and Measure
- Audit source-level duplicates, speaker leakage and corruption provenance.
  Report split counts, clean/severity balance and immutable core test membership.
- Record extension language support/verification evidence and inclusion or
  evaluation-only status. Add-on C streaming metrics remain for later evaluation,
  not invented data-preparation results.

---

### 4. Required Output Artifacts
- `data/train_manifest.jsonl`
- `data/val_manifest.jsonl`
- `data/test_manifest.jsonl` (frozen-set reference; preserve existing contents)
- `reports/data_audit.md`

---

### 5. Completion Check
> **Definition of Done for Day 37:**  
> The audit demonstrates no source or speaker leakage into training/validation,
preserves the frozen core evaluation set and documents provenance. Any optional
extension is separately identified and does not create another training track.

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
