# Day 37: Adaptation dataset and leakage audit

> **Week 6 • Day 2 of 7**  
> **Navigation:** [← Day 36](day_36.md) | [Week 6 Plan](../Week_6_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 38 →](day_38.md)

> **v3 STATUS: CORE** Leaked data makes every later number meaningless, so this session precedes training.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Train, validation, and test separation.
- Speaker leakage.
- Synthetic corruption sampling.
- Balanced severity distribution.

---

### 2. Build in MendSpeech
- Build manifests pairing clean transcripts with corrupted audio for one adaptation experiment in `data/`.
- Preserve the already-frozen test membership and speaker-separated splits.
- Audit source duplicates and speaker leakage, and write the audit to `reports/data_audit.md`.

---

### 3. Experiment and Measure
- Prove no source or speaker appears in more than one split, including via corrupted copies.
- Report the severity distribution and correct any imbalance before training.

---

### 4. Required Output Artifacts
['- `data/train_manifest.jsonl`', '- `data/val_manifest.jsonl`', '- `data/test_manifest.jsonl`', '- `reports/data_audit.md`']

---

### 5. Completion Check
> **Definition of Done for Day 37:**  
> The evaluation set cannot appear in training through clean or corrupted duplicates, and the audit shows how you know.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Dataset leakage and speaker separation references
