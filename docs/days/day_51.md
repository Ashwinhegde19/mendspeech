# Day 51: Release SpeechDamageBench v1 and freeze the evaluation set

> **Week 8 • Day 2 of 7**  
> **Navigation:** [← Day 50](day_50.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 52 →](day_52.md)

> **v3 STATUS: CORE** The frozen set is the project's anchor; new experiments get new configs, never a new test set.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Severity grids.
- Speaker-separated evaluation.
- Seed control and deterministic manifests.
- Package versioning and checksum verification.

---

### 2. Build in MendSpeech
- Finalize the standalone package and lock manifest checksums in `benchmarks/`.
- Document a one-command example that reproduces one benchmark item in `speechdamagebench/README.md`.

---

### 3. Experiment and Measure
- Reinstall the package in a clean environment.
- Regenerate a sample from the manifest and verify its checksum.
- Verify clean references are byte-identical after regeneration.

---

### 4. Required Output Artifacts
['- `speechdamagebench/CHANGELOG.md`', '- `benchmarks/manifest.csv`', '- `benchmarks/README.md`']

---

### 5. Completion Check
> **Definition of Done for Day 51:**  
> A clean environment reproduces a benchmark item from the manifest, and clean references are provably unchanged.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Reproducible packaging references
