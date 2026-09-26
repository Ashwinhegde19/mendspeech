# Day 19: Real-log-Mel block validation (merged into Day 18)

> **Week 3 • Day 5 of 7**  
> **Navigation:** [← Day 18](day_18.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 20 →](day_20.md)

> **v2 STATUS: MERGED into Day 18.** No standalone session. Validate input projection, mask propagation, shapes, and gradients on Day 18's same Conformer block; do not build a separate encoder or run a depth sweep.

---

### Compute Target
`Local CPU — within Day 18`

---

### 1. Learn
- Input projection.
- Mask propagation.
- Temporal dimensions.

---

### 2. Build in MendSpeech
- In Day 18, connect real log-Mel features through input projection to the same tested Conformer block.
- Propagate padding masks and validate temporal dimensions; no separate implementation.

---

### 3. Experiment and Measure
- In Day 18, trace shapes through every stage on real speech and test masks and finite, nonzero gradients on valid inputs.

---

### 4. Required Output Artifacts
- `results/day19_shape_trace.md` — produced in Day 18; no standalone code artifact

---

### 5. Completion Check
> **Definition of Done for Day 19:**  
> Absorbed into Day 18: a real log-Mel tensor passes through the same block's
> projection and mask handling with a documented shape trace and valid gradients.

---

### 6. Study Method & Protocol
Use Day 18's session and completion check. This merged item has no separate build session or commit.

---

### 7. References & Resources
- Conformer primary paper
- A mature Conformer implementation such as NVIDIA NeMo
