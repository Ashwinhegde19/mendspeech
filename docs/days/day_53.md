# Day 53: Run cascaded repair and seam ablations

> **Week 8 • Day 4 of 7**  
> **Navigation:** [← Day 52](day_52.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 54 →](day_54.md)

> **v2 STATUS: MERGED into [Day 52](day_52.md) — single combined ablation session.** Produce all repair/seam controls and artifacts within Day 52; no standalone session.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Repair threshold.
- Repair span padding.
- Preserve percentage.
- Full resynthesis baseline.
- Boundary energy matching, crossfade choice, and seam artifact rate.

---

### 2. Build in MendSpeech
- Run Preserve, Balanced, Rescue, full resynthesis, naive selective stitching, and boundary matched selective stitching.
- Record original waveform retained, repair percentage, end to end latency, speaker similarity proxy, and seam metrics.

---

### 3. Experiment and Measure
- Test whether repairing more audio always helps intelligibility.
- Test whether boundary matching reduces seam artifacts without materially increasing latency.
- Keep recognition outputs fixed for the stitching comparison so only the repair method changes.

---

### 4. Required Output Artifacts
- `results/capstone_cascaded_repair.csv`
- `results/repair_tradeoff.png`
- `results/seam_ablation.png`

---

### 5. Completion Check
> **Definition of Done for Day 53:**  
> You have a defensible result for the cascaded selective repair path and can separate
recognition, reconstruction, and stitching effects.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Your frozen protocol and prior results
- A reproducible pretrained direct latent or codec audio inpainting baseline
- Primary papers only when needed to interpret a result
