# Day 18: Assemble one Conformer block

> **Week 3 • Day 4 of 7**  
> **Navigation:** [← Day 17](day_17.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 19 →](day_19.md)

> **v3 STATUS: CORE** Absorbs Days 17 and 19: one tested block, with real log-Mel features passing through it.
---

### Compute Target
`Local CPU`

---

### 1. Learn
- Macaron structure.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.

---

### 2. Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper in `src/models/conformer_ffn.py`.
- Assemble feed-forward, attention, convolution, second feed-forward, and normalization in `src/models/conformer_block.py`.
- Add log-Mel input projection and propagate padding masks; assert input, output, and valid-length shapes.

---

### 3. Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; assert finite, nonzero gradients.
- Compare output statistics with and without half-step residual scaling.

---

### 4. Required Output Artifacts
['- `src/models/conformer_ffn.py`', '- `tests/test_conformer_ffn.py`', '- `src/models/conformer_block.py`', '- `tests/test_conformer_block.py`', '- `docs/conformer_block_walkthrough.md`', '- `results/day19_shape_trace.md`']

---

### 5. Completion Check
> **Definition of Done for Day 18:**  
> You can point to every operation in one tested block and say why it exists, and a real log-Mel tensor passes through it with valid gradients.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Gulati et al., Conformer
- NVIDIA NeMo Conformer implementation
