# Day 18: Assemble one Conformer block

> **Week 3 • Day 4 of 7**  
> **Navigation:** [← Day 17](day_17.md) | [Week 3 Plan](../Week_3_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 19 →](day_19.md)

> **v2 STATUS: CORE — absorbs Days 17 and 19.** Implement the macaron feed-forward, assemble one tested Conformer block, and pass real log-Mel features through that same block with input projection and mask propagation. No separate encoder or depth sweep.

---

### Compute Target
`Local CPU`

---

### 1. Learn
- Macaron structure.
- Layer normalization placement.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.

---

### 2. Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper absorbed from Day 17, with shape and gradient tests.
- Assemble feed forward, the scratch attention and convolution modules, second feed forward, and final normalization into one block.
- Add input projection from real log-Mel features and propagate padding masks through this same block; assert input, output, and valid-length shapes.

---

### 3. Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; test mask handling and finite, nonzero gradients on valid inputs.
- Compare output statistics with and without half-step residual scaling in the same tests. A second toy-training ablation is not a release requirement.

---

### 4. Required Output Artifacts
- `src/models/conformer_ffn.py` — Day 17's absorbed macaron implementation
- `tests/test_conformer_ffn.py`
- `src/models/conformer_block.py`
- `tests/test_conformer_block.py`
- `docs/conformer_block_walkthrough.md`
- `results/day19_shape_trace.md` — retained Day 19 path, produced here

---

### 5. Completion Check
> **Definition of Done for Day 18:**  
> You can explain every operation in one tested block. A real log-Mel tensor
> passes through its projection and mask handling with documented shapes and
> valid gradients; the retained Day 19 shape trace records the evidence.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Conformer primary paper
- A mature Conformer implementation such as NVIDIA NeMo
