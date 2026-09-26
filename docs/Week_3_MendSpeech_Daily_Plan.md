# Week 3: Conformer From First Principles

> **Days 15 to 21**  
> **Navigation:** [← Week 2](Week_2_MendSpeech_Daily_Plan.md) | [Master Index](INDEX.md) | [Master Roadmap](MendSpeech_8_Week_Master_Roadmap.md) | [Week 4 →](Week_4_MendSpeech_Daily_Plan.md)

---

> [!IMPORTANT]
> **Week Milestone:**  
> Implement the core encoder pieces so model behavior is not a black box.
>
> **v2 gate evidence:** Follow Gate 3 in [the execution plan](REVISED_EXECUTION_PLAN.md), not a calendar target. This week has **4 build sessions: Days 15, 16, 18, and 21**; together with Week 4, the encoder block has **9 build sessions**.

---

---

## Week Map

| Day | Focus | Compute | Status | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 15** | Attention for speech sequences | `Local CPU, L4 optional for scaling` | CORE | [Open Day 15](days/day_15.md) |
| **Day 16** | Conformer convolution module | `Local CPU` | CORE | [Open Day 16](days/day_16.md) |
| **Day 17** | Macaron feed forward and residual scaling | `Local CPU` | LEARN-ONLY | [Open Day 17](days/day_17.md) |
| **Day 18** | Assemble one Conformer block | `Local CPU` | CORE | [Open Day 18](days/day_18.md) |
| **Day 19** | Real-log-Mel block validation | `Local CPU — within Day 18` | MERGED | [Open Day 19](days/day_19.md) |
| **Day 20** | Compare your block with a production implementation | `Local CPU` | DROPPED | [Open Day 20](days/day_20.md) |
| **Day 21** | Architecture review: what dominates streaming latency | `Local CPU` | CORE | [Open Day 21](days/day_21.md) |

---

## Phase Focus

Attention, Conformer internals, and streaming cost

---

## Daily Detailed Operating Plans
### DAY 15: Attention for speech sequences
- **Compute:** `Local CPU, L4 optional for scaling`
- **Dedicated Daily File:** [`docs/days/day_15.md`](days/day_15.md)

> **v3 STATUS: CORE** Attention cost is why streaming needs a stateful encoder rather than a windowed one.
#### Learn
- Query, key, value projections.
- Scaled dot product attention.
- Attention masks.
- Quadratic cost in sequence length and why long-form audio suffers.
#### Build in MendSpeech
- Implement single-head then multi-head attention in `src/models/attention.py`.
- Add shape assertions and gradient tests in `tests/test_attention.py`.
#### Experiment and Measure
- Change sequence length and measure forward time and peak memory.
- Plot the quadratic cost you predicted against the cost you measured.
#### Required Output
['- `src/models/attention.py`', '- `tests/test_attention.py`', '- `results/day15_attention_cost.csv`']
#### Completion Check
> You can derive every major tensor shape from memory and explain the quadratic term streaming must avoid.

---

### DAY 16: Conformer convolution module
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_16.md`](days/day_16.md)

> **v3 STATUS: CORE** Depthwise convolution is what makes a Conformer cheaper than attention alone at long sequence lengths.
#### Learn
- Depthwise separable convolution.
- Receptive field and locality.
- Causality assumptions for a streaming encoder.
#### Build in MendSpeech
- Implement a Conformer-style convolution module in `src/models/conv_module.py`.
- Test causality assumptions and receptive field growth in `tests/test_conv_module.py`.
#### Experiment and Measure
- Feed synthetic impulses and inspect how local information spreads.
- Measure receptive field against layer count and compare with the analytic prediction.
#### Required Output
['- `src/models/conv_module.py`', '- `tests/test_conv_module.py`', '- `results/day16_receptive_field.csv`']
#### Completion Check
> You can explain why depthwise convolution is cheap and what local context it captures relative to attention.

---

### DAY 17: Macaron feed forward and residual scaling
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_17.md`](days/day_17.md)

> **v3 STATUS: LEARN-ONLY** No build session; the macaron build moves into Day 18.
#### Learn
- Macaron structure.
- Layer normalization placement.
- Residual scaling and why half-step helps deep stacks.
#### Build in MendSpeech
- No standalone build. Day 18 implements the macaron feed-forward.
#### Experiment and Measure
- No standalone measurement. Day 18 compares output statistics with and without residual scaling.
#### Required Output
['None for this learn-only session; artifacts are produced within Day 18.']
#### Completion Check
> You can explain the ordering of the Conformer block without memorizing a diagram.

---

### DAY 18: Assemble one Conformer block
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_18.md`](days/day_18.md)

> **v3 STATUS: CORE** Absorbs Days 17 and 19: one tested block, with real log-Mel features passing through it.
#### Learn
- Macaron structure.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.
#### Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper in `src/models/conformer_ffn.py`.
- Assemble feed-forward, attention, convolution, second feed-forward, and normalization in `src/models/conformer_block.py`.
- Add log-Mel input projection and propagate padding masks; assert input, output, and valid-length shapes.
#### Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; assert finite, nonzero gradients.
- Compare output statistics with and without half-step residual scaling.
#### Required Output
['- `src/models/conformer_ffn.py`', '- `tests/test_conformer_ffn.py`', '- `src/models/conformer_block.py`', '- `tests/test_conformer_block.py`', '- `docs/conformer_block_walkthrough.md`', '- `results/day19_shape_trace.md`']
#### Completion Check
> You can point to every operation in one tested block and say why it exists, and a real log-Mel tensor passes through it with valid gradients.

---

### DAY 19: Real-log-Mel block validation
- **Compute:** `Local CPU — within Day 18`
- **Dedicated Daily File:** [`docs/days/day_19.md`](days/day_19.md)

> **v3 STATUS: MERGED** into Day 18. No standalone session.
#### Learn
- Input projection and mask propagation.
#### Build in MendSpeech
- Validate projection, masks, shapes, and gradients on Day 18's same block; do not build a separate encoder.
#### Experiment and Measure
- No depth sweep. The shape trace in `results/day19_shape_trace.md` is produced within Day 18.
#### Required Output
['- `results/day19_shape_trace.md` — produced in Day 18']
#### Completion Check
> A real log-Mel tensor passes through the block with documented shapes and valid gradients.

---

### DAY 20: Compare your block with a production implementation
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_20.md`](days/day_20.md)

> **v3 STATUS: DROPPED** Reading assignment only; do not schedule a session.
#### Learn
- Production Conformer code structure.
#### Build in MendSpeech
- No build. Optional reading time only.
#### Experiment and Measure
- No measurement.
#### Required Output
['None.']
#### Completion Check
> You have skimmed a production implementation and can name what your scratch block omits.

---

### DAY 21: Architecture review: what dominates streaming latency
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_21.md`](days/day_21.md)

> **v3 STATUS: CORE** The review now asks a systems question rather than a memorization question.
#### Learn
- Review attention, convolution, feed-forward, normalization, and residual paths.
- Which operations scale with sequence length, and which are constant per chunk.
#### Build in MendSpeech
- Write `reports/week3_conformer.md` naming the operations that dominate streaming cost, using the shape trace as evidence.
- No separate inspector UI; the report is the artifact.
#### Experiment and Measure
- Give a ten-minute whiteboard explanation from waveform features through one block to a latency claim.
#### Required Output
['- `reports/week3_conformer.md`', '- `results/day19_shape_trace.md`']
#### Completion Check
> You can explain which parts are local, which are global, and which become the bottleneck under streaming.

---
