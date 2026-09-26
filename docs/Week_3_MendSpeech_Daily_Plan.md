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

## Week Map

| Day | Focus | Minimum Evidence / Artifact | Compute | Daily Link |
| :--- | :--- | :--- | :--- | :--- |
| **Day 15** | Attention for speech sequences | You can derive every major tensor shape and explain quadratic sequence cost. | `Local CPU, L4 optional for scaling` | [Open Day 15](days/day_15.md) |
| **Day 16** | Conformer convolution module | You can explain why depthwise convolution is computationally attractive and what local context it captures. | `Local CPU` | [Open Day 16](days/day_16.md) |
| **Day 17** | LEARN-ONLY: macaron feed forward and residual scaling | Explain block ordering; build work is absorbed into Day 18. | `Local CPU` | [Open Day 17](days/day_17.md) |
| **Day 18** | CORE: assemble one Conformer block; absorbs Days 17 and 19 | One tested block, macaron FFN, real-log-Mel projection/masks, shape trace, and valid gradients. | `Local CPU` | [Open Day 18](days/day_18.md) |
| **Day 19** | MERGED into Day 18: real-log-Mel block validation | `results/day19_shape_trace.md` is produced in Day 18; no separate encoder. | `Local CPU — within Day 18` | [Open Day 19](days/day_19.md) |
| **Day 20** | DROPPED: production implementation comparison | Optional reading only; no build or artifact. | `Local CPU` | [Open Day 20](days/day_20.md) |
| **Day 21** | CORE: static architecture review | Report links the tested shape trace and explains local/global context and streaming limits. | `Local CPU` | [Open Day 21](days/day_21.md) |

---

## v2 Compression Map (Gate Evidence)

| Day | v2 Status | Note |
| :--- | :--- | :--- |
| **Day 15** | CORE | Attention from scratch |
| **Day 16** | CORE | Conformer convolution module from scratch |
| **Day 17** | LEARN-ONLY | No build session. Read the Learn block in theory time; the macaron build moves into Day 18. |
| **Day 18** | CORE — absorbs Days 17 and 19 | Macaron FFN + one block with real-log-Mel projection, masks, shapes, and gradient tests; no second toy-training ablation required |
| **Day 19** | MERGED into Day 18 | Same-block validation only; retain `results/day19_shape_trace.md`, no separate encoder or depth sweep |
| **Day 20** | DROPPED | Reading assignment only: orient in production Conformer code |
| **Day 21** | CORE | Static architecture report and linked shape trace; no inspector UI |

---

## Reference Spine
- Gulati et al., Conformer: Convolution
- augmented Transformer for Speech Recognition\nAnnotated Transformer and Attention Is All You Need\nA mature open
- source Conformer implementation

---

## Daily Detailed Operating Plans

### DAY 15: Attention for speech sequences
- **Compute:** `Local CPU, L4 optional for scaling`
- **Dedicated Daily File:** [`docs/days/day_15.md`](days/day_15.md)

#### Learn
- Query, key, value projections.
- Scaled dot product attention.
- Attention masks.
- Sequence length cost.

#### Build in MendSpeech
- Implement single head attention and then multi head attention in PyTorch.
- Add shape assertions and gradient tests.

#### Experiment and Measure
- Change sequence length and measure forward time and memory.

#### Required Output
- `src/models/attention.py`
- `tests/test_attention.py`
- `results/day15_attention_scaling.csv`

#### Completion Check
> You can derive every major tensor shape and explain quadratic sequence cost.

---

### DAY 16: Conformer convolution module
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_16.md`](days/day_16.md)

#### Learn
- Pointwise convolution.
- GLU gating.
- Depthwise convolution.
- Batch normalization and activation.
- Why local patterns matter in speech.

#### Build in MendSpeech
- Implement a Conformer style convolution module.
- Test causality assumptions and receptive field.

#### Experiment and Measure
- Feed synthetic impulses and inspect how local information spreads.

#### Required Output
- `src/models/conformer_conv.py`
- `tests/test_conformer_conv.py`
- `notebooks/day16_receptive_field.ipynb`

#### Completion Check
> You can explain why depthwise convolution is computationally attractive and what
local context it captures.

---

### DAY 17: Macaron feed forward and residual scaling (LEARN-ONLY)
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_17.md`](days/day_17.md)

#### Learn
- Feed forward expansion.
- Swish or SiLU activation.
- Dropout.
- Half step residual weighting in Conformer.

#### Build in MendSpeech
- No standalone build. Day 18 implements the feed-forward module, residual wrapper, and numerical shape/gradient tests.

#### Experiment and Measure
- Day 18 compares output statistics with and without residual scaling in its tests.

#### Required Output
- None for this learn-only session; the FFN implementation and tests are Day 18 outputs.

#### Completion Check
> You can explain the ordering of the Conformer block without memorizing a diagram.

---

### DAY 18: Assemble one Conformer block
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_18.md`](days/day_18.md)

> **v2 STATUS: CORE — absorbs Days 17 and 19.** Implement the macaron feed-forward, assemble one tested Conformer block, and pass real log-Mel features through that same block with input projection and mask propagation. No separate encoder or depth sweep.

#### Learn
- Macaron structure.
- Layer normalization placement.
- Attention plus convolution interaction.
- Input projection, padding masks, and temporal dimensions for real log-Mel features.

#### Build in MendSpeech
- Implement the macaron feed-forward and half-step residual wrapper absorbed from Day 17, with shape and gradient tests.
- Assemble feed forward, the scratch attention and convolution modules, second feed forward, and final normalization into one block.
- Add input projection from real log-Mel features and propagate padding masks through this same block; assert input, output, and valid-length shapes.

#### Experiment and Measure
- Run forward and backward tests on several sequence lengths.
- Trace a real log-Mel tensor through projection and every stage; test mask handling and finite, nonzero gradients on valid inputs.
- Compare output statistics with and without half-step residual scaling in the same tests. A second toy-training ablation is not a release requirement.

#### Required Output
- `src/models/conformer_ffn.py` — Day 17's absorbed macaron implementation
- `tests/test_conformer_ffn.py`
- `src/models/conformer_block.py`
- `tests/test_conformer_block.py`
- `docs/conformer_block_walkthrough.md`
- `results/day19_shape_trace.md` — retained Day 19 path, produced here

#### Completion Check
> You can explain every operation in one tested block. A real log-Mel tensor
> passes through its projection and mask handling with documented shapes and
> valid gradients; the retained Day 19 shape trace records the evidence.

---

### DAY 19: Real-log-Mel block validation (merged into Day 18)
- **Compute:** `Local CPU — within Day 18`
- **Dedicated Daily File:** [`docs/days/day_19.md`](days/day_19.md)

> **v2 STATUS: MERGED into Day 18.** No standalone session. Validate input projection, mask propagation, shapes, and gradients on Day 18's same Conformer block; do not build a separate encoder or run a depth sweep.

#### Learn
- Input projection.
- Mask propagation.
- Temporal dimensions.

#### Build in MendSpeech
- In Day 18, connect real log-Mel features through input projection to the same tested Conformer block.
- Propagate padding masks and validate temporal dimensions; no separate implementation.

#### Experiment and Measure
- In Day 18, trace shapes through every stage on real speech and test masks and finite, nonzero gradients on valid inputs.

#### Required Output
- `results/day19_shape_trace.md` — produced in Day 18; no standalone code artifact

#### Completion Check
> Absorbed into Day 18: a real log-Mel tensor passes through the same block's
> projection and mask handling with a documented shape trace and valid gradients.

---

### DAY 20: Production implementation comparison (DROPPED)
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_20.md`](days/day_20.md)

#### Learn
- Read the original Conformer paper sections relevant to block design.
- Inspect a mature implementation such as NeMo.
- Identify differences caused by engineering and efficiency.

#### Build in MendSpeech
- None; optional reading only, not a scheduled build.

#### Experiment and Measure
- No experiment required.

#### Required Output
- None; no standalone artifact is required.

#### Completion Check
> No scheduled completion requirement. Optional reading supports orientation in production Conformer code.

---

### DAY 21: Week 3 architecture review
- **Compute:** `Local CPU`
- **Dedicated Daily File:** [`docs/days/day_21.md`](days/day_21.md)

> **v2 STATUS: CORE — static architecture review.** Keep the review milestone and shape evidence; no architecture-inspector UI.

#### Learn
- Review attention, convolution, feed forward, normalization, residual paths, and sequence cost.

#### Build in MendSpeech
- Write a static architecture report showing the single block's stage shapes, mask propagation, and context assumptions, using `results/day19_shape_trace.md` from Day 18.

#### Experiment and Measure
- Give yourself a ten minute whiteboard explanation from waveform features through one Conformer block.

#### Required Output
- `reports/week3_conformer.md`

#### Completion Check
> You can explain which parts are local, which are global, and which become
> problematic for streaming, with the static report linked to the tested shape trace.

---
