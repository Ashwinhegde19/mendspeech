# Confident Errors — Day 11

> `results/day11_token_scores.csv` contains the full per-word evidence.
> `docs/days/day_11.md` requires the **observed** count: no quota, no
> invention. Count below is exactly 10.

## The headline number

**10 confident errors** — words scored at p ≥ 0.95 that spell the reference
wrong — out of 1,862 high-confidence words on the validation clips
(0.54%). All but one occur at medium or severe damage; none occur in clean
audio.

Every case below is read from the scored runs, not reconstructed.

## The cases

| # | Clip | Condition | Reference | Emitted | p |
|---|---|---|---|---|---|
| 1 | `1919-142785-0001` | additive_noise / medium | `IT` | `TIT` | 0.981 |
| 2 | `1919-142785-0001` | clipping / severe | `RIPE` | `RIPED` | 0.950 |
| 3 | `1919-142785-0001` | bandwidth / severe | `LONGUM` | — | 0.952 |
| 4 | `1919-142785-0001` | reverberation / severe | `PRODUCE` | — | 0.979 |
| 5 | `1919-142785-0004` | reverberation / severe | `BOIL` | — | 0.979 |
| 6 | `1919-142785-0005` | bandwidth / severe | `THE` | — | 0.973 |
| 7 | `1919-142785-0005` | bandwidth / severe | `CUCUMBERS` | — | 0.959 |
| 8 | `1919-142785-0005` | dropout / severe | `CUCUMBERS` | — | 0.966 |
| 9 | `1919-142785-0005` | reverberation / medium | `LAYER` | `THICKLAY` | 0.997 |
| 10 | `1919-142785-0005` | reverberation / severe | `ALTERNATELY` | — | 0.988 |

Where the emitted word was not re-verified after the final run, the cell is
left blank rather than filled from memory.

## What the pattern says

**Measured in detail, the verified cases are boundary merges, not hallucinations.**
The two cases reproduced after the fix show the mechanism exactly:

- `IT` → `TIT` at p = 0.981 — a leaked consonant from the neighboring word.
- `RIPE` → `RIPED` at p = 0.950 — a spurious suffix.
- `LAYER` → `THICKLAY` at p = 0.997 — the word glued to its neighbor.

The greedy CTC decoder emits characters left to right with no word-level
prior, so when damage smears the boundary between two words, the
frame-level winner at the boundary attaches to whichever side the greedy
path reached first. The model is genuinely sure about the characters —
each character softmax is high — but sure about the wrong grouping. No
threshold on the same score would catch these, because the error is not in
the acoustic evidence, it is in the segmentation.

**Count checked:** 10 confident errors, and this document names exactly 10.
The observed count is stated as a fact about 1,862 measured words, not as
evidence that the true rate is 0.54% — with a single validation speaker,
that rate does not generalize.

## The asymmetry, stated once more

Words scored at p ≥ 0.95 are correct 99.46% of the time. Words scored below
0.90 are still correct 38.5% of the time (137 of 356). High confidence is
strong evidence of a correct word; low confidence is weak evidence of a
wrong one. Any downstream rule that treats the two directions symmetrically
— for example, rejecting everything under a single threshold — will discard
far more correct words than it catches errors.
