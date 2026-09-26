# Day 14: Week 2 integration and review

> **Week 2 • Day 7 of 7**  
> **Navigation:** [← Day 13](day_13.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 15 →](day_15.md)

> **v2 STATUS: CORE — Gate 2 advances on evidence, not a date.** Extend
> `app/audio_lab.py`; external-comparator deferral does not block the core ASR work.

---

### Compute Target
`Modal L4 recommended`

---

### 1. Learn
- Review CTC, WER, confidence, timestamp alignment, and repair decisions.

---

### 2. Build in MendSpeech
- Extend `app/audio_lab.py`: damaged audio to transcript to confidence to timed
  preserve/inspect/repair/abstain proposals. Reuse the Day 12 overlay; do not
  create a separate milestone app or claim synthesis before it exists.
- Add clean JSON output for every run: source/corruption/seed, model revision,
  policy preset/version, thresholds, intervals, actions and reason codes.
- Verify the Modal wrapper records model revision, GPU type, software versions, and run id automatically.
- Carry forward `docs/baseline_install_notes.md`: one comparator's `feasible`
  or `deferred` status and verified capabilities. A blocker report is enough
  for this conditional branch; it must not be labeled masked-inpainting success.

---

### 3. Experiment and Measure
- Run at least twenty corrupted utterances with matched clean/raw-damaged
  controls and fixed validation-selected thresholds; inspect false repair,
  missed repair, inspect and abstain cases. Preserve model-version provenance.
- Report ASR WER/CER, uncertainty overlap, proposed repair coverage and clean
  false repairs; do not imply generated-audio improvement. Keep L4 comparisons
  separate from functional CPU smoke runs.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `infra/modal_asr.py`
- `results/week2_casebook.md`
- `reports/week2_asr_uncertainty.md`

---

### 5. Completion Check
> **Definition of Done for Day 14:**  
> The one app shows what the ASR heard and the exact proposed actions, with
unchanged audio for inspect/abstain. The casebook/report retain controls, model
and policy versions, errors and comparator feasibility status. Gate 2 does not
require a successful external neural restoration model.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- CTC primary paper or a reliable derivation
- Framework ASR documentation for logits, timestamps, and confidence
