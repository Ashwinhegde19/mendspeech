# Day 56: Final demo, clean reproduction and release

> **Week 8 • Day 7 of 7**
> **Navigation:** [← Day 55](day_55.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Index →](../INDEX.md)

> **v4 STATUS: CORE** Planned evidence, not completed implementation.
> **Prerequisites:** [Day 55](day_55.md)
> **Effort:** 2–3 focused hours; estimates include learning and tests, not a deadline.

---

### Compute Target
`Modal L4 for measured GPU work; local CPU for checks`

---

### 1. Learn
- Honest demonstration including failure modes; reproducible release.

---

### 2. Build in MendSpeech
- Extend only app/audio_lab.py: live/replayed audio, partial/final transcripts, confidence, triage, guarded edit, latency budget, one failure case.
- Reproduce one frozen benchmark from a fresh environment and tag a release.

---

### 3. Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Confirm displayed numbers match committed artifacts; show a failure case, not only the success path.

---

### 4. Required Output Artifacts
- `app/audio_lab.py`
- `REPRODUCE.md`
- `demos/final_demo.mp4`
- `docs/architecture.md`

---

### 5. Completion Check
> **Definition of Done for Day 56:**
> A new user can run, evaluate and reproduce the system, and every number shown traces to a committed artifact.

---

### 6. Study Method & Protocol
Read the relevant concepts, implement the smallest testable slice, measure, and explain one concrete example (shape, units, seed, input and output). Use the effort range to schedule multiple sittings when needed. Do not substitute file existence or a blocked run for required evidence. Stop at declared spend/time limits; seek scope review after two extra sittings without progress.

---

### 7. References & Resources
- [Execution and measurement rules](../REVISED_EXECUTION_PLAN.md)
- [Timing and quality contract](../LATENCY_AND_QUALITY_CONTRACT.md)
- [Editor and RL contract](../EDITOR_AND_RL_CONTRACT.md)
- Pinned model/backend primary documentation; verify supported behavior before using optional dependencies.
