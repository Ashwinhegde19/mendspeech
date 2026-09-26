# Day 56: Final demo, clean reproduction, and release

> **Week 8 • Day 7 of 7**  
> **Navigation:** [← Day 55](day_55.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Index →](../INDEX.md)

> **v3 STATUS: CORE** Release gate. The demo must show measured numbers, not a scripted success path.
---

### Compute Target
`Modal L4 plus local interface`

---

### 1. Learn
- Demonstrating a system honestly, including its failure modes.
- Releasing with a stable, reproducible artifact.

---

### 2. Build in MendSpeech
- Extend only `app/audio_lab.py` with live or prerecorded audio, partial/final transcripts, confidence, triage actions, and the measured latency budget.
- Reproduce one frozen benchmark from a fresh environment and tag a stable release.

---

### 3. Experiment and Measure
- Verify every public chart regenerates from saved result files.
- Demonstrate at least one failure case, not only the success path.
- Confirm the demo's displayed numbers match the committed result files.

---

### 4. Required Output Artifacts
['- `app/audio_lab.py`', '- `REPRODUCE.md`', '- `demos/final_demo.mp4`', '- `docs/architecture.md`']

---

### 5. Completion Check
> **Definition of Done for Day 56:**  
> A new user can run, evaluate, and reproduce the system, and every number shown traces to a committed artifact.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Reproducible release practice
