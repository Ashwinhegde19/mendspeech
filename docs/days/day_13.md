# Day 13: Define selective repair policy v0

> **Week 2 • Day 6 of 7**  
> **Navigation:** [← Day 12](day_12.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 14 →](day_14.md)

> **v2 STATUS: CORE.** Define safe action semantics now; exercise synthesis-time
> abstention by Day 49, not first at the final comparison.

---

### Compute Target
`Local CPU after ASR outputs are
cached`

---

### 1. Learn
- Threshold policies.
- Hysteresis to avoid rapid toggling.
- Minimum repair span and padding.
- False repair versus missed repair tradeoff.
- Uncertainty indicates a need for evidence, not permission to invent content.

---

### 2. Build in MendSpeech
- Keep Preserve, Balanced, and Rescue as sensitivity presets, not action labels.
  Each returns timed decisions with an action and reason code:
  - `preserve`: reliable audio remains unchanged.
  - `inspect`: flag uncertain content for review; do not synthesize it.
  - `repair`: propose a bounded edit only when content evidence, speaker-use
    permission, alignment and supported synthesis constraints are sufficient.
  - `abstain`: an unsafe or unsupported repair is refused; retain original audio
    and disclose why no reconstruction was produced.
- Missing evidence or unavailable synthesis support cannot silently become a
  repair. Week 2 tests decisions without claiming generated audio. Carry these
  semantics into `src/controller/abstain.py` and exercise them on Day 49.

---

### 3. Experiment and Measure
- Sweep thresholds on speaker-separated validation data only; log selected
  values in `configs/repair_modes.yaml` and freeze them before test scoring.
- Measure proposed repair coverage and overlap with known damage, false repair
  on clean speech, missed repair, and inspect/abstain rates. Ground-truth damage
  masks score decisions; they are not policy inputs in normal evaluation.
- Include reliable clean audio, uncertain text, missing permission, missing
  capability and invalid alignment cases; verify all non-repair actions leave
  audio unchanged. Raw confidence remains provisional until Day 41 calibration.

---

### 4. Required Output Artifacts
- `src/controller/policy.py`
- `configs/repair_modes.yaml`
- `results/day13_policy_sweep.csv`

---

### 5. Completion Check
> **Definition of Done for Day 13:**  
> You can explain and demonstrate preserve/inspect/repair/abstain decisions,
including refusal to synthesize unsupported content. Thresholds come only from
validation; false repairs and abstentions are visible rather than hidden.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- CTC primary paper or a reliable derivation
- Framework ASR documentation for logits, timestamps, and confidence
