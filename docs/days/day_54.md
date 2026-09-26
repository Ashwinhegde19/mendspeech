# Day 54: Capability-scoped direct restoration comparison

> **Week 8 • Day 5 of 7**  
> **Navigation:** [← Day 53](day_53.md) | [Week 8 Plan](../Week_8_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 55 →](day_55.md)

> **v2 STATUS: CORE — one conditional external comparator, no model hunting.** Week 2 feasibility bounds apply; unavailable external restoration/inpainting is explicitly deferred.

---

### Compute Target
`Modal L4 for comparisons; local CPU for analysis`

---

### 1. Learn
- Why text is an information bottleneck for prosody and acoustic continuity.
- Direct audio inpainting in latent or codec token spaces at a conceptual level.
- Fair baseline design when systems have different latency and compute profiles.
- Failure taxonomy across semantic correctness, speaker similarity, prosody, seam quality, and compute.

---

### 2. Build in MendSpeech
- Consume the one selected comparator and bounded feasible/deferred decision
  in Week 2's `docs/baseline_install_notes.md`. Selection is not a claim of
  tested support. Do not search for substitutes or train restoration from scratch.
- If feasible, implement only `src/baselines/direct_audio_restore.py` behind
  the shared benchmark interface. Record checkpoint/revision/license, sample
  rate, supported damage conditions, mask support, and whether output changes
  samples outside a requested interval. Do not pass masks unless supported.
  This plan chooses one restoration adapter, not a separate inpainting
  adapter; an inpainting claim requires verified mask-aware
  missing-span reconstruction, not a suggestive filename or denoising output.
- Feed identical supported SpeechDamageBench cases, label out-of-scope
  conditions `unsupported`, and exclude them from aggregate comparisons.
  A model available only outside L4 is not an L4 efficiency comparison;
  defer it rather than silently changing hardware or extending the budget.
- If unavailable, still compare raw damaged audio, full resynthesis, naive
  selective repair, and boundary-matched selective repair. Record external
  restoration/inpainting as `deferred`, not tested; no second project/fallback.
- Reuse the tested `src/controller/abstain.py` from Day 49. Keep abstention
  active when inferred content, conditioning consent, or seam safety is weak.

---

### 3. Experiment and Measure
- Compare the four internal paths and only the supported external conditions
  on the same cases. Use predicted text for normal TTS paths; separately label
  oracle text/spans and live versus simulated context policies.
- Select at least ten worst or most revealing cases and inspect them manually.
- In `results/capstone_architecture_compare.csv`, include method/checkpoint,
  corruption, mask capability, text/span source, execution mode, condition
  status, and reason. Measured rows may show improvement, equality, or harm;
  `unsupported`/`deferred` rows have missing metrics, never invented numbers.
- Create a failure casebook and tradeoff plot from measured evidence only.
  If no supported external run exists, title the plot as internal comparisons
  and state the deferral; do not imply both architectures were evaluated.

---

### 4. Required Output Artifacts
- Feasible branch only: `src/baselines/direct_audio_restore.py` (one adapter
  with capability metadata; no placeholder implementation if deferred)
- `results/capstone_architecture_compare.csv`
- `results/capstone_failure_casebook.md`
- `results/architecture_tradeoff.png`
- Reuse, do not postpone: `src/controller/abstain.py` (required by Day 49)

---

### 5. Completion Check
> **Definition of Done for Day 54:**  
> Supported conditions have reproducible measured comparisons and explicit
> limitations; unsupported conditions are not fabricated. If the comparator
> is unavailable, the internal capstone plus external/inpainting deferral is
> complete, but an external or mask-aware comparison is not claimed as tested.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- Your frozen protocol and prior results
- The one Week 2 comparator's model card and verified capability record
- Primary papers only when needed to interpret a result
