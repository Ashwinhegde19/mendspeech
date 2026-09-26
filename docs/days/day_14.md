# Day 14: Decoding comparison: greedy, beam, and beam plus LM

> **Week 2 • Day 7 of 7**  
> **Navigation:** [← Day 13](day_13.md) | [Week 2 Plan](../Week_2_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 15 →](day_15.md)

> **v3 STATUS: CORE** Week 2 integration and the accuracy-versus-latency trade-off. A lower WER is not automatically better: a fluent but acoustically wrong transcript is the failure this project must catch.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Greedy versus beam search: accuracy gained against search cost.
- External n-gram language models: why a plausible transcript can be acoustically wrong.
- Cache reuse for isolating decoder cost from acoustic cost.

---

### 2. Build in MendSpeech
- Extend the baseline runner to support greedy, beam, and beam plus one small n-gram LM in `src/asr/decoding.py`.
- Record LM text provenance, normalization, and split roles in `data/lm_text_manifest.csv`; exclude evaluation references and duplicates.
- Freeze one LM order and a small validation-only beam/LM-weight candidate list in `configs/decoding.yaml`.

---

### 3. Experiment and Measure
- Report WER/CER and names-and-numbers error for all three decoders on identical held-out cases.
- Collect cases where the LM helped and cases where it hurt; a lower WER does not prove safety.
- Measure decoder-only time on cached acoustic outputs separately from fresh audio-to-transcript latency.

---

### 4. Required Output Artifacts
['- `src/asr/decoding.py`', '- `tests/test_asr_decoding.py`', '- `configs/decoding.yaml`', '- `data/lm_text_manifest.csv`', '- `results/day14_decoding_comparison.csv`', '- `docs/day14_harmful_lm_changes.md`', '- `app/audio_lab.py`']

---

### 5. Completion Check
> **Definition of Done for Day 14:**  
> All three decoders are measured on the same held-out cases with separated decoder and end-to-end timing, and the LM's harmful changes are documented as first-class evidence.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- NVIDIA NeMo ASR Language Modeling and Customization
- Beam search and n-gram LM fusion references
