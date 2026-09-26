# Day 26: Efficiency benchmark harness

> **Week 4 • Day 5 of 7**  
> **Navigation:** [← Day 25](day_25.md) | [Week 4 Plan](../Week_4_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 27 →](day_27.md)

> **v2 STATUS: CORE — efficiency and three-way offline decoding.** Extend the shared harness; one acoustic model and one small n-gram LM. Extra integration work is not assumed to fit the original session slot.

---

### Compute Target
`Modal L4`

---

### 1. Learn
- Warmup runs.
- Synchronized GPU timing.
- Median and percentile latency.
- Real time factor.
- Peak memory.
- Greedy search versus beam search; LM weight, insertion terms, and pruning.
- External LM preferences can alter content: lower WER alone is not proof that
  repair decisions are safer. Fused search scores are not calibrated probabilities.

---

### 2. Build in MendSpeech
- Create one benchmark function used by every later experiment.
- Log environment and model metadata automatically.
- Use the compatible acoustic head/backend verified in Day 24. Compare greedy,
  beam without an external LM, and beam plus one small n-gram LM in
  `src/asr/decoding.py`; no second acoustic checkpoint or scratch search engine.
- Freeze text provenance, permitted use, source IDs, normalization, hashes,
  deduplication and split roles in `data/lm_text_manifest.csv`. Exclude all
  evaluation references/duplicates from LM training; document unknown base-model
  pretraining overlap. Keep raw corpora and LM binaries in ignored storage.
- Freeze one small LM order and corpus budget plus a small validation-only
  beam-width/LM-weight candidate list in `configs/decoding.yaml`. Keep lexicon,
  insertion bonuses, tokenizer and normalization controlled; beam-only disables
  the external LM. Select once on validation, never using test errors as prompts
  for vocabulary additions, tuning, or retraining. No broad parameter sweep.
- Reuse the same acoustic outputs when the selected head/backend permits it;
  otherwise rerun identical frozen acoustic settings and label cache use. Test
  empty inputs, token mapping, and LM influence in `tests/test_asr_decoding.py`;
  CTC tests include blank/repeated-token collapse. Test timing/alignment mapping
  before decoded hypotheses can enter repair. Document backend score semantics.
- Keep offline decoding separate from the streaming runner. If compatibility is
  blocked, retain the greedy baseline and report the failed required comparison
  for scope review, not completed beam/LM integration.

---

### 3. Experiment and Measure
- Run repeated inference and calculate variance.
- Detect and discard obviously invalid cold start comparisons.
- Report WER/CER, predeclared names/numbers errors, and both helpful and harmful
  transcript changes on identical held-out clean/damaged cases. A null or worse
  LM result is valid; invented improvement or omitted failed cases are not.
- Measure decoder-only elapsed time using cached acoustic outputs separately
  from fresh audio-to-transcript latency, RTF and peak memory. Use repeated warm
  runs on fixed L4/batch/precision, separate cold measurements, and record CPU
  decoder host, thread/worker counts, cache status, timing boundaries and counts.
  Cached decoder speed is not live ASR latency; no streaming LM claim yet.

---

### 4. Required Output Artifacts
- `src/bench/benchmark_asr.py`
- `src/bench/environment.py`
- `results/day26_repeatability.csv`
- `src/asr/decoding.py`
- `tests/test_asr_decoding.py`
- `configs/decoding.yaml`
- `data/lm_text_manifest.csv`
- `results/day26_decoding_comparison.csv`
- `docs/day26_decoding.md` (reproduction, tuning/split audit, score semantics,
  backend/head compatibility, help/hurt cases, and streaming limitations)

---

### 5. Completion Check
> **Definition of Done for Day 26:**  
> Repeated runs support reproducible greedy/beam/LM accuracy and cost comparisons
> with validation-only selection and tests. The evidence distinguishes cached
> decoder time from fresh end-to-end latency and search scores from confidence.
> A blocked beam or LM branch leaves this comparison incomplete; independent
> work may continue, but Gate 3 requires evidence or an approved scope revision.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.
Re-estimate after Day 24; the LM corpus and integration are additional work.
After two extra sessions, report remaining scope before extending the experiment.

---

### 7. References & Resources
- FastConformer primary paper
- NVIDIA NeMo FastConformer model documentation
- NVIDIA NeMo ASR Language Modeling and Customization: use the beam/n-gram
  recipe and optional dependencies matching the pinned framework and head.
