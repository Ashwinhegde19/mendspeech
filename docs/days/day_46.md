# Day 46: LLM post-processing stage

> **Week 7 • Day 4 of 7**  
> **Navigation:** [← Day 45](day_45.md) | [Week 7 Plan](../Week_7_MendSpeech_Daily_Plan.md) | [Master Index](../INDEX.md) | [Day 47 →](day_47.md)

> **v3 STATUS: CORE** One small pinned model, behind an adapter, so the ASR result stays reproducible without it.
---

### Compute Target
`Modal L4`

---

### 1. Learn
- Time-to-first-token versus full response.
- Streaming versus batched generation.
- Prefix caching and why repeated system context should be free.

---

### 2. Build in MendSpeech
- Add one small pinned LLM post-processing adapter in `src/llm/polish.py`; the core ASR path must run without it.
- Pin model, revision, quantization, and prompt template in `configs/llm.yaml`.

---

### 3. Experiment and Measure
- Measure TTFT and full-response latency separately.
- Measure prefix-cache hit rate across repeated requests and its effect on TTFT.
- Report quality change on the polished output, not only latency.

---

### 4. Required Output Artifacts
['- `src/llm/polish.py`', '- `configs/llm.yaml`', '- `tests/test_llm_polish.py`', '- `results/day46_llm_latency.csv`']

---

### 5. Completion Check
> **Definition of Done for Day 46:**  
> The LLM stage is measured for TTFT and full response, and the ASR result is still reproducible with it disabled.

---

### 6. Study Method & Protocol
25 minutes focused reading. 65 minutes implementation or controlled experiment. 20 minutes research
notebook. 10 minutes commit and explain the result aloud. When debugging is incomplete, continue the
same task in the next session instead of pretending the day is finished.

---

### 7. References & Resources
- vLLM and TGI serving documentation
- Prefix caching and KV-cache reuse references
