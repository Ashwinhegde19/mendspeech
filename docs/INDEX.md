# MendSpeech AI Agent & Developer Navigation Index

Welcome to the **MendSpeech Documentation and Daily Execution Suite**. This workspace is structured specifically for executing the MendSpeech research and systems project day-by-day using **AI Coding Agents** (such as Antigravity, Claude, or Gemini).

> **v2 scope:** Read the [execution plan](REVISED_EXECUTION_PLAN.md) first.
> Day numbers are specification identifiers, not calendar promises. Status
> banners override day bodies; optional drills are not release gates. Extend
> one application, `app/audio_lab.py`, rather than creating milestone copies.

---

## 📂 Workspace Structure

```text
MendSpeech_All_Plans/
├── docs/                                 # Complete Markdown Knowledge Base & Daily Plans
│   ├── INDEX.md                          # This master navigation index
│   ├── MendSpeech_Project_Blueprint.md   # Architectural blueprint, metrics, definition of done
│   ├── MendSpeech_8_Week_Master_Roadmap.md # Timeline, workload, research questions, hardware
│   ├── MendSpeech_Complete_56_Day_Plan.md# All 56 daily plans compiled in one searchable file
│   ├── SPEECH_ML_SYSTEMS_DRILLS.md        # Optional speech systems fundamentals drills
│   ├── Week_1_MendSpeech_Daily_Plan.md   # Audio DSP & SpeechDamageBench v0
│   ├── Week_2_MendSpeech_Daily_Plan.md   # ASR, CTC math, confidence & Modal pipeline
│   ├── Week_3_MendSpeech_Daily_Plan.md   # One tested Conformer block from scratch
│   ├── Week_4_MendSpeech_Daily_Plan.md   # FastConformer subsampling & efficiency
│   ├── Week_5_MendSpeech_Daily_Plan.md   # Streaming cache-aware ASR & lookahead
│   ├── Week_6_MendSpeech_Daily_Plan.md   # Robust fine-tuning, RNN-T & calibration
│   ├── Week_7_MendSpeech_Daily_Plan.md   # TTS, boundary matching & seam diagnostics
│   ├── Week_8_MendSpeech_Daily_Plan.md   # Controlled repair and restoration comparisons
│   └── days/                             # Granular individual daily task files (Day 01 to Day 56)
│       ├── day_01.md
│       ├── day_02.md
│       └── ...
└── pdfs/                                 # Original preserved PDF documents (archived)
    ├── MendSpeech_Project_Blueprint.pdf
    ├── MendSpeech_8_Week_Master_Roadmap.pdf
    ├── MendSpeech_Complete_56_Day_Plan.pdf
    └── ...
```

---

## 🤖 How to Execute Day-by-Day with AI Coding Agents

When working with an AI agent:
1. **Feed Today's Prompt Directly:** Mention the specific daily file (e.g. `@docs/days/day_01.md`).
2. **Standard 2-Hour Protocol:**
   - **25 min:** Read theory/concepts outlined under `# 1. Learn`.
   - **65 min:** Build code and run experiments under `# 2. Build` and `# 3. Experiment and Measure`.
   - **20 min:** Update the research notebook (`notebooks/` or `results/`).
   - **10 min:** Validate against `# 5. Completion Check` and commit artifacts under `# 4. Required Output`.
3. **Keep Compute Fixed:** Check the day's `Compute Target`. Week 1 is Local CPU. Modal L4 starts as an option on Day 08 and is the default for measured GPU work from Week 4 onward.

---

## 🗺️ Master 8-Week / 56-Day Progression Matrix

| Week | Focus | Milestone | Weekly Plan | Daily Files |
| :--- | :--- | :--- | :--- | :--- |
| **Week 1** | Audio, Degradation, & Measurement Foundations | Build the audio laboratory and release `SpeechDamageBench` as a standalone package. | [Week 1 Guide](Week_1_MendSpeech_Daily_Plan.md) | [Day 01](days/day_01.md) • [Day 02](days/day_02.md) • [Day 03](days/day_03.md) • [Day 04](days/day_04.md) • [Day 05](days/day_05.md) • [Day 06](days/day_06.md) • [Day 07](days/day_07.md) |
| **Week 2** | ASR, CTC, Confidence, & Repair Localization | Build the recognition and uncertainty layer, plus a reusable Modal cloud pipeline. | [Week 2 Guide](Week_2_MendSpeech_Daily_Plan.md) | [Day 08](days/day_08.md) • [Day 09](days/day_09.md) • [Day 10](days/day_10.md) • [Day 11](days/day_11.md) • [Day 12](days/day_12.md) • [Day 13](days/day_13.md) • [Day 14](days/day_14.md) |
| **Week 3** | Conformer From First Principles | Implement Conformer attention, convolutions, and Macaron feed-forwards from scratch in PyTorch. | [Week 3 Guide](Week_3_MendSpeech_Daily_Plan.md) | [Day 15](days/day_15.md) • [Day 16](days/day_16.md) • [Day 17](days/day_17.md) • [Day 18](days/day_18.md) • [Day 19](days/day_19.md) • [Day 20](days/day_20.md) • [Day 21](days/day_21.md) |
| **Week 4** | FastConformer & Efficient Encoder Behavior | Profile subsampling, receptive fields, and establish a reproducible FastConformer baseline. | [Week 4 Guide](Week_4_MendSpeech_Daily_Plan.md) | [Day 22](days/day_22.md) • [Day 23](days/day_23.md) • [Day 24](days/day_24.md) • [Day 25](days/day_25.md) • [Day 26](days/day_26.md) • [Day 27](days/day_27.md) • [Day 28](days/day_28.md) |
| **Week 5** | Streaming, Cache-Aware Inference, & Adaptive Context | Implement cache-aware streaming ASR and evaluate uncertainty-guided adaptive context spending. | [Week 5 Guide](Week_5_MendSpeech_Daily_Plan.md) | [Day 29](days/day_29.md) • [Day 30](days/day_30.md) • [Day 31](days/day_31.md) • [Day 32](days/day_32.md) • [Day 33](days/day_33.md) • [Day 34](days/day_34.md) • [Day 35](days/day_35.md) |
| **Week 6** | Robustness, Fine-Tuning, RNN-T, & Calibration | Adapt the recognizer to damaged speech, explore RNN-T, and calibrate confidence scores. | [Week 6 Guide](Week_6_MendSpeech_Daily_Plan.md) | [Day 36](days/day_36.md) • [Day 37](days/day_37.md) • [Day 38](days/day_38.md) • [Day 39](days/day_39.md) • [Day 40](days/day_40.md) • [Day 41](days/day_41.md) • [Day 42](days/day_42.md) |
| **Week 7** | TTS, Speaker Preservation, & Boundary-Matched Reconstruction | Build MendSpeech V1 cascaded selective repair with duration alignment and seam diagnostics. | [Week 7 Guide](Week_7_MendSpeech_Daily_Plan.md) | [Day 43](days/day_43.md) • [Day 44](days/day_44.md) • [Day 45](days/day_45.md) • [Day 46](days/day_46.md) • [Day 47](days/day_47.md) • [Day 48](days/day_48.md) • [Day 49](days/day_49.md) |
| **Week 8** | Research Capstone: Controlled Repair Comparisons | Freeze benchmarks, run ablations, report the verified external comparator or its deferral, and reproduce the release. | [Week 8 Guide](Week_8_MendSpeech_Daily_Plan.md) | [Day 50](days/day_50.md) • [Day 51](days/day_51.md) • [Day 52](days/day_52.md) • [Day 53](days/day_53.md) • [Day 54](days/day_54.md) • [Day 55](days/day_55.md) • [Day 56](days/day_56.md) |

---

## 📚 Core Documentation Links
- [**Revised Execution Plan — v2 focused scope, gates & add-on labs**](REVISED_EXECUTION_PLAN.md)
- [**MendSpeech Project Blueprint**](MendSpeech_Project_Blueprint.md)
- [**8-Week Master Roadmap**](MendSpeech_8_Week_Master_Roadmap.md)
- [**Complete 56-Day Searchable Plan**](MendSpeech_Complete_56_Day_Plan.md)
- [**Optional Speech-ML Systems Drill Track**](SPEECH_ML_SYSTEMS_DRILLS.md)
- [**Index of PDF Documents**](MendSpeech_PDF_Set_Index.md)
