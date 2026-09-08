# Connectionist Temporal Classification (CTC) from First Principles

> **MendSpeech Technical Reference • Week 2: Recognition & Uncertainty**  
> **Source Module:** [`src/asr/ctc_decode.py`](../src/asr/ctc_decode.py) • **Tests:** [`tests/test_ctc_decode.py`](../tests/test_ctc_decode.py)

---

## 1. The Core Problem: Unaligned Sequence Learning

In automatic speech recognition (ASR), continuous speech waveforms are converted into discrete acoustic frames $X = (x_1, x_2, \dots, x_T)$ (e.g., 100 frames/sec via STFT log-Mel spectrograms). The target label sequence $Y = (y_1, y_2, \dots, y_U)$ is the text transcript (e.g., characters or subwords).

Two fundamental properties define this problem:
1. **Length Mismatch:** $T \ge U$ (acoustic frames heavily outnumber phonetic/character tokens).
2. **Missing Alignment:** Standard speech datasets provide only the audio and the transcript. The exact millisecond start and end boundaries of individual phonemes are unknown.

Prior to CTC (Graves et al., 2006), models relied on iterative forced-alignment pipelines (such as Gaussian Mixture Model - Hidden Markov Models / GMM-HMMs). CTC introduced an end-to-end differentiable loss function that sums over all possible valid temporal alignments without requiring explicit frame-level segmentation.

---

## 2. The Blank Symbol ($\epsilon$) and the Collapse Operator $\mathcal{B}$

### Why Deduplication Alone Fails
If an acoustic encoder outputs a character prediction at every frame, sustained sounds naturally span multiple frames. For example, the vowel `/a/` in *"cat"* might persist across 10 frames:
$$\pi = [c, c, a, a, a, a, a, a, t, t]$$

A naive deduplication operator that simply merges consecutive identical tokens produces:
$$\text{dedup}(\pi) = [c, a, t] \implies \text{"cat"} \quad \text{(Correct)}$$

However, when words contain **consecutive identical characters** (such as *"bee"*, *"book"*, or *"hello"*), naive deduplication collapses the true spelling:
$$\pi_{\text{broken}} = [b, b, e, e, e, e] \implies \text{dedup}(\pi_{\text{broken}}) = [b, e] \implies \mathbf{"be"} \quad \text{(Broken!)}$$

### The Collapse Operator $\mathcal{B}$
CTC solves this by expanding the alphabet $\Omega$ with a null symbol: the **blank token** $\epsilon$ (so $\Omega' = \Omega \cup \{\epsilon\}$).

The collapse function $\mathcal{B}: {\Omega'}^T \to \Omega^{\le T}$ is formally executed in two strict sequential steps:
1. **Step 1 — Merge consecutive identical symbols:**
   $$\text{merge}([b, b, \epsilon, e, e, \epsilon, e, e]) = [b, \epsilon, e, \epsilon, e]$$
2. **Step 2 — Strip all blank tokens ($\epsilon$):**
   $$\text{strip\_blanks}([b, \epsilon, e, \epsilon, e]) = [b, e, e] \implies \mathbf{"bee"}$$

The blank token acts as a temporal delimiter. It enables the model to hold non-emitting states and allows identical consecutive labels to be emitted legally.

---

## 3. The Many-to-One Alignment Mapping

A single ground truth transcript $Y$ can be produced by many distinct frame-level sequences $\pi \in {\Omega'}^T$. We denote the set of all alignments that collapse into $Y$ as $\mathcal{B}^{-1}(Y)$.

### Path Enumeration Example: Target `"CAT"` ($T = 4$)
For target $Y = [C, A, T]$ across $T = 4$ time steps, there are exactly **7 legal paths**:

| Path $\pi$ | Step 1: Merge Identicals | Step 2: Remove $\epsilon$ | Collapsed Output $\mathcal{B}(\pi)$ |
| :--- | :--- | :--- | :--- |
| `[-, C, A, T]` | `[-, C, A, T]` | `[C, A, T]` | `"CAT"` |
| `[C, -, A, T]` | `[C, -, A, T]` | `[C, A, T]` | `"CAT"` |
| `[C, C, A, T]` | `[C, A, T]` | `[C, A, T]` | `"CAT"` |
| `[C, A, -, T]` | `[C, A, -, T]` | `[C, A, T]` | `"CAT"` |
| `[C, A, A, T]` | `[C, A, T]` | `[C, A, T]` | `"CAT"` |
| `[C, A, T, -]` | `[C, A, T, -]` | `[C, A, T]` | `"CAT"` |
| `[C, A, T, T]` | `[C, A, T]` | `[C, A, T]` | `"CAT"` |

### Path Enumeration Example: Repeated Letters `"BEE"` ($T = 4$)
For $Y = [B, E, E]$ across $T = 4$ frames, because the letter `'E'` is repeated, the two `'E'`s **must** be separated by at least one blank. Any path repeating `'E'` consecutively (like `[B, B, E, E]`) collapses into `"BE"`.

Therefore, for $T = 4$, there is **only 1 legal path**:
$$\pi = [B, E, -, E] \xrightarrow{\text{merge}} [B, E, -, E] \xrightarrow{\text{strip}} [B, E, E] \implies \mathbf{"BEE"}$$

---

## 4. Total Probability & The Forward-Backward Algorithm

Under CTC, the total conditional probability of sequence $Y$ given acoustic input $X$ is the **sum** of the probabilities of all valid alignments:
$$P(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi \mid X)$$

Because enumerating all paths naively is combinatorial ($\mathcal{O}(|\Omega'|^T)$), CTC computes this efficiently via dynamic programming using a modified **Forward-Backward algorithm**:
- The target sequence $Y$ is interleaved with blanks to form a modified sequence $Y'$ of length $2U + 1$ (e.g., `_ c _ a _ t _`).
- Forward variables $\alpha_t(s)$ compute the total probability mass of all prefixes terminating at sub-sequence position $s$ at frame $t$.
- Backward variables $\beta_t(s)$ compute the probability of the remaining suffix from frame $t$ to $T$.
- Total computation scales linearly: $\mathcal{O}(T \times U)$.

The CTC loss for training is the negative log-likelihood:
$$\mathcal{L}_{\text{CTC}} = -\ln P(Y \mid X) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t \mid X, t)$$

---

## 5. The Conditional Independence Assumption

CTC models compute the alignment probability as the direct product of individual frame emissions:
$$P(\pi \mid X) = \prod_{t=1}^{T} y_{\pi_t}^t$$

### Architectural Consequence
This formulation explicitly assumes that **label emission at frame $t$ is conditionally independent of label emission at frame $t-1$, given the acoustic representation $X$**.

| Strength | Weakness |
| :--- | :--- |
| **High Parallelism:** All $T$ frames can be evaluated concurrently in a single neural network forward pass. No autoregressive feedback loop. | **Zero Language Modeling:** The decoder cannot condition its current character prediction on previously generated characters. |
| **Streaming-Friendly:** Non-autoregressive decoding has deterministic memory and computational complexity. | **Acoustic Ambiguity & Spelling Errors:** Prone to homophone substitutions (e.g. *"their"* vs *"there"*, *"fone"* vs *"phone"*) under acoustic degradation. |

### How Modern Systems Address This
1. **CTC + Beam Search with n-gram Language Model:** Combines acoustic emission scores with external language model probabilities:
   $$\text{Score}(\pi) = \ln P_{\text{CTC}}(Y \mid X) + \alpha \ln P_{\text{LM}}(Y) + \beta |Y|$$
2. **RNN-Transducer (RNN-T):** Replaces the independent CTC head with a joint network combining an Acoustic Encoder and a causal Prediction Network (which tracks past text tokens).

---

## 6. Significance for MendSpeech

CTC is uniquely suited for MendSpeech's selective repair architecture:
1. **Spiky Emissions & Natural Alignment:** CTC models concentrate their non-blank probability mass into sharp, localized temporal peaks at phoneme centers, emitting blanks elsewhere.
2. **Entropy as Damage Detector:** Under clean speech, emission entropy is low and peak token confidence is high ($>0.95$). Under physical corruption (noise, clipping, dropout), emission entropy spikes and peak probability collapses.
3. **No Hallucination:** Unlike autoregressive models (e.g. Whisper) which invent plausible-sounding words when audio is missing, CTC simply outputs blanks or low-confidence noise. This makes CTC confidence a direct, unvarnished indicator of acoustic damage.
