# Conservative Transcript Editing and Post-Training Contract

> **Plan, not implementation.** This contract fixes the missing model/reward/data
> definition in v3. No training or model compatibility is claimed by writing it.

## 1. Task and limits

One small causal instruction-following language model formats a supplied ASR
transcript. The release targets English dictation on a diagnostic corpus, not
universal language coverage. Input contains the ASR text and optional uncertainty
flags; output contains only the formatted transcript. The unmodified ASR text
always remains available. Raw audio and a correct spoken reference are not LLM
inputs. Transcript text is untrusted data, not an instruction to execute.

Allowed edits: whitespace, casing, and annotated punctuation/sentence boundaries
that preserve meaning. No paraphrase, summarization, word replacement, filler
deletion, guessed names, number conversion, or completion of missing content.
Names, numbers, negation, units and domain terms are protected. Declining an edit
returns the original text with a separate reason code, not an invented sentence.
Formatting can still change meaning; lexical equality alone is not a safety proof.

The four required baseline families are raw ASR, deterministic formatting,
prompt-only LLM, and SFT editor. The RL editor and compute-matched continued-SFT
control extend those same frozen evaluation cases. Evaluate both proposed output
and post-guard delivered output; a guard that rejects everything cannot hide a
bad model. Report fallback rate, edit coverage, helpful edits and harmful edits.

## 2. Data and independent evaluation

Day 13 specifies `data/editor_manifest.jsonl`, separate from the immutable speech
benchmark. Each row records source/group ID, consent or license, source ASR model
and decoder, original transcript, allowed formatting targets, protected spans,
annotation/reviewer provenance, split, hashes, and `needs_edit`/`identity` tags.
Formatting targets must preserve the **supplied ASR words**, including recognition
errors. Spoken gold transcripts are used only in ASR evaluation, not as targets
that reward the editor for guessing unavailable words.

Group source utterances, speakers where known, templates and near-duplicate text
before assigning train/validation/test. Never move a hard case after inspection.
Start with a small disjoint pilot subset. Before the measured training comparison,
aim for at least 200 train, 40 validation and 60 held-out test items, including
already-correct text, names/numbers/negation, short inputs, damaged-speech ASR
errors and instruction-like text. These are small-set diagnostic minima, not
population-level accuracy claims. If annotation capacity is insufficient, revise
the experimental scope explicitly rather than fabricate labels or use test data.

Validation selects prompts, formatting policy, thresholds, reward weights and
checkpoint. Training metrics are not test metrics. Seal test targets before model
selection and evaluate them in the final frozen release phase. Record possible
foundation-model pretraining overlap as unknown unless independently verified.

Independent evaluation includes lexical/protected-span violations, task formatting
accuracy on `needs_edit` and `identity` slices, fallback/edit coverage, and blinded
manual meaning-preservation review on a preselected sample with rater/item counts.
WER against spoken gold remains an ASR metric; free-form polished prose is not
scored against it as though all textual changes were recognition errors.

## 3. Model and training route

Use one 0.5–1.5B causal LM with permitted inference/adaptation weights and one
tokenizer/chat template. `Qwen/Qwen2.5-0.5B-Instruct` is a **candidate**, not a
selected or installed dependency: it appears in TRL's official GRPO example.
Day 15 verifies one candidate; Day 16 pins its exact revision, license, chat
template, EOS/pad behavior, trainable LoRA module names and parameter counts.
Stop for a scope decision if it fails; no architecture shopping or second model.

The planned backend is one version-compatible PyTorch/Transformers/PEFT/TRL
environment: completion-only supervised fine-tuning, then GRPO using the SFT
checkpoint as the starting and fixed reference policy. These optional packages
are not in the current project dependencies; pin and smoke-test them in an
isolated experiment environment at the implementation gate before adoption.
Do not pass a Wav2Vec2/CTC acoustic checkpoint to a causal-LM trainer. Do not
implement a new PPO engine or train a separate neural reward model.

Day 16 is a feasibility experiment, not completed post-training: run a tiny SFT
update and a group-relative update, verify changed adapter weights and finite
gradients/log-probabilities, restore/reload a checkpoint, and inspect rollouts.
Starting caps: group size 2, one prompt group at a time, <=256 prompt tokens,
<=128 completion tokens, LoRA rank 8, <=10 SFT steps and <=5 GRPO steps.
Compatibility/batch divisibility must be checked for the pinned trainer. Record
precision, optimizer, accumulation, clipping, tokenizer and generation seed.
Use ordinary model generation initially; vLLM is not another required trainer.

The full bounded run's step/token/wall-time/spend limits are chosen from this
pilot before Day 40. ASR is unloaded during training on the one L4. Account for
policy, reference, optimizer, activations and sampled generations—not just
checkpoint size. A forward-only fit is not evidence that training fits.

## 4. Reward and anti-shortcut tests

For supplied transcript `x`, candidate `y` and human-annotated allowed formatting
targets `A(x)`, define a versioned checker `C(x,y)` that detects empty output for
nonempty input, truncation, added commentary, lexical insertion/deletion/change,
or protected-span violation. Freeze its Unicode/token normalization rules.

Proposed initial reward (weights fixed before the measured run):

```text
if C(x, y) reports a violation: R(x, y) = -1
otherwise: R(x, y) = 0.5 * formatting_accuracy(y, A(x))
                     + 0.5 * best_normalized_character_similarity(y, A(x))
```

Both non-violation terms are bounded to [0, 1], with explicit empty/length limits
and masking defined in `configs/editor_reward.yaml`. Punctuation/casing labels
are attached to positions between preserved input tokens; no reward is awarded
for a fluent but unsupported word. Identity output remains valid and is expected
on already-correct inputs; on `needs_edit` cases it must compete with the
deterministic baseline rather than win through a blanket refusal penalty.

Unit-test deleted negation, altered numbers/names, copied prompt text, commentary,
blank outputs, overlong outputs, excessive punctuation, Unicode tricks, correct
identity, valid formatting and alternative annotations. Keep adversarial examples
out of the final test set if they are used for reward design. Reward is a proxy
for this **text-preservation task**, not a measure of acoustic support or truth.

GRPO samples current-policy completions per prompt and compares their rewards.
Cached ASR outputs can supply editor inputs; cached logits cannot replace fresh
policy rollouts or the update's likelihoods. Record group reward variance,
zero-variance groups, truncation, completion length, adapter gradient norms,
clipping and reference KL. Freeze a small nonzero KL coefficient (pilot starting
value 0.02); do not assume a trainer default enables it. All-zero advantages,
nonfinite loss or OOM mean a failed pilot, not a successful RL run.

## 5. Controls, selection and stopping

Compare prompt-only, SFT, SFT+GRPO and SFT+continued-SFT with the same input
groups, validation protocol and base checkpoint. For the continued-SFT control,
match the extra GRPO L4 wall-time budget and report actual processed tokens,
steps and cost; this controls extra compute, not identical optimization behavior.
Use two explicit seeds for the small final comparison if the authorized budget
permits. Otherwise label single-seed evidence exploratory and report that limit.

Freeze validation-based stop criteria: no nonfinite updates, no repeated OOM,
no spend/wall-time overrun, and no >2 percentage-point protected-content violation
increase relative to the SFT validation baseline at two consecutive evaluations.
Record evaluation counts and uncertainty; on small sets these limits are stop
rules, not statistical guarantees. Stop and diagnose reward gain accompanied by
worse independent quality or loss of rollout diversity.

An executed null/worse result is valid evidence. A blocked trainer is not a
negative RL result and cannot complete the RL milestone. Keep the functioning
baseline available and report partial delivery until scope is explicitly revised.
This is LLM editor post-training, not speech-model RL or user personalization.

## 6. Primary technical references

Documentation checked September 26, 2026. Pin versions at implementation; current
web examples do not establish this repository's compatibility, quality or cost.

```text
https://huggingface.co/docs/trl/sft_trainer
https://huggingface.co/docs/trl/grpo_trainer
https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
```