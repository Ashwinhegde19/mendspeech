"""Failure intelligence for the frozen Hinglish baseline.

This module turns preserved baseline artifacts into a per-clip failure
diagnosis with explicit evidence and confidence labels. It never replaces the
strict ``mendspeech.hinglish.v1`` metric: strict WER/CER numbers are inputs to
the diagnosis, and the script-equivalence view is an additional diagnostic
estimate, not a metric.

Failure classes
---------------
- ``generation_cap_with_repetition``: the clip's inference runtime sits in
  the decoder runtime ceiling and the hypothesis contains repetitive loop
  evidence. This is the strongest available structural signature of greedy
  decoding that never emitted EOS and stopped only at the decoder length cap.
- ``generation_cap_without_repetition_evidence``: the runtime ceiling was
  reached but no repetitive loop evidence was detected.
- ``repetitive_generation_early_stop``: repetitive loop evidence with a
  runtime below the ceiling, so the loop self-terminated before any cap.
- ``script_mismatch_dominant``: script-related lexical mismatches (Latin and
  Devanagari re-rendering of the same token, dropped Latin tokens) account
  for at least half of the strict word edits.
- ``recognition_difficulty``: remaining strict errors that are neither
  generation-side nor script-mismatch dominant.
- ``no_failure_detected``: no failure evidence under the current heuristics.

Classification precedence is generation-side > script mismatch > recognition.
Every classification carries a confidence label:

- ``structural_evidence``: derived from decoder runtime structure plus loop
  evidence, the strongest label this repository can currently assign.
- ``heuristic_evidence``: content heuristics (repetition, insertion mass).
- ``heuristic_estimate``: approximate script-aware alignment statistics.
- ``undetermined``: evidence is insufficient for the question.

A clip is never treated as a confirmed hallucination solely because a
heuristic triggered; the report keeps the heuristic and structural layers
separate so a human can review both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.hinglish.normalization import normalize_hinglish_text

FAILURE_INTELLIGENCE_VERSION = "mendspeech.hinglish_failure_intelligence.v1"

CLASS_GENERATION_CAP_REPETITION = "generation_cap_with_repetition"
CLASS_GENERATION_CAP_UNEXPLAINED = "generation_cap_without_repetition_evidence"
CLASS_EARLY_REPETITION = "repetitive_generation_early_stop"
CLASS_SCRIPT_MISMATCH = "script_mismatch_dominant"
CLASS_RECOGNITION = "recognition_difficulty"
CLASS_NONE = "no_failure_detected"

FAILURE_CLASSES: Tuple[str, ...] = (
    CLASS_GENERATION_CAP_REPETITION,
    CLASS_GENERATION_CAP_UNEXPLAINED,
    CLASS_EARLY_REPETITION,
    CLASS_SCRIPT_MISMATCH,
    CLASS_RECOGNITION,
    CLASS_NONE,
)

GENERATION_SIDE_CLASSES = frozenset(
    {
        CLASS_GENERATION_CAP_REPETITION,
        CLASS_GENERATION_CAP_UNEXPLAINED,
        CLASS_EARLY_REPETITION,
    }
)

CONFIDENCE_STRUCTURAL = "structural_evidence"
CONFIDENCE_HEURISTIC = "heuristic_evidence"
CONFIDENCE_ESTIMATE = "heuristic_estimate"
CONFIDENCE_UNDETERMINED = "undetermined"

# Script-mismatch classification floors. A clip is script-mismatch dominant
# only when the approximate script-related edit count reaches both a minimum
# size and a minimum share of the strict word edits.
SCRIPT_MISMATCH_SHARE_THRESHOLD = 0.5
SCRIPT_MISMATCH_MIN_EDITS = 2

# Recognition difficulty is the catch-all class for any remaining edits.
RECOGNITION_WER_THRESHOLD = 1.0

# Dominance rule thresholds for the corpus-level attribution.
INSERTION_DOMINANCE_THRESHOLD = 0.6

DOMINANCE_REPETITIVE_GENERATION = "repetitive_generation_at_decoder_cap"
DOMINANCE_SCRIPT_MISMATCH = "script_or_orthography_mismatch"
DOMINANCE_RECOGNITION = "recognition_difficulty"

_DOMINANCE_RULES: Mapping[str, str] = {
    DOMINANCE_REPETITIVE_GENERATION: (
        "generation-side clips carry at least 60% of all insertions; "
        "the strict WER is dominated by generated material the reference "
        "does not contain"
    ),
    DOMINANCE_SCRIPT_MISMATCH: (
        "script-related lexical mismatches cover at least 50% of strict "
        "word edits and insertions are not generation-dominated"
    ),
    DOMINANCE_RECOGNITION: (
        "neither generation-side insertions nor script mismatches dominate "
        "the strict edits"
    ),
}

UNDETERMINED_NOTES: Tuple[str, ...] = (
    "Generated token counts and EOS presence are not recorded in the "
    "preserved artifacts; the decoder length cap is inferred from the "
    "runtime-ceiling signature and is not confirmed by token-level evidence.",
    "Script-related mismatch counts use an independent difflib word "
    "alignment and are lower-bound estimates; they can differ from the "
    "strict scorer's alignment and never replace the strict metric.",
    "The repetition heuristics are content heuristics. A clip is not a "
    "confirmed hallucination solely because a heuristic triggered.",
    "Attribution WER values that exclude failure classes are reported for "
    "diagnosis only; the corpus WER of record remains the strict metric.",
    "Loop onset time within the audio is unknown; the artifacts contain no "
    "timing information beyond total runtime.",
)


def contains_latin(text: str) -> bool:
    """Return True when the token contains at least one ASCII letter."""
    return any("a" <= char.lower() <= "z" for char in text)


def contains_devanagari(text: str) -> bool:
    """Return True when the token contains at least one Devanagari codepoint."""
    return any("\u0900" <= char <= "\u097f" for char in text)


@dataclass(frozen=True)
class ScriptMismatchStatistics:
    """Approximate script-related lexical mismatch statistics for one clip.

    All counts are computed on normalized whitespace tokens with an
    independent difflib alignment; they are diagnostics, not the strict
    metric.

    Args:
        reference_tokens: Normalized reference token count.
        hypothesis_tokens: Normalized hypothesis token count.
        replaced_script_mismatch: Replaced pairs where exactly one side
            contains an ASCII letter.
        replaced_same_script: Replaced pairs where both sides agree on
            ASCII-letter presence.
        deleted_latin_tokens: Reference tokens containing an ASCII letter
            that are missing from the hypothesis.
        deleted_devanagari_tokens: Deleted reference tokens without ASCII
            letters.
        inserted_extra_tokens: Hypothesis tokens beyond the alignment.
    """

    reference_tokens: int
    hypothesis_tokens: int
    replaced_script_mismatch: int
    replaced_same_script: int
    deleted_latin_tokens: int
    deleted_devanagari_tokens: int
    inserted_extra_tokens: int

    @property
    def script_mismatch_edits(self) -> int:
        """Lower-bound count of script-related lexical mismatches."""
        return self.replaced_script_mismatch + self.deleted_latin_tokens


def script_mismatch_statistics(
    reference: str, hypothesis: str
) -> ScriptMismatchStatistics:
    """Compute approximate script-related mismatch statistics.

    Both texts are normalized with the project normalizer before tokenizing,
    keeping script detection aligned with the scoring pipeline.

    Args:
        reference: Reference transcript text.
        hypothesis: Hypothesis transcript text.

    Returns:
        ScriptMismatchStatistics for the pair.
    """
    reference_tokens = normalize_hinglish_text(reference).split()
    hypothesis_tokens = normalize_hinglish_text(hypothesis).split()
    matcher = SequenceMatcher(a=reference_tokens, b=hypothesis_tokens, autojunk=False)
    replaced_script_mismatch = 0
    replaced_same_script = 0
    deleted_latin = 0
    deleted_devanagari = 0
    inserted_extra = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for token in reference_tokens[i1:i2]:
                if contains_latin(token):
                    deleted_latin += 1
                else:
                    deleted_devanagari += 1
        elif tag == "insert":
            inserted_extra += j2 - j1
        else:  # replace
            reference_block = reference_tokens[i1:i2]
            hypothesis_block = hypothesis_tokens[j1:j2]
            paired = min(len(reference_block), len(hypothesis_block))
            for index in range(paired):
                reference_token = reference_block[index]
                hypothesis_token = hypothesis_block[index]
                if contains_latin(reference_token) != contains_latin(hypothesis_token):
                    replaced_script_mismatch += 1
                else:
                    replaced_same_script += 1
            for token in reference_block[paired:]:
                if contains_latin(token):
                    deleted_latin += 1
                else:
                    deleted_devanagari += 1
            inserted_extra += max(0, len(hypothesis_block) - paired)
    return ScriptMismatchStatistics(
        reference_tokens=len(reference_tokens),
        hypothesis_tokens=len(hypothesis_tokens),
        replaced_script_mismatch=replaced_script_mismatch,
        replaced_same_script=replaced_same_script,
        deleted_latin_tokens=deleted_latin,
        deleted_devanagari_tokens=deleted_devanagari,
        inserted_extra_tokens=inserted_extra,
    )


@dataclass(frozen=True)
class GenerationFailureSignals:
    """Generation-side signals extracted from the inspection report.

    Args:
        cap_hit: Whether the clip's runtime is a member of the decoder
            runtime ceiling cluster.
        has_word_loop: Word-level repetition evidence.
        has_char_loop: Character-level repetition evidence inside a word.
        has_repetition_evidence: Any repetition evidence.
        heuristic_flag: The offline hallucination heuristic flagged the clip.
        loop_coverage_words: Largest loop coverage in word units.
        output_words: Whitespace word count of the raw hypothesis.
    """

    cap_hit: bool
    has_word_loop: bool
    has_char_loop: bool
    has_repetition_evidence: bool
    heuristic_flag: bool
    loop_coverage_words: int
    output_words: int


@dataclass(frozen=True)
class ClipErrorProfile:
    """One clip's strict error counts plus diagnostic inputs.

    Args:
        clip_id: Clip identifier.
        speaker_id: Speaker identifier.
        reference_words: Strict reference word count.
        hypothesis_words: Strict hypothesis word count.
        substitutions: Strict substitutions.
        deletions: Strict deletions.
        insertions: Strict insertions.
        wer: Strict word error rate.
        english_ratio_bin: Strict evaluation bucket.
        script: Script-mismatch estimate.
        generation: Generation-side signals.
    """

    clip_id: str
    speaker_id: str
    reference_words: int
    hypothesis_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    english_ratio_bin: str
    script: ScriptMismatchStatistics
    generation: GenerationFailureSignals

    @property
    def total_edits(self) -> int:
        """Strict total word edits."""
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class FailureClassification:
    """Failure class, confidence label, and evidence sentences."""

    failure_class: str
    confidence: str
    reasons: Tuple[str, ...] = ()


def classify_failure(profile: ClipErrorProfile) -> FailureClassification:
    """Classify one clip with documented precedence and evidence.

    Precedence is generation-side structural evidence, then script-mismatch
    dominance, then recognition difficulty, then no failure.

    Args:
        profile: Strict error counts plus diagnostic inputs.

    Returns:
        FailureClassification with reasons in deterministic order.
    """
    generation = profile.generation
    reasons: List[str] = []
    if generation.cap_hit:
        loop_kind = (
            "word-level loop"
            if generation.has_word_loop
            else "character-level loop"
            if generation.has_char_loop
            else "no loop evidence"
        )
        reasons.append(
            "runtime is a member of the decoder runtime-ceiling cluster"
        )
        reasons.append(
            f"repetition evidence: {loop_kind}; loop coverage "
            f"{generation.loop_coverage_words} word unit(s)"
        )
        reasons.append(
            f"strict insertions {profile.insertions} against "
            f"{profile.reference_words} reference word(s)"
        )
        if generation.has_word_loop or generation.has_char_loop or (
            generation.has_repetition_evidence
        ):
            return FailureClassification(
                failure_class=CLASS_GENERATION_CAP_REPETITION,
                confidence=CONFIDENCE_STRUCTURAL,
                reasons=tuple(reasons),
            )
        reasons.append(
            "no repetitive loop was detected inside the capped output; "
            "content degeneracy is undetermined"
        )
        return FailureClassification(
            failure_class=CLASS_GENERATION_CAP_UNEXPLAINED,
            confidence=CONFIDENCE_HEURISTIC,
            reasons=tuple(reasons),
        )
    if generation.heuristic_flag or generation.has_word_loop or generation.has_char_loop:
        reasons.append(
            "repetition evidence below the runtime ceiling: the loop "
            "stopped before any decoder cap"
        )
        reasons.append(
            f"loop coverage {generation.loop_coverage_words} word unit(s); "
            f"strict insertions {profile.insertions}"
        )
        return FailureClassification(
            failure_class=CLASS_EARLY_REPETITION,
            confidence=CONFIDENCE_HEURISTIC,
            reasons=tuple(reasons),
        )
    edits = profile.total_edits
    if edits == 0:
        return FailureClassification(
            failure_class=CLASS_NONE,
            confidence=CONFIDENCE_HEURISTIC,
            reasons=("no strict edits and no generation-side evidence",),
        )
    script_edits = profile.script.script_mismatch_edits
    script_share = script_edits / edits
    if script_edits >= SCRIPT_MISMATCH_MIN_EDITS and (
        script_share >= SCRIPT_MISMATCH_SHARE_THRESHOLD
    ):
        reasons.append(
            f"{script_edits} of {edits} strict word edits are script-related "
            f"lexical mismatches ({script_share:.0%} share, estimate)"
        )
        reasons.append(
            f"dropped Latin tokens {profile.script.deleted_latin_tokens}; "
            f"Latin/Devanagari re-renderings "
            f"{profile.script.replaced_script_mismatch}"
        )
        return FailureClassification(
            failure_class=CLASS_SCRIPT_MISMATCH,
            confidence=CONFIDENCE_ESTIMATE,
            reasons=tuple(reasons),
        )
    reasons.append(
        f"strict substitutions {profile.substitutions}, deletions "
        f"{profile.deletions}, insertions {profile.insertions} over "
        f"{profile.reference_words} reference word(s); script-related share "
        f"{script_share:.0%}"
    )
    return FailureClassification(
        failure_class=CLASS_RECOGNITION,
        confidence=CONFIDENCE_HEURISTIC,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class FailureAttribution:
    """Corpus-level attribution of the strict error mass."""

    clip_counts: Dict[str, int]
    flagged_clip_count: int
    cap_hit_clip_count: int
    total_reference_words: int
    total_edits: int
    corpus_wer: float
    insertions_total: int
    insertions_from_generation_side: int
    insertions_generation_side_share: float
    script_mismatch_edits_total: int
    script_mismatch_share_of_edits: float
    corpus_wer_excluding_generation_side: float
    corpus_wer_excluding_cap_hits: float
    dominant_error_type: str
    dominance_rule: str


def _corpus_wer(
    profiles: Sequence[ClipErrorProfile],
    excluded_classes: Optional[Iterable[str]] = None,
    classes_by_clip: Optional[Mapping[str, str]] = None,
) -> float:
    total_edits = 0
    total_reference_words = 0
    excluded = set(excluded_classes or ())
    for profile in profiles:
        if classes_by_clip is not None and classes_by_clip.get(profile.clip_id) in excluded:
            continue
        total_edits += profile.total_edits
        total_reference_words += profile.reference_words
    if total_reference_words == 0:
        return 0.0
    return total_edits / total_reference_words


def summarize_failure_attribution(
    profiles: Sequence[ClipErrorProfile],
    classifications: Mapping[str, FailureClassification],
) -> FailureAttribution:
    """Summarize corpus-level error attribution across failure classes.

    Args:
        profiles: Per-clip strict error profiles.
        classifications: Classification per clip ID.

    Returns:
        FailureAttribution with counts, shares, attribution WER values, and
        the deterministic dominance verdict.
    """
    clip_counts: Dict[str, int] = {name: 0 for name in FAILURE_CLASSES}
    for profile in profiles:
        classification = classifications.get(profile.clip_id)
        if classification is None:
            raise KeyError(f"missing classification for {profile.clip_id}")
        clip_counts[classification.failure_class] = (
            clip_counts.get(classification.failure_class, 0) + 1
        )
    class_by_clip = {
        profile.clip_id: classifications[profile.clip_id].failure_class
        for profile in profiles
    }
    insertions_total = sum(profile.insertions for profile in profiles)
    insertions_generation_side = sum(
        profile.insertions
        for profile in profiles
        if class_by_clip[profile.clip_id] in GENERATION_SIDE_CLASSES
    )
    script_edits_total = sum(
        profile.script.script_mismatch_edits for profile in profiles
    )
    total_edits = sum(profile.total_edits for profile in profiles)
    insertions_share = (
        insertions_generation_side / insertions_total if insertions_total else 0.0
    )
    script_share = script_edits_total / total_edits if total_edits else 0.0
    if insertions_share >= INSERTION_DOMINANCE_THRESHOLD:
        dominant = DOMINANCE_REPETITIVE_GENERATION
    elif script_share >= SCRIPT_MISMATCH_SHARE_THRESHOLD:
        dominant = DOMINANCE_SCRIPT_MISMATCH
    else:
        dominant = DOMINANCE_RECOGNITION
    return FailureAttribution(
        clip_counts=clip_counts,
        flagged_clip_count=sum(
            1 for profile in profiles if profile.generation.heuristic_flag
        ),
        cap_hit_clip_count=sum(
            1 for profile in profiles if profile.generation.cap_hit
        ),
        total_reference_words=sum(
            profile.reference_words for profile in profiles
        ),
        total_edits=total_edits,
        corpus_wer=_corpus_wer(profiles),
        insertions_total=insertions_total,
        insertions_from_generation_side=insertions_generation_side,
        insertions_generation_side_share=insertions_share,
        script_mismatch_edits_total=script_edits_total,
        script_mismatch_share_of_edits=script_share,
        corpus_wer_excluding_generation_side=_corpus_wer(
            profiles, GENERATION_SIDE_CLASSES, class_by_clip
        ),
        corpus_wer_excluding_cap_hits=_corpus_wer(
            profiles,
            {CLASS_GENERATION_CAP_REPETITION, CLASS_GENERATION_CAP_UNEXPLAINED},
            class_by_clip,
        ),
        dominant_error_type=dominant,
        dominance_rule=_DOMINANCE_RULES[dominant],
    )


def build_failure_intelligence_payload(
    profiles: Sequence[ClipErrorProfile],
    classifications: Mapping[str, FailureClassification],
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    """Assemble the reusable failure intelligence report payload.

    Args:
        profiles: Per-clip strict error profiles.
        classifications: Classification per clip ID.
        provenance: Input hashes and generation configuration summary. Paths
            are allowed here because the payload is not committed verbatim;
            the report script decides what it writes.

    Returns:
        JSON-serialisable payload answering the seven failure-intelligence
        questions. Clips are sorted by clip ID.
    """
    ordered = sorted(profiles, key=lambda profile: profile.clip_id)
    attribution = summarize_failure_attribution(ordered, classifications)
    per_clip: List[Dict[str, object]] = []
    failed_clips: List[str] = []
    for profile in ordered:
        classification = classifications[profile.clip_id]
        if classification.failure_class != CLASS_NONE:
            failed_clips.append(profile.clip_id)
        per_clip.append(
            {
                "clip_id": profile.clip_id,
                "speaker_id": profile.speaker_id,
                "failure_class": classification.failure_class,
                "confidence": classification.confidence,
                "reasons": list(classification.reasons),
                "wer": round(profile.wer, 4),
                "substitutions": profile.substitutions,
                "deletions": profile.deletions,
                "insertions": profile.insertions,
                "reference_words": profile.reference_words,
                "english_ratio_bin": profile.english_ratio_bin,
                "cap_hit": profile.generation.cap_hit,
                "heuristic_flag": profile.generation.heuristic_flag,
                "has_word_loop": profile.generation.has_word_loop,
                "has_char_loop": profile.generation.has_char_loop,
                "loop_coverage_words": profile.generation.loop_coverage_words,
                "script_mismatch_edits": profile.script.script_mismatch_edits,
                "deleted_latin_tokens": profile.script.deleted_latin_tokens,
                "replaced_script_mismatch": profile.script.replaced_script_mismatch,
            }
        )
    training_candidates = [
        {
            "clip_id": entry["clip_id"],
            "failure_class": entry["failure_class"],
            "total_edits": (
                int(entry["substitutions"])
                + int(entry["deletions"])
                + int(entry["insertions"])
            ),
            "rationale": _training_rationale(str(entry["failure_class"])),
        }
        for entry in per_clip
        if entry["failure_class"] in {CLASS_RECOGNITION, CLASS_SCRIPT_MISMATCH}
    ]
    training_candidates.sort(
        key=lambda item: (-int(item["total_edits"]), str(item["clip_id"]))
    )
    return {
        "report_version": FAILURE_INTELLIGENCE_VERSION,
        "provenance": dict(provenance),
        "method": {
            "strict_metric": "mendspeech.hinglish.v1 (unchanged)",
            "scoring_path": (
                "src.hinglish.evaluation.score_prediction over normalized "
                "text; identical to the frozen baseline scorer"
            ),
            "generation_signals": (
                "scripts/inspect_hinglish_generation_failures.py runtime "
                "ceiling cluster, loop coverage, and flag sets"
            ),
            "script_estimate": (
                "difflib word alignment; Latin/Devanagari presence per "
                "token; lower-bound estimate, diagnostic only"
            ),
            "classification_precedence": (
                "generation-side structural evidence > script mismatch > "
                "recognition difficulty > no failure"
            ),
            "confidence_labels": [
                CONFIDENCE_STRUCTURAL,
                CONFIDENCE_HEURISTIC,
                CONFIDENCE_ESTIMATE,
                CONFIDENCE_UNDETERMINED,
            ],
        },
        "questions": {
            "which_utterances_failed": {
                "failed_clip_count": len(failed_clips),
                "clip_count": len(ordered),
                "failed_clip_ids": failed_clips,
            },
            "dominant_error_type": {
                "dominant_error_type": attribution.dominant_error_type,
                "rule": attribution.dominance_rule,
                "corpus_wer": round(attribution.corpus_wer, 6),
                "insertions_from_generation_side_share": round(
                    attribution.insertions_generation_side_share, 4
                ),
                "script_mismatch_share_of_edits": round(
                    attribution.script_mismatch_share_of_edits, 4
                ),
                "corpus_wer_excluding_generation_side": round(
                    attribution.corpus_wer_excluding_generation_side, 6
                ),
                "corpus_wer_excluding_cap_hits": round(
                    attribution.corpus_wer_excluding_cap_hits, 6
                ),
            },
            "classification_evidence": {
                "clip_counts": attribution.clip_counts,
                "flagged_clip_count": attribution.flagged_clip_count,
                "cap_hit_clip_count": attribution.cap_hit_clip_count,
                "total_reference_words": attribution.total_reference_words,
                "total_edits": attribution.total_edits,
                "insertions_total": attribution.insertions_total,
                "insertions_from_generation_side": (
                    attribution.insertions_from_generation_side
                ),
                "script_mismatch_edits_total": (
                    attribution.script_mismatch_edits_total
                ),
            },
            "uncertainties": list(UNDETERMINED_NOTES),
            "training_value_hypotheses": {
                "hypothesis": (
                    "Clips classified as recognition difficulty or "
                    "script-mismatch dominant are the primary candidates for "
                    "adaptation data because their failures look correctable "
                    "with in-domain supervision; generation-side clips are "
                    "decoder-configuration candidates first. Training value "
                    "is a hypothesis: it is measured by the bounded "
                    "intelligent-versus-random experiment, not by this "
                    "report."
                ),
                "selection_pipeline": (
                    "src/hinglish/selection.py implements the ranked "
                    "candidate interface; scripts/select_hinglish_training_"
                    "candidates.py produces the two arms."
                ),
                "ranked_candidates": training_candidates,
            },
            "evaluation_hooks": _evaluation_hooks(),
        },
        "per_clip": per_clip,
    }


def _training_rationale(failure_class: str) -> str:
    if failure_class == CLASS_SCRIPT_MISMATCH:
        return (
            "script-related lexical mismatch dominates; code-switched "
            "supervision may correct script and orthography behaviour"
        )
    return (
        "recognition difficulty under a working decoder; substitutions and "
        "deletions are potentially correctable with in-domain supervision"
    )


def _evaluation_hooks() -> Dict[str, object]:
    return {
        "comparison": (
            "identical-budget random versus intelligent selection, same "
            "base checkpoint and revision, same hyperparameters and step "
            "budget, matched example count and total audio duration"
        ),
        "evaluation_set": (
            "the frozen 47-speaker held-out reserve from "
            "mendspeech.mucs_holdout_pool.v0; never used for tuning"
        ),
        "primary_metrics": [
            "corpus WER (strict mendspeech.hinglish.v1)",
            "macro utterance WER",
            "insertions and generation-cap hit share",
            "code-switched versus Hindi-dominant bucket WER",
        ],
        "success_criterion": (
            "intelligent arm improves corpus WER over the random arm by at "
            "least 10% relative at identical budget, before any aggregate "
            "claim per-difficulty-slice results must be reported"
        ),
        "policy": "docs/hinglish_evaluation_policy.md",
    }


def render_markdown_report(payload: Mapping[str, object]) -> str:
    """Render the failure intelligence payload as a human-readable report.

    Args:
        payload: Output of :func:`build_failure_intelligence_payload`.

    Returns:
        Markdown document with clip-ID-level tables. Never includes
        transcript text because the payload never carries any.
    """
    questions = payload["questions"]
    assert isinstance(questions, Mapping)
    failed = questions["which_utterances_failed"]
    dominant = questions["dominant_error_type"]
    evidence = questions["classification_evidence"]
    training = questions["training_value_hypotheses"]
    hooks = questions["evaluation_hooks"]
    assert isinstance(failed, Mapping)
    assert isinstance(dominant, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(hooks, Mapping)
    lines: List[str] = []
    lines.append("# Hinglish failure intelligence report")
    lines.append("")
    lines.append(
        "Generated by `scripts/report_hinglish_failure_intelligence.py` "
        f"({payload['report_version']}). Strict metric: "
        "`mendspeech.hinglish.v1`, unchanged. Heuristic labels are evidence, "
        "not confirmed hallucination verdicts."
    )
    lines.append("")
    lines.append("## 1. Which utterances failed?")
    lines.append("")
    lines.append(
        f"{failed['failed_clip_count']} of {failed['clip_count']} clips carry "
        "failure evidence under the current heuristics."
    )
    lines.append("")
    lines.append("## 2. What type of error dominates?")
    lines.append("")
    lines.append(
        f"Dominant type: **{dominant['dominant_error_type']}** — "
        f"{dominant['rule']}."
    )
    lines.append("")
    lines.append(
        f"Corpus WER {dominant['corpus_wer']:.4f}; insertions from "
        "generation-side clips "
        f"{dominant['insertions_from_generation_side_share']:.1%}; "
        "script-mismatch share of edits "
        f"{dominant['script_mismatch_share_of_edits']:.1%}; WER excluding "
        f"generation-side clips "
        f"{dominant['corpus_wer_excluding_generation_side']:.4f}; WER "
        f"excluding cap-hit clips "
        f"{dominant['corpus_wer_excluding_cap_hits']:.4f}."
    )
    lines.append("")
    lines.append("## 3. Failure classes")
    lines.append("")
    lines.append("| Class | Clips |")
    lines.append("| :--- | ---: |")
    clip_counts = evidence["clip_counts"]
    assert isinstance(clip_counts, Mapping)
    for name in FAILURE_CLASSES:
        lines.append(f"| `{name}` | {clip_counts.get(name, 0)} |")
    lines.append("")
    lines.append(
        f"Cap-hit clips: {evidence['cap_hit_clip_count']}; heuristic-flagged "
        f"clips: {evidence['flagged_clip_count']}; insertions "
        f"{evidence['insertions_from_generation_side']} of "
        f"{evidence['insertions_total']} come from generation-side clips."
    )
    lines.append("")
    lines.append("## 4. Evidence and confidence per clip")
    lines.append("")
    lines.append(
        "| Clip | Class | Confidence | WER | S | D | I | Cap hit | Flagged |"
    )
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: |")
    per_clip = payload["per_clip"]
    assert isinstance(per_clip, Sequence)
    for entry in per_clip:
        assert isinstance(entry, Mapping)
        lines.append(
            "| {clip} | `{cls}` | {conf} | {wer:.4f} | {s} | {d} | {i} | "
            "{cap} | {flag} |".format(
                clip=entry["clip_id"],
                cls=entry["failure_class"],
                conf=entry["confidence"],
                wer=float(entry["wer"]),
                s=entry["substitutions"],
                d=entry["deletions"],
                i=entry["insertions"],
                cap="yes" if entry["cap_hit"] else "no",
                flag="yes" if entry["heuristic_flag"] else "no",
            )
        )
    lines.append("")
    lines.append("## 5. What is still uncertain?")
    lines.append("")
    uncertainties = questions["uncertainties"]
    assert isinstance(uncertainties, Sequence)
    for note in uncertainties:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 6. Which training examples might be valuable?")
    lines.append("")
    lines.append(str(training["hypothesis"]))
    lines.append("")
    lines.append("| Rank | Clip | Class | Strict edits |")
    lines.append("| ---: | :--- | :--- | ---: |")
    ranked = training["ranked_candidates"]
    assert isinstance(ranked, Sequence)
    for index, item in enumerate(ranked, start=1):
        assert isinstance(item, Mapping)
        lines.append(
            f"| {index} | {item['clip_id']} | `{item['failure_class']}` | "
            f"{item['total_edits']} |"
        )
    lines.append("")
    lines.append("## 7. How will we evaluate whether selection helped?")
    lines.append("")
    lines.append(f"- Comparison: {hooks['comparison']}")
    lines.append(f"- Evaluation set: {hooks['evaluation_set']}")
    metrics = hooks["primary_metrics"]
    assert isinstance(metrics, Sequence)
    lines.append("- Primary metrics: " + "; ".join(str(item) for item in metrics))
    lines.append(f"- Success criterion: {hooks['success_criterion']}")
    lines.append(f"- Policy: `{hooks['policy']}`")
    lines.append("")
    return "\n".join(lines)
