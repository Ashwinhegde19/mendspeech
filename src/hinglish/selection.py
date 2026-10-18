"""Deterministic candidate selection for bounded Hinglish adaptation experiments.

This module ranks and selects speech examples for an identical-budget
adaptation experiment comparing two arms:

- ``random``: a seeded control sample of eligible candidates, one clip per
  speaker preferred.
- ``intelligent``: a documented scoring formula over ASR failure evidence
  (when predictions exist), code-mix composition, duration quality, and
  transcript richness, combined with speaker and shard diversity constraints.

Both arms draw from the *training* speaker pool only. Validation speakers,
official-test speakers, and (when configured) diagnostic speakers are
rejected before ranking. Reference transcripts never enter this module: a
candidate only records whether a reference exists (``reference_available``),
never the text. Prediction text is also never stored; scoring consumes
already-computed error counts.

Bias warning (also emitted in every selection report)
-----------------------------------------------------
Intelligent selection concentrates the annotation budget on intrinsically
harder examples: high deletion/substitution difficulty, insertion-heavy or
repetitive-hallucination behaviour, and code-switched composition. Adaptation
improvements measured on a selection from this arm are therefore not
comparable to unselected data or to the random arm unless per-difficulty-slice
results are reported and the annotation budget is identical. The mitigation
contract enforced here is: matched total audio duration (±2% default
tolerance), and the documented requirement that the annotator instructions,
annotator pool, base checkpoint and revision, fine-tuning hyperparameters, and
step budget are identical across arms.

Label-free acquisition (formula ``hinglish.labelfree.v1``, frozen
2026-10-18 before any pool evaluation and before any J2 inference)
ranks without reference text from model-derived numeric ASR output
statistics carried by :class:`AsrOutputFeatures` (repetition, token
budget/stop behaviour, chosen-token logprob risk) plus manifest duration
and optional manifest audio-quality inputs. The prohibition is stronger
than in the oracle mode: ground-truth failure features raise outright, a
failed model scoring record is rejected as ineligible (never scored, never
zeroed), and a candidate without a usable ``AsrOutputFeatures`` record
cannot be scored at all. Weights are renormalized over the components that
are available for a candidate; a missing feature is never treated as zero.

Determinism is a product feature. Both arms are pure functions of
(candidates, config): the random arm uses ``random.Random(seed)`` over a
deterministically sorted list, and the intelligent arm sorts by
``(-score, clip_id)`` with explicit tie-breaks. No unseeded randomness and no
network access exist in this module.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

SELECTION_VERSION = "mendspeech.hinglish_selection.v1"

ARM_RANDOM = "random"
ARM_INTELLIGENT = "intelligent"

ASR_AVAILABLE = "available"
ASR_UNAVAILABLE = "unavailable"

DEFAULT_TARGET_COUNT = 100
DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_MAX_DURATION_SEC = 20.0
DEFAULT_MAX_PER_SPEAKER = 1
DEFAULT_DURATION_TOLERANCE = 0.02
DEFAULT_SEED = 42
DEFAULT_MIN_TOTAL_WORDS = 2
DEFAULT_PREFERRED_DURATION_SEC = 8.0

# Composition band (English word ratio) that the intelligent arm prefers.
# Clips inside the band score 1.0; the score decays linearly to 0.0 at a
# distance of 0.4 outside the band.
CODE_MIX_BAND_LOW = 0.4
CODE_MIX_BAND_HIGH = 0.6
CODE_MIX_FALLOFF = 0.4

# Failure-evidence saturation constants. The deletion/substitution rate is
# capped at 2.0 edits per reference word and the insertion rate at 4.0
# insertions per reference word before being mapped into [0, 1].
FAILURE_EDIT_SATURATION = 2.0
FAILURE_INSERTION_SATURATION = 4.0
FAILURE_HALLUCINATION_FLOOR = 0.75

# ---------------------------------------------------------------------------
# Label-free scoring formula ``hinglish.labelfree.v1``.
#
# FROZEN 2026-10-18 before pool evaluation and before J2 inference. No
# weights, saturation constants, band edges, or precedence rules may be
# tuned after seeing J2 results. Every constant below is mirrored in
# results/hinglish/label_free_scoring_formula.v1.json.
# ---------------------------------------------------------------------------
LABEL_FREE_FORMULA_VERSION = "hinglish.labelfree.v1"
LABEL_FREE_FROZEN_NOTE = (
    "Frozen 2026-10-18 before pool evaluation and before J2 inference; no "
    "weights may be tuned after seeing J2 results."
)

# Component names in canonical order and their frozen weights. The weights
# sum to exactly 1.0; for a candidate with unavailable components the
# weights of the available components are renormalized (missing features
# are never zero-filled).
LABEL_FREE_COMPONENTS: Tuple[str, ...] = (
    "duration_fit",
    "repetition_risk",
    "length_anomaly",
    "confidence_risk",
    "audio_quality_risk",
)
LABEL_FREE_SCORING_WEIGHTS: Mapping[str, float] = {
    "duration_fit": 0.30,
    "repetition_risk": 0.30,
    "length_anomaly": 0.20,
    "confidence_risk": 0.12,
    "audio_quality_risk": 0.08,
}

# Repetition component saturations. ``repeated_token_fraction`` saturates at
# 0.4 and ``max_consecutive_repeated_tokens`` at 24 before the 0.6/0.4
# combination. ``repetition_flag`` floors the component at 0.75.
LF_REPETITION_FRACTION_SATURATION = 0.4
LF_REPETITION_CONSECUTIVE_SATURATION = 24.0
LF_REPETITION_FLAG_FLOOR = 0.75
# J1 evidence: a decode that hits its token budget without emitting EOS is
# looping until truncation, so the repetition-risk component is floored at
# 0.85. A decode that emitted EOS without hitting the budget stopped
# naturally, so the component is clamped to at most 0.55.
LF_CAP_HIT_NO_EOS_FLOOR = 0.85
LF_NATURAL_EOS_REPETITION_CEILING = 0.55

# Length-anomaly component. The expected transcription-rate band
# [10.32258064516129, 13.693693693693694] tokens per audio second is the
# min/max over the three pre-registered J1 control clips at config C0
# (results/hinglish/decoder_probe_metrics.json, per_clip, category
# ``control``): 111627_bYPRSx2sjXooUdv8_0000 (86 tokens / 7.275 s),
# 136706_PMLT41UVHAjHIiJW_0000 (76 tokens / 5.55 s),
# 415128_riwQQ9Di2QY1QXwx_0000 (16 tokens / 1.55 s). Inside the band the
# anomaly is exactly 0; outside it rises linearly from 0 at the band edge
# to 1.0 at half of the low edge (LF_TOKEN_RATE_SATURATION_LOW = 0.5 x
# LF_TOKEN_RATE_BAND_LOW = 5.161290322580645) or 50% beyond the high edge
# (LF_TOKEN_RATE_SATURATION_HIGH = 1.5 x LF_TOKEN_RATE_BAND_HIGH =
# 20.54054054054054).
LF_TOKEN_RATE_BAND_LOW = 10.32258064516129
LF_TOKEN_RATE_BAND_HIGH = 13.693693693693694
LF_TOKEN_RATE_BAND_MEAN = 11.945860060259795
LF_LENGTH_ANOMALY_EDGE_FRACTION = 0.5
LF_TOKEN_RATE_SATURATION_LOW = 5.161290322580645
LF_TOKEN_RATE_SATURATION_HIGH = 20.54054054054054
# A decode that hit the token budget without emitting EOS is treated as at
# least this anomalous regardless of its token rate.
LF_LENGTH_ANOMALY_CAP_HIT_FLOOR = 0.80

# Confidence component. Chosen-token logprob risk: ``min_logprob`` at or
# above 0.10 is healthy (risk 0) and saturates to risk 1.0 at -1.00;
# ``mean_logprob`` at or above 0.05 is healthy and saturates to risk 1.0 at
# -0.40. The two risks are combined 0.5/0.5. Unavailable logprobs leave the
# whole component None (never zero).
LF_CONFIDENCE_MIN_LOGPROB_HEALTHY = 0.10
LF_CONFIDENCE_MIN_LOGPROB_SPAN = 1.10
LF_CONFIDENCE_MEAN_LOGPROB_HEALTHY = 0.05
LF_CONFIDENCE_MEAN_LOGPROB_SPAN = 0.45
LF_CONFIDENCE_SUBWEIGHT_MIN = 0.5
LF_CONFIDENCE_SUBWEIGHT_MEAN = 0.5

# Audio-quality component inputs come from the pool manifest (dBFS and
# fraction units). Clipping fraction saturates at 0.05; a peak at or above
# -0.5 dBFS floors the base risk at 0.70; |mean RMS| of at most 38 dBFS is
# quiet (risk 0) and 58 dBFS or louder saturates to risk 1.0. Base and RMS
# risks combine 0.5/0.5; when only some inputs are available the available
# risks are renormalized, and with no inputs the component is None.
LF_QUALITY_CLIPPING_SATURATION_FRACTION = 0.05
LF_QUALITY_PEAK_FLOOR_DBFS = -0.5
LF_QUALITY_PEAK_FLOOR_RISK = 0.70
LF_QUALITY_RMS_QUIET_DBFS = 38.0
LF_QUALITY_RMS_SATURATION_DBFS = 20.0
LF_QUALITY_SUBWEIGHT_BASE = 0.5
LF_QUALITY_SUBWEIGHT_RMS = 0.5

# Component weights when ASR failure evidence is available. Weights are
# renormalized over the components that are available for a candidate, so an
# unavailable component is never silently treated as zero signal. When
# failure evidence is unavailable the remaining weights (0.3 / 0.1 / 0.1)
# renormalize to 0.6 / 0.2 / 0.2.
SCORING_WEIGHTS: Mapping[str, float] = {
    "failure": 0.5,
    "code_mix": 0.3,
    "duration": 0.1,
    "quality": 0.1,
}

REASON_EMPTY_CLIP_ID = "empty_clip_id"
REASON_EMPTY_SPEAKER_ID = "empty_speaker_id"
REASON_SPEAKER_OUTSIDE_TRAINING_POOL = "speaker_outside_training_pool"
REASON_SPEAKER_FORBIDDEN = "speaker_forbidden"
REASON_CLIP_FORBIDDEN = "clip_forbidden"
REASON_REFERENCE_UNAVAILABLE = "reference_unavailable"
REASON_METADATA_INVALID = "metadata_invalid"
REASON_RATIO_OUT_OF_RANGE = "ratio_out_of_range"
REASON_DURATION_MISSING = "duration_missing"
REASON_DURATION_OUT_OF_BOUNDS = "duration_out_of_bounds"
REASON_INSUFFICIENT_WORDS = "insufficient_word_count"
REASON_EMPTY_REFERENCE = "empty_reference_after_normalization"
# Label-free mode only: the model-derived scoring record exists but reported
# a failed decode, so the candidate cannot be scored (never scored, never
# zeroed). A candidate whose record is missing entirely keeps the reason
# code unset here and instead fails loudly in score_candidate, so that an
# integration bug after the scoring run surfaces as an explicit error.
REASON_SCORING_FAILED = "scoring_failed"

ASR_OUTPUT_STATUS_OK = "ok"
ASR_OUTPUT_STATUS_FAILED = "failed"

SELECTION_BIAS_WARNING = (
    "Intelligent selection concentrates the annotation budget on intrinsically "
    "harder examples (deletion/substitution difficulty, insertion-heavy or "
    "repetitive-hallucination behaviour, code-switched composition). Results "
    "from this arm must not be compared against unselected data or the random "
    "arm as a single aggregate: report per-difficulty-slice results and keep "
    "the annotation budget identical."
)

SELECTION_MITIGATIONS: Tuple[str, ...] = (
    "Match total audio duration across arms within the configured tolerance "
    "(default 2%) before annotation.",
    "Use identical annotation instructions and annotator pool for both arms.",
    "Start from the identical base checkpoint and revision.",
    "Use identical fine-tuning hyperparameters and step budget.",
    "Report per-difficulty-slice and per-composition-slice results before any "
    "aggregate claim.",
)

# Acquisition modes. ``oracle`` ranks with reference-derived evidence (ASR
# failure features against ground truth, transcript-derived language ratios
# and word counts) and is therefore supervised. ``label_free`` ranks from
# acquisition-time signals only and rejects the ground-truth-derived inputs
# outright instead of zeroing them.
ACQUISITION_ORACLE = "oracle"
ACQUISITION_LABEL_FREE = "label_free"
ACQUISITION_MODES: Tuple[str, ...] = (ACQUISITION_ORACLE, ACQUISITION_LABEL_FREE)

LABEL_FREE_ACQUISITION_NOTE = (
    "label_free acquisition ranks from acquisition-time signals only under "
    f"formula {LABEL_FREE_FORMULA_VERSION}: model-derived numeric ASR output "
    "statistics and manifest duration and audio-quality metadata; it "
    "prohibits ground-truth WER/S/D/I features, reference word counts, "
    "transcript-derived language ratios, and any reference text during "
    "ranking; a failed model scoring record is rejected as ineligible "
    f"({REASON_SCORING_FAILED}) and never scored; unavailable components are "
    "never zeroed, weights renormalize over the remaining components, and "
    "ranking optimizes a proxy, not annotated outcomes"
)

# The only columns a selection manifest may contain. Reference transcripts,
# predictions, and any free text derived from them are rejected by the writer.
SAFE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "clip_id",
    "speaker_id",
    "arm",
    "score",
    "rank",
    "ratio_english_words",
    "ratio_hindi_words",
    "count_english_words",
    "count_hindi_words",
    "segment_part",
    "shard",
    "duration_sec",
    "asr_status",
    "asr_wer",
    "difficulty_bucket",
    "code_mix_bucket",
)

_UNSAFE_COLUMN_TOKENS = ("transcript", "reference", "prediction", "hypothesis")


@dataclass(frozen=True)
class SelectionConfig:
    """Operational knobs for one selection run.

    Args:
        target_count: Number of clips each arm must return.
        min_duration_sec: Lower duration bound for eligibility, seconds.
        max_duration_sec: Upper duration bound for eligibility, seconds.
        max_per_speaker: Cap on clips from one speaker per arm.
        duration_tolerance: Fractional tolerance for duration-budget matching,
            for example 0.02 for ±2%.
        seed: Seed for the random control arm.
        preferred_duration_sec: Duration rewarded highest by the intelligent
            arm's duration component, seconds.
        min_total_words: Minimum English+Hindi word count for eligibility.
        require_duration: When true, candidates without a duration are
            rejected instead of passing with the duration component marked
            unavailable.
        acquisition_mode: ``oracle`` (default) ranks with reference-derived
            evidence; ``label_free`` ranks from acquisition-time signals
            only and rejects ground-truth-derived inputs. In ``label_free``
            mode ``min_total_words`` must be 0 and ``require_duration``
            must be true, because the word-count gate is transcript-derived
            and durations are always required from the manifest. Scoring in
            this mode additionally requires a usable model-derived
            ``AsrOutputFeatures`` record (status ``ok``) on every scored
            candidate.
    """

    target_count: int = DEFAULT_TARGET_COUNT
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC
    max_per_speaker: int = DEFAULT_MAX_PER_SPEAKER
    duration_tolerance: float = DEFAULT_DURATION_TOLERANCE
    seed: int = DEFAULT_SEED
    preferred_duration_sec: float = DEFAULT_PREFERRED_DURATION_SEC
    min_total_words: int = DEFAULT_MIN_TOTAL_WORDS
    require_duration: bool = False
    acquisition_mode: str = ACQUISITION_ORACLE

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be positive")
        if self.max_per_speaker < 1:
            raise ValueError("max_per_speaker must be positive")
        if self.min_duration_sec <= 0:
            raise ValueError("min_duration_sec must be positive")
        if self.max_duration_sec <= self.min_duration_sec:
            raise ValueError("max_duration_sec must exceed min_duration_sec")
        if not 0.0 <= self.duration_tolerance < 1.0:
            raise ValueError("duration_tolerance must be in [0, 1)")
        if not self.min_duration_sec <= self.preferred_duration_sec <= self.max_duration_sec:
            raise ValueError(
                "preferred_duration_sec must lie within the duration bounds"
            )
        if self.min_total_words < 0:
            raise ValueError("min_total_words must be non-negative")
        if self.acquisition_mode not in ACQUISITION_MODES:
            raise ValueError(
                "acquisition_mode must be one of " + ", ".join(ACQUISITION_MODES)
            )
        if self.acquisition_mode == ACQUISITION_LABEL_FREE:
            if self.min_total_words != 0:
                raise ValueError(
                    "label_free acquisition prohibits the transcript-derived "
                    "word-count eligibility gate: min_total_words must be 0"
                )
            if not self.require_duration:
                raise ValueError(
                    "label_free acquisition requires known durations: set "
                    "require_duration=True so ranking uses the only "
                    "label-free signal"
                )


@dataclass(frozen=True)
class AsrFailureFeatures:
    """ASR-derived failure evidence for one clip.

    The fields are computed outside this module (for example from a predicted
    and a reference transcript). The transcript texts themselves never reach
    this dataclass.
    """

    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    hallucination_flag: bool = False


@dataclass(frozen=True)
class AsrOutputFeatures:
    """Model-derived numeric output statistics for one pool clip.

    Every field is a number, a boolean, or the decode status; no property
    that requires parsing generated text (word counts, ratios, text
    lengths) is carried here. The decoded text itself always stays private
    in the inference-run witness. Repetition statistics are decoded-text-
    derived TOKEN statistics (token IDs and repetition fractions), which is
    why they are allowed.

    Units: durations in seconds, logprobs and entropy as produced by the
    model over the sequence, counts in tokens.

    Args:
        status: ``"ok"`` for a usable decode record, ``"failed"`` otherwise.
            A failed record is never scored; in ``label_free`` mode the
            candidate is rejected with reason code ``scoring_failed``.
        audio_duration_sec: Audio duration in seconds as measured by the
            scoring run. Informational in the frozen formula (the manifest
            ``CandidateRecord.duration_sec`` remains the duration source of
            record); present so runs can be cross-checked.
        inference_runtime_sec: Decode wall-clock time, seconds.
        generated_token_count: Number of content tokens generated by the
            decode (trailing EOS stripped, per the pinned semantics).
        eos_emitted_effective: Whether the decode ended with an effective
            EOS (any EOS channel of the pinned semantics).
        hit_length_limit_effective: Whether the decode hit its effective
            token budget (444 content tokens at the pinned default; 160
            for a ``max_new_tokens=160`` config).
        max_consecutive_repeated_tokens: Longest run of one repeated token
            ID in the output sequence; ``None`` when unknown.
        repeated_token_fraction: Fraction of output tokens that repeat any
            earlier token ID in the same sequence; ``None`` when unknown.
        repetition_flag: Heuristic looping flag over the decoded token
            sequence; ``None`` when not measured.
        mean_logprob: Mean chosen-token logprob; ``None`` when not
            measured.
        min_logprob: Minimum chosen-token logprob; ``None`` when not
            measured.
        entropy_mean: Mean chosen-token entropy (nats); carried for the
            inference witness. Recorded but deliberately unused (weight 0)
            in the frozen ``hinglish.labelfree.v1`` formula.
    """

    status: str
    audio_duration_sec: Optional[float] = None
    inference_runtime_sec: Optional[float] = None
    generated_token_count: Optional[int] = None
    eos_emitted_effective: Optional[bool] = None
    hit_length_limit_effective: Optional[bool] = None
    max_consecutive_repeated_tokens: Optional[int] = None
    repeated_token_fraction: Optional[float] = None
    repetition_flag: Optional[bool] = None
    mean_logprob: Optional[float] = None
    min_logprob: Optional[float] = None
    entropy_mean: Optional[float] = None


@dataclass(frozen=True)
class CandidateRecord:
    """One selection candidate with no transcript text.

    Args:
        clip_id: Unique clip identifier.
        speaker_id: Speaker identifier.
        reference_available: Whether a nonempty reference exists. The text is
            deliberately not carried here.
        ratio_english_words: English word ratio in [0, 1]. Only used by the
            oracle acquisition mode (and by manual audits); ``None`` is
            tolerated in label-free flows and written as an empty manifest
            value.
        ratio_hindi_words: Hindi word ratio in [0, 1].
        count_english_words: English word count from metadata.
        count_hindi_words: Hindi word count from metadata.
        segment_part: Source segment part as recorded in the inventory.
        shard: Inventory shard identifier used for diversity spreading.
        duration_sec: Audio duration in seconds when known; the required
            duration source in ``label_free`` mode.
        metadata_valid: Whether upstream metadata validation passed.
        invalid_fields: Names of invalid metadata fields, for auditing.
        asr: Optional ground-truth-derived ASR failure evidence. Prohibited
            (raises) in ``label_free`` scoring.
        asr_output: Optional model-derived scoring record. Required with
            status ``ok`` for ``label_free`` scoring; never consulted in
            oracle mode.
        audio_peak_dbfs: Optional pool-manifest audio quality input: peak
            level in dBFS (negative, 0.0 = full scale).
        audio_clipping_fraction: Optional pool-manifest audio quality
            input: fraction of samples at or near full scale, in [0, 1].
        audio_mean_rms_dbfs: Optional pool-manifest audio quality input:
            mean RMS level in dBFS (largest magnitude = louder, expressed
            as a negative dBFS value).
    """

    clip_id: str
    speaker_id: str
    reference_available: bool
    ratio_english_words: float
    ratio_hindi_words: float = 0.0
    count_english_words: int = 0
    count_hindi_words: int = 0
    segment_part: str = ""
    shard: str = ""
    duration_sec: Optional[float] = None
    metadata_valid: bool = True
    invalid_fields: Tuple[str, ...] = ()
    asr: Optional[AsrFailureFeatures] = None
    asr_output: Optional[AsrOutputFeatures] = None
    audio_peak_dbfs: Optional[float] = None
    audio_clipping_fraction: Optional[float] = None
    audio_mean_rms_dbfs: Optional[float] = None


@dataclass(frozen=True)
class EligibilityDecision:
    """Eligibility outcome with explicit rejection reason codes."""

    eligible: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DedupeResult:
    """Deduplicated candidates plus the duplicate clip IDs that were removed."""

    kept: Tuple[CandidateRecord, ...]
    duplicate_clip_ids: Tuple[str, ...]


@dataclass(frozen=True)
class CandidatePool:
    """Eligible candidates and the audit trail of what was rejected."""

    eligible: Tuple[CandidateRecord, ...]
    rejected: Tuple[Tuple[str, Tuple[str, ...]], ...]
    duplicate_clip_ids: Tuple[str, ...]
    asr_available_count: int
    asr_unavailable_count: int
    duration_unknown_count: int


@dataclass(frozen=True)
class CandidateScore:
    """Score and per-component evidence for one eligible candidate.

    ``components`` maps component name to its value, or to ``None`` when the
    component is unavailable for this candidate (for example ASR failure
    evidence without predictions, or duration without audio metadata).
    """

    clip_id: str
    score: float
    components: Mapping[str, Optional[float]]
    asr_status: str
    asr_wer: Optional[float]
    difficulty_bucket: str
    code_mix_bucket: str


@dataclass(frozen=True)
class RankedCandidate:
    """One selected candidate with its position in an arm's ranked list."""

    rank: int
    candidate: CandidateRecord
    score: Optional[float]
    components: Optional[Mapping[str, Optional[float]]]
    asr_status: str
    asr_wer: Optional[float]
    difficulty_bucket: str
    code_mix_bucket: str


@dataclass(frozen=True)
class BudgetMatchResult:
    """Outcome of matching one arm's total duration to a target."""

    applied: bool
    target_total_duration_sec: Optional[float]
    final_total_duration_sec: Optional[float]
    residual_sec: Optional[float]
    residual_fraction: Optional[float]
    tolerance: float
    within_tolerance: bool
    swaps_applied: int
    duration_unknown_count: int
    note: str


@dataclass(frozen=True)
class ArmSelection:
    """One arm's ranked selection with duration accounting.

    Args:
        arm: ``random`` or ``intelligent``.
        ranked: Selected candidates in rank order.
        total_duration_sec: Sum over clips with known duration, seconds. This
            is an underestimate when ``duration_unknown_count`` is nonzero.
        duration_known_count: Clips with a known duration.
        duration_unknown_count: Clips without a known duration.
        sample_order: For the random arm, the clip IDs of all eligible
            candidates in the seeded sampled order. Duration matching reuses
            this order for replacements. Empty for other arms.
        budget_match: Duration-budget matching outcome, when attempted.
    """

    arm: str
    ranked: Tuple[RankedCandidate, ...]
    total_duration_sec: Optional[float]
    duration_known_count: int
    duration_unknown_count: int
    sample_order: Tuple[str, ...] = ()
    budget_match: Optional[BudgetMatchResult] = None

    @property
    def selected(self) -> Tuple[CandidateRecord, ...]:
        """Selected candidate records in rank order."""
        return tuple(entry.candidate for entry in self.ranked)


@dataclass(frozen=True)
class PairedSelectionResult:
    """Both arms plus the shared candidate pool they were drawn from."""

    pool: CandidatePool
    intelligent: ArmSelection
    random: ArmSelection


# ---------------------------------------------------------------------------
# Eligibility and pool preparation
# ---------------------------------------------------------------------------


def evaluate_eligibility(
    candidate: CandidateRecord,
    *,
    training_speakers: Iterable[str],
    forbidden_speakers: Iterable[str] = (),
    forbidden_clip_ids: Iterable[str] = (),
    config: SelectionConfig = SelectionConfig(),
) -> EligibilityDecision:
    """Evaluate one candidate against the selection eligibility rules.

    Rules, in order of evaluation: identifiers must be nonempty; the speaker
    must belong to the training pool and must not be forbidden (validation,
    official-test, or configured diagnostic speakers); the clip must not be
    forbidden (for example an already-evaluated development clip); a reference
    must exist; metadata must be valid; the English ratio must be in [0, 1];
    duration must satisfy the configured bounds when known (and must be known
    when ``require_duration`` is set); the word-count floor must be met; ASR
    features must not describe an empty normalized reference; and, in
    ``label_free`` mode, a model-derived scoring record must not report a
    failed decode (reason code ``scoring_failed``; the candidate is never
    scored and never zeroed instead).

    Args:
        candidate: Candidate to evaluate.
        training_speakers: Speaker IDs allowed to contribute candidates.
        forbidden_speakers: Speaker IDs rejected even if they appear in the
            training pool.
        forbidden_clip_ids: Clip IDs rejected outright.
        config: Selection configuration.

    Returns:
        EligibilityDecision with all applicable reason codes.
    """
    training = set(training_speakers)
    forbidden_speaker_set = set(forbidden_speakers)
    forbidden_clip_set = set(forbidden_clip_ids)

    reasons: list[str] = []
    if not candidate.clip_id.strip():
        reasons.append(REASON_EMPTY_CLIP_ID)
    if not candidate.speaker_id.strip():
        reasons.append(REASON_EMPTY_SPEAKER_ID)
    if candidate.speaker_id not in training:
        reasons.append(REASON_SPEAKER_OUTSIDE_TRAINING_POOL)
    if candidate.speaker_id in forbidden_speaker_set:
        reasons.append(REASON_SPEAKER_FORBIDDEN)
    if candidate.clip_id in forbidden_clip_set:
        reasons.append(REASON_CLIP_FORBIDDEN)
    if not candidate.reference_available:
        reasons.append(REASON_REFERENCE_UNAVAILABLE)
    if not candidate.metadata_valid:
        reasons.append(REASON_METADATA_INVALID)
    if candidate.ratio_english_words is not None and not (
        0.0 <= candidate.ratio_english_words <= 1.0
    ):
        reasons.append(REASON_RATIO_OUT_OF_RANGE)
    if candidate.duration_sec is None:
        if config.require_duration:
            reasons.append(REASON_DURATION_MISSING)
    elif not config.min_duration_sec <= candidate.duration_sec <= config.max_duration_sec:
        reasons.append(REASON_DURATION_OUT_OF_BOUNDS)
    total_words = candidate.count_english_words + candidate.count_hindi_words
    if total_words < config.min_total_words:
        reasons.append(REASON_INSUFFICIENT_WORDS)
    if candidate.asr is not None and candidate.asr.reference_words <= 0:
        reasons.append(REASON_EMPTY_REFERENCE)
    if (
        config.acquisition_mode == ACQUISITION_LABEL_FREE
        and candidate.asr_output is not None
        and candidate.asr_output.status != ASR_OUTPUT_STATUS_OK
    ):
        reasons.append(REASON_SCORING_FAILED)

    return EligibilityDecision(eligible=not reasons, reasons=tuple(reasons))


def dedupe_candidates(
    candidates: Iterable[CandidateRecord],
) -> DedupeResult:
    """Keep the first occurrence of each clip ID in input order."""
    kept: list[CandidateRecord] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for candidate in candidates:
        if candidate.clip_id in seen:
            duplicates.append(candidate.clip_id)
            continue
        seen.add(candidate.clip_id)
        kept.append(candidate)
    return DedupeResult(kept=tuple(kept), duplicate_clip_ids=tuple(duplicates))


def prepare_candidate_pool(
    candidates: Iterable[CandidateRecord],
    *,
    training_speakers: Iterable[str],
    forbidden_speakers: Iterable[str] = (),
    forbidden_clip_ids: Iterable[str] = (),
    config: SelectionConfig = SelectionConfig(),
) -> CandidatePool:
    """Deduplicate and filter candidates into an eligible selection pool.

    Raises:
        ValueError: If the training speaker pool is empty. Refusing an empty
            pool prevents a misconfigured run from silently selecting from
            non-training speakers.
    """
    training = set(training_speakers)
    if not training:
        raise ValueError("training_speakers must not be empty")

    deduped = dedupe_candidates(candidates)
    eligible: list[CandidateRecord] = []
    rejected: list[Tuple[str, Tuple[str, ...]]] = []
    for candidate in deduped.kept:
        decision = evaluate_eligibility(
            candidate,
            training_speakers=training,
            forbidden_speakers=forbidden_speakers,
            forbidden_clip_ids=forbidden_clip_ids,
            config=config,
        )
        if decision.eligible:
            eligible.append(candidate)
        else:
            rejected.append((candidate.clip_id, decision.reasons))

    asr_available = sum(1 for candidate in eligible if candidate.asr is not None)
    return CandidatePool(
        eligible=tuple(eligible),
        rejected=tuple(rejected),
        duplicate_clip_ids=deduped.duplicate_clip_ids,
        asr_available_count=asr_available,
        asr_unavailable_count=len(eligible) - asr_available,
        duration_unknown_count=sum(
            1 for candidate in eligible if candidate.duration_sec is None
        ),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def failure_component(asr: Optional[AsrFailureFeatures]) -> Optional[float]:
    """Return the ASR failure-evidence component, or None when unavailable.

    The component combines deletion/substitution difficulty (weight 0.7,
    saturated at 2.0 edits per reference word) with insertion-heaviness
    (weight 0.3, saturated at 4.0 insertions per reference word). A flagged
    likely repetitive hallucination floors the component at 0.75 so that the
    hallucination-prone failure mode cannot be ranked low by a moderate edit
    rate.
    """
    if asr is None or asr.reference_words <= 0:
        return None
    edits = (asr.substitutions + asr.deletions) / asr.reference_words
    insertions = asr.insertions / asr.reference_words
    score = 0.7 * min(edits / FAILURE_EDIT_SATURATION, 1.0)
    score += 0.3 * min(insertions / FAILURE_INSERTION_SATURATION, 1.0)
    if asr.hallucination_flag:
        score = max(score, FAILURE_HALLUCINATION_FLOOR)
    return score


def code_mix_component(ratio_english_words: float) -> float:
    """Return the code-mix preference in [0, 1].

    Ratios inside the 0.4-0.6 English band score 1.0. Outside the band the
    score decays linearly to 0.0 at a distance of 0.4.
    """
    ratio = min(max(ratio_english_words, 0.0), 1.0)
    if CODE_MIX_BAND_LOW <= ratio <= CODE_MIX_BAND_HIGH:
        return 1.0
    distance = (
        CODE_MIX_BAND_LOW - ratio if ratio < CODE_MIX_BAND_LOW else ratio - CODE_MIX_BAND_HIGH
    )
    return max(0.0, 1.0 - distance / CODE_MIX_FALLOFF)


def duration_component(
    duration_sec: Optional[float], config: SelectionConfig
) -> Optional[float]:
    """Return the duration-quality component, or None when duration is unknown.

    Scores 1.0 at ``config.preferred_duration_sec`` and decays linearly to 0.0
    at the furthest configured bound.
    """
    if duration_sec is None:
        return None
    span = max(
        config.preferred_duration_sec - config.min_duration_sec,
        config.max_duration_sec - config.preferred_duration_sec,
    )
    if span <= 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(duration_sec - config.preferred_duration_sec) / span)


def quality_component(count_english_words: int, count_hindi_words: int) -> float:
    """Return the transcript-richness component in [0, 1].

    Scales linearly with total word count and saturates at 30 words.
    """
    total_words = max(0, count_english_words) + max(0, count_hindi_words)
    return min(total_words / 30.0, 1.0)


def _clip01(value: float) -> float:
    """Clamp a value into the closed interval [0, 1]."""
    return min(1.0, max(0.0, float(value)))


def lf_duration_component(
    candidate: CandidateRecord, config: SelectionConfig
) -> Optional[float]:
    """Label-free duration-fit component in [0, 1], or None without duration.

    Reuses :func:`duration_component` semantics unchanged: 1.0 at
    ``config.preferred_duration_sec`` (8.0 seconds by default) and linear
    decay to 0.0 at the furthest configured duration bound, so the
    annotation-preferred ~8 second band scores highest. Higher is a better
    annotation fit (this component is a preference, not a risk). Weight
    0.30 in ``hinglish.labelfree.v1``.
    """
    return duration_component(candidate.duration_sec, config)


def lf_repetition_component(
    asr_output: AsrOutputFeatures,
) -> Optional[float]:
    """Label-free repetition-risk component in [0, 1], or None.

    Risk of repetitive hallucination implied by decoded-token repetition
    statistics alone (no generated text leaves the inference witness).
    Weight 0.30 in ``hinglish.labelfree.v1``. Higher is more suspicious.

    Args:
        asr_output: Model-derived record with status ``ok``. The component
            is None unless BOTH ``repeated_token_fraction`` and
            ``max_consecutive_repeated_tokens`` are present (a partial
            repetition measurement is never scored as a partial sum).

    Returns:
        The component value. Returned levels: the 0.6/0.4 weighted
        combination of the saturated repetition fraction and consecutive
        repeat run (bases in [0, 1]); floored at 0.75 when
        ``repetition_flag`` is true; floored at 0.85 when the decode hit
        its token budget without emitting EOS (J1 evidence: cap-hit
        without EOS is looping until truncation); clamped to at most 0.55
        when the decode emitted EOS without hitting the budget (a natural
        stop reduces suspicion). The precedence rules are evaluated in that
        order, so a cap-hit floor dominates the flag floor and the
        natural-stop clamp is applied last (unconditionally over the flag
        floor).
    """
    fraction = asr_output.repeated_token_fraction
    consecutive = asr_output.max_consecutive_repeated_tokens
    if fraction is None or consecutive is None:
        return None
    base = 0.6 * _clip01(fraction / LF_REPETITION_FRACTION_SATURATION)
    base += 0.4 * _clip01(consecutive / LF_REPETITION_CONSECUTIVE_SATURATION)
    result = base
    if asr_output.repetition_flag is True:
        result = max(result, LF_REPETITION_FLAG_FLOOR)
    if (
        asr_output.hit_length_limit_effective is True
        and asr_output.eos_emitted_effective is not True
    ):
        result = max(result, LF_CAP_HIT_NO_EOS_FLOOR)
    if (
        asr_output.eos_emitted_effective is True
        and asr_output.hit_length_limit_effective is False
    ):
        result = min(result, LF_NATURAL_EOS_REPETITION_CEILING)
    return _clip01(result)


def lf_length_anomaly_component(
    asr_output: AsrOutputFeatures, duration_sec: Optional[float]
) -> Optional[float]:
    """Label-free length-anomaly component in [0, 1], or None.

    Compares the decoded token count against the audio duration in seconds
    using the pre-registered healthy transcription-rate band from the
    committed J1 control clips (results/hinglish/decoder_probe_metrics.json,
    per_clip C0 records, category ``control``):

    - 111627_bYPRSx2sjXooUdv8_0000: 86 tokens / 7.275 s
      = 11.821305841924397 tokens per audio second
    - 136706_PMLT41UVHAjHIiJW_0000: 76 tokens / 5.55 s
      = 13.693693693693694 tokens per audio second
    - 415128_riwQQ9Di2QY1QXwx_0000: 16 tokens / 1.55 s
      = 10.32258064516129 tokens per audio second

    The healthy band is the min/max of those three rates:
    [10.32258064516129, 13.693693693693694] tokens per audio second (band
    mean 11.945860060259795). A rate inside the band is exactly 0. Outside
    the band it rises linearly from 0 at the band edge to 1.0 at half of
    the low edge (5.161290322580645 tokens per audio second) or 50% beyond
    the high edge (20.54054054054054 tokens per audio second). A decode that
    hit its token budget without emitting EOS raises a smaller computed
    anomaly to at least 0.80 (J1 cap-hit looping signature). Weight
    0.20 in ``hinglish.labelfree.v1``. Higher is more anomalous.

    Args:
        asr_output: Model-derived record with status ``ok``.
        duration_sec: Manifest audio duration in seconds; the denominator
            of record for the token rate. None leaves the component None
            (never zero).

    Raises:
        ValueError: If ``duration_sec`` is not positive (a zero or negative
            duration cannot define a token rate).
    """
    tokens = asr_output.generated_token_count
    if tokens is None or duration_sec is None:
        return None
    if duration_sec <= 0:
        raise ValueError(
            "duration_sec must be positive to form a tokens-per-second "
            f"rate (candidate record reports {duration_sec})"
        )
    rate = tokens / duration_sec
    if rate < LF_TOKEN_RATE_BAND_LOW:
        anomaly = (
            LF_TOKEN_RATE_BAND_LOW - rate
        ) / (
            LF_TOKEN_RATE_BAND_LOW - LF_TOKEN_RATE_SATURATION_LOW
        )
    elif rate > LF_TOKEN_RATE_BAND_HIGH:
        anomaly = (
            rate - LF_TOKEN_RATE_BAND_HIGH
        ) / (
            LF_TOKEN_RATE_SATURATION_HIGH - LF_TOKEN_RATE_BAND_HIGH
        )
    else:
        anomaly = 0.0
    if (
        asr_output.hit_length_limit_effective is True
        and asr_output.eos_emitted_effective is not True
    ):
        anomaly = max(anomaly, LF_LENGTH_ANOMALY_CAP_HIT_FLOOR)
    return _clip01(anomaly)


def lf_confidence_component(
    asr_output: AsrOutputFeatures,
) -> Optional[float]:
    """Label-free confidence-risk component in [0, 1], or None.

    Chosen-token logprob risk measured per step over the decoded sequence.
    ``min_logprob`` risk is ``min((0.10 - min_logprob) / 1.10, 1.0)``
    clamped to [0, 1] (min_logprob >= 0.10 healthy -> 0; <= -1.00 -> 1.0);
    ``mean_logprob`` risk is ``min((0.05 - mean_logprob) / 0.45, 1.0)``
    clamped (0.05 healthy -> 0; -0.40 -> 1.0). Combined 0.5 min / 0.5 mean.
    Weight 0.12 in ``hinglish.labelfree.v1``. Higher is riskier. When
    either measured logprob is None the component is None (never zero).
    """
    min_logprob = asr_output.min_logprob
    mean_logprob = asr_output.mean_logprob
    if min_logprob is None or mean_logprob is None:
        return None
    min_risk = _clip01(
        (LF_CONFIDENCE_MIN_LOGPROB_HEALTHY - min_logprob)
        / LF_CONFIDENCE_MIN_LOGPROB_SPAN
    )
    mean_risk = _clip01(
        (LF_CONFIDENCE_MEAN_LOGPROB_HEALTHY - mean_logprob)
        / LF_CONFIDENCE_MEAN_LOGPROB_SPAN
    )
    return LF_CONFIDENCE_SUBWEIGHT_MIN * min_risk + LF_CONFIDENCE_SUBWEIGHT_MEAN * mean_risk


def lf_audio_quality_component(
    *,
    audio_peak_dbfs: Optional[float],
    audio_clipping_fraction: Optional[float],
    audio_mean_rms_dbfs: Optional[float],
) -> Optional[float]:
    """Label-free audio-quality-risk component in [0, 1], or None.

    Inputs come from the pool manifest, not from inference: peak level
    (dBFS), clipping fraction in [0, 1], and mean RMS level (dBFS). Base
    clipping risk is ``min(clipping_fraction / 0.05, 1.0)`` clamped; a peak
    at or above -0.5 dBFS floors the base risk at 0.70; the RMS risk is
    ``clip((abs(mean_rms_dbfs) - 38) / 20, 0, 1)`` so |RMS| <= 38 dBFS is
    quiet (0) and >= 58 dBFS saturates (1.0). Base and RMS risks combine
    0.5/0.5. With only some inputs available the available risks are
    renormalized (a missing input is never a zero); with no inputs at all
    the component is None. Weight 0.08 in ``hinglish.labelfree.v1``.
    Higher is riskier audio.
    """
    base_risk: Optional[float] = None
    if audio_clipping_fraction is not None:
        base_risk = _clip01(
            audio_clipping_fraction / LF_QUALITY_CLIPPING_SATURATION_FRACTION
        )
        if (
            audio_peak_dbfs is not None
            and audio_peak_dbfs >= LF_QUALITY_PEAK_FLOOR_DBFS
        ):
            base_risk = max(base_risk, LF_QUALITY_PEAK_FLOOR_RISK)
    elif audio_peak_dbfs is not None and audio_peak_dbfs >= LF_QUALITY_PEAK_FLOOR_DBFS:
        # A clipping fraction is the primary base input; with only a peak
        # level available the peak floor stands alone as the base risk.
        base_risk = LF_QUALITY_PEAK_FLOOR_RISK
    rms_risk: Optional[float] = None
    if audio_mean_rms_dbfs is not None:
        rms_risk = _clip01(
            (abs(audio_mean_rms_dbfs) - LF_QUALITY_RMS_QUIET_DBFS)
            / LF_QUALITY_RMS_SATURATION_DBFS
        )
    if base_risk is None and rms_risk is None:
        return None
    # Equal 0.5/0.5 subweights over the available risks; a single available
    # risk renormalizes to full weight (never a zero contribution).
    available = [risk for risk in (base_risk, rms_risk) if risk is not None]
    return sum(available) / len(available)


def code_mix_bucket(ratio_english_words: float) -> str:
    """Bucket an English ratio relative to the preferred band."""
    if ratio_english_words < CODE_MIX_BAND_LOW:
        return "below_band"
    if ratio_english_words > CODE_MIX_BAND_HIGH:
        return "above_band"
    return "in_band"


def difficulty_bucket(candidate: CandidateRecord) -> str:
    """Bucket a candidate by ASR failure evidence, or mark it unavailable."""
    component = failure_component(candidate.asr)
    if component is None:
        return ASR_UNAVAILABLE
    if component < 1.0 / 3.0:
        return "low"
    if component < 2.0 / 3.0:
        return "medium"
    return "high"


def score_candidate(
    candidate: CandidateRecord,
    config: SelectionConfig = SelectionConfig(),
) -> CandidateScore:
    """Score one candidate with components renormalized over availability.

    In ``oracle`` mode the components are exactly the four supervised
    signals (``failure``, ``code_mix``, ``duration``, ``quality``). In
    ``label_free`` mode the components are exactly the five frozen
    ``hinglish.labelfree.v1`` signals (``duration_fit``, ``repetition_risk``,
    ``length_anomaly``, ``confidence_risk``, ``audio_quality_risk``); the
    transcript-derived code-mix and quality signals are not part of the
    component set (not even as unavailable slots).

    Args:
        candidate: Candidate to score.
        config: Selection configuration.

    Returns:
        CandidateScore where ``components`` marks unavailable components with
        ``None``. The overall score is a weighted mean over the components
        that are available; unavailable components never contribute and are
        never zero-filled.

    Raises:
        ValueError: In ``label_free`` mode when the candidate carries ASR
            failure features (ground-truth-derived and therefore prohibited
            during acquisition), or when it lacks a usable (present with
            status ``ok``) model-derived ``AsrOutputFeatures`` scoring
            record.
    """
    if (
        config.acquisition_mode == ACQUISITION_LABEL_FREE
        and candidate.asr is not None
    ):
        raise ValueError(
            "label_free acquisition prohibits ground-truth WER/S/D/I "
            f"features (candidate {candidate.clip_id} carries them)"
        )
    weights_map: Mapping[str, float]
    if config.acquisition_mode == ACQUISITION_LABEL_FREE:
        asr_output = candidate.asr_output
        if asr_output is None or asr_output.status != ASR_OUTPUT_STATUS_OK:
            raise ValueError(
                "label_free scoring requires AsrOutputFeatures (status ok) "
                f"for candidate {candidate.clip_id}"
            )
        weights_map = LABEL_FREE_SCORING_WEIGHTS
        components: dict[str, Optional[float]] = {
            "duration_fit": lf_duration_component(candidate, config),
            "repetition_risk": lf_repetition_component(asr_output),
            "length_anomaly": lf_length_anomaly_component(
                asr_output, candidate.duration_sec
            ),
            "confidence_risk": lf_confidence_component(asr_output),
            "audio_quality_risk": lf_audio_quality_component(
                audio_peak_dbfs=candidate.audio_peak_dbfs,
                audio_clipping_fraction=candidate.audio_clipping_fraction,
                audio_mean_rms_dbfs=candidate.audio_mean_rms_dbfs,
            ),
        }
    else:
        weights_map = SCORING_WEIGHTS
        components = {
            "failure": failure_component(candidate.asr),
            "code_mix": code_mix_component(candidate.ratio_english_words),
            "duration": duration_component(candidate.duration_sec, config),
            "quality": quality_component(
                candidate.count_english_words, candidate.count_hindi_words
            ),
        }
    available_weight = sum(
        weights_map[name]
        for name, value in components.items()
        if value is not None
    )
    if available_weight <= 0.0:
        raise ValueError("no scoring component is available for candidate")
    weighted = sum(
        weights_map[name] * float(value)
        for name, value in components.items()
        if value is not None
    )
    asr_status = ASR_AVAILABLE if candidate.asr is not None else ASR_UNAVAILABLE
    code_mix_bucket_value = (
        code_mix_bucket(candidate.ratio_english_words)
        if candidate.ratio_english_words is not None
        else ASR_UNAVAILABLE
    )
    return CandidateScore(
        clip_id=candidate.clip_id,
        score=weighted / available_weight,
        components=components,
        asr_status=asr_status,
        asr_wer=candidate.asr.wer if candidate.asr is not None else None,
        difficulty_bucket=difficulty_bucket(candidate),
        code_mix_bucket=code_mix_bucket_value,
    )


def score_candidates(
    candidates: Iterable[CandidateRecord],
    config: SelectionConfig = SelectionConfig(),
) -> Tuple[CandidateScore, ...]:
    """Score several candidates in input order."""
    return tuple(score_candidate(candidate, config) for candidate in candidates)


# ---------------------------------------------------------------------------
# Arm selection
# ---------------------------------------------------------------------------


def _apply_speaker_caps(
    sequence: Sequence[CandidateRecord],
    *,
    target_count: int,
    max_per_speaker: int,
) -> list[CandidateRecord]:
    """Pick in sequence order, preferring one clip per speaker.

    Candidates whose speaker already contributed are deferred behind all
    first occurrences and only used to fill remaining slots up to
    ``max_per_speaker``.
    """
    first_occurrences: list[CandidateRecord] = []
    deferred: list[CandidateRecord] = []
    seen: set[str] = set()
    for candidate in sequence:
        if candidate.speaker_id not in seen:
            seen.add(candidate.speaker_id)
            first_occurrences.append(candidate)
        else:
            deferred.append(candidate)

    selected: list[CandidateRecord] = []
    counts: Counter = Counter()
    for candidate in first_occurrences + deferred:
        if len(selected) >= target_count:
            break
        if counts[candidate.speaker_id] >= max_per_speaker:
            continue
        selected.append(candidate)
        counts[candidate.speaker_id] += 1
    return selected


def _duration_summary(
    candidates: Sequence[CandidateRecord],
) -> Tuple[Optional[float], int, int]:
    """Return (known-duration total, known count, unknown count)."""
    known = [c.duration_sec for c in candidates if c.duration_sec is not None]
    if not known:
        return None, 0, len(candidates)
    return sum(known), len(known), len(candidates) - len(known)


def assert_finalized_arm_speakers(
    arm: ArmSelection, config: SelectionConfig
) -> None:
    """Assert the finalized arm honors the one-clip-per-speaker contract.

    With ``max_per_speaker=1`` a finalized arm must contain exactly one clip
    per speaker — including after duration-budget matching swapped clips in
    and out. For any other ``max_per_speaker`` this check is a no-op because
    multi-clip speakers are then allowed.

    Raises:
        ValueError: If ``max_per_speaker`` is 1 and any speaker contributes
            more than one clip to the finalized arm.
    """
    if config.max_per_speaker != 1:
        return
    speaker_counts = Counter(entry.candidate.speaker_id for entry in arm.ranked)
    duplicated = sorted(speaker for speaker, n in speaker_counts.items() if n > 1)
    if duplicated:
        raise ValueError(
            f"finalized arm {arm.arm!r} violates the one-clip-per-speaker "
            "contract after duration matching; speakers with multiple "
            "clips: " + ", ".join(duplicated[:5])
        )


def _rank_random_entries(
    selected: Sequence[CandidateRecord],
    config: SelectionConfig,
) -> Tuple[RankedCandidate, ...]:
    """Rank random-arm entries, recording scores for audit only."""
    ranked: list[RankedCandidate] = []
    for index, candidate in enumerate(selected):
        score = score_candidate(candidate, config)
        ranked.append(
            RankedCandidate(
                rank=index + 1,
                candidate=candidate,
                score=score.score,
                components=score.components,
                asr_status=score.asr_status,
                asr_wer=score.asr_wer,
                difficulty_bucket=score.difficulty_bucket,
                code_mix_bucket=score.code_mix_bucket,
            )
        )
    return tuple(ranked)


def select_random_arm(
    eligible: Sequence[CandidateRecord],
    config: SelectionConfig = SelectionConfig(),
) -> ArmSelection:
    """Select the seeded random control arm.

    Candidates are sorted deterministically by ``(speaker_id, clip_id)`` and
    shuffled with ``random.Random(config.seed)``. Selection then prefers one
    clip per speaker and fills remaining slots up to ``max_per_speaker``.
    Scores are recorded for audit comparison only; they do not influence this
    arm's selection.

    Raises:
        ValueError: If the eligible pool cannot satisfy ``target_count``.
    """
    if not eligible:
        raise ValueError("eligible candidate pool is empty")
    ordered = sorted(eligible, key=lambda c: (c.speaker_id, c.clip_id))
    random.Random(config.seed).shuffle(ordered)
    selected = _apply_speaker_caps(
        ordered,
        target_count=config.target_count,
        max_per_speaker=config.max_per_speaker,
    )
    if len(selected) < config.target_count:
        raise ValueError(
            f"eligible pool of {len(eligible)} clips cannot satisfy "
            f"target_count={config.target_count} under max_per_speaker="
            f"{config.max_per_speaker}"
        )
    total, known, unknown = _duration_summary(selected)
    arm = ArmSelection(
        arm=ARM_RANDOM,
        ranked=_rank_random_entries(selected, config),
        total_duration_sec=total,
        duration_known_count=known,
        duration_unknown_count=unknown,
        sample_order=tuple(candidate.clip_id for candidate in ordered),
    )
    assert_finalized_arm_speakers(arm, config)
    return arm


def _interleave_by_shard(
    candidates: Sequence[CandidateRecord],
) -> list[CandidateRecord]:
    """Round-robin the selection across shards, preserving within-shard order.

    Rank order drives which clips survive a prefix cut, so spreading shards
    early keeps any prefix shard-diverse. Shards appear in first-appearance
    order; candidates without a shard form their own group.
    """
    groups: "OrderedDict[str, list[CandidateRecord]]" = OrderedDict()
    for candidate in candidates:
        groups.setdefault(candidate.shard or "unknown", []).append(candidate)
    interleaved: list[CandidateRecord] = []
    while True:
        advanced = False
        for group in groups.values():
            if group:
                interleaved.append(group.pop(0))
                advanced = True
        if not advanced:
            break
    return interleaved


def select_intelligent_arm(
    eligible: Sequence[CandidateRecord],
    config: SelectionConfig = SelectionConfig(),
) -> ArmSelection:
    """Select the intelligent arm.

    Pipeline: score every eligible candidate; group by speaker; round-robin
    one clip per speaker (speakers ordered by their best score, then speaker
    ID); fill further rounds up to ``max_per_speaker``; finally interleave the
    ranks across shards. Ties break on ``clip_id`` throughout, so the result
    is deterministic.

    Raises:
        ValueError: If the eligible pool cannot satisfy ``target_count``.
    """
    if not eligible:
        raise ValueError("eligible candidate pool is empty")
    scored = score_candidates(eligible, config)
    by_speaker: dict[str, list[CandidateScore]] = defaultdict(list)
    by_clip: dict[str, CandidateRecord] = {}
    for candidate, score in zip(eligible, scored):
        by_speaker[candidate.speaker_id].append(score)
        by_clip[candidate.clip_id] = candidate
    for group in by_speaker.values():
        group.sort(key=lambda s: (-s.score, s.clip_id))

    speaker_order = sorted(
        by_speaker,
        key=lambda speaker: (-by_speaker[speaker][0].score, speaker),
    )
    picked: list[CandidateScore] = []
    for round_index in range(config.max_per_speaker):
        if len(picked) >= config.target_count:
            break
        for speaker in speaker_order:
            if len(picked) >= config.target_count:
                break
            group = by_speaker[speaker]
            if round_index < len(group):
                picked.append(group[round_index])
    if len(picked) < config.target_count:
        raise ValueError(
            f"eligible pool of {len(eligible)} clips cannot satisfy "
            f"target_count={config.target_count} under max_per_speaker="
            f"{config.max_per_speaker}"
        )

    picked_candidates = [by_clip[score.clip_id] for score in picked]
    interleaved = _interleave_by_shard(picked_candidates)
    score_by_clip = {score.clip_id: score for score in picked}
    ranked = tuple(
        RankedCandidate(
            rank=index + 1,
            candidate=candidate,
            score=score_by_clip[candidate.clip_id].score,
            components=score_by_clip[candidate.clip_id].components,
            asr_status=score_by_clip[candidate.clip_id].asr_status,
            asr_wer=score_by_clip[candidate.clip_id].asr_wer,
            difficulty_bucket=score_by_clip[candidate.clip_id].difficulty_bucket,
            code_mix_bucket=score_by_clip[candidate.clip_id].code_mix_bucket,
        )
        for index, candidate in enumerate(interleaved)
    )
    total, known, unknown = _duration_summary(interleaved)
    arm = ArmSelection(
        arm=ARM_INTELLIGENT,
        ranked=ranked,
        total_duration_sec=total,
        duration_known_count=known,
        duration_unknown_count=unknown,
    )
    assert_finalized_arm_speakers(arm, config)
    return arm


# ---------------------------------------------------------------------------
# Duration-budget matching
# ---------------------------------------------------------------------------


def _budget_match_result(
    *,
    applied: bool,
    target: Optional[float],
    final_total: Optional[float],
    tolerance: float,
    swaps: int,
    unknown_durations: int,
    note: str,
) -> BudgetMatchResult:
    residual = None
    residual_fraction = None
    within = False
    if target is not None and final_total is not None:
        residual = final_total - target
        residual_fraction = abs(residual) / target if target > 0 else 0.0
        within = residual_fraction <= tolerance
    return BudgetMatchResult(
        applied=applied,
        target_total_duration_sec=target,
        final_total_duration_sec=final_total,
        residual_sec=residual,
        residual_fraction=residual_fraction,
        tolerance=tolerance,
        within_tolerance=within,
        swaps_applied=swaps,
        duration_unknown_count=unknown_durations,
        note=note,
    )


def match_duration_budget(
    selection: Sequence[CandidateRecord],
    replacement_pool: Sequence[CandidateRecord],
    *,
    target_total_duration_sec: Optional[float],
    tolerance: float = DEFAULT_DURATION_TOLERANCE,
    max_per_speaker: int = DEFAULT_MAX_PER_SPEAKER,
    max_swaps: int = 200,
    max_replacements_examined: int = 4000,
) -> Tuple[Tuple[CandidateRecord, ...], BudgetMatchResult]:
    """Match a selection's total audio duration to a target with one-for-one swaps.

    The selection's clip count never changes: each step swaps the single
    (remove, add) pair that most reduces the absolute duration residual while
    respecting the per-speaker cap. Replacement candidates are examined in the
    order supplied, up to ``max_replacements_examined`` with known durations.
    A replacement inherits the rank position of the clip it replaces.

    The function is deterministic: scans are index-ordered and swaps are only
    applied while the residual strictly decreases. Residual mismatch is
    reported when the pool cannot reach the tolerance; it is never hidden.

    Args:
        selection: Chosen candidates, in rank order.
        replacement_pool: Further candidates, in deterministic preference
            order, used as swap material.
        target_total_duration_sec: Target total audio duration, seconds.
        tolerance: Fractional tolerance for a successful match.
        max_per_speaker: Per-speaker clip cap enforced during swaps.
        max_swaps: Maximum number of swaps to apply.
        max_replacements_examined: Cap on examined replacement candidates.

    Returns:
        ``(adjusted_selection, BudgetMatchResult)``. When any duration is
        unknown, when no target is given, or when the target is already met,
        the selection is returned unchanged with the reason in ``note``.
    """
    current = list(selection)
    unknown_durations = sum(1 for c in current if c.duration_sec is None)
    if unknown_durations:
        total, _, _ = _duration_summary(current)
        return tuple(current), _budget_match_result(
            applied=False,
            target=target_total_duration_sec,
            final_total=total,
            tolerance=tolerance,
            swaps=0,
            unknown_durations=unknown_durations,
            note="skipped_unknown_durations",
        )
    if target_total_duration_sec is None:
        total, _, _ = _duration_summary(current)
        return tuple(current), _budget_match_result(
            applied=False,
            target=None,
            final_total=total,
            tolerance=tolerance,
            swaps=0,
            unknown_durations=0,
            note="skipped_no_target",
        )

    current_total = sum(c.duration_sec for c in current)
    tolerance_abs = tolerance * target_total_duration_sec

    selected_ids = {c.clip_id for c in current}
    replacements = [
        candidate
        for candidate in replacement_pool
        if candidate.clip_id not in selected_ids
        and candidate.duration_sec is not None
    ][:max_replacements_examined]
    counts: Counter = Counter(c.speaker_id for c in current)

    swaps = 0
    while (
        swaps < max_swaps
        and abs(current_total - target_total_duration_sec) > tolerance_abs
        and replacements
    ):
        best: Optional[Tuple[float, int, int]] = None
        for index, removed in enumerate(current):
            removed_duration = float(removed.duration_sec)
            residual_without = current_total - removed_duration - target_total_duration_sec
            for candidate_index, added in enumerate(replacements):
                if added.clip_id in selected_ids:
                    continue
                if added.speaker_id == removed.speaker_id:
                    speaker_ok = counts[added.speaker_id] <= max_per_speaker
                else:
                    speaker_ok = counts[added.speaker_id] + 1 <= max_per_speaker
                if not speaker_ok:
                    continue
                new_residual = residual_without + float(added.duration_sec)
                improvement = abs(current_total - target_total_duration_sec) - abs(new_residual)
                if improvement <= 1e-9:
                    continue
                if best is None or improvement > best[0] + 1e-12:
                    best = (improvement, index, candidate_index)
        if best is None:
            break

        _, index, candidate_index = best
        removed = current[index]
        added = replacements[candidate_index]
        counts[removed.speaker_id] -= 1
        counts[added.speaker_id] += 1
        current[index] = added
        selected_ids.remove(removed.clip_id)
        selected_ids.add(added.clip_id)
        current_total = current_total - float(removed.duration_sec) + float(added.duration_sec)
        # The removed clip returns to the swap pool so later steps can still
        # use its duration, but strictly improving swaps prevent cycling.
        replacements.pop(candidate_index)
        replacements.append(removed)
        swaps += 1

    within = abs(current_total - target_total_duration_sec) <= tolerance_abs
    note = "within_tolerance" if within else "tolerance_not_reached_after_swaps"
    return tuple(current), _budget_match_result(
        applied=True,
        target=target_total_duration_sec,
        final_total=current_total,
        tolerance=tolerance,
        swaps=swaps,
        unknown_durations=0,
        note=note,
    )


def _rerank_after_budget_match(
    arm: ArmSelection,
    adjusted: Sequence[CandidateRecord],
    budget_match: BudgetMatchResult,
    config: SelectionConfig,
) -> ArmSelection:
    """Rebuild an arm after swaps, preserving retained entries' metadata."""
    existing = {entry.candidate.clip_id: entry for entry in arm.ranked}
    ranked: list[RankedCandidate] = []
    for index, candidate in enumerate(adjusted):
        entry = existing.get(candidate.clip_id)
        if entry is None:
            # A replacement: derive audit fields from the candidate directly.
            score = score_candidate(candidate, config)
            entry = RankedCandidate(
                rank=index + 1,
                candidate=candidate,
                score=score.score,
                components=score.components,
                asr_status=score.asr_status,
                asr_wer=score.asr_wer,
                difficulty_bucket=score.difficulty_bucket,
                code_mix_bucket=score.code_mix_bucket,
            )
        ranked.append(replace(entry, rank=index + 1))
    total, known, unknown = _duration_summary(adjusted)
    return replace(
        arm,
        ranked=tuple(ranked),
        total_duration_sec=total,
        duration_known_count=known,
        duration_unknown_count=unknown,
        budget_match=budget_match,
    )


def build_paired_selection(
    candidates: Iterable[CandidateRecord],
    *,
    training_speakers: Iterable[str],
    forbidden_speakers: Iterable[str] = (),
    forbidden_clip_ids: Iterable[str] = (),
    config: SelectionConfig = SelectionConfig(),
) -> PairedSelectionResult:
    """Build both arms from one candidate pool with matched duration budgets.

    The intelligent arm runs first and defines the duration target. The
    random arm is then adjusted by one-for-one swaps (using the remaining
    seeded sample order as swap material) toward that target within
    ``config.duration_tolerance``. Any residual mismatch is recorded in the
    random arm's ``budget_match``.
    """
    pool = prepare_candidate_pool(
        candidates,
        training_speakers=training_speakers,
        forbidden_speakers=forbidden_speakers,
        forbidden_clip_ids=forbidden_clip_ids,
        config=config,
    )
    intelligent = select_intelligent_arm(pool.eligible, config)
    random_arm = select_random_arm(pool.eligible, config)

    if intelligent.total_duration_sec is None:
        match = _budget_match_result(
            applied=False,
            target=None,
            final_total=random_arm.total_duration_sec,
            tolerance=config.duration_tolerance,
            swaps=0,
            unknown_durations=random_arm.duration_unknown_count,
            note="skipped_no_target",
        )
        random_final = replace(random_arm, budget_match=match)
        assert_finalized_arm_speakers(random_final, config)
        assert_finalized_arm_speakers(intelligent, config)
        return PairedSelectionResult(pool, intelligent, random_final)

    by_id = {candidate.clip_id: candidate for candidate in pool.eligible}
    selected_ids = {entry.candidate.clip_id for entry in random_arm.ranked}
    replacement_order = [
        by_id[clip_id]
        for clip_id in random_arm.sample_order
        if clip_id not in selected_ids
    ]
    adjusted, match = match_duration_budget(
        random_arm.selected,
        replacement_order,
        target_total_duration_sec=intelligent.total_duration_sec,
        tolerance=config.duration_tolerance,
        max_per_speaker=config.max_per_speaker,
    )
    random_final = _rerank_after_budget_match(random_arm, adjusted, match, config)
    assert_finalized_arm_speakers(random_final, config)
    assert_finalized_arm_speakers(intelligent, config)
    return PairedSelectionResult(pool, intelligent, random_final)


# ---------------------------------------------------------------------------
# Reporting and safe manifest output
# ---------------------------------------------------------------------------


def _arm_summary(arm: ArmSelection, config: SelectionConfig) -> dict[str, object]:
    """Summarize one arm for the selection report."""
    candidates = [entry.candidate for entry in arm.ranked]
    speaker_counts = Counter(candidate.speaker_id for candidate in candidates)
    scores = [entry.score for entry in arm.ranked if entry.score is not None]
    wers = [entry.asr_wer for entry in arm.ranked if entry.asr_wer is not None]
    known_durations = [
        candidate.duration_sec
        for candidate in candidates
        if candidate.duration_sec is not None
    ]
    summary: dict[str, object] = {
        "arm": arm.arm,
        "size": len(candidates),
        "distinct_speakers": len(speaker_counts),
        "clips_per_speaker_max": max(speaker_counts.values()) if speaker_counts else 0,
        "clips_per_speaker_mean": (
            round(sum(speaker_counts.values()) / len(speaker_counts), 3)
            if speaker_counts
            else 0.0
        ),
        "speakers_at_cap": sum(
            1 for count in speaker_counts.values() if count >= config.max_per_speaker
        ),
        "total_duration_sec": (
            round(arm.total_duration_sec, 3)
            if arm.total_duration_sec is not None
            else None
        ),
        "mean_clip_duration_sec": (
            round(sum(known_durations) / len(known_durations), 3)
            if known_durations
            else None
        ),
        "duration_known_count": arm.duration_known_count,
        "duration_unknown_count": arm.duration_unknown_count,
        "asr_status_counts": dict(Counter(entry.asr_status for entry in arm.ranked)),
        "difficulty_bucket_counts": dict(
            Counter(entry.difficulty_bucket for entry in arm.ranked)
        ),
        "code_mix_bucket_counts": dict(
            Counter(entry.code_mix_bucket for entry in arm.ranked)
        ),
        "shard_counts": dict(
            sorted(Counter(candidate.shard or "unknown" for candidate in candidates).items())
        ),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_asr_wer": round(sum(wers) / len(wers), 4) if wers else None,
    }
    if arm.budget_match is not None:
        summary["budget_match"] = asdict(arm.budget_match)
    return summary


def build_selection_report(
    pool: CandidatePool,
    arms: Mapping[str, ArmSelection],
    config: SelectionConfig,
) -> dict[str, object]:
    """Build the machine-readable selection report.

    Args:
        pool: The shared candidate pool the arms were drawn from.
        arms: Arm name to selection.
        config: Selection configuration used.

    Returns:
        A JSON-serializable report with arm sizes, duration totals, bucket
        composition, per-speaker cap usage, pool rejection reasons (including
        ``scoring_failed`` in ``label_free`` mode), the bias warning, and the
        mitigation contract. In ``label_free`` mode it also echoes
        ``formula_version = "hinglish.labelfree.v1"`` and
        ``label_free_component_availability`` mapping each frozen component to
        how many of the eligible pool candidates had it available; in oracle
        mode both keys are ``None``.
    """
    reason_counts: Counter = Counter()
    for _, reasons in pool.rejected:
        reason_counts.update(reasons)

    label_free_availability: Optional[Mapping[str, object]] = None
    formula_version_reported: Optional[str] = None
    if config.acquisition_mode == ACQUISITION_LABEL_FREE:
        formula_version_reported = LABEL_FREE_FORMULA_VERSION
        eligible_count = len(pool.eligible)
        scored = score_candidates(pool.eligible, config)
        label_free_availability = {
            component: {
                "available_count": sum(
                    1
                    for entry in scored
                    if entry.components.get(component) is not None
                ),
                "eligible_count": eligible_count,
            }
            for component in LABEL_FREE_COMPONENTS
        }

    mitigations = list(SELECTION_MITIGATIONS)
    if config.acquisition_mode == ACQUISITION_LABEL_FREE:
        mitigations.append(LABEL_FREE_ACQUISITION_NOTE)

    return {
        "selection_version": SELECTION_VERSION,
        "formula_version": formula_version_reported,
        "config": asdict(config),
        "acquisition_mode": config.acquisition_mode,
        "pool": {
            "eligible": len(pool.eligible),
            "rejected": len(pool.rejected),
            "duplicates_removed": len(pool.duplicate_clip_ids),
            "duplicate_clip_ids": list(pool.duplicate_clip_ids),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "asr_available_count": pool.asr_available_count,
            "asr_unavailable_count": pool.asr_unavailable_count,
            "duration_unknown_count": pool.duration_unknown_count,
        },
        "arms": {
            name: _arm_summary(arm, config) for name, arm in sorted(arms.items())
        },
        "label_free_component_availability": label_free_availability,
        "bias_warning": SELECTION_BIAS_WARNING,
        "mitigations": mitigations,
        "asr_coverage_note": (
            "ASR failure components are marked unavailable when predictions "
            "are missing for a candidate; they are never scored as zero. "
            "Compare difficulty composition only across arms with equal ASR "
            "coverage."
        ),
    }


def selection_manifest_rows(arm: ArmSelection) -> list[dict[str, object]]:
    """Convert one arm into safe manifest rows (no transcript text).

    Optional numeric columns that are unavailable (``None``) are written as
    empty strings. This can happen in ``label_free`` selection runs, where
    transcript-derived ratio/count/wer columns carry no value; the writer
    never emits ``NaN`` or ``None`` strings. The column set is always
    :data:`SAFE_MANIFEST_COLUMNS`, unchanged.
    """
    rows: list[dict[str, object]] = []
    for entry in arm.ranked:
        candidate = entry.candidate
        rows.append(
            {
                "clip_id": candidate.clip_id,
                "speaker_id": candidate.speaker_id,
                "arm": arm.arm,
                "score": "" if entry.score is None else round(entry.score, 6),
                "rank": entry.rank,
                "ratio_english_words": (
                    "" if candidate.ratio_english_words is None
                    else candidate.ratio_english_words
                ),
                "ratio_hindi_words": (
                    "" if candidate.ratio_hindi_words is None
                    else candidate.ratio_hindi_words
                ),
                "count_english_words": (
                    "" if candidate.count_english_words is None
                    else candidate.count_english_words
                ),
                "count_hindi_words": (
                    "" if candidate.count_hindi_words is None
                    else candidate.count_hindi_words
                ),
                "segment_part": candidate.segment_part,
                "shard": candidate.shard,
                "duration_sec": (
                    "" if candidate.duration_sec is None else candidate.duration_sec
                ),
                "asr_status": entry.asr_status,
                "asr_wer": "" if entry.asr_wer is None else round(entry.asr_wer, 6),
                "difficulty_bucket": entry.difficulty_bucket,
                "code_mix_bucket": entry.code_mix_bucket,
            }
        )
    return rows


def _assert_safe_rows(rows: Iterable[Mapping[str, object]]) -> None:
    """Refuse any row that would write transcript-bearing or unknown columns."""
    allowed = set(SAFE_MANIFEST_COLUMNS)
    for row in rows:
        for key in row:
            lowered = key.lower()
            if any(token in lowered for token in _UNSAFE_COLUMN_TOKENS):
                raise ValueError(
                    f"refusing to write unsafe manifest column: {key}"
                )
            if key not in allowed:
                raise ValueError(f"unknown manifest column: {key}")


def write_selection_manifest(path: Path, arm: ArmSelection) -> None:
    """Write a safe selection manifest CSV.

    The writer refuses unknown columns and any column whose name suggests
    transcript, reference, or prediction text.

    Args:
        path: Destination CSV path.
        arm: Arm selection to write.
    """
    rows = selection_manifest_rows(arm)
    _assert_safe_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SAFE_MANIFEST_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_selection_report(path: Path, report: Mapping[str, object]) -> None:
    """Write the selection report JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
