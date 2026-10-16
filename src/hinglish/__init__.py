"""Hinglish speech-model evaluation helpers.

This package provides normalized Hinglish ASR transcription handling and
metadata utilities. It intentionally avoids model inference and cloud concerns
so it can be unit tested without GPU dependencies.
"""


from __future__ import annotations


from .data import (
    REQUIRED_MUCS_COLUMNS,
    MUCS_HINGLISH_DATASET,
    MUCS_HINGLISH_REVISION,
    HinglishSplitAudit,
    MucsHinglishRecord,
    audit_split_records,
    load_mucs_hinglish_manifest,
    select_one_per_speaker,
    split_speakers_for_validation,
    validate_mucs_manifest,
)
from .evaluation import (
    HinglishPrediction,
    score_prediction,
    summarize_predictions,
)
from .metrics import (
    code_switch_bin,
    code_switch_ratio,
    english_word_ratio,
    language_aware_word_error_rate,
)
from .normalization import (
    HINGLISH_NORMALIZATION_VERSION,
    normalize_hinglish_text,
)


__all__ = [
    "REQUIRED_MUCS_COLUMNS",
    "MUCS_HINGLISH_DATASET",
    "MUCS_HINGLISH_REVISION",
    "HinglishSplitAudit",
    "MucsHinglishRecord",
    "audit_split_records",
    "code_switch_bin",
    "code_switch_ratio",
    "HinglishPrediction",
    "HINGLISH_NORMALIZATION_VERSION",
    "language_aware_word_error_rate",
    "score_prediction",
    "summarize_predictions",
    "load_mucs_hinglish_manifest",
    "normalize_hinglish_text",
    "select_one_per_speaker",
    "split_speakers_for_validation",
    "validate_mucs_manifest",
]
