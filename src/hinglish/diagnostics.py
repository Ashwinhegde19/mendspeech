"""Offline diagnostics for Hinglish ASR baseline outputs.

These helpers analyse existing predictions without running inference, changing
the baseline, or modifying a reference.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

from src.hinglish.evaluation import HinglishPrediction, score_prediction, summarize_predictions
from src.hinglish.normalization import normalize_hinglish_text


@dataclass(frozen=True)
class RepetitionFinding:
    """Heuristic description of repetition in one prediction."""

    max_consecutive_repeat: int
    longest_repeated_phrase: int
    repeated_phrase_count: int


def script_word_counts(text: str) -> dict[str, int]:
    """Count Latin and Devanagari script words in raw text."""
    cleaned = unicodedata.normalize("NFC", text or "")
    words = cleaned.split()
    latin = sum(1 for word in words if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in word))
    devanagari = sum(
        1
        for word in words
        if any("ऀ" <= ch <= "ॿ" for ch in word)
    )
    other = len(words) - latin - devanagari
    return {
        "words": len(words),
        "latin_script_words": latin,
        "devanagari_words": devanagari,
        "other_script_words": max(0, other),
    }


def analyze_repetition(words: Iterable[str]) -> RepetitionFinding:
    """Measure exact consecutive repetition in a token sequence.

    This does not detect semantic repetition across different wording. It is
    used as offline evidence that can be inspected before any mitigation.
    """
    tokens = list(words)
    max_repeat = 1
    current_run = 1
    for previous, current in zip(tokens, tokens[1:]):
        if current == previous:
            current_run += 1
            max_repeat = max(max_repeat, current_run)
        else:
            current_run = 1

    longest_phrase = 0
    max_possible = len(tokens) // 2
    for k in range(1, max_possible + 1):
        for start in range(len(tokens) - (2 * k) + 1):
            if tokens[start : start + k] == tokens[start + k : start + 2 * k]:
                longest_phrase = max(longest_phrase, k)

    phrase_count = 0
    if longest_phrase:
        for start in range(len(tokens)):
            if (
                start + longest_phrase <= len(tokens)
                and tokens[start : start + longest_phrase]
                == tokens[start + longest_phrase : start + 2 * longest_phrase]
            ):
                phrase_count += 1
    return RepetitionFinding(
        max_consecutive_repeat=max_repeat,
        longest_repeated_phrase=longest_phrase,
        repeated_phrase_count=phrase_count,
    )


def likely_repetitive_hallucination(
    *,
    reference_words: int,
    prediction_words: int,
    insertion_count: int,
    max_consecutive_repeat: int,
    longest_repeated_phrase: int,
) -> bool:
    """Return whether a prediction has clear repeated-hallucination evidence."""
    if prediction_words <= 0:
        return False
    # A literal immediate repetition is only a hallucination concern when the
    # prediction is long or materially exceeds the reference. Repeated
    # content in genuine speech does not satisfy those conditions.
    insertion_heavy = (
        insertion_count >= 8
        and prediction_words >= max(10, 2 * max(1, reference_words))
    )
    local_repeat = (
        max_consecutive_repeat >= 3
        and prediction_words > max(3, reference_words)
    )
    phrase_repeat = (
        longest_repeated_phrase >= 4
        and insertion_count >= max(1, reference_words)
    )
    return insertion_heavy or local_repeat or phrase_repeat


def diagnose_prediction(prediction: HinglishPrediction) -> dict[str, object]:
    """Score one prediction and return offline diagnostic fields."""
    if not prediction.reference.strip():
        raise ValueError("reference transcript must not be empty")
    score = score_prediction(prediction)
    normalized_reference = normalize_hinglish_text(prediction.reference)
    normalized_prediction = normalize_hinglish_text(prediction.prediction)
    reference_words = normalized_reference.split()
    prediction_words = normalized_prediction.split()
    repetition = analyze_repetition(prediction_words)
    is_hallucinated = likely_repetitive_hallucination(
        reference_words=int(score["reference_words"]),
        prediction_words=int(score["hypothesis_words"]),
        insertion_count=int(score["insertions"]),
        max_consecutive_repeat=repetition.max_consecutive_repeat,
        longest_repeated_phrase=repetition.longest_repeated_phrase,
    )
    raw_reference_counts = script_word_counts(prediction.reference)
    raw_prediction_counts = script_word_counts(prediction.prediction)
    return {
        "clip_id": prediction.clip_id,
        "speaker_id": prediction.speaker_id,
        "reference": prediction.reference,
        "prediction": prediction.prediction,
        "normalized_reference": normalized_reference,
        "normalized_prediction": normalized_prediction,
        "wer": score["wer"],
        "cer": score["cer"],
        "reference_words": score["reference_words"],
        "hypothesis_words": score["hypothesis_words"],
        "substitutions": score["substitutions"],
        "deletions": score["deletions"],
        "insertions": score["insertions"],
        "total_edits": score["total_edits"],
        "english_ratio": prediction.ratio_english_words,
        "english_ratio_bin": score["english_ratio_bin"],
        "max_consecutive_repeat": repetition.max_consecutive_repeat,
        "longest_repeated_phrase": repetition.longest_repeated_phrase,
        "repeated_phrase_count": repetition.repeated_phrase_count,
        "likely_repetitive_hallucination": is_hallucinated,
        "reference_script_counts": raw_reference_counts,
        "prediction_script_counts": raw_prediction_counts,
        "baseline_label": "Whisper-small-0 (language=hi, task=transcribe)",
    }


def diagnose_predictions(
    predictions: Iterable[HinglishPrediction],
) -> list[dict[str, object]]:
    """Diagnose several predictions with per-clip scores."""
    return [diagnose_prediction(prediction) for prediction in predictions]


def summarize_diagnostics(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Compute aggregate evidence without modifying baseline measurements."""
    materialized = list(records)
    if not materialized:
        return {"utterances": 0, "likely_hallucination_clips": 0, "clips": []}
    total_reference_words = sum(int(record["reference_words"]) for record in materialized)
    total_edits = sum(int(record["total_edits"]) for record in materialized)
    total_insertions = sum(int(record["insertions"]) for record in materialized)
    total_hallucination_clips = sum(
        bool(record["likely_repetitive_hallucination"]) for record in materialized
    )
    return {
        "utterances": len(materialized),
        "total_reference_words": total_reference_words,
        "total_edits": total_edits,
        "corpus_wer": (
            total_edits / total_reference_words
            if total_reference_words
            else 0.0
        ),
        "total_insertions": total_insertions,
        "likely_hallucination_clips": total_hallucination_clips,
        "likely_hallucination_ids": [
            str(record["clip_id"])
            for record in materialized
            if record["likely_repetitive_hallucination"]
        ],
        "max_consecutive_repeat": max(
            int(record["max_consecutive_repeat"]) for record in materialized
        ),
        "max_repeated_phrase": max(
            int(record["longest_repeated_phrase"]) for record in materialized
        ),
    }
