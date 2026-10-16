"""Offline evaluation helpers for Hinglish ASR baseline results.

Inference is intentionally separated from this module. Remote functions can
write prediction CSVs and local code can score them without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from src.hinglish.metrics import code_switch_bin, english_word_ratio
from src.hinglish.normalization import normalize_hinglish_text
from src.metrics.wer import character_error_rate, word_error_rate


@dataclass(frozen=True)
class HinglishPrediction:
    """One model output and its ground-truth record."""

    clip_id: str
    speaker_id: str
    reference: str
    prediction: str
    ratio_english_words: float = 0.5
    language: str = "hi"
    model_id: str = ""
    device: str = ""


def score_prediction(prediction: HinglishPrediction) -> dict[str, object]:
    """Normalize and score a single Hinglish prediction."""
    reference = normalize_hinglish_text(prediction.reference)
    hypothesis = normalize_hinglish_text(prediction.prediction)
    wer = word_error_rate(reference, hypothesis)
    cer = character_error_rate(reference, hypothesis)
    ratio = max(0.0, min(1.0, prediction.ratio_english_words))
    return {
        "clip_id": prediction.clip_id,
        "speaker_id": prediction.speaker_id,
        "reference_words": wer.ref_len,
        "reference_characters": cer.ref_len,
        "hypothesis_words": wer.hyp_len,
        "hypothesis_characters": cer.hyp_len,
        "wer": wer.error_rate,
        "cer": cer.error_rate,
        "substitutions": wer.substitutions,
        "deletions": wer.deletions,
        "insertions": wer.insertions,
        "total_edits": wer.total_edits,
        "character_edits": cer.total_edits,
        "english_ratio_bin": code_switch_bin(ratio),
        "ratio_english_words": ratio,
    }


def summarize_predictions(
    predictions: Iterable[HinglishPrediction],
) -> dict[str, float | int]:
    """Calculate aggregate rates and per-bin averages."""
    scored = [score_prediction(prediction) for prediction in predictions]
    total_reference_words = sum(int(row["reference_words"]) for row in scored)
    total_reference_characters = sum(int(row["reference_characters"]) for row in scored)
    total_edits = sum(int(row["total_edits"]) for row in scored)
    total_character_edits = sum(int(row["character_edits"]) for row in scored)
    if total_reference_words == 0:
        overall_wer = 0.0
    else:
        overall_wer = total_edits / total_reference_words
    if total_reference_characters == 0:
        overall_cer = 0.0
    else:
        overall_cer = total_character_edits / total_reference_characters

    by_bin: dict[str, list[float]] = {}
    for row in scored:
        by_bin.setdefault(str(row["english_ratio_bin"]), []).append(float(row["wer"]))
    bin_summary = {
        f"wer_{bin_name}": sum(values) / len(values)
        for bin_name, values in by_bin.items()
    }

    return {
        "utterances": len(scored),
        "reference_words": total_reference_words,
        "reference_characters": total_reference_characters,
        "total_edits": total_edits,
        "total_character_edits": total_character_edits,
        "wer": overall_wer,
        "cer": overall_cer,
        "mean_utterance_wer": (sum(float(row["wer"]) for row in scored) / len(scored))
        if scored
        else 0.0,
        "mean_utterance_cer": (sum(float(row["cer"]) for row in scored) / len(scored))
        if scored
        else 0.0,
        "bins": bin_summary,
    }


def normalize_ratio(english_words: int, total_words: int) -> float:
    """Wrapper around code_switch_ratio for adapter callers."""
    return english_word_ratio(english_words, total_words)
