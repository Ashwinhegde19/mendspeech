"""Hinglish evaluation metrics helpers.

These utilities extend MendSpeech's reusable WER and normalization
conventions with code-switch-specific grouping.
"""


from __future__ import annotations

from typing import Iterable


def english_word_ratio(english_words: int, total_words: int) -> float:
    """Return the English word ratio.

    This is a language-composition measure, not a code-switch transition
    count. Switch-count metrics are unavailable in the current transcripts.
    """
    if total_words <= 0:
        raise ValueError("total_words must be positive")
    if english_words < 0 or english_words > total_words:
        raise ValueError("english_words must be between 0 and total_words")
    return english_words / total_words


# Compatibility alias. The primary metric name describes language composition.
code_switch_ratio = english_word_ratio


def code_switch_bin(ratio: float, *, low: float = 0.2, high: float = 0.8) -> str:
    """Bucket a code-switch ratio into a coarse density class."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if ratio <= low:
        return "hindi-dominant"
    if ratio >= high:
        return "english-dominant"
    return "code-switched"


def group_code_switch_bin(ratio: float, *, low: float = 0.2, high: float = 0.8) -> str:
    """Alias for code_switch_bin for future metric naming clarity."""
    return code_switch_bin(ratio, low=low, high=high)


def language_aware_word_error_rate(word_errors: int, reference_words: int) -> float:
    """Compute a plain word-level rate without replacing WER alignment."""
    if reference_words < 0 or word_errors < 0:
        raise ValueError("word errors and reference words must be non-negative")
    if reference_words == 0:
        return 0.0 if word_errors == 0 else float("inf")
    return word_errors / reference_words


def average_by_bin(
    rows: Iterable[dict[str, object]],
    bin_field: str,
    metric_field: str,
) -> dict[str, float]:
    """Average a metric by a code-switch bin using only provided rows."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        bin_name = str(row[bin_field])
        value = float(row[metric_field])
        totals[bin_name] = totals.get(bin_name, 0.0) + value
        counts[bin_name] = counts.get(bin_name, 0) + 1
    return {
        bin_name: totals[bin_name] / counts[bin_name]
        for bin_name in totals
        if counts[bin_name]
    }
