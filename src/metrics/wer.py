"""Word and character error rates over aligned text.

Day 10, concept 1: WER and CER. Two error rates, one definition:

    error_rate = total_edits / ref_len

where ``total_edits`` is the Levenshtein distance (the fewest single-unit
insertions, deletions and substitutions that turn the reference into the
hypothesis) and ``ref_len`` is the number of reference units. WER counts
words; CER counts characters.

Both rates share one alignment routine, so a word rate and a character rate
over the same texts cannot disagree by construction. The alignment must be a
true Levenshtein alignment, not a positional comparison: comparing tokens
index-by-index charges one deletion against every following token and
inflates the rate.

The empty-reference case is a stated convention, not an accident: with no
reference units the rate is 0.0 when the hypothesis is also empty and
:data:`EMPTY_REFERENCE_RATE` otherwise. It never raises and never returns NaN.

Scope note: this module scores text that is already in comparable form. It
does not lowercase, strip punctuation, or otherwise rewrite its inputs;
normalization is a separate frozen convention applied by the caller. Tokens
are compared exactly as given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

# Rate reported when the reference is empty but the hypothesis is not.
# Named so callers compare against one documented sentinel, not a bare inf.
EMPTY_REFERENCE_RATE = float("inf")

# Accepted units for ErrorRate.metric. CER scores characters, WER scores words.
VALID_METRICS = ("wer", "cer")


@dataclass
class ErrorRate:
    """One scored hypothesis: its rate plus the counts the rate came from.

    Attributes:
        metric: Unit that was scored, ``"wer"`` for words or ``"cer"`` for
            characters.
        error_rate: ``total_edits / ref_len`` as a fraction, not a percentage.
            0.0 when the hypothesis matches the reference. For an empty
            reference it is 0.0 if the hypothesis is empty and
            :data:`EMPTY_REFERENCE_RATE` otherwise.
        total_edits: Fewest single-unit edits that turn the reference into
            the hypothesis (the Levenshtein distance).
        ref_len: Number of reference units, in the unit named by ``metric``.
            This is the denominator.
        hyp_len: Number of hypothesis units.
    """

    metric: str
    error_rate: float
    total_edits: int
    ref_len: int
    hyp_len: int

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Return the Levenshtein distance between two token sequences.

    Dynamic programming over the full (n+1) x (m+1) table, where
    ``dist[i][j]`` is the distance between ``ref[:i]`` and ``hyp[:j]``. Only
    the final distance is returned; the per-operation breakdown is not
    derived here.

    Args:
        ref: Reference tokens, in order.
        hyp: Hypothesis tokens, in order.

    Returns:
        The fewest single-token insertions, deletions and substitutions that
        turn ``ref`` into ``hyp``. 0 when the sequences are identical.
    """
    n, m = len(ref), len(hyp)

    # Rolling row: dist_row[j] is dist[i][j] for the current i.
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            substitution = previous[j - 1] + (ref[i - 1] != hyp[j - 1])
            current[j] = min(substitution, previous[j] + 1, current[j - 1] + 1)
        previous = current
    return previous[m]


def _build_rate(
    ref: List[str], hyp: List[str], metric: str
) -> ErrorRate:
    """Score one aligned pair and wrap the result as an ErrorRate.

    Args:
        ref: Reference units, in order.
        hyp: Hypothesis units, in order.
        metric: ``"wer"`` or ``"cer"``.

    Returns:
        ErrorRate with the rate and the counts behind it.
    """
    ref_len, hyp_len = len(ref), len(hyp)
    total_edits = _levenshtein(ref, hyp)

    if ref_len == 0:
        rate = 0.0 if hyp_len == 0 else EMPTY_REFERENCE_RATE
    else:
        rate = total_edits / ref_len

    return ErrorRate(
        metric=metric,
        error_rate=rate,
        total_edits=total_edits,
        ref_len=ref_len,
        hyp_len=hyp_len,
    )


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """Score a hypothesis against a reference in words.

    Splits both strings on whitespace and aligns the word sequences. Case and
    punctuation are compared as given; the caller is responsible for having
    normalized both texts the same way.

    Args:
        reference: Ground-truth text.
        hypothesis: Produced text to score.

    Returns:
        ErrorRate where ``ref_len`` counts words and ``error_rate`` is the
        Levenshtein distance over the reference word count. For the empty
        string the rate is 0.0.

    Raises:
        ValueError: If either argument is not a string.

    Example:
        >>> word_error_rate("the cat sat", "the bat sat").error_rate
        0.3333333333333333
    """
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("reference and hypothesis must both be str")
    return _build_rate(reference.split(), hypothesis.split(), "wer")


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """Score a hypothesis against a reference in characters.

    Aligns the two strings character by character, including spaces, which
    the caller controls by choosing what to normalize beforehand.

    Args:
        reference: Ground-truth text, already normalized.
        hypothesis: Produced text, already normalized the same way.

    Returns:
        ErrorRate where ``ref_len`` counts characters and ``error_rate`` is
        the Levenshtein distance over the reference character count.

    Raises:
        ValueError: If either argument is not a string.

    Example:
        >>> character_error_rate("abcd", "abxd").error_rate
        0.25
    """
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("reference and hypothesis must both be str")
    return _build_rate(list(reference), list(hypothesis), "cer")
