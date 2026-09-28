"""Word and character error rates, with the edit operations behind them.

Day 10, concepts 1 and 2. One definition underlies both rates:

    error_rate = total_edits / ref_len

where ``total_edits`` is the Levenshtein distance (the fewest single-unit
insertions, deletions and substitutions that turn the reference into the
hypothesis) and ``ref_len`` is the number of reference units. WER counts
words; CER counts characters.

:func:`levenshtein_counts` returns the distance *and* attributes it to
individual operations, so a dropped word is distinguishable from a wrong
word. The three counts always sum to ``total_edits``, which is why every
rate computed here still matches a distance-only alignment. WER and CER
share that one routine and cannot disagree by construction.

One caveat on the breakdown: when several alignments tie for the optimal
distance, how the distance splits across S/D/I depends on the backtrace
order, which is fixed (diagonal, then up, then left) and therefore
deterministic. The *total* is alignment-independent; the distribution is not.
Report the total as the metric and treat the split as one valid reading.

The alignment must be a true Levenshtein alignment, not a positional
comparison: comparing tokens index-by-index charges one deletion against
every following token and inflates the rate.

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
from typing import List, Sequence

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
    substitutions: int
    deletions: int
    insertions: int

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


@dataclass
class EditCounts:
    """Per-operation counts from one alignment.

    The three counts always sum to the Levenshtein distance, so
    ``total_edits`` derived from this record equals the distance a
    distance-only alignment would report. Knowing *which* operations
    occurred is what separates a dropped word from a wrong word.

    Attributes:
        substitutions: Reference units replaced by a different unit.
        deletions: Reference units absent from the hypothesis (the
            recognizer missed them).
        insertions: Hypothesis units absent from the reference (the
            recognizer hallucinated them).
        ref_len: Number of reference units.
        hyp_len: Number of hypothesis units.
    """

    substitutions: int
    deletions: int
    insertions: int
    ref_len: int
    hyp_len: int

    @property
    def total_edits(self) -> int:
        """Sum of substitutions, deletions and insertions.

        Equal to the Levenshtein distance between the two sequences.
        """
        return self.substitutions + self.deletions + self.insertions

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return {**asdict(self), "total_edits": self.total_edits}


def levenshtein_counts(ref: Sequence[str], hyp: Sequence[str]) -> EditCounts:
    """Align two token sequences and count each kind of edit.

    Fills the full (n+1) x (m+1) distance table, then backtraces from the
    bottom-right corner to attribute the distance to individual operations.
    Ties resolve in a fixed order (diagonal, then up, then left), so the
    counts are deterministic for a given input pair.

    Args:
        ref: Reference tokens, in order. The ground-truth side.
        hyp: Hypothesis tokens, in order. The produced side.

    Returns:
        EditCounts whose fields sum to the Levenshtein distance. Inputs are
        not mutated.
    """
    ref_tokens = list(ref)
    hyp_tokens = list(hyp)
    n, m = len(ref_tokens), len(hyp_tokens)

    # dist[i][j] is the edit distance between ref[:i] and hyp[:j].
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dist[i - 1][j - 1] + (ref_tokens[i - 1] != hyp_tokens[j - 1])
            dist[i][j] = min(diagonal, dist[i - 1][j] + 1, dist[i][j - 1] + 1)

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dist[i][j]
            == dist[i - 1][j - 1] + (ref_tokens[i - 1] != hyp_tokens[j - 1])
        ):
            if ref_tokens[i - 1] != hyp_tokens[j - 1]:
                substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        ref_len=n,
        hyp_len=m,
    )


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
    counts = levenshtein_counts(ref, hyp)
    total_edits = counts.total_edits

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
        substitutions=counts.substitutions,
        deletions=counts.deletions,
        insertions=counts.insertions,
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
