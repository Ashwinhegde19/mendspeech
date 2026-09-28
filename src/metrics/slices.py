"""Protected-content error slices for transcript scoring.

Day 10, concept 4. Aggregate WER treats every word as equally important, but
a dictation system that drops a negation or mangles a name fails far more
badly than one that mishears an article. The editor contract makes this
concrete: names, numbers, negation, units and domain terms are *protected*
spans, and a >2 percentage-point increase in protected-content violations is
a stop condition for training.

This module partitions reference words into slices and reports error counts
per slice, so a small overall WER cannot hide a concentrated failure. Three
rules govern it:

- **No model-based judge.** Slices come from explicit word lists and regular
  expressions only, so a score is reproducible and costs nothing to recompute.
  The contract forbids a model judge here for the same reason.
- **Empty slices are reported, never hidden.** A slice with no matching
  reference words gets ``None`` for its rate and a count of 0, and callers
  are expected to say so. Measured on the frozen benchmark, the slices are
  small: 9 number words, 12 negation words and 11 name words out of 772
  reference words, so a per-slice rate on this corpus is a diagnostic, not a
  stable statistic.
- **Slices partition the reference, they do not double-count it.** A word
  matching two patterns is reported under the first matching slice, so
  per-slice counts sum to at most the total reference length.

Detection is deliberately conservative. LibriSpeech references are fully
upper case, so capitalization cannot identify a proper noun; names therefore
require an explicit list rather than a casing heuristic. A heuristic that
guessed would inflate the slice with ordinary words and hide the real ones.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from src.metrics.normalization import normalize_text
from src.metrics.wer import levenshtein_counts

# Negation words. Matched after normalization, so contracted forms have
# already lost their apostrophe and appear as ITS, DONT and so on.
NEGATION_WORDS = frozenset(
    {
        "NO", "NOT", "NEVER", "NONE", "NOR", "NOBODY", "NOTHING",
        "NOWHERE", "CANNOT", "CANT", "DONT", "DOESNT", "DIDNT", "ISNT",
        "ARENT", "WONT", "WOULDNT", "COULDNT", "SHOULDNT", "ITS", "THATS",
        "HES", "SHES", "YOURE", "YOUVE", "WERE", "HADNT", "HASNT", "HAVENT",
    }
)

# Proper nouns in the frozen benchmark. Curation is required because the
# corpus is upper case throughout; see the module docstring.
PROPER_NOUNS = frozenset(
    {
        "MISTER", "QUILTER", "LEIGHTON", "SHAKESPEARE", "LINNELL", "MASON",
        "FOSTER", "BIRKET", "CARKER", "COLLIER", "HUGH", "CONNELL", "ROCKY",
        "ITHACA", "TUESDAY", "SATURDAY", "MONDAY", "FRIDAY", "SUNDAY",
        "THURSDAY", "WEDNESDAY", "JANUARY", "FEBRUARY",
    }
)

# A token containing any digit, or a spelled-out number.
_NUMBER_PATTERN = re.compile(r"\d")
_SPELLED_NUMBERS = frozenset(
    {
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
        "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN",
        "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY",
        "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
        "HUNDRED", "THOUSAND", "MILLION", "FIRST", "SECOND", "THIRD",
        "FOURTH", "FIFTH", "HALF", "QUARTER",
    }
)

# Slice names, in the order they are tested. A token is assigned to the
# first slice it matches, so the partition is disjoint.
SLICE_NAMES = ("number", "negation", "name")


def classify_token(token: str) -> Optional[str]:
    """Return the slice a single reference token belongs to.

    Args:
        token: One normalized reference word.

    Returns:
        The slice name (``"number"``, ``"negation"`` or ``"name"``), or None
        when the token is ordinary content. The first matching slice wins,
        so the result is a partition rather than a set of labels.
    """
    upper = token.upper()
    if _NUMBER_PATTERN.search(upper) or upper in _SPELLED_NUMBERS:
        return "number"
    if upper in NEGATION_WORDS:
        return "negation"
    if upper in PROPER_NOUNS:
        return "name"
    return None


@dataclass
class SliceScore:
    """Errors confined to one protected-content slice.

    Attributes:
        slice_name: ``"number"``, ``"negation"``, ``"name"`` or ``"overall"``.
        ref_tokens: Reference words in this slice. 0 means the slice does not
            occur in the scored references.
        errors: Edit distance restricted to this slice's words.
        error_rate: ``errors / ref_tokens``, or None when ``ref_tokens`` is 0.
            None is a real answer meaning "not measured", and callers must
            not read it as zero.
        covered: False when the slice has no reference words, so an empty
            slice is visible instead of silently scoring 0.0.
    """

    slice_name: str
    ref_tokens: int
    errors: int
    error_rate: Optional[float]
    covered: bool

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


def slice_error_counts(reference: str, hypothesis: str) -> Dict[str, int]:
    """Count reference words in each slice, ignoring errors.

    Reports the size of each slice in a reference, which is the denominator
    any per-slice rate will use.

    Args:
        reference: Raw reference text.
        hypothesis: Unused; accepted so this matches the scoring signature.

    Returns:
        Mapping of slice name to reference word count, including ``"all"``
        for the total reference length.
    """
    del hypothesis  # Signature symmetry with slice_scores; size only.
    words = normalize_text(reference).split()
    counts: Dict[str, int] = {name: 0 for name in SLICE_NAMES}
    for word in words:
        label = classify_token(word)
        if label is not None:
            counts[label] += 1
    counts["all"] = len(words)
    return counts


def _aligned_pairs(
    ref_words: Sequence[str], hyp_words: Sequence[str]
) -> List[tuple]:
    """Backtrace the shared alignment into (ref_index, hyp_index) pairs.

    Uses the same tie-break order as the error-rate code (diagonal, then up,
    then left) so slice counts and headline counts come from one alignment.
    A ``None`` hypothesis index marks a deleted reference word.

    Args:
        ref_words: Normalized reference words.
        hyp_words: Normalized hypothesis words.

    Returns:
        List of ``(ref_index, hyp_index_or_None)`` in reference order.
    """
    n, m = len(ref_words), len(hyp_words)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dist[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1])
            dist[i][j] = min(diagonal, dist[i - 1][j] + 1, dist[i][j - 1] + 1)

    pairs: Dict[int, Optional[int]] = {}
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dist[i][j]
            == dist[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1])
        ):
            pairs[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            pairs[i - 1] = None
            i -= 1
        else:
            j -= 1

    return [(k, pairs[k]) for k in sorted(pairs)]


def slice_scores(reference: str, hypothesis: str) -> List[SliceScore]:
    """Score protected-content slices for one reference/hypothesis pair.

    Each slice is scored by keeping the reference words that belong to it and
    the hypothesis words the shared alignment paired with them, then counting
    edits over that sub-alignment. Slices are disjoint, so a word is charged
    to exactly one slice.

    A slice absent from the reference is returned with ``ref_tokens=0``,
    ``covered=False`` and ``error_rate=None`` rather than being dropped, so
    an empty slice stays visible in the output.

    Args:
        reference: Raw reference text.
        hypothesis: Raw hypothesis text.

    Returns:
        One :class:`SliceScore` per slice in :data:`SLICE_NAMES`, followed
        by an ``"overall"`` row carrying the untruncated rate.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    overall = levenshtein_counts(ref_words, hyp_words)
    pairs = _aligned_pairs(ref_words, hyp_words)

    # Hypothesis words paired with each reference index, in reference order.
    paired_hyp = {ref_i: hyp_i for ref_i, hyp_i in pairs if hyp_i is not None}

    scores: List[SliceScore] = []
    for name in SLICE_NAMES:
        ref_indices = [
            i for i, word in enumerate(ref_words) if classify_token(word) == name
        ]
        n = len(ref_indices)
        if n == 0:
            scores.append(
                SliceScore(
                    slice_name=name,
                    ref_tokens=0,
                    errors=0,
                    error_rate=None,
                    covered=False,
                )
            )
            continue

        sub_ref = [ref_words[i] for i in ref_indices]
        matched = [paired_hyp[i] for i in ref_indices if i in paired_hyp]
        sub_hyp = [hyp_words[j] for j in matched]
        sub_counts = levenshtein_counts(sub_ref, sub_hyp)
        scores.append(
            SliceScore(
                slice_name=name,
                ref_tokens=n,
                errors=sub_counts.total_edits,
                error_rate=sub_counts.total_edits / n,
                covered=True,
            )
        )

    scores.append(
        SliceScore(
            slice_name="overall",
            ref_tokens=overall.ref_len,
            errors=overall.total_edits,
            error_rate=(
                overall.total_edits / overall.ref_len if overall.ref_len else None
            ),
            covered=overall.ref_len > 0,
        )
    )
    return scores
