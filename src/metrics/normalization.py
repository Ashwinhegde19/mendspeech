"""Frozen text normalization for error-rate scoring.

Day 10, concept 3. Scoring is only meaningful when the reference and the
hypothesis are compared under one agreed convention. This module is that
convention, and it is deliberately the *only* place text is rewritten before
scoring.

The rule applied by :func:`normalize_text` is:

1. Case-fold to upper case, so ``"the"`` and ``"THE"`` are one token. LibriSpeech
   references are upper case and the CTC decoder emits upper case, but the
   rule does not rely on that holding for every future corpus.
2. Remove apostrophes *without* splitting the word, so ``"QUILTER'S"``
   becomes ``"QUILTERS"`` rather than two tokens. This is the standard
   LibriSpeech evaluation convention and it matters: 12 of the 772 reference
   words in the frozen benchmark are apostrophe forms, and turning the
   apostrophe into a space would both change the word count and charge a
   spurious insertion on every one of them.
3. Remove other punctuation, keeping word boundaries intact.
4. Collapse runs of whitespace to a single space and strip the ends, so
   trailing punctuation or a stray space cannot register as an edit.

The rule is intentionally *not* aggressive. It does not expand contractions
(``"IT'S"`` stays ``"ITS"``, it does not become ``"IT IS"``), does not rewrite
numbers, and does not remove filler words. Those choices change the meaning
of the metric and belong to a stated decision, not to a cleanup function.

Normalization is a scoring-time convention, not a transcript cleanup step:
the raw ASR text stays available untouched for the editor and for human
review, as required by the quality contract.
"""

from __future__ import annotations

import re
import string
from typing import Optional

# The frozen normalization profile, named so a result file can record which
# convention produced it. Change this only alongside a protocol amendment.
NORMALIZATION_VERSION = "mendspeech.v1"

# Punctuation removed outright, other than the apostrophe which is deleted
# by APOSTROPHE_PATTERN first so the word is preserved.
_PUNCTUATION = set(string.punctuation) - {"'"}

_APOSTROPHE_PATTERN = re.compile(r"['\u2019]")

# Whitespace runs, including non-breaking space, collapsed to one space.
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(
    text: str,
    *,
    keep_apostrophe: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize text to the frozen scoring convention.

    Applies, in order: upper-casing, apostrophe removal, other punctuation
    removal, and whitespace collapsing. See the module docstring for the
    rationale behind each step.

    Args:
        text: Raw reference or hypothesis text.
        keep_apostrophe: Retain apostrophes instead of deleting them. Used to
            measure the cost of the convention itself; scoring should leave
            this False.
        collapse_whitespace: Collapse whitespace runs to single spaces and
            strip the ends. Disabling this is for diagnostics only.

    Returns:
        The normalized text. Never returns None, and never raises for
        ordinary text.

    Raises:
        ValueError: If ``text`` is not a string.

    Example:
        >>> normalize_text("  Mister Quilter's  Gospel. ")
        "MISTER QUILTERS GOSPEL"
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a str, got {type(text).__name__}")

    normalized = text.upper()

    if not keep_apostrophe:
        normalized = _APOSTROPHE_PATTERN.sub("", normalized)

    normalized = "".join(
        char for char in normalized if char not in _PUNCTUATION
    )

    if collapse_whitespace:
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()

    return normalized


def normalization_delta(reference: str, hypothesis: str) -> dict:
    """Report what normalization changed on both sides of a pair.

    Useful for showing why a raw comparison and a normalized comparison
    disagree, and for confirming that a corpus needs no normalization.

    Args:
        reference: Raw reference text.
        hypothesis: Raw hypothesis text.

    Returns:
        Dict with the raw and normalized text for each side, plus boolean
        flags for whether normalization altered either side.
    """
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    return {
        "reference_raw": reference,
        "reference_normalized": norm_ref,
        "hypothesis_raw": hypothesis,
        "hypothesis_normalized": norm_hyp,
        "reference_changed": norm_ref != reference,
        "hypothesis_changed": norm_hyp != hypothesis,
        "version": NORMALIZATION_VERSION,
    }


def count_apostrophe_words(text: str) -> int:
    """Count whitespace-separated words containing an apostrophe.

    Reports how many words in a corpus the apostrophe rule actually affects,
    which is the evidence for keeping the rule rather than assuming it.

    Args:
        text: Raw text to inspect.

    Returns:
        Number of words containing at least one apostrophe or typographic
        apostrophe. 0 for text without any.

    Example:
        >>> count_apostrophe_words("MISTER QUILTER'S GOSPEL")
        1
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a str, got {type(text).__name__}")
    return sum(1 for word in text.split() if _APOSTROPHE_PATTERN.search(word))
