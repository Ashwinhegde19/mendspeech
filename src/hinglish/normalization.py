"""Hinglish transcription normalization.

This module provides a versioned normalizer for Hindi-English code-mixed
ASR transcripts. It does not transliterate Hindi or adapt a reference to a
model output. Its only purpose is to make scoring consistent by applying a
stated Unicode and punctuation convention to both references and hypotheses.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable


HINGLISH_NORMALIZATION_VERSION = "mendspeech.hinglish.v1"


def _is_punctuation(char: str) -> bool:
    """Return whether a character is punctuation or an underscore."""
    return char == "_" or unicodedata.category(char).startswith("P")


def normalize_hinglish_text(
    text: str,
    *,
    casefold: bool = True,
) -> str:
    """Normalize a Hinglish transcript using a fixed Unicode convention.

    Args:
        text: Raw transcript text.
        casefold: If true, uppercase letters. Devanagari is mostly unchanged
            by Python Unicode casing.

    Returns:
        Normalized transcript text.

    Raises:
        ValueError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u2018", "'")

    # Remove apostrophes without splitting a word, matching the existing
    # MendSpeech convention.
    normalized = normalized.replace("'", "")

    # Replace other punctuation with whitespace. Using Unicode categories
    # avoids treating Devanagari matras and marks as punctuation.
    normalized = "".join(
        " " if _is_punctuation(char) else char for char in normalized
    )

    if casefold:
        normalized = normalized.upper()

    return " ".join(normalized.split())


def normalize_tokens(tokens: Iterable[str]) -> list[str]:
    """Normalize a token sequence while preserving token boundaries."""
    return [normalize_hinglish_text(token) for token in tokens]


def is_empty_after_normalization(text: str) -> bool:
    """Return whether text has no word content after normalization."""
    return not bool(normalize_hinglish_text(text))


def normalization_version() -> str:
    """Return the active Hinglish normalization profile."""
    return HINGLISH_NORMALIZATION_VERSION
