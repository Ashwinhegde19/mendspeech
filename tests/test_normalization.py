"""Tests for the frozen text normalization convention."""

import pandas as pd
import pytest

from src.metrics.normalization import (
    NORMALIZATION_VERSION,
    count_apostrophe_words,
    normalize_text,
    normalization_delta,
)

MANIFEST = "data/benchmark_manifest.csv"


class TestNormalizeText:
    """Each step of the frozen rule."""

    def test_upper_cases(self):
        assert normalize_text("the cat") == "THE CAT"

    def test_removes_apostrophe_without_splitting(self):
        assert normalize_text("MISTER QUILTER'S GOSPEL") == "MISTER QUILTERS GOSPEL"

    def test_apostrophe_word_stays_one_token(self):
        """The key property: no new word boundary is introduced."""
        assert len(normalize_text("QUILTER'S").split()) == 1

    def test_removes_other_punctuation(self):
        assert normalize_text("Hello, world!") == "HELLO WORLD"
        assert normalize_text("a.b;c:d") == "ABCD"

    def test_collapses_whitespace_and_strips(self):
        assert normalize_text("  a   b  ") == "A B"
        assert normalize_text("a\n\tb") == "A B"

    def test_empty_and_whitespace_only(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_punctuation_only_becomes_empty(self):
        assert normalize_text("...") == ""

    def test_digits_preserved(self):
        assert normalize_text("Room 101") == "ROOM 101"

    def test_does_not_expand_contractions(self):
        """'IT'S' normalizes to 'ITS', not 'IT IS'. Frozen and deliberate."""
        assert normalize_text("IT'S") == "ITS"
        assert len(normalize_text("IT'S").split()) == 1

    def test_keep_apostrophe_option(self):
        assert normalize_text("QUILTER'S", keep_apostrophe=True) == "QUILTER'S"

    def test_typographic_apostrophe_handled(self):
        assert normalize_text("QUILTER’S") == "QUILTERS"

    def test_idempotent(self):
        once = normalize_text("  MISTER QUILTER'S, GOSPEL! ")
        assert normalize_text(once) == once

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be a str"):
            normalize_text(None)


class TestNormalizationDelta:
    """Reporting what the convention changed."""

    def test_reports_no_change_for_clean_text(self):
        delta = normalization_delta("THE CAT", "THE CAT")
        assert delta["reference_changed"] is False
        assert delta["hypothesis_changed"] is False

    def test_reports_punctuation_removal(self):
        delta = normalization_delta("MISTER QUILTER'S GOSPEL.", "MISTER QUILTERS GOSPEL")
        assert delta["reference_changed"] is True
        assert delta["hypothesis_changed"] is False
        assert delta["reference_normalized"] == "MISTER QUILTERS GOSPEL"

    def test_carries_version(self):
        assert normalization_delta("a", "b")["version"] == NORMALIZATION_VERSION


class TestApostropheAudit:
    """Evidence that the apostrophe rule matters for this corpus."""

    def test_benchmark_contains_apostrophe_words(self):
        manifest = pd.read_csv(MANIFEST)
        counts = manifest.transcript.map(count_apostrophe_words)
        assert counts.sum() > 0, "frozen benchmark has no apostrophe words"

    def test_counting_is_per_word(self):
        assert count_apostrophe_words("A'S B'S C") == 2
        assert count_apostrophe_words("NO APOSTROPHES") == 0
        assert count_apostrophe_words("") == 0

    def test_rule_changes_no_token_count(self):
        """Deleting the apostrophe must not change the reference word count."""
        manifest = pd.read_csv(MANIFEST)
        for transcript in manifest.transcript:
            assert len(normalize_text(transcript).split()) == len(transcript.split())
