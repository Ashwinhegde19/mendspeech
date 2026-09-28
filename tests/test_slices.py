"""Tests for protected-content error slices (Day 10, concept 4)."""

import pandas as pd
import pytest

from src.metrics.slices import (
    NEGATION_WORDS,
    PROPER_NOUNS,
    SLICE_NAMES,
    SliceScore,
    classify_token,
    slice_error_counts,
    slice_scores,
)
from src.metrics.wer import word_error_rate

MANIFEST = "data/benchmark_manifest.csv"


def by_name(scores):
    """Index slice scores by slice name for readable assertions."""
    return {s.slice_name: s for s in scores}


class TestClassifyToken:
    """Single-token classification."""

    def test_ordinary_words_are_unclassified(self):
        for token in ["THE", "CAT", "AND", "QUICKLY"]:
            assert classify_token(token) is None

    def test_digits_are_numbers(self):
        assert classify_token("3") == "number"
        assert classify_token("101") == "number"
        assert classify_token("ROOM101") == "number"

    def test_spelled_numbers_are_numbers(self):
        assert classify_token("THREE") == "number"
        assert classify_token("TWENTY") == "number"
        assert classify_token("FIRST") == "number"

    def test_negation_words(self):
        for token in ["NO", "NOT", "NEVER", "NOR", "CANNOT"]:
            assert classify_token(token) == "negation"

    def test_contraction_after_normalization_is_negation(self):
        """Apostrophes are deleted first, so IT'S arrives as ITS."""
        assert classify_token("ITS") == "negation"

    def test_curated_proper_nouns(self):
        for token in ["MISTER", "QUILTER", "SHAKESPEARE"]:
            assert classify_token(token) == "name"


class TestSliceErrorCounts:
    """Slice sizes, which are the per-slice denominators."""

    def test_counts_each_slice(self):
        counts = slice_error_counts("MISTER QUILTER HAS 3 ROOMS AND NO TIME", "")
        assert counts["name"] == 2
        assert counts["number"] == 1
        assert counts["negation"] == 1
        assert counts["all"] == 8

    def test_no_protected_content(self):
        counts = slice_error_counts("THE CAT SAT", "")
        assert counts["name"] == 0
        assert counts["number"] == 0
        assert counts["negation"] == 0

    def test_counts_sum_at_most_all(self):
        """Slices partition the reference, so they cannot exceed it."""
        counts = slice_error_counts("MISTER HAS THREE ROOMS AND NO TIME", "")
        assert counts["name"] + counts["number"] + counts["negation"] <= counts["all"]


class TestSliceScores:
    """Per-slice error rates from the shared alignment."""

    def test_perfect_match_scores_zero(self):
        scores = by_name(
            slice_scores("I HAVE 3 ROOMS AND NO TIME", "I HAVE 3 ROOMS AND NO TIME")
        )
        assert scores["overall"].error_rate == 0.0
        assert scores["number"].error_rate == 0.0
        assert scores["negation"].error_rate == 0.0

    def test_empty_slice_reports_none_not_zero(self):
        """An unmeasured slice must not read as a perfect score."""
        scores = by_name(slice_scores("I HAVE TIME", "I HAVE TIME"))
        assert scores["name"].ref_tokens == 0
        assert scores["name"].error_rate is None
        assert scores["name"].covered is False
        assert scores["number"].error_rate is None

    def test_every_slice_always_present(self):
        """Slices are reported even when empty, so gaps stay visible."""
        names = [s.slice_name for s in slice_scores("THE CAT", "THE CAT")]
        for expected in SLICE_NAMES:
            assert expected in names
        assert "overall" in names

    def test_wrong_number_is_charged_to_number_slice(self):
        scores = by_name(slice_scores("I HAVE 3 ROOMS", "I HAVE 5 ROOMS"))
        assert scores["number"].errors == 1
        assert scores["number"].error_rate == 1.0

    def test_wrong_name_is_charged_to_name_slice(self):
        scores = by_name(slice_scores("MISTER QUILTER IS HERE", "MYSTERY QUILTER IS HERE"))
        assert scores["name"].ref_tokens == 2
        assert scores["name"].errors == 1
        assert scores["name"].error_rate == pytest.approx(0.5)

    def test_normalization_applied_to_both_sides(self):
        """Punctuation and case must not create slice errors."""
        scores = by_name(slice_scores("Mister Quilter's, no.", "MISTER QUILTERS NO"))
        assert scores["name"].error_rate == 0.0
        assert scores["negation"].error_rate == 0.0

    def test_slices_do_not_double_count(self):
        """Sum of slice reference tokens stays within the reference."""
        ref = "MISTER QUILTER HAS 3 ROOMS AND NO TIME"
        hyp = "MYSTERY QUILTERS HAS 5 ROOMS AND NO TIME"
        scores = by_name(slice_scores(ref, hyp))
        sliced = sum(scores[n].ref_tokens for n in SLICE_NAMES)
        assert sliced <= scores["overall"].ref_tokens

    def test_returns_slice_score_records(self):
        for score in slice_scores("MISTER HAS NO TIME", "MISTER HAS NO TIME"):
            assert isinstance(score, SliceScore)
            assert set(score.to_dict()) == {
                "slice_name",
                "ref_tokens",
                "errors",
                "error_rate",
                "covered",
            }

    def test_empty_reference_reports_uncovered_overall(self):
        scores = by_name(slice_scores("", "SOMETHING"))
        assert scores["overall"].ref_tokens == 0
        assert scores["overall"].error_rate is None
        assert scores["overall"].covered is False

    def test_deterministic(self):
        ref, hyp = "MISTER HAS 3 ROOMS AND NO TIME", "MISTER HAS 4 ROOMS"
        assert slice_scores(ref, hyp) == slice_scores(ref, hyp)


class TestSliceCorpusAudit:
    """What the frozen benchmark can and cannot support."""

    def test_benchmark_has_no_digit_tokens(self):
        """Digits are absent, so number detection must also accept words."""
        manifest = pd.read_csv(MANIFEST)
        digits = [
            w
            for t in manifest.transcript
            for w in t.split()
            if any(c.isdigit() for c in w)
        ]
        assert digits == []

    def test_number_slice_uses_spelled_out_numbers(self):
        """The numeric slice is non-empty only via spelled-out forms."""
        manifest = pd.read_csv(MANIFEST)
        counts = slice_error_counts(" ".join(manifest.transcript), "")
        assert counts["number"] > 0, "spelled-out numbers not detected"

    def test_all_slices_present_but_small(self):
        """Slice coverage is thin, so callers must not over-read the rates."""
        manifest = pd.read_csv(MANIFEST)
        counts = slice_error_counts(" ".join(manifest.transcript), "")
        for name in SLICE_NAMES:
            assert counts[name] > 0, f"{name} slice unexpectedly empty"
            assert counts[name] < 0.05 * counts["all"], f"{name} slice too large"

    def test_perfect_score_is_zero_and_still_covered(self):
        """A clean run reports 0.0 with real denominators, not None."""
        manifest = pd.read_csv(MANIFEST)
        reference = " ".join(manifest.transcript)
        scores = by_name(slice_scores(reference, reference))
        assert scores["overall"].error_rate == 0.0
        for name in SLICE_NAMES:
            assert scores[name].covered is True
            assert scores[name].error_rate == 0.0
            assert scores[name].ref_tokens > 0

    def test_text_without_protected_content_leaves_slices_empty(self):
        """Slices absent from the text must report None, not 0.0."""
        scores = by_name(slice_scores("THE CAT SAT ON THE MAT", "THE CAT SAT"))
        assert scores["name"].error_rate is None
        assert scores["negation"].error_rate is None
        assert scores["number"].error_rate is None


    def test_dropped_negation_is_charged_to_negation_slice(self):
        scores = by_name(slice_scores("I HAVE NO TIME", "I HAVE TIME"))
        assert scores["negation"].errors == 1
        assert scores["negation"].error_rate == 1.0

    def test_overall_matches_word_error_rate(self):
        """Slice view and headline metric must not disagree."""
        ref = "MISTER QUILTER HAS 3 ROOMS AND NO TIME"
        hyp = "MYSTERY QUILTERS HAS 5 ROOMS AND NO TIME"
        overall = by_name(slice_scores(ref, hyp))["overall"]
        headline = word_error_rate(ref, hyp)
        assert overall.ref_tokens == headline.ref_len
        assert overall.errors == headline.total_edits
        assert overall.error_rate == pytest.approx(headline.error_rate)


    def test_lowercase_is_accepted(self):
        assert classify_token("no") == "negation"
        assert classify_token("mister") == "name"

    def test_slices_are_disjoint(self):
        """A token appears in at most one curated list."""
        for token in set(NEGATION_WORDS) | set(PROPER_NOUNS):
            in_neg = token in NEGATION_WORDS
            in_name = token in PROPER_NOUNS
            assert not (in_neg and in_name), f"{token} in two lists"
