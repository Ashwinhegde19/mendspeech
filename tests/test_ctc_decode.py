"""Unit tests for CTC collapse, legal path enumeration, and first-principles decoding."""

import pytest
import torch

from src.asr.ctc_decode import (
    CTCAlignmentResult,
    FirstPrinciplesCTCDecoder,
    ctc_collapse,
    ctc_decode_transcript,
    enumerate_legal_paths,
)


def test_ctc_collapse_empty_and_single():
    """Verify edge cases: empty sequence and single-token sequences."""
    assert ctc_collapse([], blank_token="-") == []
    assert ctc_collapse(["-"], blank_token="-") == []
    assert ctc_collapse(["-", "-", "-"], blank_token="-") == []
    assert ctc_collapse(["A"], blank_token="-") == ["A"]
    assert ctc_collapse(["A", "A", "A"], blank_token="-") == ["A"]


def test_ctc_collapse_deliberate_break_and_fix_repeated_letters():
    """Demonstrate how repeated letters break without a blank delimiter, and how blank fixes it.

    Target word: "BEE"
    """
    blank = "-"

    # Without blank: the second 'E' is merged and destroyed -> "BE"
    broken_alignment = ["B", "B", "E", "E", "E"]
    collapsed_broken = ctc_collapse(broken_alignment, blank_token=blank)
    assert collapsed_broken == ["B", "E"]
    assert "".join(collapsed_broken) != "BEE"

    # With blank separator: the second 'E' survives the collapse -> "BEE"
    fixed_alignment = ["B", "B", "-", "E", "E", "-", "E", "E"]
    collapsed_fixed = ctc_collapse(fixed_alignment, blank_token=blank)
    assert collapsed_fixed == ["B", "E", "E"]
    assert "".join(collapsed_fixed) == "BEE"


def test_ctc_collapse_words_book_and_hello():
    """Verify collapse on double-letter words: BOOK and HELLO."""
    blank = "-"

    # BOOK
    book_path = ["B", "B", "-", "O", "O", "-", "O", "O", "K"]
    assert ctc_collapse(book_path, blank_token=blank) == ["B", "O", "O", "K"]

    # HELLO
    hello_path = ["H", "E", "E", "-", "L", "L", "-", "L", "O", "O"]
    assert ctc_collapse(hello_path, blank_token=blank) == ["H", "E", "L", "L", "O"]


def test_ctc_collapse_integer_class_ids():
    """Verify that ctc_collapse works symmetrically on integer class IDs."""
    blank_id = 0
    # Target: [1, 2, 2, 3]
    raw_ids = [1, 1, blank_id, 2, 2, blank_id, blank_id, 2, 3, 3]
    collapsed = ctc_collapse(raw_ids, blank_token=blank_id)
    assert collapsed == [1, 2, 2, 3]


def test_ctc_decode_transcript_spaces():
    """Verify conversion of word separator symbols to whitespace."""
    tokens = ["-", "H", "I", "-", "|", "-", "T", "H", "E", "R", "E", "-"]
    transcript = ctc_decode_transcript(tokens, blank_symbol="-", word_separator="|")
    assert transcript == "HI THERE"


def test_enumerate_legal_paths_cat():
    """Enumerate all valid alignments for 'CAT' of length T=4 and verify each one."""
    target = ["C", "A", "T"]
    paths = enumerate_legal_paths(target=target, time_steps=4, blank_symbol="-")

    # For T=4 and 3 distinct characters, valid paths are:
    # Repeat one letter: CCAT, CAAT, CATT (3 paths)
    # Insert one blank: -CAT, C-AT, CA-T, CAT- (4 paths)
    # Total = 7 paths
    expected_paths = [
        ["-", "C", "A", "T"],
        ["C", "-", "A", "T"],
        ["C", "C", "A", "T"],
        ["C", "A", "-", "T"],
        ["C", "A", "A", "T"],
        ["C", "A", "T", "-"],
        ["C", "A", "T", "T"],
    ]

    assert len(paths) == len(expected_paths)
    for p in paths:
        assert p in expected_paths
        # Every enumerated path must collapse exactly to CAT
        assert ctc_collapse(p, blank_token="-") == target


def test_enumerate_legal_paths_repeated_letters_bee():
    """Enumerate valid alignments for 'BEE' of length T=4.

    Because 'E' is repeated, the two 'E's must be separated by a blank!
    Therefore, ['B', 'B', 'E', 'E'] is NOT a legal path (collapses to ['B', 'E']).
    The ONLY legal 4-step path for 'BEE' is ['B', 'E', '-', 'E'].
    """
    target = ["B", "E", "E"]
    paths = enumerate_legal_paths(target=target, time_steps=4, blank_symbol="-")

    assert len(paths) == 1
    assert paths[0] == ["B", "E", "-", "E"]
    assert ctc_collapse(paths[0], blank_token="-") == target


def test_first_principles_ctc_decoder_tensor():
    """Verify FirstPrinciplesCTCDecoder greedy argmax on a synthetic emission tensor."""
    labels = ["-", "|", "C", "A", "T"]
    decoder = FirstPrinciplesCTCDecoder(labels=labels, blank_idx=0, word_separator="|")

    # Create synthetic logits for 5 frames: [C, C, A, -, T]
    # Frame 0: C (idx 2)
    # Frame 1: C (idx 2)
    # Frame 2: A (idx 3)
    # Frame 3: - (idx 0)
    # Frame 4: T (idx 4)
    logits = torch.zeros(5, len(labels))
    target_indices = [2, 2, 3, 0, 4]
    for t, idx in enumerate(target_indices):
        logits[t, idx] = 10.0

    result = decoder.decode_greedy(logits)
    assert isinstance(result, CTCAlignmentResult)
    assert result.transcript == "CAT"
    assert result.collapsed_tokens == ["C", "A", "T"]
    assert result.raw_path == ["C", "C", "A", "-", "T"]
    assert len(result.frame_confidences) == 5
    assert all(c > 0.99 for c in result.frame_confidences)
    assert result.mean_confidence > 0.99
