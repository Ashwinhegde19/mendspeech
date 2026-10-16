"""Unit tests for Hinglish code-switch metrics."""

from src.hinglish.metrics import (
    average_by_bin,
    code_switch_bin,
    code_switch_ratio,
)


def test_code_switch_ratio_and_bin() -> None:
    ratio = code_switch_ratio(2, 10)
    assert ratio == 0.2
    assert code_switch_bin(ratio) == "hindi-dominant"


def test_average_by_bin_groups_rows() -> None:
    rows = [
        {"bin": "mixed", "wer": 0.1},
        {"bin": "mixed", "wer": 0.3},
    ]
    assert average_by_bin(rows, "bin", "wer") == {"mixed": 0.2}
