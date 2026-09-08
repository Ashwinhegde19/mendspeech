"""CTC Collapse and Decoding from First Principles.

Day 9 module: implements Connectionist Temporal Classification (CTC) alignment
collapse, legal path enumeration, and greedy decoding without third-party library
dependencies.

Key principles implemented:
1. Blank delimiter: separates consecutive identical tokens (e.g. 'B-E-E' vs 'B-E').
2. Collapse operator B: merges consecutive identical labels, then removes all blanks.
3. Many-to-one mapping: multiple distinct frame alignments collapse to the same transcript.
4. Conditional independence: frame probabilities factorize over time.
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import torch

T = TypeVar("T")


def ctc_collapse(alignment: Sequence[T], blank_token: T) -> List[T]:
    """Applies the CTC collapse operator B to an alignment sequence.

    The CTC collapse operation consists of two sequential steps:
    1. Merge consecutive identical tokens:
       [b, b, blank, o, o, blank, o, k] -> [b, blank, o, blank, o, k]
    2. Remove all blank tokens:
       [b, blank, o, blank, o, k] -> [b, o, o, k]

    Args:
        alignment: Sequence of token symbols or integer class IDs.
        blank_token: The symbol or integer ID representing the CTC blank.

    Returns:
        The collapsed sequence with merged duplicates and stripped blanks.

    Raises:
        ValueError: If alignment is not a sequence.
    """
    if not isinstance(alignment, (list, tuple)):
        alignment = list(alignment)

    if len(alignment) == 0:
        return []

    # Step 1: Merge consecutive identical elements
    deduped: List[T] = []
    prev_token: Optional[T] = None
    for token in alignment:
        if token != prev_token:
            deduped.append(token)
            prev_token = token

    # Step 2: Drop all blank tokens
    collapsed = [token for token in deduped if token != blank_token]
    return collapsed


def ctc_decode_transcript(
    alignment: Sequence[str],
    blank_symbol: str = "-",
    word_separator: str = "|",
) -> str:
    """Collapses a character token alignment and converts word separators into spaces.

    Args:
        alignment: Sequence of string character tokens.
        blank_symbol: CTC blank delimiter (default: '-').
        word_separator: Word boundary symbol (default: '|').

    Returns:
        Clean human-readable transcript.
    """
    collapsed_tokens = ctc_collapse(alignment, blank_token=blank_symbol)
    raw_text = "".join(collapsed_tokens)
    return raw_text.replace(word_separator, " ").strip()


def enumerate_legal_paths(
    target: Sequence[str],
    time_steps: int,
    blank_symbol: str = "-",
) -> List[List[str]]:
    """Generates all legal frame-level alignments pi of length T that collapse to target.

    Demonstrates the many-to-one mapping B^-1(target):
    Multiple distinct frame alignments map to the exact same target transcript.

    Args:
        target: Target sequence of symbols (e.g. ['C', 'A', 'T'] or ['B', 'E', 'E']).
        time_steps: Length of the alignment path in frames (must be >= len(target)).
        blank_symbol: The blank token symbol.

    Returns:
        List of valid alignment paths, each of length time_steps.
    """
    target_list = list(target)
    if time_steps < len(target_list):
        return []

    # Unique vocabulary needed for this target plus the blank symbol
    alphabet = list(dict.fromkeys(target_list + [blank_symbol]))

    legal_paths: List[List[str]] = []

    def _backtrack(current_path: List[str]) -> None:
        if len(current_path) == time_steps:
            if ctc_collapse(current_path, blank_token=blank_symbol) == target_list:
                legal_paths.append(list(current_path))
            return

        # Prune search space: if remaining slots are insufficient for remaining target tokens
        collapsed_so_far = ctc_collapse(current_path, blank_token=blank_symbol)
        if len(collapsed_so_far) > len(target_list):
            return
        if len(target_list) - len(collapsed_so_far) > (time_steps - len(current_path)):
            return

        for symbol in alphabet:
            current_path.append(symbol)
            _backtrack(current_path)
            current_path.pop()

    _backtrack([])
    return legal_paths


@dataclass
class CTCAlignmentResult:
    """Result of first-principles CTC decoding on an emission matrix."""

    transcript: str
    collapsed_tokens: List[str]
    raw_path: List[str]
    frame_confidences: List[float]
    mean_confidence: float


class FirstPrinciplesCTCDecoder:
    """Standalone CTC Decoder implementing greedy argmax decoding and alignment collapse."""

    def __init__(
        self,
        labels: Sequence[str],
        blank_idx: int = 0,
        word_separator: str = "|",
    ) -> None:
        """Initializes decoder with vocabulary labels and blank index.

        Args:
            labels: Ordered sequence of string labels matching vocabulary classes.
            blank_idx: Index of the blank symbol in labels (default: 0).
            word_separator: Character representing word boundaries (default: '|').
        """
        self.labels = list(labels)
        self.blank_idx = blank_idx
        self.blank_symbol = self.labels[blank_idx]
        self.word_separator = word_separator

    def decode_greedy(self, emissions: torch.Tensor) -> CTCAlignmentResult:
        """Decodes frame logits or log-probabilities into a transcript.

        Args:
            emissions: Tensor of shape [T, V] or [1, T, V] containing logits or probs.

        Returns:
            CTCAlignmentResult with transcript, paths, and confidence diagnostics.
        """
        if emissions.dim() == 3:
            if emissions.shape[0] != 1:
                raise ValueError("Batch size > 1 not supported in single decode")
            emissions = emissions[0]

        probs = torch.softmax(emissions, dim=-1)
        max_probs, argmax_indices = torch.max(probs, dim=-1)

        raw_indices = argmax_indices.tolist()
        raw_path = [self.labels[idx] for idx in raw_indices]
        frame_confidences = [round(float(p), 4) for p in max_probs.tolist()]

        collapsed_indices = ctc_collapse(raw_indices, blank_token=self.blank_idx)
        collapsed_tokens = [self.labels[idx] for idx in collapsed_indices]

        raw_text = "".join(collapsed_tokens)
        transcript = raw_text.replace(self.word_separator, " ").strip()

        mean_conf = round(float(sum(frame_confidences) / len(frame_confidences)), 4) if frame_confidences else 0.0

        return CTCAlignmentResult(
            transcript=transcript,
            collapsed_tokens=collapsed_tokens,
            raw_path=raw_path,
            frame_confidences=frame_confidences,
            mean_confidence=mean_conf,
        )
