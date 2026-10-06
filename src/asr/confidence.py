"""Confidence definitions for ASR tokens and words.

Day 11, concept 1: frame score, emitted-token confidence and word correctness
are three different quantities, and averaging them together or calling one
another hides errors.

The baseline in :mod:`src.asr.baseline` reports ``average_confidence`` as the
mean softmax maximum over **every acoustic frame**, while its docstring
claims the mean over emitted non-blank tokens. Those are different numbers.
On a measured clean clip the frame count is 292 with 150 blanks, so 51.4
percent of the frames averaged in are blanks whose confidence says nothing
about whether the recognized words were correct. The two figures differ by
about 0.011 on that clip.

Rather than change the baseline, which would break comparability with the
committed Day 10 results, this module defines the quantities explicitly and
separately:

- :func:`mean_frame_confidence` — mean over all frames. A property of the
  encoder's output, dominated by blanks.
- :func:`mean_token_confidence` — mean over emitted tokens only. A property
  of what was actually said.
- :func:`min_token_confidence` — the weakest emitted token, which is where
  a single bad word hides inside a healthy average.

Every record carries :class:`ConfidenceProvenance` so a score can never be
compared across a model, head, decoder or precision change without that
change being visible. Nothing here is calibrated: these are raw softmax
probabilities, and a value in [0, 1] is not a probability until it has been
fitted and checked. Fitting is later work; this module only keeps the
distinction explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

# Token states for alignment against a reference.
# VALID: this token was matched to a reference word.
# INSERTED: the model emitted a token with no reference counterpart.
VALID = "valid"
INSERTED = "inserted"
# Alignment states for reference words.
MATCHED = "matched"
DELETED = "deleted"


@dataclass(frozen=True)
class ConfidenceProvenance:
    """Identifies the exact system that produced a confidence score.

    A score is meaningless without this. Changing the checkpoint, decoder
    head, tokenizer or numeric precision invalidates comparisons made with
    scores that lack the same values, so they travel with every record.

    Attributes:
        model: Model identifier, e.g. a torchaudio pipeline name.
        head: Output head, e.g. ``"ctc"``.
        tokenizer: Token vocabulary identifier.
        decoder: Decoding rule that produced the token sequence.
        precision: Numeric precision the model ran in.
    """

    model: str
    head: str
    tokenizer: str
    decoder: str
    precision: str

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


@dataclass
class TokenScore:
    """Confidence for one emitted token and its alignment state.

    Attributes:
        token: The emitted token text, e.g. ``"HER"``.
        index: Position in the emitted token sequence.
        probability: Softmax probability of the emitted token at its frame.
            This is a raw softmax value, not a calibrated probability.
        frame_index: Acoustic frame the token was emitted from.
        timestamp_sec: Estimated center time of the frame, in seconds.
        blank_before: True when the previous emitted token was a blank,
            meaning this token starts a new CTC emission group.
        state: :data:`VALID` when aligned to a reference word, or
            :data:`INSERTED` when the model emitted it with no counterpart.
    """

    token: str
    index: int
    probability: float
    frame_index: int
    timestamp_sec: float
    blank_before: bool
    state: str = VALID
    reference_word: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True when this token was matched to a reference word."""
        return self.state == VALID

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)



def mean_frame_confidence(frame_probabilities: Sequence[float]) -> float:
    """Mean softmax maximum over every acoustic frame.

    This is what ``ASRBaseline.average_confidence`` actually computes. Blank
    frames are included, and they typically dominate: a measured clean clip
    had 150 blank frames out of 292. The result describes the encoder's
    output distribution, not the quality of the recognized words.

    Args:
        frame_probabilities: Softmax maximum per frame, in frame order.

    Returns:
        Mean probability as a fraction in [0, 1]. Returns 0.0 for an empty
        sequence rather than raising.

    Raises:
        ValueError: If any value lies outside [0, 1].
    """
    values = _validate(frame_probabilities, "frame_probabilities")
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_token_confidence(scores: Sequence[TokenScore]) -> float:
    """Mean probability over emitted tokens only.

    Excludes blank frames, matching what the baseline docstring claims its
    ``average_confidence`` means. This is the quantity to use when asking
    how confident the model was in the words it actually said.

    Args:
        scores: Emitted token scores in emission order.

    Returns:
        Mean probability as a fraction in [0, 1]. Returns 0.0 when no tokens
        were emitted, which is a real outcome for silence.
    """
    values = _validate([s.probability for s in scores], "token probabilities")
    if not values:
        return 0.0
    return sum(values) / len(values)


def min_token_confidence(scores: Sequence[TokenScore]) -> Optional[float]:
    """Lowest emitted-token probability, or None when nothing was emitted.

    A mean hides a single badly misheard word inside a healthy average, so
    the weakest token is reported separately. On damaged audio this is often
    the only score that moves.

    Args:
        scores: Emitted token scores in emission order.

    Returns:
        The minimum probability, or None for an empty sequence.
    """
    if not scores:
        return None
    return min(_validate([s.probability for s in scores], "token probabilities"))


def confidence_summary(
    frame_probabilities: Sequence[float],
    scores: Sequence[TokenScore],
    provenance: ConfidenceProvenance,
) -> Dict[str, object]:
    """Summarize frame-level and token-level confidence side by side.

    Reports both means together precisely because they disagree. A large gap
    means the utterance was mostly blank, so the frame average is dominated
    by frames carrying no word-level information.

    Args:
        frame_probabilities: Softmax maximum per frame, in frame order.
        scores: Emitted token scores in emission order.
        provenance: System that produced these scores.

    Returns:
        Dict with the frame mean, token mean, minimum, counts, the gap
        between the two means, and provenance.
    """
    frames = _validate(frame_probabilities, "frame_probabilities")
    frame_mean = mean_frame_confidence(frames)
    token_mean = mean_token_confidence(scores)
    weakest = min_token_confidence(scores)
    return {
        "mean_frame_confidence": round(frame_mean, 6),
        "mean_token_confidence": round(token_mean, 6),
        "min_token_confidence": None if weakest is None else round(weakest, 6),
        "num_frames": len(frames),
        "num_tokens": len(scores),
        "frame_token_gap": round(frame_mean - token_mean, 6),
        "provenance": provenance.to_dict(),
    }


@dataclass
class WordScore:
    """Confidence for one reference word and whether it was recognized.

    Attributes:
        word: The reference word.
        index: Position in the reference word sequence.
        state: :data:`MATCHED` when a token aligned to it, or
            :data:`DELETED` when the model did not emit it at all.
        token_probability: Mean probability of the tokens aligned to this
            word, or None when the word was deleted.
        token_count: Number of tokens aligned to this word.
        is_correct: True when the aligned token spelled this word exactly.
            A substitution sets this False while leaving the state
            :data:`MATCHED`, because the word was attempted but realized
            wrongly. None when the word was deleted, since nothing was
            produced to judge.
    """

    word: str
    index: int
    state: str
    token_probability: Optional[float]
    token_count: int
    is_correct: Optional[bool] = None

    @property
    def is_matched(self) -> bool:
        """True when the model emitted something for this word."""
        return self.state == MATCHED

    def to_dict(self) -> dict:
        """Return a flat dict suitable for csv.DictWriter."""
        return asdict(self)


@dataclass
class AlignmentResult:
    """Emitted tokens aligned to reference words.

    Attributes:
        token_scores: Every emitted token with its state.
        word_scores: Every reference word with its state.
        num_valid_tokens: Tokens matched to a reference word.
        num_inserted: Tokens emitted with no reference counterpart.
        num_matched_words: Reference words the model produced.
        num_deleted_words: Reference words the model never produced.
    """

    token_scores: List[TokenScore] = field(default_factory=list)
    word_scores: List[WordScore] = field(default_factory=list)
    num_valid_tokens: int = 0
    num_inserted: int = 0
    num_matched_words: int = 0
    num_deleted_words: int = 0

    def to_dict(self) -> dict:
        """Return a summary dict suitable for csv.DictWriter."""
        return {
            "num_valid_tokens": self.num_valid_tokens,
            "num_inserted": self.num_inserted,
            "num_matched_words": self.num_matched_words,
            "num_deleted_words": self.num_deleted_words,
        }


def align_tokens_to_reference(
    scores: Sequence[TokenScore],
    reference_words: Sequence[str],
) -> AlignmentResult:
    """Align emitted tokens to reference words, marking valid and missing.

    Uses the same Levenshtein alignment and tie-break order as
    :func:`src.metrics.wer.levenshtein_counts`, so the confidence view and
    the error-rate view describe one alignment and cannot disagree. Tokens
    are compared case-insensitively after dropping the ``"|"`` word
    separator, because the CTC decoder emits it for spaces.

    A token that lands on a reference word becomes :data:`VALID`, including
    when it substitutes a different word there: the reference word *was*
    attempted and realized, which is different from never producing it. A
    token with no counterpart becomes :data:`INSERTED`. A reference word with
    no token becomes :data:`DELETED` and carries ``token_probability=None``.
    That is the reason the state is explicit: a missed word is not a
    zero-confidence word, it is an absent one, and averaging a fabricated
    zero into a mean would understate confidence.

    Args:
        scores: Emitted token scores in emission order.
        reference_words: Reference words in reference order.

    Returns:
        AlignmentResult with both sides and their states filled in. Input
        scores are not mutated.
    """
    content_positions = [
        i for i, s in enumerate(scores) if s.token not in ("|", " ")
    ]
    content = [scores[i].token for i in content_positions]
    n, m = len(content), len(reference_words)

    def same(i: int, j: int) -> bool:
        return content[i - 1].upper() == reference_words[j - 1].upper()

    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j - 1] + (0 if same(i, j) else 1),
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
            )

    token_state = [INSERTED] * n
    word_state = [DELETED] * m
    content_to_reference: Dict[int, int] = {}

    # Record which reference words were spelled exactly right. A diagonal
    # step that matched the same token is correct; one that substituted is
    # an attempt at that word but a wrong realization.
    correct_flags: Dict[int, bool] = {}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (0 if same(i, j) else 1):
            correct_flags[j - 1] = same(i, j)
            token_state[i - 1] = VALID
            word_state[j - 1] = MATCHED
            content_to_reference[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            i -= 1  # hypothesis token with no word; stays INSERTED
        else:
            j -= 1  # reference word never produced; stays DELETED

    token_scores: List[TokenScore] = []
    for content_index, original in enumerate(content_positions):
        source = scores[original]
        ref_index = content_to_reference.get(content_index)
        token_scores.append(
            TokenScore(
                token=source.token,
                index=source.index,
                probability=source.probability,
                frame_index=source.frame_index,
                timestamp_sec=source.timestamp_sec,
                blank_before=source.blank_before,
                state=token_state[content_index],
                reference_word=reference_words[ref_index] if ref_index is not None else None,
            )
        )

    per_word: Dict[int, List[float]] = {}
    for content_index, ref_index in content_to_reference.items():
        per_word.setdefault(ref_index, []).append(
            float(token_scores[content_index].probability)
        )

    word_scores = [
        WordScore(
            word=word,
            index=index,
            state=word_state[index],
            token_probability=(
                round(sum(per_word[index]) / len(per_word[index]), 6)
                if index in per_word
                else None
            ),
            token_count=len(per_word.get(index, [])),
            is_correct=(
                correct_flags.get(index) if index in correct_flags else None
            ),
        )
        for index, word in enumerate(reference_words)
    ]

    return AlignmentResult(
        token_scores=token_scores,
        word_scores=word_scores,
        num_valid_tokens=sum(1 for s in token_scores if s.is_valid),
        num_inserted=sum(1 for s in token_scores if s.state == INSERTED),
        num_matched_words=sum(1 for w in word_scores if w.is_matched),
        num_deleted_words=sum(1 for w in word_scores if w.state == DELETED),
    )


    token_scores: List[TokenScore] = field(default_factory=list)
    word_scores: List[WordScore] = field(default_factory=list)
    num_valid_tokens: int = 0
    num_inserted: int = 0
    num_matched_words: int = 0
    num_deleted_words: int = 0

    def to_dict(self) -> dict:
        """Return a summary dict suitable for csv.DictWriter."""
        return {
            "num_valid_tokens": self.num_valid_tokens,
            "num_inserted": self.num_inserted,
            "num_matched_words": self.num_matched_words,
            "num_deleted_words": self.num_deleted_words,
        }


def collapse_tokens_to_words(
    scores: Sequence[TokenScore], separator: str = "|"
) -> List[TokenScore]:
    """Group character-level tokens into word-level tokens.

    The CTC baseline emits one token per character with ``"|"`` marking a
    word boundary, so its vocabulary has 29 entries rather than one per word.
    Comparing those characters directly against reference *words* would mark
    every word wrong: a single ``"I"`` substituted for ``"ILLUSTRATION"`` is
    one edit in the model's favour, not a correct word and not a wrong one.

    Grouping first is what makes word-level confidence meaningful. A word's
    probability is the mean of its character probabilities, its frame index
    and timestamp come from its first character, and it counts as a valid
    token because it occupies a real word slot.

    A word with no characters cannot occur, but an empty input returns an
    empty list rather than raising.

    Args:
        scores: Character-level token scores in emission order.
        separator: Token text marking a word boundary.

    Returns:
        Word-level TokenScore list, in order. A word is emitted whenever a
        non-separator token follows a boundary or the start of the sequence.

    Raises:
        ValueError: If any probability lies outside [0, 1].
    """
    words: List[TokenScore] = []
    buffer: List[TokenScore] = []

    def flush() -> None:
        if not buffer:
            return
        words.append(
            TokenScore(
                token="".join(s.token for s in buffer),
                index=len(words),
                probability=round(
                    sum(s.probability for s in buffer) / len(buffer), 6
                ),
                frame_index=buffer[0].frame_index,
                timestamp_sec=buffer[0].timestamp_sec,
                blank_before=buffer[0].blank_before,
            )
        )
        buffer.clear()

    for score in scores:
        _validate([score.probability], "token probabilities")
        if score.token == separator:
            flush()
        else:
            buffer.append(score)
    flush()
    return words


def confidence_accuracy_bins(
    word_scores: Sequence[WordScore],
    bin_edges: Sequence[float] = (0.0, 0.5, 0.7, 0.9, 0.95, 1.0),
) -> List[Dict[str, object]]:
    """Bin reference words by confidence and measure accuracy in each bin.

    This is the test of whether a confidence score carries information. If it
    does, accuracy rises across the bins; if the score is noise, every bin
    sits near the same accuracy. A high-confidence bin with low accuracy is
    the definition of a confident error.

    Deleted words are counted separately rather than binned, because they
    have no probability and assigning one would invent evidence. They are
    reported in the ``deleted_words`` field of the returned summary.

    Only exact spelling counts as correct. A substituted word is binned by
    its own (typically lower) probability and marked incorrect, so a
    confident mishearing lands in a high bin with low accuracy rather than
    disappearing.

    Args:
        word_scores: Reference words with states and probabilities, typically
            from :func:`align_tokens_to_reference`.
        bin_edges: Ascending bin edges. The last value is the upper bound of
            the final bin. Defaults give five bins spanning [0, 1].

    Returns:
        Dict with ``bins`` (one row per non-empty bin, carrying bounds, word
        count, correct count and accuracy) plus ``deleted_words`` and
        ``judged_words``. Bins with no words are omitted, never reported as
        0.0 accuracy.

    Raises:
        ValueError: If ``bin_edges`` has fewer than two values.
    """
    edges = list(bin_edges)
    if len(edges) < 2:
        raise ValueError("bin_edges needs at least a lower and upper edge")

    num_bins = len(edges) - 1
    correct = [0] * num_bins
    total = [0] * num_bins
    deleted = 0

    for word in word_scores:
        if word.state == DELETED or word.token_probability is None:
            deleted += 1
            continue
        # Assign to the highest bin whose lower edge the probability reaches.
        index = 0
        for i in range(num_bins):
            if word.token_probability >= edges[i]:
                index = i
        total[index] += 1
        if word.is_correct:
            correct[index] += 1

    rows = [
        {
            "bin_lower": round(edges[i], 4),
            "bin_upper": round(edges[i + 1], 4),
            "words": total[i],
            "correct": correct[i],
            "accuracy": round(correct[i] / total[i], 6),
        }
        for i in range(num_bins)
        if total[i] > 0
    ]
    return {
        "bins": rows,
        "deleted_words": deleted,
        "judged_words": sum(total),
    }




def _validate(values: Sequence[float], label: str) -> List[float]:
    """Coerce values to floats, raising when any lies outside [0, 1].

    Args:
        values: Raw numeric values from a score record.
        label: Name used in the error message.

    Returns:
        The values as a list of floats.

    Raises:
        ValueError: If any value is outside [0, 1], since a softmax-derived
            probability cannot be.
    """
    out: List[float] = []
    for value in values:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label} must lie in [0, 1], got {number}")
        out.append(number)
    return out
