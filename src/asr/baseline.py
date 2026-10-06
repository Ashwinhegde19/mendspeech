"""MendSpeech ASR Baseline: Pretrained acoustic model with CTC decoding and uncertainty metrics.

Day 8 foundational module: takes raw audio waveforms (16 kHz mono), passes them through
a pretrained Wav2Vec2 CTC acoustic model, extracts frame-level logits and probabilities,
and implements greedy CTC decoding with token-level timestamps and calibrated confidences.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio

from src.audio.loader import load_audio


@dataclass
class ASROutput:
    """Structured record of an ASR recognition result with timing and uncertainty metadata.

    Attributes:
        clip_id: Identifier for the audio sample.
        transcript: Final normalized text transcript.
        duration_sec: Audio duration in seconds.
        num_frames: Number of acoustic output frames from the encoder.
        sample_rate: Audio sample rate in Hz.
        tokens: Decoded token sequence (characters).
        token_indices: Class index for each decoded token in vocabulary.
        token_timestamps_sec: Estimated center timestamp for each emitted token.
        average_confidence: Mean softmax probability over emitted non-blank tokens.
        frame_confidences: Max softmax probability at every individual acoustic frame.
        frame_class_indices: Argmax class index at every acoustic frame. Added
            on Day 11 so per-token confidence can be re-derived from the frame
            decisions without re-running the model. Purely additive and does
            not change any previously reported value.
    """

    clip_id: str
    transcript: str
    duration_sec: float
    num_frames: int
    sample_rate: int
    tokens: List[str]
    token_indices: List[int]
    token_timestamps_sec: List[float]
    average_confidence: float
    frame_confidences: List[float]
    frame_class_indices: List[int] = field(default_factory=list)


def greedy_ctc_decode(
    emissions: torch.Tensor,
    labels: Sequence[str],
    blank_id: int = 0,
    duration_sec: Optional[float] = None,
) -> Tuple[str, List[str], List[int], List[float], List[float]]:
    """Performs greedy CTC decoding on a frame-level emission tensor.

    Applies the standard CTC collapse rule:
    1. Merge consecutive identical tokens.
    2. Drop the blank token.
    3. Map the word separator '|' to whitespace.

    Args:
        emissions: Emission tensor of shape [frames, vocab_size] or [1, frames, vocab_size] (logits).
        labels: Tuple/List of vocabulary tokens matching the model's output classes.
        blank_id: Index of the CTC blank token (default: 0).
        duration_sec: Total audio duration in seconds. Used for frame-to-second timestamp mapping.

    Returns:
        Tuple of:
            - transcript: Formatted text string.
            - tokens: List of emitted token strings.
            - token_indices: List of integer class indices.
            - token_timestamps: List of timestamps in seconds for emitted tokens.
            - frame_confidences: List of top-token probabilities for all acoustic frames.
    """
    if emissions.dim() == 3:
        if emissions.shape[0] != 1:
            raise ValueError(f"Batch size > 1 not supported, got shape {tuple(emissions.shape)}")
        emissions = emissions[0]

    num_frames, vocab_size = emissions.shape
    probs = torch.softmax(emissions, dim=-1)
    max_probs, argmax_indices = torch.max(probs, dim=-1)

    argmax_list = argmax_indices.tolist()
    max_prob_list = max_probs.tolist()

    tokens: List[str] = []
    token_indices: List[int] = []
    token_timestamps: List[float] = []
    token_probs: List[float] = []

    frame_duration = (duration_sec / num_frames) if (duration_sec and num_frames > 0) else 0.02

    previous_id = -1
    for t in range(num_frames):
        current_id = argmax_list[t]
        if current_id == previous_id:
            # CTC rule 1: Merge consecutive identical tokens
            continue

        previous_id = current_id
        if current_id == blank_id:
            # CTC rule 2: Drop blanks
            continue

        # Emitted non-blank token
        token_str = labels[current_id] if current_id < len(labels) else "?"
        tokens.append(token_str)
        token_indices.append(current_id)
        token_timestamps.append(round(t * frame_duration, 4))
        token_probs.append(max_prob_list[t])

    # Convert token sequence into readable text: '|' represents space between words
    raw_text = "".join(tokens)
    transcript = raw_text.replace("|", " ").strip()

    return transcript, tokens, token_indices, token_timestamps, max_prob_list


class ASRBaseline:
    """Pretrained acoustic model for baseline ASR recognition and confidence calibration."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Initializes the baseline using torchaudio's Wav2Vec2 ASR Base 960h bundle.

        Args:
            device: Computing device ('cpu', 'cuda', etc.). If None, automatically detects.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sample_rate = self.bundle.sample_rate
        self.labels = self.bundle.get_labels()
        self.blank_id = 0  # Standard for Wav2Vec2 ASR Base: '-' is index 0

        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()

    def transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        clip_id: str = "audio",
    ) -> ASROutput:
        """Transcribes a raw waveform tensor.

        Args:
            waveform: 1D [time] or 2D [1, time] audio waveform tensor.
            sample_rate: Input sample rate in Hz.
            clip_id: Identifier for output record.

        Returns:
            ASROutput containing transcript, timestamps, tokens, and confidences.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.to(self.device)
        duration_sec = waveform.shape[-1] / self.sample_rate

        with torch.no_grad():
            emissions, _ = self.model(waveform)

        transcript, tokens, token_indices, timestamps, frame_confidences = greedy_ctc_decode(
            emissions=emissions,
            labels=self.labels,
            blank_id=self.blank_id,
            duration_sec=duration_sec,
        )

        # Calculate average confidence across all frames
        avg_confidence = float(sum(frame_confidences) / len(frame_confidences)) if frame_confidences else 0.0

        return ASROutput(
            clip_id=clip_id,
            transcript=transcript,
            duration_sec=round(duration_sec, 3),
            num_frames=emissions.shape[1],
            sample_rate=self.sample_rate,
            tokens=tokens,
            token_indices=token_indices,
            token_timestamps_sec=timestamps,
            average_confidence=round(avg_confidence, 4),
            frame_confidences=[round(c, 4) for c in frame_confidences],
            frame_class_indices=[
                int(i)
                for i in torch.argmax(
                    torch.softmax(emissions[0], dim=-1), dim=-1
                ).tolist()
            ],
        )

    def transcribe_file(self, audio_path: Union[str, Path]) -> ASROutput:
        """Loads an audio file and transcribes it.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ASROutput instance.
        """
        path = Path(audio_path)
        clip_id = path.stem
        waveform, sr = load_audio(path, target_sr=self.sample_rate)
        return self.transcribe(waveform=waveform, sample_rate=sr, clip_id=clip_id)
