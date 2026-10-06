"""MendSpeech Day 11 experiment: confidence versus correctness.

Scores the validation role of the frozen benchmark and asks whether a
confidence score carries information about whether a word was recognized
correctly. This tests the near-perfect aggregate correlation seen in the
Day 10 results, where mean confidence tracked WER at r = -0.99 across
sixteen aggregate points. That correlation may be an artifact of severe
damage degrading both quantities at once rather than evidence that the score
is meaningful per word.

For each validation utterance under clean and each damage operator at three
severities, this records every frame probability and every emitted token
probability, aligns them to the reference words, then bins words by
confidence and measures accuracy per bin.

Outputs:
  results/day11_confidence_by_damage.csv   accuracy per confidence bin
  results/day11_token_scores.csv           per-word score and correctness
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from speechdamagebench.audio_damage import CORRUPTIONS, DamageConfig, apply_damage
from src.asr.baseline import ASRBaseline
from src.asr.confidence import (
    ConfidenceProvenance,
    TokenScore,
    align_tokens_to_reference,
    collapse_tokens_to_words,
    confidence_accuracy_bins,
    mean_frame_confidence,
    mean_token_confidence,
    min_token_confidence,
)
from src.audio.loader import load_audio
from src.metrics.normalization import NORMALIZATION_VERSION
from src.metrics.roles import CANONICAL_MANIFEST, role_for_clip

SEED = 42
SEVERITIES = ("mild", "medium", "severe")
SAMPLE_RATE = 16000
BIN_EDGES = (0.0, 0.5, 0.7, 0.9, 0.95, 1.0)

# Day 11 measures on validation only. Calibration is reserved for fitting a
# calibration mapping, and final test stays sealed.
SCORED_ROLE = "validation"

PROVENANCE = ConfidenceProvenance(
    model="WAV2VEC2_ASR_BASE_960H",
    head="ctc",
    tokenizer="chars-29",
    decoder="greedy_ctc_collapse",
    precision="float32",
)

BIN_CSV = Path("results/day11_confidence_by_damage.csv")
TOKEN_CSV = Path("results/day11_token_scores.csv")


def token_scores_from_output(out, labels, blank_id: int) -> list:
    """Rebuild emitted-token scores with their frame index and probabilities.

    The baseline returns token strings and frame confidences but not the
    class index chosen at each frame, so the emission walk is repeated here
    using the documented CTC collapse rule, mirroring
    ``src.asr.baseline.greedy_ctc_decode``.

    Args:
        out: ASROutput from the baseline.
        labels: Vocabulary labels for the model.
        blank_id: CTC blank class index.

    Returns:
        List of TokenScore, one per emitted non-blank token.
    """
    scores = []
    previous_id = -1
    blank_before = True
    for frame_index, class_index in enumerate(out.frame_class_indices):
        if class_index == previous_id:
            continue
        previous_id = class_index
        if class_index == blank_id:
            blank_before = True
            continue
        scores.append(
            TokenScore(
                token=labels[class_index],
                index=len(scores),
                probability=out.frame_confidences[frame_index],
                frame_index=frame_index,
                timestamp_sec=(
                    round(frame_index * (out.duration_sec / out.num_frames), 4)
                    if out.num_frames
                    else 0.0
                ),
                blank_before=blank_before,
            )
        )
        blank_before = False
    return scores


def main() -> None:
    print("--- Starting Day 11 confidence versus correctness ---")
    manifest = pd.read_csv(CANONICAL_MANIFEST)
    roles = role_for_clip(manifest)
    scored = manifest[manifest["clip_id"].map(roles) == SCORED_ROLE]
    print(
        f"Scoring {len(scored)} validation utterances. "
        f"calibration and final_test excluded."
    )

    baseline = ASRBaseline(device="cpu")
    bin_rows, token_rows = [], []

    for _, row in scored.iterrows():
        clip_id = row["clip_id"]
        reference_words = row["transcript"].split()

        waveform, sr = load_audio(Path(row["file_path"]), target_sr=SAMPLE_RATE)
        clean_np = waveform.squeeze(0).numpy()

        conditions = [("clean", "none", -1, clean_np)]
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                config = DamageConfig(
                    corruption=corruption, severity=severity,
                    seed=SEED, source_id=clip_id,
                )
                damaged_np, _ = apply_damage(
                    clean_np, sample_rate=SAMPLE_RATE, config=config
                )
                conditions.append((corruption, severity, SEED, damaged_np))

        for corruption, severity, seed, audio in conditions:
            out = baseline.transcribe(
                torch.from_numpy(audio).unsqueeze(0),
                sample_rate=SAMPLE_RATE,
                clip_id=f"{clip_id}_{corruption}",
            )
            char_scores = token_scores_from_output(
                out, baseline.labels, baseline.blank_id
            )
            # The baseline emits characters; group them into words before
            # aligning against reference words.
            scores = collapse_tokens_to_words(char_scores)
            alignment = align_tokens_to_reference(scores, reference_words)
            summary = confidence_accuracy_bins(alignment.word_scores, BIN_EDGES)
            frame_mean = mean_frame_confidence(out.frame_confidences)
            token_mean = mean_token_confidence(scores)
            weakest = min_token_confidence(scores)

            for entry in summary["bins"]:
                bin_rows.append({
                    "clip_id": clip_id,
                    "condition": corruption,
                    "severity": severity,
                    "seed": seed,
                    "bin_lower": entry["bin_lower"],
                    "bin_upper": entry["bin_upper"],
                    "words": entry["words"],
                    "correct": entry["correct"],
                    "accuracy": round(100 * entry["accuracy"], 4),
                    "deleted_words": summary["deleted_words"],
                    "mean_frame_confidence": round(frame_mean, 6),
                    "mean_token_confidence": round(token_mean, 6),
                    "min_token_confidence": weakest,
                    "normalization": NORMALIZATION_VERSION,
                })

            for word in alignment.word_scores:
                token_rows.append({
                    "clip_id": clip_id,
                    "condition": corruption,
                    "severity": severity,
                    "word": word.word,
                    "state": word.state,
                    "token_probability": word.token_probability,
                    "is_correct": word.is_correct,
                    "provenance_model": PROVENANCE.model,
                    "provenance_decoder": PROVENANCE.decoder,
                })

        print(f"  [{clip_id}] {len(conditions)} conditions scored")

    bins = pd.DataFrame(bin_rows)
    tokens = pd.DataFrame(token_rows)
    for path, frame in ((BIN_CSV, bins), (TOKEN_CSV, tokens)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Saved {len(frame)} rows to {path}")

    print("\nAccuracy by confidence bin, pooled:")
    pooled = bins.groupby(["bin_lower", "bin_upper"]).agg(
        words=("words", "sum"), correct=("correct", "sum")
    )
    pooled["accuracy"] = (100 * pooled["correct"] / pooled["words"]).round(2)
    print(pooled.to_string())


if __name__ == "__main__":
    main()
