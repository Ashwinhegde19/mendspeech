"""MendSpeech Day 10 experiment: error rates against the frozen benchmark.

Scores the 30-utterance frozen benchmark with real LibriSpeech reference
transcripts, unlike the Day 08 run which compared damaged output against the
model's own clean transcript and so measured drift rather than accuracy.

For every utterance in the calibration and validation roles, this transcribes
the clean audio and each of the five SpeechDamageBench damage operators at
three severities, then reports word and character error rates, the
substitution/deletion/insertion breakdown, and protected-content slice rates.

The final-test role is deliberately not scored here. It is sealed until the
frozen release phase, and the protocol forbids using it for development.

Outputs:
  results/day10_wer_by_damage.csv   per-condition aggregate rates
  results/day10_error_types.csv     substitution/deletion/insertion counts
  results/day10_slice_rates.csv     per-slice protected-content rates
  results/day10_raw_runs.csv        one row per scored run
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
from src.audio.loader import load_audio
from src.metrics.normalization import NORMALIZATION_VERSION
from src.metrics.roles import CANONICAL_MANIFEST, assign_roles, role_for_clip
from src.metrics.slices import SLICE_NAMES, slice_scores
from src.metrics.wer import character_error_rate, word_error_rate

SEED = 42
SEVERITIES = ("mild", "medium", "severe")
SAMPLE_RATE = 16000

# Roles scored in this experiment. final_test is intentionally excluded.
SCORED_ROLES = ("calibration", "validation")

WER_CSV = Path("results/day10_wer_by_damage.csv")
ERROR_TYPES_CSV = Path("results/day10_error_types.csv")
SLICE_CSV = Path("results/day10_slice_rates.csv")
RAW_CSV = Path("results/day10_raw_runs.csv")


def score_pair(reference: str, hypothesis: str) -> dict:
    """Score one reference/hypothesis pair into a flat record.

    Args:
        reference: Ground-truth transcript from the frozen manifest.
        hypothesis: Transcript produced by the ASR baseline.

    Returns:
        Dict with WER, CER and the substitution/deletion/insertion counts.
    """
    wer = word_error_rate(reference, hypothesis)
    cer = character_error_rate(reference, hypothesis)
    return {
        "reference_words": wer.ref_len,
        "wer": wer.error_rate,
        "cer": cer.error_rate,
        "substitutions": wer.substitutions,
        "deletions": wer.deletions,
        "insertions": wer.insertions,
        "total_edits": wer.total_edits,
    }



def main() -> None:
    print("--- Starting Day 10 frozen-benchmark scoring ---")
    manifest = pd.read_csv(CANONICAL_MANIFEST)
    roles = role_for_clip(manifest)

    # Confirm the split before touching any audio.
    for role, assignment in assign_roles(manifest).items():
        print(
            f"  role {role:12s} n={assignment.num_utterances:2d} "
            f"speakers={assignment.speakers}"
        )
    scored = manifest[manifest["clip_id"].map(roles).isin(SCORED_ROLES)]
    print(
        f"\nScoring {len(scored)} utterances from "
        f"{sorted(set(scored['clip_id'].map(roles)))}. "
        f"final_test is sealed and not scored."
    )

    baseline = ASRBaseline(device="cpu")
    records = []

    for _, row in scored.iterrows():
        clip_id = row["clip_id"]
        role = roles[clip_id]
        reference = row["transcript"]

        waveform, sr = load_audio(Path(row["file_path"]), target_sr=SAMPLE_RATE)
        clean_np = waveform.squeeze(0).numpy()

        conditions = [("clean", "none", -1, clean_np)]
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                config = DamageConfig(
                    corruption=corruption,
                    severity=severity,
                    seed=SEED,
                    source_id=clip_id,
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
            record = {
                "clip_id": clip_id,
                "speaker_id": row["speaker_id"],
                "role": role,
                "condition": corruption,
                "severity": severity,
                "seed": seed,
                "duration_sec": out.duration_sec,
                "avg_confidence": out.average_confidence,
                "normalization": NORMALIZATION_VERSION,
            }
            record.update(score_pair(reference, out.transcript))
            for slice_score in slice_scores(reference, out.transcript):
                record[f"{slice_score.slice_name}_ref_tokens"] = (
                    slice_score.ref_tokens
                )
                record[f"{slice_score.slice_name}_errors"] = slice_score.errors
                record[f"{slice_score.slice_name}_rate"] = slice_score.error_rate
            records.append(record)

        print(f"  [{clip_id} {role}] {len(conditions)} conditions scored")

    frame = pd.DataFrame(records)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_CSV, index=False)

    # --- Aggregate by condition, averaging per-utterance rates. ---
    condition_rows, type_rows, slice_rows = [], [], []

    for (condition, severity), group in frame.groupby(
        ["condition", "severity"], sort=False
    ):
        condition_rows.append(
            {
                "condition": condition,
                "severity": severity,
                "seed": SEED,
                "runs": len(group),
                "independent_utterances": group["clip_id"].nunique(),
                "independent_speakers": group["speaker_id"].nunique(),
                "mean_wer": round(100 * group["wer"].mean(), 4),
                "mean_cer": round(100 * group["cer"].mean(), 4),
                "mean_confidence": round(group["avg_confidence"].mean(), 4),
                "normalization": NORMALIZATION_VERSION,
            }
        )
        type_rows.append(
            {
                "condition": condition,
                "severity": severity,
                "runs": len(group),
                "substitutions": int(group["substitutions"].sum()),
                "deletions": int(group["deletions"].sum()),
                "insertions": int(group["insertions"].sum()),
                "total_edits": int(group["total_edits"].sum()),
                "reference_words": int(group["reference_words"].sum()),
                "normalization": NORMALIZATION_VERSION,
            }
        )
        for name in SLICE_NAMES:
            covered = group[group[f"{name}_ref_tokens"] > 0]
            tokens = int(covered[f"{name}_ref_tokens"].sum())
            slice_rows.append(
                {
                    "condition": condition,
                    "severity": severity,
                    "slice": name,
                    "ref_tokens": tokens,
                    "errors": int(covered[f"{name}_errors"].sum()),
                    "error_rate": (
                        round(100 * covered[f"{name}_errors"].sum() / tokens, 4)
                        if tokens
                        else None
                    ),
                    "utterances_covered": int(len(covered)),
                    "normalization": NORMALIZATION_VERSION,
                }
            )

    for path, rows in (
        (WER_CSV, condition_rows),
        (ERROR_TYPES_CSV, type_rows),
        (SLICE_CSV, slice_rows),
    ):
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
