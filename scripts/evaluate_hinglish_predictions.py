#!/usr/bin/env python3
"""Score Hinglish ASR predictions against a MendSpeech manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.hinglish.evaluation import (
    HinglishPrediction,
    score_prediction,
    summarize_predictions,
)
from src.hinglish.normalization import HINGLISH_NORMALIZATION_VERSION

PREDICTION_COLUMNS = frozenset({"clip_id", "prediction"})
MANIFEST_COLUMNS = frozenset({"clip_id", "transcript"})


def _validate_prediction_manifest(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one-to-one prediction-to-reference mapping."""
    missing_prediction = PREDICTION_COLUMNS.difference(predictions.columns)
    if missing_prediction:
        raise ValueError(
            "predictions CSV must contain: " + ", ".join(sorted(missing_prediction))
        )
    missing_manifest = MANIFEST_COLUMNS.difference(manifest.columns)
    if missing_manifest:
        raise ValueError(
            "manifest CSV must contain: " + ", ".join(sorted(missing_manifest))
        )

    if predictions["clip_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["clip_id"].duplicated(), "clip_id"
        ].astype(str).unique()
        raise ValueError(
            "duplicate clip IDs in predictions: " + ", ".join(map(str, duplicates))
        )
    if manifest["clip_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["clip_id"].duplicated(), "clip_id"
        ].astype(str).unique()
        raise ValueError(
            "duplicate clip IDs in manifest: " + ", ".join(map(str, duplicates))
        )

    prediction_ids = set(predictions["clip_id"].astype(str))
    manifest_ids = set(manifest["clip_id"].astype(str))
    missing = prediction_ids - manifest_ids
    if missing:
        raise ValueError(
            "unknown prediction clip IDs: " + ", ".join(sorted(missing))
        )
    unmatched = manifest_ids - prediction_ids
    if unmatched:
        raise ValueError(
            "manifest clip IDs without predictions: " + ", ".join(sorted(unmatched))
        )

    empty_reference = manifest[
        manifest["transcript"].isna()
        | (manifest["transcript"].astype(str).str.strip() == "")
    ]
    if not empty_reference.empty:
        raise ValueError(
            "empty reference transcripts for: "
            + ", ".join(empty_reference["clip_id"].astype(str))
        )

    invalid_status = predictions.get("status", pd.Series(["ok"] * len(predictions)))
    invalid_status = invalid_status.isna() | (
        invalid_status.astype(str).str.strip().isin({"FAILED", "failed"})
    )
    return predictions, manifest


def _data_quality(manifest: pd.DataFrame) -> dict[str, object]:
    """Report invalid Hinglish composition metadata."""
    if "ratio_english_words" not in manifest.columns:
        return {"records": len(manifest), "invalid_ratio_records": 0}
    invalid = manifest[
        manifest["ratio_english_words"].isna()
        | (manifest["ratio_english_words"] < 0)
        | (manifest["ratio_english_words"] > 1)
    ]
    return {
        "records": len(manifest),
        "invalid_ratio_records": len(invalid),
        "invalid_ratio_clip_ids": tuple(invalid["clip_id"].astype(str)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score Hinglish ASR predictions without re-running inference"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--model-id", default="openai/whisper-small")
    parser.add_argument("--model-revision", default="973afd24965f72e36ca33b3055d56a652f456b4d")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "hinglish" / "baseline_metrics.json",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    predictions, manifest = _validate_prediction_manifest(predictions, manifest)

    lookup = manifest.set_index("clip_id").to_dict(orient="index")
    scored: list[HinglishPrediction] = []
    failed: list[dict[str, str]] = []
    for row in predictions.to_dict(orient="records"):
        clip_id = str(row["clip_id"])
        manifest_row = lookup[clip_id]
        status = str(row.get("status", "ok")).strip().lower()
        if status == "failed":
            failed.append(
                {
                    "clip_id": clip_id,
                    "error": str(row.get("error", "")),
                }
            )
            continue
        scored.append(
            HinglishPrediction(
                clip_id=clip_id,
                speaker_id=str(manifest_row.get("speaker_id", "")),
                reference=str(manifest_row.get("transcript", "")).strip(),
                prediction=str(row["prediction"]),
                ratio_english_words=float(
                    manifest_row.get("ratio_english_words", 0.5)
                ),
                language=str(row.get("language", args.language)),
                task=str(row.get("task", args.task)),
                model_id=str(row.get("model_id", args.model_id)),
                model_revision=str(row.get("model_revision", args.model_revision)),
                status="ok",
                error="",
            )
        )

    summary = summarize_predictions(scored)
    summary["normalization_version"] = HINGLISH_NORMALIZATION_VERSION
    summary["failed_predictions"] = failed
    summary["failed_prediction_count"] = len(failed)
    summary["data_quality"] = _data_quality(manifest)
    summary["evaluation_guardrails"] = {
        "manifest_required": True,
        "one_to_one_matching": True,
        "empty_reference_rejected": True,
        "unknown_clip_ids_rejected": True,
        "duplicate_clip_ids_rejected": True,
        "prediction_failures_reported_separately": True,
        "invalid_metadata_preserved": True,
    }
    summary["results"] = summary["results"] if "results" in summary else []

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"Detailed results written to {output_path}")
    return 0


HINGLISH_PREDICTION_COLUMNS = frozenset({"clip_id", "prediction"})


if __name__ == "__main__":
    raise SystemExit(main())
