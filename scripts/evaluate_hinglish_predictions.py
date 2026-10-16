#!/usr/bin/env python3
"""Score Hinglish ASR predictions against a MendSpeech manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.hinglish.evaluation import HinglishPrediction, score_prediction, summarize_predictions
from src.hinglish.normalization import HINGLISH_NORMALIZATION_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score Hinglish ASR predictions without re-running inference"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "hinglish" / "baseline_metrics.json",
    )
    args = parser.parse_args()

    predictions_frame = pd.read_csv(args.predictions)
    if "clip_id" not in predictions_frame.columns:
        raise ValueError("predictions CSV must contain clip_id")
    if "prediction" not in predictions_frame.columns:
        raise ValueError("predictions CSV must contain prediction")

    manifest_lookup: dict[str, dict[str, object]] = {}
    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        manifest_lookup = manifest.set_index("clip_id").to_dict(orient="index")

    prediction_records = []
    for row in predictions_frame.to_dict(orient="records"):
        clip_id = str(row["clip_id"])
        manifest_row = manifest_lookup.get(clip_id, {})
        prediction_records.append(
            HinglishPrediction(
                clip_id=clip_id,
                speaker_id=str(manifest_row.get("speaker_id", "")),
                reference=str(manifest_row.get("transcript", "")),
                prediction=str(row["prediction"]),
                ratio_english_words=float(manifest_row.get("ratio_english_words", 0.5)),
            )
        )

    summary = summarize_predictions(prediction_records)
    summary["normalization_version"] = HINGLISH_NORMALIZATION_VERSION
    summary["results"] = [score_prediction(record) for record in prediction_records]

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"Detailed results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
