#!/usr/bin/env python3
"""Run offline Hinglish baseline hallucination diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hinglish.data import MucsHinglishRecord
from src.hinglish.diagnostics import (
    diagnose_predictions,
    summarize_diagnostics,
)
from src.hinglish.evaluation import HinglishPrediction
from src.hinglish.normalization import HINGLISH_NORMALIZATION_VERSION


def _predictions_from_csv(predictions, manifest) -> list[HinglishPrediction]:
    lookup = manifest.set_index("clip_id").to_dict(orient="index")
    records = []
    for row in predictions.to_dict(orient="records"):
        manifest_row = lookup[str(row["clip_id"])]
        records.append(
            HinglishPrediction(
                clip_id=str(row["clip_id"]),
                speaker_id=str(manifest_row.get("speaker_id", "")),
                reference=str(manifest_row["transcript"]),
                prediction=str(row["prediction"]),
                ratio_english_words=float(manifest_row.get("ratio_english_words", 0.5)),
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Hinglish baseline diagnostics")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "hinglish" / "baseline_diagnostics.json",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    records = _predictions_from_csv(predictions, manifest)
    diagnostics = diagnose_predictions(records)
    summary = summarize_diagnostics(diagnostics)
    payload = {
        "dataset": "dianavdavidson/MUCS-Hinglish",
        "normalization_version": HINGLISH_NORMALIZATION_VERSION,
        "summary": summary,
        "records": diagnostics,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps(summary, indent=2, ensure_ascii=False))
    print(f"Diagnostics written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY