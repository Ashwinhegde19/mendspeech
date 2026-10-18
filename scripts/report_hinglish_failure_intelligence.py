#!/usr/bin/env python3
"""Build the reusable Hinglish failure intelligence report.

Consumes the private run artifacts (predictions, frozen manifest, generation
inspection report) plus the committed baseline metrics for a consistency
cross-check, and writes a JSON report and a Markdown reading copy that answer
the seven failure-intelligence questions. Reference transcripts and
hypotheses are read but never written into the outputs; per-clip rows carry
clip IDs, counts, and evidence text only.

Typical use::

    python scripts/inspect_hinglish_generation_failures.py \
        --raw-output <run>/raw_modal_output.json \
        --manifest <freeze>/frozen_validation_manifest.csv \
        --diagnostics <run>/hallucination_diagnostics.json \
        --output /tmp/inspection.json

    python scripts/report_hinglish_failure_intelligence.py \
        --predictions <run>/predictions.csv \
        --manifest <freeze>/frozen_validation_manifest.csv \
        --inspection-json /tmp/inspection.json \
        --metrics results/hinglish/validation_baseline_metrics.json \
        --output-json results/hinglish/failure_intelligence.json \
        --output-md results/hinglish/failure_intelligence.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.hinglish.evaluation import HinglishPrediction, score_prediction
from src.hinglish.failure_intelligence import (
    ClipErrorProfile,
    GenerationFailureSignals,
    build_failure_intelligence_payload,
    classify_failure,
    render_markdown_report,
    script_mismatch_statistics,
)

MANIFEST_REFERENCE_ALIASES = ("transcript", "reference_transcript")
DEFAULT_TOLERANCE = 5.1e-05

GENERATION_CONFIGURATION_NOTE = (
    "infra/modal_hinglish_asr.py ran the transformers ASR pipeline with "
    "generate_kwargs={'task': 'transcribe', 'language': 'hi'} and no "
    "length, timestamp, repetition, or EOS controls, so the only stop "
    "conditions were EOS or the Whisper default max_length (448 decoder "
    "tokens); all clips are shorter than 30 seconds, so a single decode "
    "window applies."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    reference_column = None
    for alias in MANIFEST_REFERENCE_ALIASES:
        if alias in manifest.columns:
            reference_column = alias
            break
    if reference_column is None:
        raise SystemExit(
            "manifest needs a transcript or reference_transcript column"
        )
    if reference_column != "transcript":
        manifest = manifest.rename(columns={reference_column: "transcript"})
    required = {"clip_id", "speaker_id", "ratio_english_words"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    empty = manifest["transcript"].isna() | (
        manifest["transcript"].astype(str).str.strip() == ""
    )
    if empty.any():
        raise SystemExit(
            "empty reference transcripts for: "
            + ", ".join(sorted(manifest.loc[empty, "clip_id"].astype(str))[:5])
        )
    ratios = manifest["ratio_english_words"]
    if ratios.isna().any() or (ratios < 0).any() or (ratios > 1).any():
        raise SystemExit("manifest ratio_english_words must be within [0, 1]")
    return manifest


def _load_predictions(path: Path) -> Dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"clip_id", "prediction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"predictions CSV missing columns: {sorted(missing)}")
    duplicates = sorted(
        set(frame.loc[frame["clip_id"].duplicated(), "clip_id"].astype(str))
    )
    if duplicates:
        raise SystemExit(
            "duplicate clip IDs in predictions: " + ", ".join(duplicates[:10])
        )
    return {
        str(row["clip_id"]): str(row["prediction"])
        for _, row in frame.iterrows()
    }


def _load_inspection(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "per_clip" not in payload or "runtime_ceiling" not in payload:
        raise SystemExit(
            "inspection JSON needs per_clip and runtime_ceiling sections; "
            "generate it with scripts/inspect_hinglish_generation_failures.py"
        )
    return payload


def _loop_coverage(row: Mapping[str, Any]) -> int:
    candidates = [
        row.get("loop_region_coverage_words"),
        row.get("phrase_repeat_coverage_words"),
        row.get("short_loop_coverage_words"),
        row.get("periodic_word_coverage_words"),
    ]
    values = [int(value) for value in candidates if isinstance(value, (int, float))]
    return max(values) if values else 0


def _build_profiles(
    manifest: pd.DataFrame,
    predictions: Mapping[str, str],
    inspection: Mapping[str, Any],
) -> List[ClipErrorProfile]:
    ceiling = set(map(str, inspection.get("runtime_ceiling", {}).get("members", [])))
    flagged = set(map(str, inspection.get("flagged_clip_ids", [])))
    inspection_rows = {
        str(row["clip_id"]): row for row in inspection.get("per_clip", [])
    }
    unknown = set(predictions) - set(manifest["clip_id"].astype(str))
    missing = set(manifest["clip_id"].astype(str)) - set(predictions)
    if unknown:
        raise SystemExit(
            "predictions contain clips outside the manifest: "
            + ", ".join(sorted(unknown)[:5])
        )
    if missing:
        raise SystemExit(
            "manifest clips missing predictions: " + ", ".join(sorted(missing)[:5])
        )
    profiles: List[ClipErrorProfile] = []
    for _, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        reference = str(row["transcript"])
        hypothesis = predictions[clip_id]
        scored = score_prediction(
            HinglishPrediction(
                clip_id=clip_id,
                speaker_id=str(row["speaker_id"]),
                reference=reference,
                prediction=hypothesis,
                ratio_english_words=float(row["ratio_english_words"]),
            )
        )
        inspection_row = inspection_rows.get(clip_id, {})
        signals = GenerationFailureSignals(
            cap_hit=clip_id in ceiling,
            has_word_loop=bool(inspection_row.get("has_word_loop", False)),
            has_char_loop=bool(inspection_row.get("has_char_loop", False)),
            has_repetition_evidence=bool(
                inspection_row.get("has_repetition_evidence", False)
            ),
            heuristic_flag=clip_id in flagged
            or bool(inspection_row.get("flagged", False)),
            loop_coverage_words=_loop_coverage(inspection_row),
            output_words=int(inspection_row.get("output_words", 0)),
        )
        profiles.append(
            ClipErrorProfile(
                clip_id=clip_id,
                speaker_id=str(row["speaker_id"]),
                reference_words=int(scored["reference_words"]),
                hypothesis_words=int(scored["hypothesis_words"]),
                substitutions=int(scored["substitutions"]),
                deletions=int(scored["deletions"]),
                insertions=int(scored["insertions"]),
                wer=float(scored["wer"]),
                english_ratio_bin=str(scored["english_ratio_bin"]),
                script=script_mismatch_statistics(reference, hypothesis),
                generation=signals,
            )
        )
    return profiles


def _cross_check_metrics(
    metrics_path: Path, profiles: List[ClipErrorProfile], tolerance: float
) -> Dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = {str(row["clip_id"]): row for row in metrics.get("per_clip", [])}
    mismatches: List[str] = []
    for profile in profiles:
        row = rows.get(profile.clip_id)
        if row is None:
            mismatches.append(f"{profile.clip_id}: absent from committed metrics")
            continue
        fields = {
            "wer": ("wer", profile.wer, tolerance),
            "substitutions": ("substitutions", profile.substitutions, 0),
            "deletions": ("deletions", profile.deletions, 0),
            "insertions": ("insertions", profile.insertions, 0),
        }
        for label, (column, computed, tol) in fields.items():
            committed = row.get(column)
            if committed is None:
                continue
            if abs(float(committed) - float(computed)) > tol:
                mismatches.append(
                    f"{profile.clip_id}: {label} committed={committed} "
                    f"computed={computed}"
                )
    if mismatches:
        print("Input artifacts disagree with the committed metrics:", file=sys.stderr)
        for item in mismatches[:10]:
            print(f"  {item}", file=sys.stderr)
        raise SystemExit(2)
    return {
        "checked": True,
        "match": True,
        "tolerance": tolerance,
        "clip_count": len(profiles),
    }


def _load_inference_config(metrics_path: Optional[Path]) -> Dict[str, Any]:
    if metrics_path is None:
        return {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics.get("inference_config")
    return dict(config) if isinstance(config, Mapping) else {}


def _provenance(
    predictions_path: Path,
    manifest_path: Path,
    inspection_path: Path,
    metrics_path: Optional[Path],
    inspection: Mapping[str, Any],
    inference_config: Mapping[str, Any],
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "predictions_sha256": _sha256_file(predictions_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "inspection_sha256": _sha256_file(inspection_path),
        "generation_configuration": GENERATION_CONFIGURATION_NOTE,
        "labels": {
            "structural_evidence": "decoder runtime structure plus loop evidence",
            "heuristic_evidence": "content heuristics, human review advised",
            "heuristic_estimate": "approximate script-aware alignment",
        },
    }
    if inference_config:
        provenance["baseline_inference_config"] = dict(inference_config)
    sources = inspection.get("sources")
    if isinstance(sources, Mapping):
        provenance["inspection_sources"] = dict(sources)
    if metrics_path is not None:
        provenance["metrics_sha256"] = _sha256_file(metrics_path)
    return provenance


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Hinglish failure intelligence report from private "
            "run artifacts (clip-ID outputs only; transcripts never written)"
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--inspection-json",
        type=Path,
        required=True,
        help=(
            "Generation inspection report produced by "
            "scripts/inspect_hinglish_generation_failures.py"
        ),
    )
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    predictions = _load_predictions(args.predictions)
    inspection = _load_inspection(args.inspection_json)
    profiles = _build_profiles(manifest, predictions, inspection)
    classifications = {
        profile.clip_id: classify_failure(profile) for profile in profiles
    }
    consistency: Dict[str, Any] = {"checked": False}
    if args.metrics is not None:
        consistency = _cross_check_metrics(args.metrics, profiles, args.tolerance)
    payload = build_failure_intelligence_payload(
        profiles,
        classifications,
        _provenance(
            args.predictions,
            args.manifest,
            args.inspection_json,
            args.metrics,
            inspection,
            _load_inference_config(args.metrics),
        ),
    )
    payload["consistency"] = consistency
    markdown = render_markdown_report(payload)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    dominant = payload["questions"]["dominant_error_type"]
    failed = payload["questions"]["which_utterances_failed"]
    counts = payload["questions"]["classification_evidence"]["clip_counts"]
    print(f"clips: {failed['clip_count']}; failed: {failed['failed_clip_count']}")
    print("clip classes: " + json.dumps(counts, sort_keys=True))
    print(f"dominant error type: {dominant['dominant_error_type']}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
