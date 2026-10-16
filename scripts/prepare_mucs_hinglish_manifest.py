#!/usr/bin/env python3
"""Prepare a small, reproducible Hinglish manifest from MUCS metadata.

This script streams dataset metadata only. It does not download audio and does
not run model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import replace

from src.hinglish.data import (
    MUCS_HINGLISH_DATASET,
    MUCS_HINGLISH_REVISION,
    export_mucs_manifest,
    load_mucs_hinglish_manifest,
    select_one_per_speaker,
)

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream MUCS Hinglish metadata into a MendSpeech manifest"
    )
    parser.add_argument("--dataset", default=MUCS_HINGLISH_DATASET)
    parser.add_argument("--source-split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default=None,
        help="Override the manifest split label; defaults to --source-split.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=25,
        help="Metadata rows to stream. Use 0 to stream the available source split.",
    )
    parser.add_argument(
        "--one-per-speaker",
        action="store_true",
        help="Select one deterministic record for each distinct speaker.",
    )
    parser.add_argument(
        "--speaker-limit",
        type=int,
        default=0,
        help="Maximum speakers for --one-per-speaker (0 means every speaker).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "hinglish" / "mucs_hinglish_manifest.csv",
    )
    args = parser.parse_args()

    if args.max_items < 0:
        raise ValueError("--max-items must be non-negative")

    actual_split = args.split or args.source_split
    stream_limit = None if args.max_items == 0 else args.max_items
    records = load_mucs_hinglish_manifest(
        args.source_split,
        max_items=stream_limit,
        dataset_name=args.dataset,
        recompute_policy="count_ratio",
    )
    if not records:
        print("No records loaded")
        return 1
    if args.one_per_speaker:
        records = select_one_per_speaker(
            records,
            speaker_limit=args.speaker_limit if args.speaker_limit > 0 else None,
        )
    if actual_split != args.source_split:
        records = [
            replace(record, split=actual_split) for record in records
        ]

    export_path = args.output.resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(export_mucs_manifest(records))
    frame.to_csv(export_path, index=False)

    print(f"Wrote {len(records)} metadata-only records to {export_path}")
    print(f"Speakers: {len({record.speaker_id for record in records})}")
    print("Audio was not downloaded.")
    print("Audit this manifest before model inference.")
    print(
        {
            "dataset": args.dataset,
            "split": args.split,
            "records": len(records),
            "revision": MUCS_HINGLISH_REVISION,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
