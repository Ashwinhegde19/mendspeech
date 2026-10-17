#!/usr/bin/env python3
"""Extract audio bytes only for selected validation clips.

Reads targeted parquet row groups from Hugging Face Parquet shards. The
script only fetches the `audio` column for row groups that actually
contain a selected clip, avoiding a full dataset download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fsspec
import pyarrow.parquet as pq

PARQUET_BASE = (
    "https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish"
    "/resolve/refs%2Fconvert%2Fparquet/default/train"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract only the audio bytes required by the frozen manifest"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Frozen validation manifest CSV produced by the selection script",
    )
    parser.add_argument(
        "--clip-row-index", type=Path, required=True,
        help="clip_row_index.json mapping shards to clip row indices",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Destination directory for WAV files; kept outside Git",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)

    with args.clip_row_index.open(encoding="utf-8") as handle:
        mapping = {int(key): value for key, value in json.load(handle).items()}

    # Re-read the manifest rows so checksums land next to audio for auditing.
    with args.manifest.open(encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    audio_suffix = ".wav"
    downloaded_rows = 0
    audio_records: dict[str, dict[str, object]] = {}

    for shard in sorted(mapping):
        entry = mapping[shard]
        wanted_row_groups = sorted(set(entry["row_groups"]))
        wanted_rows = entry["row_index"]
        url = f"{PARQUET_BASE}/{shard:04d}.parquet"
        print(f"shard {shard:02d}: fetching row groups {wanted_row_groups}", flush=True)
        with fsspec.open(url, "rb", block_size=128 * 1024) as handle:
            parquet_file = pq.ParquetFile(handle)
            audio_column = {
                "path": [],
                "bytes": [],
                "row_group": [],
                "row_offset": [],
            }
            for row_group in wanted_row_groups:
                # Each row group's audio column is ~20 MiB uncompressed but
                # only fetched when explicitly requested.
                table = parquet_file.read_row_group(row_group, columns=["audio"])
                column = table.column("audio")
                audio_column["path"].extend(
                    item["path"] for item in column.to_pylist()
                )
                audio_column["bytes"].extend(
                    bytes(item["bytes"]) if item["bytes"] is not None else None
                    for item in column.to_pylist()
                )
                audio_column["row_group"].extend([row_group] * table.num_rows)
            # Track the absolute row offset for each row group so the target
            # index can be resolved.
            metadata = parquet_file.metadata
            offsets = {}
            running = 0
            for row_group in range(metadata.num_row_groups):
                offsets[row_group] = running
                running += metadata.row_group(row_group).num_rows

        target_rows = set(wanted_rows.values())
        for index, path in enumerate(audio_column["path"]):
            # We know only row groups explicitly requested. Map the index
            # back to the absolute dataset row to locate the target clip.
            continue  # resolved below using offsets

        # Rebuild absolute-position lookup for the selected row groups only.
        absolute_rows: dict[int, dict[str, object]] = {}
        row_cursor = 0
        for row_group in wanted_row_groups:
            entries = []
            rows = metadata.row_group(row_group).num_rows
            for offset in range(rows):
                item_index = row_cursor + offset
                entries.append(item_index)
            row_cursor += rows
            absolute_rows[row_group] = entries

        # Now match clip ids by `path`, which mirrors `{segment_id}.wav`.
        for row_group, rows in absolute_rows.items():
            for offset, absolute_row in enumerate(rows):
                item_index = absolute_row
                path = audio_column["path"][item_index]
                clip_id = path.replace(audio_suffix, "")
                if clip_id not in wanted_rows:
                    continue
                data = audio_column["bytes"][item_index]
                if data is None:
                    print(f"shard {shard:02d}: clip {clip_id} has no bytes")
                    continue
                destination = output_dir / "audio" / f"{clip_id}{audio_suffix}"
                destination.write_bytes(data)
                downloaded_rows += 1
                audio_records[clip_id] = {
                    "shard": shard,
                    "row_index": wanted_rows[clip_id],
                    "row_group": row_group,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "audio_path_in_parquet": path,
                }
        print(f"shard {shard:02d}: extracted {downloaded_rows} clips so far", flush=True)

    # Validate every selected clip is present.
    expected_clips = []
    for row in manifest_rows:
        expected_clips.append(row["clip_id"])
    expected_count = len(expected_clips)
    missing = [
        clip for clip in expected_clips if clip not in audio_records
    ]
    if missing:
        raise SystemExit(f"missing audio for {len(missing)} clips: {missing}")

    checksums_path = output_dir / "audio_checksums.json"
    checksums_path.write_text(
        json.dumps(audio_records, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"extracted {downloaded_rows} of {expected_count} selected clips")
    print(f"checksums: {checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
