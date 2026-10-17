#!/usr/bin/env python3
"""Build a metadata-only MUCS Hinglish training inventory from remote Parquet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = (
    "https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish"
    "/resolve/refs%2Fconvert%2Fparquet/default/train"
)
COLUMNS = [
    "segment_id",
    "speaker_id",
    "transcript",
    "ratio_english_words",
    "ratio_hindi_words",
    "count_english_words",
    "count_dev_words",
    "segment_part",
]
REVISION = "3c66ee127a7b0e0a746d4df1002858567135edc1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream remote MUCS training metadata columns without audio"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=23)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=None)
    args = parser.parse_args()

    if args.start_shard < 0 or args.shard_count < 1:
        raise ValueError("invalid shard range")

    end_shard = args.end_shard if args.end_shard is not None else args.shard_count
    if end_shard <= args.start_shard:
        raise ValueError("end_shard must be greater than start_shard")
    if end_shard > args.shard_count:
        raise ValueError("end_shard exceeds shard count")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".checkpoint.json")

    completed: set[int] = set()
    if checkpoint.exists():
        try:
            completed = set(json.loads(checkpoint.read_text()).get("completed_shards", []))
        except json.JSONDecodeError:
            completed = set()

    fieldnames = COLUMNS + ["shard"]
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()

        rows_written = 0
        for shard_index in range(args.start_shard, end_shard):
            if shard_index in completed:
                print(f"shard {shard_index:04d}: already completed", flush=True)
                continue
            url = f"{BASE_URL}/{shard_index:04d}.parquet"
            print(f"shard {shard_index:04d}: starting {url}", flush=True)
            with fsspec.open(url, "rb", block_size=128 * 1024) as remote_file:
                parquet_file = pq.ParquetFile(remote_file)
                table = parquet_file.read(columns=COLUMNS)
                frame = table.to_pandas()
            shard_rows = 0
            for record in frame.to_dict("records"):
                record["shard"] = shard_index
                writer.writerow(record)
                shard_rows += 1
            handle.flush()
            rows_written += shard_rows
            completed.add(shard_index)
            checkpoint.write_text(
                json.dumps(
                    {
                        "dataset": "dianavdavidson/MUCS-Hinglish",
                        "revision": REVISION,
                        "completed_shards": sorted(completed),
                        "rows_written": rows_written,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"shard {shard_index:04d}: wrote {shard_rows} rows, "
                f"cumulative={rows_written}",
                flush=True,
            )

    print(f"metadata inventory written to {output}")
    print(f"rows written: {rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY