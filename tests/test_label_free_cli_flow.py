"""CLI-flow tests for label-free selection plumbing (Phase 7.2).

These tests exercise the selection CLI in ``label_free`` acquisition mode
with synthetic pool manifests and synthetic scoring records. They pin:

- required inputs and refusals of prohibited inputs,
- the reference join check being clip-ID-only (no transcript values read),
- scoring coverage requirements (missing records hard-stop; failed records
  flow through and are rejected during selection, never scored),
- the frozen pool manifest being the single duration source,
- arms-overlap reporting for paired selection.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLI = REPO_ROOT / "scripts" / "select_hinglish_training_candidates.py"

POOL_COLUMNS = [
    "clip_id",
    "speaker_id",
    "audio_duration_sec",
    "shard",
    "segment_part",
    "mean_rms_dbfs",
    "peak_dbfs",
    "clipping_fraction",
    "picked_reason",
]
SPEAKERS = [f"spk{index:03d}" for index in range(8)]


def _write_pool(path: Path, *, durations: list[float] | None = None) -> list[str]:
    clip_ids = [f"{speaker}_seg0000" for speaker in SPEAKERS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POOL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index, (clip_id, speaker) in enumerate(zip(clip_ids, SPEAKERS)):
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "speaker_id": speaker,
                    "audio_duration_sec": (
                        durations[index] if durations else 5.0 + index * 0.5
                    ),
                    "shard": str(index % 2),
                    "segment_part": "0000",
                    "mean_rms_dbfs": -21.0,
                    "peak_dbfs": -3.0,
                    "clipping_fraction": 0.0,
                    "picked_reason": "nearest_8s",
                }
            )
    return clip_ids


def _write_inventory(path: Path, clip_ids: list[str], *, poison: str = "") -> None:
    # The inventory carries transcript-derived columns; label-free mode must
    # read only segment_id values. `poison` varies transcript-derived values
    # so leakage tests can flip them and assert selection invariance.
    fieldnames = [
        "segment_id",
        "speaker_id",
        "transcript",
        "ratio_english_words",
        "ratio_hindi_words",
        "count_english_words",
        "count_dev_words",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, clip_id in enumerate(clip_ids):
            writer.writerow(
                {
                    "segment_id": clip_id,
                    "speaker_id": SPEAKERS[index],
                    "transcript": f"POTοινιος {poison} {index}",
                    "ratio_english_words": 0.1 + 0.08 * index if poison else 0.5,
                    "ratio_hindi_words": 0.9 - 0.08 * index if poison else 0.5,
                    "count_english_words": (index + 3) * (2 if poison else 1),
                    "count_dev_words": (index + 1) * (3 if poison else 1),
                }
            )


def _write_split(path: Path) -> None:
    payload = {
        "seed": 42,
        "training_speakers": SPEAKERS[:-1],
        "validation_speakers": SPEAKERS[-1:],
        "diagnostic_speakers_reserved_for_training": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_scoring(
    path: Path,
    clip_ids: list[str],
    *,
    statuses: dict[str, str] | None = None,
    drop: set[str] | None = None,
    tokens: dict[str, int] | None = None,
) -> None:
    records = []
    for clip_id in clip_ids:
        if drop and clip_id in drop:
            continue
        status = (statuses or {}).get(clip_id, "ok")
        records.append(
            {
                "clip_id": clip_id,
                "status": status,
                "config": "C1",
                "audio_duration_sec": 5.0,
                "inference_runtime_sec": 1.2,
                "generated_token_count": (tokens or {}).get(clip_id, 60),
                "eos_emitted_effective": True,
                "hit_length_limit_effective": False,
                "max_consecutive_repeated_tokens": 1,
                "repeated_token_fraction": 0.02,
                "repetition_flag": False,
                "mean_logprob": -0.11,
                "min_logprob": -0.35,
                "entropy_mean": 0.9,
            }
        )
    payload = {
        "run": "label_free_scoring_v1",
        "config": "C1",
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base(tmp_path: Path, clip_ids: list[str], *, poison: str = "") -> list[str]:
    _write_inventory(tmp_path / "inventory.csv", clip_ids, poison=poison)
    _write_split(tmp_path / "split.json")
    return [
        sys.executable,
        str(CLI),
        "--inventory", str(tmp_path / "inventory.csv"),
        "--speaker-split", str(tmp_path / "split.json"),
        "--acquisition-mode", "label_free",
        "--require-duration",
        "--min-total-words", "0",
        "--pool-manifest", str(tmp_path / "pool.csv"),
        "--scoring", str(tmp_path / "scoring.json"),
        "--arm", "both",
        "--count", "2",
        "--output-dir", str(tmp_path / "out"),
    ]


def test_label_free_requires_pool_and_scoring(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_split(tmp_path / "split.json")
    _write_inventory(tmp_path / "inventory.csv", clip_ids)
    command = [
        sys.executable,
        str(CLI),
        "--inventory", str(tmp_path / "inventory.csv"),
        "--speaker-split", str(tmp_path / "split.json"),
        "--acquisition-mode", "label_free",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "--pool-manifest" in completed.stderr


def test_label_free_refuses_transcript_inputs(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_split(tmp_path / "split.json")
    command = [
        sys.executable,
        str(CLI),
        "--inventory", str(tmp_path / "absent.csv"),
        "--speaker-split", str(tmp_path / "split.json"),
        "--acquisition-mode", "label_free",
        "--require-duration",
        "--min-total-words", "0",
        "--pool-manifest", str(tmp_path / "pool.csv"),
        "--scoring", str(tmp_path / "s.json"),
        "--predictions-manifest", str(tmp_path / "x.csv"),
        "--output-dir", str(tmp_path / "out"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "never reads reference transcripts" in completed.stderr


def test_label_free_rejects_missing_scoring_records(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_scoring(tmp_path / "scoring.json", clip_ids, drop={clip_ids[0]})
    completed = subprocess.run(_base(tmp_path, clip_ids), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "missing" in completed.stderr


def test_label_free_failed_records_rejected_not_zeroed(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_scoring(tmp_path / "scoring.json", clip_ids, statuses={clip_ids[0]: "failed"})
    completed = subprocess.run(_base(tmp_path, clip_ids), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "out" / "selection_report.json").read_text(encoding="utf-8"))
    assert report["inputs"]["scoring_failed_clip_count"] == 1
    # 8 pool clips − 1 validation speaker − 1 scoring-failed = 6 eligible.
    assert report["pool"]["eligible"] == len(clip_ids) - 2


def test_label_free_selection_ignores_transcript_derived_values(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_scoring(tmp_path / "scoring.json", clip_ids)
    completed = subprocess.run(_base(tmp_path, clip_ids), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    poisoned_pool = tmp_path / "poisoned"
    poisoned_pool.mkdir()
    _write_pool(poisoned_pool / "pool.csv")
    _write_scoring(poisoned_pool / "scoring.json", clip_ids)
    poisoned = _base(poisoned_pool, clip_ids, poison="PRIMARY_LEAK_SECRET")
    completed_poisoned = subprocess.run(poisoned, capture_output=True, text=True)
    assert completed_poisoned.returncode == 0, completed_poisoned.stderr

    # Manifests must be byte-identical: flipping transcript text, language
    # ratios, and word counts cannot change label-free selection.
    for name in (
        "selection_manifest_intelligent.csv",
        "selection_manifest_random.csv",
    ):
        base = (tmp_path / "out" / name).read_text(encoding="utf-8")
        other = (poisoned_pool / "out" / name).read_text(encoding="utf-8")
        assert base == other

    def _selective(payload: dict) -> str:
        # The inventory checksum may legitimately differ (poisoned fixture);
        # every selection-relevant field must be identical.
        inputs = dict(payload["inputs"])
        inputs.pop("inventory_sha256", None)
        return json.dumps(inputs, sort_keys=True)

    base_report = json.loads(
        (tmp_path / "out" / "selection_report.json").read_text(encoding="utf-8")
    )
    other_report = json.loads(
        (poisoned_pool / "out" / "selection_report.json").read_text(encoding="utf-8")
    )
    assert _selective(base_report) == _selective(other_report)
    assert base_report["selection_version"] == other_report["selection_version"]
    for name in ("selection_report.json", "selection_manifest_intelligent.csv"):
        body = (tmp_path / "out" / name).read_text(encoding="utf-8")
        assert "PRIMARY_LEAK_SECRET" not in body


def test_label_free_paired_overlap_reported(tmp_path: Path) -> None:
    clip_ids = _write_pool(tmp_path / "pool.csv")
    _write_scoring(tmp_path / "scoring.json", clip_ids)
    completed = subprocess.run(_base(tmp_path, clip_ids), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "out" / "selection_report.json").read_text(encoding="utf-8"))
    overlap = report["clip_overlap_between_arms"]
    assert overlap["count"] == len(overlap["clip_ids"])
    assert report["selection_guardrails"]["arms_clip_overlap_reported"] is True
