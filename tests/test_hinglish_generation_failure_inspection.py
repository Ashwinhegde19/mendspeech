"""Synthetic tests for Whisper generation-failure inspection statistics.

Every text used here is invented for the test. No private transcript, reference,
or model output is read, and no network or model download is involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_hinglish_generation_failures import (
    build_inspection_report,
    coefficient_of_variation,
    contrast_statistics,
    decoupling_statistics,
    detect_runtime_ceiling,
    flagged_clip_ids,
    intra_word_loops,
    load_manifest_rows,
    longest_consecutive_repeat,
    longest_periodic_run,
    longest_repeated_phrase,
    output_statistics,
    parse_raw_output,
    pearson_correlation,
    periodic_word_coverage,
    terminal_statistics,
)

CLI = REPO_ROOT / "scripts" / "inspect_hinglish_generation_failures.py"
REPO_MARKER = "ZZPRIVATEZZ"


def _raw_record(clip_id: str, transcript: str, runtime: float, duration: float) -> dict:
    return {
        "clip_id": clip_id,
        "transcript": transcript,
        "runtime_sec": runtime,
        "audio_duration_sec": duration,
    }


def test_longest_consecutive_repeat_counts_runs() -> None:
    assert longest_consecutive_repeat(["a", "a", "a", "b"]) == 3
    assert longest_consecutive_repeat(["a", "b"]) == 1
    assert longest_consecutive_repeat([]) == 0


def test_longest_periodic_run_finds_two_token_loop_to_the_end() -> None:
    run = longest_periodic_run(["a", "b", "a", "b", "a", "b"], max_period=2, min_repeats=3)
    assert run["found"] is True
    assert run["period_tokens"] == 2
    assert run["covered_tokens"] == 6
    assert run["repeats"] == 3
    assert run["reaches_end"] is True


def test_longest_periodic_run_reports_trailing_words() -> None:
    run = longest_periodic_run(["a", "b", "a", "b", "c"], max_period=2, min_repeats=2)
    assert run["period_tokens"] == 2
    assert run["covered_tokens"] == 4
    assert run["trailing_tokens"] == 1
    assert run["reaches_end"] is False


def test_longest_periodic_run_ignores_short_sequences() -> None:
    run = longest_periodic_run(["a"], max_period=4, min_repeats=2)
    assert run["found"] is False
    assert run["covered_tokens"] == 0


def test_longest_repeated_phrase_counts_all_copies() -> None:
    phrase = longest_repeated_phrase(["a", "b", "a", "b", "a", "b"])
    assert phrase["phrase_words"] == 2
    assert phrase["copies"] == 3
    assert phrase["coverage_words"] == 6
    assert phrase["reaches_end"] is True


def test_longest_repeated_phrase_keeps_larger_phrase_over_smaller() -> None:
    phrase = longest_repeated_phrase(["x", "y", "z", "x", "y", "z"])
    assert phrase["phrase_words"] == 3
    assert phrase["copies"] == 2
    assert phrase["coverage_words"] == 6


def test_longest_repeated_phrase_absent_returns_zeros() -> None:
    phrase = longest_repeated_phrase(["a", "b", "c", "d"])
    assert phrase["phrase_words"] == 0
    assert phrase["copies"] == 0
    assert phrase["reaches_end"] is False


def test_intra_word_loops_detects_single_word_loop() -> None:
    loops = intra_word_loops(["ab" * 10, "hello"], max_period=2, min_repeats=4)
    assert loops["found"] is True
    assert loops["word_index"] == 0
    assert loops["period_chars"] == 2
    assert loops["covered_characters"] == 20
    assert loops["repeats"] == 10


def test_intra_word_loops_ignores_uniform_words() -> None:
    loops = intra_word_loops(["devanagari", "words", "only"], max_period=2, min_repeats=4)
    assert loops["found"] is False
    assert loops["covered_characters"] == 0


def test_periodic_word_coverage_marks_every_looping_token() -> None:
    coverage = periodic_word_coverage(["a", "b", "a", "b", "a", "b"])
    assert coverage["share"] == 1.0
    assert coverage["covered_tokens"] == 6
    assert coverage["max_chain_tokens"] == 6


def test_periodic_word_coverage_ignores_distinct_content() -> None:
    coverage = periodic_word_coverage(["a", "b", "c", "d", "e"])
    assert coverage["share"] == 0.0
    assert coverage["found"] is False


def test_output_statistics_flags_word_loop() -> None:
    statistics = output_statistics("hello world hello world hello world hello world")
    assert statistics["output_words"] == 8
    assert statistics["has_word_loop"] is True
    assert statistics["has_repetition_evidence"] is True
    assert statistics["periodic_word_share"] == 1.0
    assert statistics["loop_active_at_stop"] is True
    assert statistics["degenerate_at_stop"] is True


def test_output_statistics_flags_character_loop_in_final_word() -> None:
    statistics = output_statistics("hello " + "क" * 40)
    assert statistics["output_words"] == 2
    assert statistics["has_char_loop"] is True
    assert statistics["intra_word_loop_characters"] >= 20
    assert statistics["intra_word_loop_at_final_word"] is True
    assert statistics["degenerate_at_stop"] is True


def test_output_statistics_clean_text_has_no_repetition_evidence() -> None:
    statistics = output_statistics("libre office calc mein formula aur functions")
    assert statistics["has_repetition_evidence"] is False
    assert statistics["periodic_word_share"] == 0.0
    assert statistics["degenerate_at_stop"] is False


def test_terminal_statistics_detects_missing_punctuation() -> None:
    plain = terminal_statistics("hello world", ["hello", "world"])
    assert plain["ends_with_terminal_punctuation"] is False
    assert plain["final_word_characters"] == 5
    assert plain["final_character_is_combining_mark"] is False

    marked = terminal_statistics("नमस्ते है।", ["नमस्ते", "है।"])
    assert marked["ends_with_terminal_punctuation"] is True


def test_terminal_statistics_detects_mid_word_cut() -> None:
    cut = terminal_statistics("नमस्ते वि", ["नमस्ते", "वि"])
    assert cut["final_character_is_combining_mark"] is True
    assert cut["ends_with_terminal_punctuation"] is False


def test_output_statistics_exposes_truncation_signal() -> None:
    open_ended = output_statistics("hello world hello world hello world")
    assert open_ended["ends_with_terminal_punctuation"] is False
    completed = output_statistics("hello world.")
    assert completed["ends_with_terminal_punctuation"] is True


def test_detect_runtime_ceiling_finds_plateau() -> None:
    runtimes = {
        "clip_a": 0.17,
        "clip_b": 0.9,
        "clip_c": 1.56,
        "clip_d": 3.687,
        "clip_e": 3.700,
        "clip_f": 3.734,
    }
    ceiling = detect_runtime_ceiling(runtimes)
    assert ceiling["found"] is True
    assert ceiling["size"] == 3
    assert ceiling["members"] == ["clip_d", "clip_e", "clip_f"]
    assert ceiling["spread_sec"] == 0.047
    assert ceiling["gap_below_sec"] == 2.127
    assert ceiling["share_of_clips"] == 0.5


def test_detect_runtime_ceiling_absent_for_spread_values() -> None:
    ceiling = detect_runtime_ceiling({"a": 0.1, "b": 1.0, "c": 2.0, "d": 3.0})
    assert ceiling["found"] is False
    assert ceiling["members"] == []


def test_pearson_correlation_handles_edges() -> None:
    assert pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson_correlation([1.0], [1.0]) is None


def test_coefficient_of_variation_handles_edges() -> None:
    assert coefficient_of_variation([1.0, 1.0, 1.0]) == 0.0
    assert coefficient_of_variation([2.0, 4.0, 6.0]) == pytest.approx(0.5)
    assert coefficient_of_variation([1.0]) is None
    assert coefficient_of_variation([0.0, 0.0]) is None


def _per_clip_row(clip_id: str, runtime: float, words: int, characters: int) -> dict:
    return {
        "clip_id": clip_id,
        "runtime_sec": runtime,
        "output_words": words,
        "output_characters": characters,
        "audio_duration_sec": 6.0,
    }


def test_decoupling_statistics_detects_flat_runtime() -> None:
    per_clip = [
        _per_clip_row("a", 3.690, 3, 30),
        _per_clip_row("b", 3.734, 40, 300),
        _per_clip_row("c", 3.700, 140, 330),
    ]
    decoupling = decoupling_statistics(per_clip, ["a", "b", "c"])
    assert decoupling["decoupled_from_output_size"] is True
    assert decoupling["ceiling_clips"]["output_words_ratio"] > 40
    assert decoupling["ceiling_clips"]["runtime_sec_coefficient_of_variation"] < 0.01


def test_decoupling_statistics_rejects_proportional_runtime() -> None:
    per_clip = [
        _per_clip_row("a", 0.5, 5, 40),
        _per_clip_row("b", 1.0, 10, 80),
        _per_clip_row("c", 1.5, 15, 120),
    ]
    decoupling = decoupling_statistics(per_clip, ["a", "b", "c"])
    assert decoupling["decoupled_from_output_size"] is False
    assert decoupling["non_ceiling_clips"]["clips"] == 0


def test_contrast_statistics_reports_group_medians_and_bins() -> None:
    per_clip = [
        {**_per_clip_row("a", 3.7, 70, 300), "english_ratio_bin": "code-switched"},
        {**_per_clip_row("b", 3.7, 60, 280), "english_ratio_bin": "code-switched"},
        {**_per_clip_row("c", 0.6, 10, 60), "english_ratio_bin": "hindi-dominant"},
        {**_per_clip_row("d", 0.7, 14, 80), "english_ratio_bin": "hindi-dominant"},
    ]
    contrast = contrast_statistics(per_clip, ["a", "b"])
    assert contrast["clip_counts"] == {"flagged": 2, "clean": 2}
    assert contrast["per_metric"]["output_words"]["flagged"]["median"] == 65.0
    assert contrast["per_metric"]["output_words"]["clean"]["median"] == 12.0
    assert contrast["per_bin"]["code-switched"] == {"flagged": 2, "clean": 0}
    assert contrast["per_bin"]["hindi-dominant"] == {"flagged": 0, "clean": 2}


def test_parse_raw_output_skips_log_preamble(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    payload = [_raw_record("clip_a", "hello", 1.0, 5.0)]
    path.write_text(
        "Launching batched Modal inference on T4...\n" + json.dumps(payload),
        encoding="utf-8",
    )
    records = parse_raw_output(path)
    assert len(records) == 1
    assert records[0]["clip_id"] == "clip_a"


def test_load_manifest_and_flagged_ids() -> None:
    manifest = {
        "clip_a": {"clip_id": "clip_a", "ratio_english_words": "0.3"},
        "clip_b": {"clip_id": "clip_b", "ratio_english_words": "0.9"},
    }
    diagnostics = {
        "summary": {"likely_hallucination_ids": ["clip_b"]},
        "records": [{"clip_id": "clip_b", "likely_repetitive_hallucination": True}],
    }
    assert flagged_clip_ids(diagnostics) == ["clip_b"]
    assert flagged_clip_ids({"records": [{"clip_id": "clip_a", "likely_repetitive_hallucination": False}]}) == []
    assert set(manifest) == {"clip_a", "clip_b"}


def test_load_manifest_rows_reads_clip_index(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "clip_id,reference_transcript,ratio_english_words\n"
        'clip_a,"some reference text",0.25\n',
        encoding="utf-8",
    )
    rows = load_manifest_rows(manifest_path)
    assert list(rows) == ["clip_a"]
    assert rows["clip_a"]["ratio_english_words"] == "0.25"


def test_build_inspection_report_excludes_transcript_text() -> None:
    raw_records = [
        _raw_record(f"clip_{index}", f"{REPO_MARKER} loop loop loop", 3.7, 5.0)
        for index in range(3)
    ]
    raw_records.append(_raw_record("clip_clean", f"{REPO_MARKER} natural speech", 0.5, 5.0))
    manifest_rows = {
        f"clip_{index}": {
            "clip_id": f"clip_{index}",
            "reference_transcript": f"{REPO_MARKER} reference words here",
            "ratio_english_words": "0.4",
        }
        for index in range(4)
    }
    report = build_inspection_report(raw_records, manifest_rows)
    payload = json.dumps(report)
    assert REPO_MARKER not in payload
    assert report["clip_count"] == 4
    assert report["runtime_ceiling"]["size"] == 3
    assert report["evidence_counts"]["clips_with_repetition_evidence"] == 3
    assert report["contrast"] is None


def test_build_inspection_report_uses_diagnostics_flags() -> None:
    raw_records = [
        _raw_record("clip_a", "word word word word", 3.7, 5.0),
        _raw_record("clip_b", "clean sentence here", 0.6, 5.0),
    ]
    manifest_rows = {
        "clip_a": {"clip_id": "clip_a", "reference_transcript": "word", "ratio_english_words": "0.25"},
        "clip_b": {"clip_id": "clip_b", "reference_transcript": "clean sentence", "ratio_english_words": "0.9"},
    }
    diagnostics = {
        "summary": {"likely_hallucination_ids": ["clip_a"]},
        "records": [{"clip_id": "clip_a", "likely_repetitive_hallucination": True, "insertions": 3}],
    }
    report = build_inspection_report(raw_records, manifest_rows, diagnostics)
    assert report["flagged_clip_ids"] == ["clip_a"]
    per_clip = {row["clip_id"]: row for row in report["per_clip"]}
    assert per_clip["clip_a"]["flagged"] is True
    assert per_clip["clip_a"]["diagnostic_insertions"] == 3
    assert per_clip["clip_b"]["english_ratio_bin"] == "english-dominant"


def test_cli_writes_clip_id_keyed_report(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    manifest_path = tmp_path / "manifest.csv"
    output_path = tmp_path / "report.json"
    raw_path.write_text(
        "log line\n"
        + json.dumps(
            [
                _raw_record("clip_a", f"{REPO_MARKER} loop loop loop loop", 3.7, 5.0),
                _raw_record("clip_b", f"{REPO_MARKER} short natural speech", 0.6, 5.0),
            ]
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        "clip_id,reference_transcript,ratio_english_words\n"
        f'clip_a,"{REPO_MARKER} reference words",0.3\n'
        f'clip_b,"{REPO_MARKER} other reference",0.3\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--raw-output",
            str(raw_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["clip_count"] == 2
    assert REPO_MARKER not in json.dumps(payload)
    assert "clip_id" in json.dumps(payload)
