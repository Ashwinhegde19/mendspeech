"""Analyze the Phase 7.1 decoder probe (J1) raw output into paired results.

This script is offline and read-only with respect to measured artifacts: it
reads the private probe raw output, the three private per-config metric
files, and the committed plan, then writes

- a rich private analysis file (still transcript-free; per-clip text lives
  only in the private predictions CSVs), and
- a committed safe aggregate (clip IDs, numbers, flags, hashes only).

For every configuration the script reports corpus and per-clip metrics, the
token-level generation diagnostics (generated count, budget, length-limit
hit, EOS evidence from ids and from the recorded chosen-token witness),
repetition patterns, runtime, and paired per-clip deltas against C0
(WER/CER changes, insertions/deletions changes, loop resolution, control
regression). Cost is reported from measured inference seconds plus a
documented container-lifetime estimate (Modal billing logs are unavailable
for ephemeral runs).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_hinglish_probe_selection import content_safety_violations

PROBE_VERSION = "mendspeech.hinglish_decoder_probe.v1"
RATE_T4_USD_PER_SEC = 0.000164
MODEL_LOAD_SEC = 90.0
CONTAINER_OVERHEAD_SEC = 60.0

_LATIN = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DEVANAGARI = set(chr(cp) for cp in range(0x0900, 0x0980))


def script_word_ratio(text: str) -> dict[str, float]:
    """Heuristic per-word script composition of a transcript (no references)."""
    words = text.split()
    latin = sum(1 for w in words if any(c in _LATIN for c in w) and not any(c in _DEVANAGARI for c in w))
    devanagari = sum(1 for w in words if any(c in _DEVANAGARI for c in w) and not any(c in _LATIN for c in w))
    mixed = sum(1 for w in words if any(c in _LATIN for c in w) and any(c in _DEVANAGARI for c in w))
    total = len(words)
    return {
        "hypothesis_words": total,
        "latin_words": latin,
        "devanagari_words": devanagari,
        "mixed_script_words": mixed,
        "latin_ratio": round(latin / total, 6) if total else 0.0,
        "devanagari_ratio": round(devanagari / total, 6) if total else 0.0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(
    raw: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Build the full analysis document from the loaded artifacts."""
    per_clip_plan = {c["clip_id"]: c for c in plan["clips"]}
    ids = [c["clip_id"] for c in plan["clips"]]
    records = raw["records"]
    by_key = {(r["config"], r["clip_id"]): r for r in records}
    rows: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, Any]] = {}
    for config_name in ("C0", "C1", "C2"):
        m = metrics[config_name]
        corpus[config_name] = {
            "utterances": m["utterances"],
            "reference_words": m["reference_words"],
            "corpus_wer": m["wer"],
            "corpus_cer": m["cer"],
            "mean_utterance_wer": m["mean_utterance_wer"],
            "mean_utterance_cer": m["mean_utterance_cer"],
            "total_edits": m["total_edits"],
            "failed_prediction_count": m.get("failed_prediction_count", 0),
            "inference_seconds": round(
                sum(by_key[(config_name, cid)]["runtime_sec"] for cid in ids), 3
            ),
            "script_distribution": {"latin_words": 0, "devanagari_words": 0, "mixed_script_words": 0},
            "token_totals": {"generated": 0, "hit_length_limit": 0, "eos_emitted": 0},
            "cap_hit_clips": 0,
        }
    for clip_id in ids:
        plan_row = per_clip_plan[clip_id]
        row: dict[str, Any] = {"clip_id": clip_id, "category": plan_row["category"]}
        for config_name in ("C0", "C1", "C2"):
            probe_rec = by_key[(config_name, clip_id)]
            instr = probe_rec["instrumentation"]
            probe = probe_rec.get("probe") or {}
            metric = next(r for r in metrics[config_name]["results"] if r["clip_id"] == clip_id)
            script = script_word_ratio(probe_rec["transcript"])
            rep = probe.get("token_repetition") or {}
            tail = rep.get("periodic_tail") or {}
            outcome = {
                "runtime_sec": probe_rec["runtime_sec"],
                "instrumentation_ok": instr["instrumentation_ok"],
                "generated_token_count": probe.get("generated_token_count"),
                "effective_token_budget": probe.get("effective_token_budget"),
                "hit_length_limit": probe.get("hit_length_limit"),
                "eos_from_ids": probe.get("eos_emitted"),
                "eos_scores_witness": instr.get("eos_emitted_scores_witness"),
                "eos_emitted_effective": instr.get("eos_emitted_effective"),
                "hit_length_limit_effective": instr.get("hit_length_limit_effective"),
                "last_chosen_token_id": instr.get("last_chosen_token_id"),
                "loop_period": tail.get("period"),
                "loop_copies": tail.get("copies"),
                "loop_coverage_ratio": tail.get("ratio"),
                "max_consecutive_token_repeats": rep.get("max_consecutive_token_repeats"),
                "longest_repeated_token_ngram_length": rep.get(
                    "longest_repeated_token_ngram_length"
                ),
                "longest_repeated_token_ngram_copies": rep.get(
                    "longest_repeated_token_ngram_copies"
                ),
                "hypothesis_words": metric["hypothesis_words"],
                "hypothesis_characters": metric["hypothesis_characters"],
                "reference_words": metric["reference_words"],
                "wer": metric["wer"],
                "cer": metric["cer"],
                "substitutions": metric["substitutions"],
                "deletions": metric["deletions"],
                "insertions": metric["insertions"],
                "script": script,
            }
            row[config_name] = outcome
            corpus[config_name]["inference_seconds"] = corpus[config_name][
                "inference_seconds"
            ]
            corpus[config_name]["script_distribution"]["latin_words"] += script["latin_words"]
            corpus[config_name]["script_distribution"]["devanagari_words"] += script["devanagari_words"]
            corpus[config_name]["script_distribution"]["mixed_script_words"] += script["mixed_script_words"]
            corpus[config_name]["token_totals"]["generated"] += probe.get("generated_token_count") or 0
            if outcome["hit_length_limit_effective"]:
                corpus[config_name]["token_totals"]["hit_length_limit"] += 1
            if outcome["eos_emitted_effective"]:
                corpus[config_name]["token_totals"]["eos_emitted"] += 1
            if outcome["hit_length_limit"]:
                corpus[config_name]["cap_hit_clips"] += 1
        c0 = row["C0"]
        for config_name in ("C1", "C2"):
            other = row[config_name]
            row[f"delta_{config_name}_vs_C0"] = {
                "wer_delta": round(other["wer"] - c0["wer"], 6),
                "cer_delta": round(other["cer"] - c0["cer"], 6),
                "insertions_delta": other["insertions"] - c0["insertions"],
                "deletions_delta": other["deletions"] - c0["deletions"],
                "substitutions_delta": other["substitutions"] - c0["substitutions"],
                "hypothesis_words_delta": other["hypothesis_words"] - c0["hypothesis_words"],
                "runtime_delta_sec": round(other["runtime_sec"] - c0["runtime_sec"], 3),
                "cap_hit_resolved": bool(c0["hit_length_limit"]) and not other["hit_length_limit"],
                "loop_resolved": (
                    bool(c0["loop_period"]) and other["loop_period"] is None
                ),
                "deletion_increase": max(0, other["deletions"] - c0["deletions"]),
                "regressed": other["wer"] > c0["wer"] + 1e-9,
            }
        rows.append(row)

    controls = [r for r in rows if r["category"] == "control"]
    summary_controls = {
        "count": len(controls),
        "unchanged_under_C1": sum(
            1 for r in controls if r["delta_C1_vs_C0"]["wer_delta"] == 0
            and r["delta_C1_vs_C0"]["hypothesis_words_delta"] == 0
        ),
        "unchanged_under_C2": sum(
            1 for r in controls if r["delta_C2_vs_C0"]["wer_delta"] == 0
            and r["delta_C2_vs_C0"]["hypothesis_words_delta"] == 0
        ),
        "regressed_under_C1": sum(1 for r in controls if r["delta_C1_vs_C0"]["regressed"]),
        "regressed_under_C2": sum(1 for r in controls if r["delta_C2_vs_C0"]["regressed"]),
    }
    cap = [r for r in rows if r["category"].startswith("ceiling")]
    early = [r for r in rows if r["category"] == "early_loop"]
    resolution = {
        "ceiling_clips": len(cap),
        "ceiling_cap_hit_resolved_under_C1": sum(
            1 for r in cap if r["delta_C1_vs_C0"]["cap_hit_resolved"]
        ),
        "ceiling_cap_hit_resolved_under_C2": sum(
            1 for r in cap if r["delta_C2_vs_C0"]["cap_hit_resolved"]
        ),
        "ceiling_loop_resolved_under_C2": sum(
            1 for r in cap if r["delta_C2_vs_C0"]["loop_resolved"]
        ),
        "ceiling_WER_improved_under_C1": sum(
            1 for r in cap if r["delta_C1_vs_C0"]["wer_delta"] < 0
        ),
        "ceiling_WER_improved_under_C2": sum(
            1 for r in cap if r["delta_C2_vs_C0"]["wer_delta"] < 0
        ),
        "ceiling_deletion_increase_under_C1": sum(
            r["delta_C1_vs_C0"]["deletion_increase"] for r in cap
        ),
        "ceiling_deletion_increase_under_C2": sum(
            r["delta_C2_vs_C0"]["deletion_increase"] for r in cap
        ),
        "early_loop_clips": len(early),
        "early_unchanged_under_C1": sum(
            1 for r in early if r["delta_C1_vs_C0"]["wer_delta"] == 0
            and r["delta_C1_vs_C0"]["hypothesis_words_delta"] == 0
        ),
        "early_unchanged_under_C2": sum(
            1 for r in early if r["delta_C2_vs_C0"]["wer_delta"] == 0
            and r["delta_C2_vs_C0"]["hypothesis_words_delta"] == 0
        ),
    }
    total_inference = sum(corpus[c]["inference_seconds"] for c in corpus)
    billed_estimate = {
        "measured_inference_seconds": round(total_inference, 3),
        "model_load_and_overhead_estimate_seconds": MODEL_LOAD_SEC + CONTAINER_OVERHEAD_SEC,
        "estimated_container_lifetime_seconds": round(
            total_inference + MODEL_LOAD_SEC + CONTAINER_OVERHEAD_SEC, 3
        ),
        "estimated_gpu_cost_usd": round(
            (total_inference + MODEL_LOAD_SEC + CONTAINER_OVERHEAD_SEC) * RATE_T4_USD_PER_SEC,
            5,
        ),
        "timeout_worst_case_cost_usd": round(420 * RATE_T4_USD_PER_SEC, 5),
        "note": (
            "Modal billing logs are unavailable for ephemeral runs; cost is an "
            "estimate from measured inference seconds plus documented load and "
            "overhead constants, labeled 'estimated', not an invoice"
        ),
    }
    return {
        "report_version": "mendspeech.hinglish_decoder_probe.report.v1",
        "probe_version": PROBE_VERSION,
        "constructed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"id": raw["model_id"], "revision": raw["model_revision"]},
        "precision": raw["precision"],
        "instrumentation": {
            "return_dict_in_generate": True,
            "output_scores": True,
            "outputs_transcript_changes": "none (C0 vs archived baseline exact 16/16; See invariance section)",
            "eos_token_ids": raw["eos_token_ids"],
            "forced_prefix_ids": raw["forced_prefix_ids"],
            "generation_config_max_length": raw["generation_config_max_length"],
        },
        "corpus_over_frozen_16": corpus,
        "summary": {
            "controls": summary_controls,
            "ceiling_and_early_resolution": resolution,
        },
        "per_clip": rows,
        "billed_estimate": billed_estimate,
        "privacy": "no transcript or reference text; clip IDs, numbers, flags, hashes only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-output-json", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=_REPO_ROOT / "results" / "hinglish" / "decoder_probe_plan.json")
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--committed-out", type=Path, default=_REPO_ROOT / "results" / "hinglish" / "decoder_probe_metrics.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.committed_out.exists() and not args.force:
        print(f"refusing to overwrite {args.committed_out} without --force")
        return 4
    raw = _load_json(args.raw_output_json)
    metrics = {cfg: _load_json(args.metrics_dir / f"metrics_{cfg}.json") for cfg in ("C0", "C1", "C2")}
    plan = _load_json(args.plan)

    analysis = analyze(raw, metrics, plan)
    analysis_private_path = args.private_out
    analysis_private_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_private_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote private analysis: {analysis_private_path}")

    committed = {
        key: analysis[key]
        for key in (
            "report_version",
            "probe_version",
            "constructed_utc",
            "model",
            "precision",
            "instrumentation",
            "corpus_over_frozen_16",
            "summary",
            "per_clip",
            "billed_estimate",
            "privacy",
        )
    }
    committed["instrumentation"] = {
        key: value
        for key, value in committed["instrumentation"].items()
        if not key.startswith("outputs_")
    }
    violation_scopes = {
        "instrumentation": committed["instrumentation"],
        "corpus_over_frozen_16": committed["corpus_over_frozen_16"],
        "summary": committed["summary"],
        "per_clip": committed["per_clip"],
    }
    violations = content_safety_violations(violation_scopes)
    if violations:
        for issue in violations[:10]:
            print(f"CONTENT SAFETY VIOLATION: {issue}")
        return 4
    args.committed_out.write_text(json.dumps(committed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote committed aggregate: {args.committed_out}")

    corpus = analysis["corpus_over_frozen_16"]
    for cfg in ("C0", "C1", "C2"):
        c = corpus[cfg]
        print(
            f"{cfg}: corpus_wer={c['corpus_wer']:.4f} corpus_cer={c['corpus_cer']:.4f} "
            f"inference={c['inference_seconds']}s capped={c['token_totals']['hit_length_limit']} "
            f"eos={c['token_totals']['eos_emitted']}"
        )
    s = analysis["summary"]
    print(json.dumps(s["controls"], indent=1))
    print(json.dumps(s["ceiling_and_early_resolution"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
