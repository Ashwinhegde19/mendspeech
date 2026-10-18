"""Modal entry point for the Phase 7.1 controlled decoder probe (abbreviated J1).

Default behavior is a strict offline dry run: it loads the frozen probe plan,
verifies audio checksums and the cost/timeout gates, and prints the exact
launch configuration. Billable inference runs only with ``--execute`` on an
explicitly approved launch.

Design constraints honored by this module:

* exactly one GPU function bound to a T4, one container, a 420 s function
  timeout, and no automatic retries;
* the Whisper model is loaded once per container and the three comparison
  configurations run through the identical baseline pipeline path described
  in ``infra/modal_hinglish_asr.py`` (language ``hi``, task ``transcribe``,
  float16, ``generate_kwargs={"task": ..., "language": ...}`` only);
* C0 must reproduce the archived baseline decoding config exactly;
  instrumentation adds only packaging flags to the ``generate`` call
  (``return_dict_in_generate=True`` and ``output_scores=True``), which do not
  change the sampled tokens, and the identical ``sequences`` tensor is handed
  back to the pipeline unchanged;
* C1 changes exactly one parameter relative to C0: ``max_new_tokens=160``;
* C2 changes exactly one parameter relative to C0:
  ``no_repeat_ngram_size=4``;
* generation configuration semantics come from the single-source diagnostics
  module (``src/hinglish/decoder_diagnostics.py``), whose source text is
  passed into the container explicitly and hashed for provenance.

Token-budget semantics used by the instrumentation: whisper returns the
decoder sequence including the four forced prefix tokens (start-of-transcript,
language, task, no-timestamps); ``max_new_tokens`` counts generated tokens
beyond that prefix, and ``max_length`` bounds the total sequence length
including the prefix. EOS status is measured two ways — from generated token
ids, and from the recorded per-step chosen tokens (the argmax of the recorded
transition scores), because whisper treats the pad token as EOS and strips
trailing EOS from the returned sequences. EOS is never inferred from
punctuation or transcript text.

Offline smoke testing: the batch runner is importable without Modal and its
device/precision can be overridden with ``MENDSPEECH_PROBE_DEVICE`` (default
``cuda``) and ``MENDSPEECH_PROBE_PRECISION`` (default ``float16``) so the full
instrumented path can be exercised on CPU before the single paid launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow ``python infra/...`` from repo root
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_ID = os.environ.get("MENDSPEECH_HINGLISH_MODEL", "openai/whisper-small")
MODEL_REVISION = os.environ.get(
    "MENDSPEECH_HINGLISH_MODEL_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d"
)
APP_NAME = "mendspeech-hinglish-decoder-probe"
FUNCTION_TIMEOUT_SEC = 420
DEFAULT_GPU = "T4"
MAX_CONTAINERS = 1
# Modal requires a scaledown window between 2 and 3600 seconds. Two seconds is
# the smallest near-immediate scale-to-zero value accepted by the SDK.
SCALEDOWN_WINDOW_SEC = 2
PROBE_DEVICE = os.environ.get("MENDSPEECH_PROBE_DEVICE", "cuda")
PROBE_PRECISION = os.environ.get("MENDSPEECH_PROBE_PRECISION", "float16")
#: Model cache root. The Modal container mounts the model volume at /cache/model;
#: offline CPU smoke tests override it to a scratch directory.
PROBE_CACHE_DIR = os.environ.get("MENDSPEECH_PROBE_CACHE_DIR", "/cache/model")
LANGUAGE = "hi"
TASK = "transcribe"
CONFIG_ORDER = ["C0", "C1", "C2"]

DEFAULT_PLAN_PATH = _REPO_ROOT / "results" / "hinglish" / "decoder_probe_plan.json"
DEFAULT_AUDIO_DIR = Path("/home/ashwin/backups/mucs-validation/audio")
DIAGNOSTICS_MODULE_PATH = _REPO_ROOT / "src" / "hinglish" / "decoder_diagnostics.py"
PROBE_VERSION_MARKER = "mendspeech.hinglish_decoder_probe.v1"
EXPECTED_CLIP_COUNT = 16
MANDATED_CATEGORY_COUNTS = {
    "ceiling_word_loop": 8,
    "ceiling_char_loop": 3,
    "early_loop": 2,
    "control": 3,
}

# Cost/time planning constants shared with scripts/plan_hinglish_decoder_probe.py.
RATE_T4_USD_PER_SEC = 0.000164
MODEL_LOAD_SEC = 90.0
CONTAINER_OVERHEAD_SEC = 60.0
C1_MAX_RUNTIME_SEC = 2.0
C2_MIN_RUNTIME_SEC = 3.75
PLANNING_BUDGET_USD = 0.10

try:
    import modal
except ImportError:  # pragma: no cover - Modal is an infra-only dependency
    modal = None  # type: ignore[assignment]

if modal is not None:
    model_cache = modal.Volume.from_name(
        "mendspeech-hinglish-model-cache",
        create_if_missing=True,
    )
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.1",
            "torchaudio==2.4.1",
            "transformers==4.46.3",
            "accelerate==1.2.1",
            "soundfile==0.12.1",
            "numpy==1.26.4",
        )
    )
    app = modal.App(APP_NAME)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_diagnostics_source() -> str:
    """Read the diagnostics module source used inside the probe container."""
    if not DIAGNOSTICS_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"decoder diagnostics module not found: {DIAGNOSTICS_MODULE_PATH}"
        )
    source = DIAGNOSTICS_MODULE_PATH.read_text(encoding="utf-8")
    if PROBE_VERSION_MARKER not in source:
        raise ValueError(
            "decoder diagnostics module missing probe version marker "
            f"{PROBE_VERSION_MARKER}"
        )
    return source


def _import_diagnostics_from_source(source: str):
    """Materialize the diagnostics module and import it from the text only.

    The probe container has no repository checkout, so the module source is
    passed as a launch argument, written to a temporary file there, and
    imported by file path. This keeps the experiment semantics single-sourced
    (the same code runs locally and in the container) and records a source
    hash for provenance.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="decoder_diagnostics_"))
    module_path = temp_dir / "decoder_diagnostics.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "probe_decoder_diagnostics", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not import probe diagnostics module")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection can resolve __module__.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _chosen_token_ids(scores) -> list[int] | None:
    """Extract the per-step chosen token ids from recorded transition scores.

    Greedy decoding selects the argmax of the post-processor logits, and the
    recorded ``scores`` hold exactly those logits, so the argmax reproduces
    the selected token for every step. Requires batch size one (the probe
    calls the pipeline with a single waveform).
    """
    if not scores:
        return None
    chosen: list[int] = []
    for step_scores in scores:
        if step_scores is None or step_scores.dim() != 2 or step_scores.shape[0] != 1:
            return None
        chosen.append(int(step_scores.argmax(dim=-1)[0].item()))
    return chosen


def _wrap_pipeline_generate(capture, original_generate):
    """Return a wrapped ``model.generate`` that keeps token behavior intact.

    ``return_dict_in_generate=True`` and ``output_scores=True`` change only
    how generation results are packaged; the sampled token sequence and the
    stopping behaviour are unchanged. The wrapper records the produced
    sequences and chosen tokens, then returns the plain ``sequences`` tensor
    so the pipeline proceeds exactly as it does in the baseline path.
    """

    def wrapped_generate(*args, **kwargs):
        kwargs = dict(kwargs)
        if not kwargs.get("return_dict_in_generate"):
            kwargs["return_dict_in_generate"] = True
        if not kwargs.get("output_scores"):
            kwargs["output_scores"] = True
        output = original_generate(*args, **kwargs)
        record = capture.record_sequences(getattr(output, "sequences", None), kwargs)
        record["chosen_token_ids"] = _chosen_token_ids(
            getattr(output, "scores", None)
        )
        return output.sequences

    return wrapped_generate


def _collect_eos_token_ids(model) -> list[int]:
    """Collect generation-config EOS ids (never inferred from text)."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        return []
    values = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    return sorted({int(value) for value in values})


def _build_forced_prefix_ids(model, tokenizer) -> list[int]:
    """Best-effort decoder prefix ids: start + language/task/notimestamps."""
    ids: list[int] = []
    start_token = model.generation_config.decoder_start_token_id
    if start_token is not None:
        ids.append(int(start_token))
    lang_token = None
    lang_to_id = getattr(tokenizer, "lang_to_id", None)
    if isinstance(lang_to_id, dict):
        lang_token = lang_to_id.get(LANGUAGE)
    if lang_token is None:
        for name in (f"<|{LANGUAGE}|>", f"<|{LANGUAGE.lower()}|>"):
            candidate = tokenizer.convert_tokens_to_ids(name)
            if candidate is not None and int(candidate) >= 0:
                lang_token = int(candidate)
                break
    if isinstance(lang_token, int):
        ids.append(lang_token)
    for name in (f"<|{TASK}|>", "<|notimestamps|>"):
        candidate = tokenizer.convert_tokens_to_ids(name)
        if candidate is not None and int(candidate) >= 0:
            ids.append(int(candidate))
    return ids


def _run_probe_batch(
    model_id: str,
    model_revision: str,
    diagnostics_source: str,
    audio_records: list[dict[str, Any]],
    configs: list[str],
) -> dict[str, Any]:
    """Run every configuration for every clip with one loaded model."""
    import io as io_module

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    diagnostics = _import_diagnostics_from_source(diagnostics_source)
    config_generate_kwargs = diagnostics.config_generate_kwargs
    analyze_generation = diagnostics.analyze_generation
    capture_class = diagnostics.GenerationCapture
    for config_name in configs:
        config_generate_kwargs(config_name)  # validates config names/tokens

    os.environ.setdefault("HF_HOME", PROBE_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{PROBE_CACHE_DIR}/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{PROBE_CACHE_DIR}/huggingface")

    torch_dtype = torch.float16 if PROBE_PRECISION == "float16" else torch.float32
    pipeline_device = 0 if PROBE_DEVICE.startswith("cuda") else -1
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=PROBE_CACHE_DIR,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=PROBE_CACHE_DIR,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(PROBE_DEVICE)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipeline_device,
    )

    eos_token_ids = _collect_eos_token_ids(model)
    prefix_ids = _build_forced_prefix_ids(model, processor.tokenizer)
    generation_config_max_length = int(model.generation_config.max_length)

    capture = capture_class()
    original_generate = model.generate
    model.generate = _wrap_pipeline_generate(capture, original_generate)

    records: list[dict[str, Any]] = []
    for audio_record in audio_records:
        clip_id = str(audio_record["clip_id"])
        try:
            data, sample_rate = sf.read(
                io_module.BytesIO(bytes(audio_record["audio_bytes"])),
                dtype="float32",
            )
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sample_rate != 16000:
                import torchaudio

                waveform = torch.from_numpy(data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                data = resampler(waveform).squeeze(0).cpu().numpy()
        except Exception as exc:
            for config_name in configs:
                records.append(
                    {
                        "config": config_name,
                        "clip_id": clip_id,
                        "transcript": "",
                        "instrumentation_ok": False,
                        "error": f"audio_load: {type(exc).__name__}: {exc}",
                    }
                )
            continue

        audio_duration_sec = round(len(data) / 16000, 3)
        for config_name in configs:
            try:
                generate_kwargs = config_generate_kwargs(config_name)
                capture.reset()
                started_at = time.perf_counter()
                with torch.inference_mode():
                    output = asr_pipeline(
                        data,
                        generate_kwargs=generate_kwargs,
                    )
                runtime_sec = round(time.perf_counter() - started_at, 3)
                capture_records = list(capture.captures)
                instrumentation_ok = len(capture_records) == 1
                probe_payload: dict[str, Any] | None = None
                chosen: list[int] | None = None
                forced_prefix_info: dict[str, Any] = {"ids": prefix_ids}
                if instrumentation_ok and capture_records:
                    chosen = capture_records[0].get("chosen_token_ids")
                    sequences = capture_records[0].get("sequences") or []
                    batch_rows = len(sequences)
                    instrumentation_ok = batch_rows == 1 and chosen is not None
                    generated_ids = [int(t) for t in sequences[0]] if batch_rows else []
                    prefix_match_count = 0
                    if generated_ids[: len(prefix_ids)] == prefix_ids:
                        prefix_match_count = len(prefix_ids)
                    forced_prefix_info["matched"] = prefix_match_count == len(prefix_ids)
                    forced_prefix_info["length"] = prefix_match_count
                    probe = analyze_generation(
                        generated_ids,
                        decoder_prefix_tokens=prefix_match_count,
                        configured_max_length=generation_config_max_length,
                        configured_max_new_tokens=generate_kwargs.get(
                            "max_new_tokens"
                        ),
                        eos_token_ids=eos_token_ids,
                    )
                    probe_payload = asdict(probe)
                else:
                    forced_prefix_info["matched"] = None
                    forced_prefix_info["length"] = None
                last_chosen = chosen[-1] if chosen else None
                eos_witness = bool(
                    last_chosen is not None and last_chosen in set(eos_token_ids)
                )
                eos_emitted_effective = bool(
                    (probe_payload or {}).get("eos_emitted", False) or eos_witness
                )
                hit_length_limit_effective = bool(
                    (probe_payload or {}).get("hit_length_limit", False)
                    and not eos_witness
                )
                records.append(
                    {
                        "config": config_name,
                        "clip_id": clip_id,
                        "transcript": str(output.get("text", "")).strip(),
                        "runtime_sec": runtime_sec,
                        "audio_duration_sec": audio_duration_sec,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "language": LANGUAGE,
                        "task": TASK,
                        "precision": PROBE_PRECISION,
                        "instrumentation": {
                            "return_dict_in_generate": True,
                            "output_scores": True,
                            "generate_call_count": len(capture_records),
                            "instrumentation_ok": instrumentation_ok,
                            "chosen_step_count": len(chosen) if chosen else 0,
                            "last_chosen_token_id": last_chosen,
                            "eos_emitted_scores_witness": eos_witness,
                            "eos_emitted_effective": eos_emitted_effective,
                            "hit_length_limit_effective": hit_length_limit_effective,
                        },
                        "generate_kwargs": (
                            capture_records[0]["kwargs"] if capture_records else None
                        ),
                        "forced_prefix": forced_prefix_info,
                        "eos_token_ids": eos_token_ids,
                        "generation_config_max_length": generation_config_max_length,
                        "probe": probe_payload,
                        "error": None,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "config": config_name,
                        "clip_id": clip_id,
                        "transcript": "",
                        "runtime_sec": None,
                        "audio_duration_sec": audio_duration_sec,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "language": LANGUAGE,
                        "task": TASK,
                        "precision": PROBE_PRECISION,
                        "instrumentation": {
                            "return_dict_in_generate": True,
                            "output_scores": True,
                            "generate_call_count": None,
                            "instrumentation_ok": False,
                        },
                        "generate_kwargs": None,
                        "forced_prefix": None,
                        "eos_token_ids": eos_token_ids,
                        "generation_config_max_length": None,
                        "probe": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return {
        "probe_version": PROBE_VERSION_MARKER,
        "model_id": model_id,
        "model_revision": model_revision,
        "precision": PROBE_PRECISION,
        "device_class": "cuda" if PROBE_DEVICE.startswith("cuda") else PROBE_DEVICE,
        "diagnostics_source_sha256": hashlib.sha256(
            diagnostics_source.encode("utf-8")
        ).hexdigest(),
        "eos_token_ids": eos_token_ids,
        "forced_prefix_ids": prefix_ids,
        "generation_config_max_length": generation_config_max_length,
        "records": records,
    }


if modal is not None:

    @app.function(
        image=image,
        gpu="T4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={"/cache/model": model_cache},
    )
    def decoder_probe_t4(
        audio_records: list[dict[str, Any]],
        configs: list[str],
        diagnostics_source: str,
        model_revision: str = MODEL_REVISION,
    ) -> dict[str, Any]:
        """Single-container probe: load Whisper once, decode every config."""
        return _run_probe_batch(
            MODEL_ID,
            model_revision,
            diagnostics_source,
            audio_records,
            configs,
        )


def _load_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        raise FileNotFoundError(
            f"frozen probe plan not found: {plan_path} (freeze it before preflight)"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("probe_version") != PROBE_VERSION_MARKER:
        raise ValueError(f"unexpected probe_version in plan: {plan.get('probe_version')}")
    model_block = plan.get("model") or {}
    if model_block.get("revision") != MODEL_REVISION:
        raise ValueError("plan model revision does not match the pinned revision")
    clips = plan.get("clips", [])
    if len(clips) != EXPECTED_CLIP_COUNT:
        raise ValueError(f"plan has {len(clips)} clips; expected {EXPECTED_CLIP_COUNT}")
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip["category"]] = counts.get(clip["category"], 0) + 1
    if counts != MANDATED_CATEGORY_COUNTS:
        raise ValueError(
            f"plan category counts {counts} != mandated {MANDATED_CATEGORY_COUNTS}"
        )
    ids = [clip["clip_id"] for clip in clips]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip ids in plan")
    return plan


def _estimate_wall_and_cost(plan: dict[str, Any]) -> dict[str, float]:
    """Recompute the conservative estimate from the plan's measured baselines."""
    config_seconds = {"C0": 0.0, "C1": 0.0, "C2": 0.0}
    for clip in plan["clips"]:
        baseline_runtime = float(clip["baseline_runtime_sec"])
        config_seconds["C0"] += baseline_runtime
        config_seconds["C1"] += min(baseline_runtime, C1_MAX_RUNTIME_SEC)
        config_seconds["C2"] += max(baseline_runtime, C2_MIN_RUNTIME_SEC)
    if isinstance(plan.get("estimate"), dict):
        plan_config_seconds = plan["estimate"].get("config_inference_seconds", {})
        for config_name, seconds in plan_config_seconds.items():
            if config_name in config_seconds:
                rounded_plan = round(float(seconds), 3)
                rounded_local = round(config_seconds[config_name], 3)
                if abs(rounded_plan - rounded_local) > 0.05:
                    raise ValueError(
                        f"plan estimate mismatch for {config_name}: "
                        f"{rounded_plan} vs {rounded_local}"
                    )
    total_wall = sum(config_seconds.values()) + MODEL_LOAD_SEC + CONTAINER_OVERHEAD_SEC
    cost = total_wall * RATE_T4_USD_PER_SEC
    return {
        **{key: round(value, 3) for key, value in config_seconds.items()},
        "wall_seconds": round(total_wall, 3),
        "estimated_gpu_cost_usd": round(cost, 5),
        "timeout_worst_case_cost_usd": round(
            FUNCTION_TIMEOUT_SEC * RATE_T4_USD_PER_SEC, 5
        ),
    }


def dry_run(plan_path: Path, audio_dir: Path) -> int:
    """Offline preflight: verify the plan, audio checksums, and the gates."""
    try:
        plan = _load_plan(plan_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"PLAN CHECK FAILED: {type(exc).__name__}: {exc}")
        return 2

    checksum_mismatches: list[str] = []
    missing_files: list[str] = []
    for clip in plan["clips"]:
        audio_path = audio_dir / f"{clip['clip_id']}.wav"
        if not audio_path.exists():
            missing_files.append(str(audio_path))
            continue
        actual_sha = _sha256_file(audio_path)
        if actual_sha != clip.get("audio_sha256"):
            checksum_mismatches.append(clip["clip_id"])
    if missing_files or checksum_mismatches:
        for path in missing_files:
            print(f"MISSING AUDIO: {path}")
        for clip_id in checksum_mismatches:
            print(f"CHECKSUM MISMATCH: {clip_id}")
        return 2

    try:
        estimate = _estimate_wall_and_cost(plan)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"ESTIMATE CHECK FAILED: {type(exc).__name__}: {exc}")
        return 2

    gates_failed = False
    if estimate["wall_seconds"] > FUNCTION_TIMEOUT_SEC:
        print("COST/TIME GATE FAILED: estimated wall exceeds function timeout")
        gates_failed = True
    if estimate["estimated_gpu_cost_usd"] > PLANNING_BUDGET_USD:
        print("COST/TIME GATE FAILED: estimate exceeds planning budget")
        gates_failed = True
    if gates_failed:
        return 2

    print("MendSpeech Hinglish decoder probe dry run")
    print(f"plan_path: {plan_path}")
    print(f"probe_version: {plan.get('probe_version')}")
    print(f"model_id: {MODEL_ID}")
    print(f"model_revision: {MODEL_REVISION}")
    print(f"gpu: {DEFAULT_GPU}")
    print(f"max_containers: {MAX_CONTAINERS}")
    print(f"scaledown_window_sec: {SCALEDOWN_WINDOW_SEC}")
    print(f"function_timeout_sec: {FUNCTION_TIMEOUT_SEC}")
    print(f"configs: {CONFIG_ORDER}")
    for clip in plan["clips"]:
        print(
            f"  clip: {clip['clip_id']} category={clip['category']} "
            f"baseline_runtime_sec={clip['baseline_runtime_sec']}"
        )
    print(f"audio_files_verified: {len(plan['clips'])} sha256 matches")
    print(f"estimate: {estimate}")
    print(
        "additional_cost_notes: container cold start, model load, and CPU time "
        "add billed seconds beyond inference"
    )
    print("status: no GPU launched; pass --execute for the approved single run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run (default) or explicitly execute the approved J1 decoder "
            "probe on one Modal T4 container"
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Frozen probe plan JSON (clips, categories, checksums, estimate).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory holding the private dev-cohort WAV files.",
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIG_ORDER,
        default=None,
        help="Restrict to a subset of configs; default runs all three.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the single billable Modal launch (preflight must pass first).",
    )
    args = parser.parse_args()

    if not args.execute:
        return dry_run(args.plan, args.audio_dir)

    if modal is None:
        print("Modal is not installed. Install with: pip install -e '.[hinglish]'")
        return 1

    try:
        plan = _load_plan(args.plan)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"PLAN CHECK FAILED: {type(exc).__name__}: {exc}")
        return 2

    try:
        estimate = _estimate_wall_and_cost(plan)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"ESTIMATE CHECK FAILED: {type(exc).__name__}: {exc}")
        return 2
    if (
        estimate["wall_seconds"] > FUNCTION_TIMEOUT_SEC
        or estimate["estimated_gpu_cost_usd"] > PLANNING_BUDGET_USD
    ):
        print("COST/TIME GATE FAILED before execution; refusing to launch.")
        return 2

    diagnostics_source = _load_diagnostics_source()

    configs = args.config or CONFIG_ORDER
    audio_records: list[dict[str, Any]] = []
    for clip in plan["clips"]:
        audio_path = args.audio_dir / f"{clip['clip_id']}.wav"
        if not audio_path.exists() or _sha256_file(audio_path) != clip.get(
            "audio_sha256"
        ):
            print(f"AUDIO VERIFY FAILED for clip {clip['clip_id']}; aborting launch.")
            return 2
        audio_records.append(
            {"clip_id": clip["clip_id"], "audio_bytes": audio_path.read_bytes()}
        )

    print(f"Launching decoder probe on {DEFAULT_GPU} (single approved run)...")
    try:
        with app.run():
            result = decoder_probe_t4.remote(
                audio_records,
                configs,
                diagnostics_source,
            )
    except Exception as exc:
        print(f"Modal execution failed: {type(exc).__name__}: {exc}")
        print("No automatic retry is authorized for this probe.")
        return 1

    payload = result if isinstance(result, dict) else {
        "probe_version": PROBE_VERSION_MARKER,
        "model_revision": MODEL_REVISION,
        "diagnostics_source_sha256": hashlib.sha256(
            diagnostics_source.encode("utf-8")
        ).hexdigest(),
        "return_type": type(result).__name__,
        "records": [],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
