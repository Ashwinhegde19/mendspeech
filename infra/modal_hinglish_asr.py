"""Dedicated Modal entry point for Hinglish ASR baseline inference.

This module is GPU-only when explicitly executed. Its default behavior is a
dry run that reports configuration, expected cost, and the command used to
launch inference. No inference runs unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = os.environ.get("MENDSPEECH_HINGLISH_MODEL", "openai/whisper-small")
MODEL_REVISION = os.environ.get(
    "MENDSPEECH_HINGLISH_MODEL_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d"
)
APP_NAME = "mendspeech-hinglish-asr"
FUNCTION_TIMEOUT_SEC = 600
DEFAULT_GPU = "T4"
MAX_CONTAINERS = 1
# Modal requires a scaledown window between 2 and 3600 seconds. Two seconds is
# the smallest near-immediate scale-to-zero value accepted by the SDK.
SCALEDOWN_WINDOW_SEC = 2

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

    def _run_whisper_batch(
        model_id: str,
        model_revision: str,
        audio_records: list[dict[str, Any]],
        language: str,
        task: str,
        precision: str,
    ) -> list[dict[str, Any]]:
        """Run Whisper inference for a small batch in one loaded model context."""
        import io as io_module

        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        os.environ.setdefault("HF_HOME", "/cache/model")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/model/transformers")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/model/huggingface")

        torch_dtype = torch.float16 if precision == "float16" else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir="/cache/model",
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir="/cache/model",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=0,
        )

        results: list[dict[str, Any]] = []
        for record in audio_records:
            clip_id = str(record["clip_id"])
            audio_bytes = bytes(record["audio_bytes"])
            try:
                data, sample_rate = sf.read(
                    io_module.BytesIO(audio_bytes), dtype="float32"
                )
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sample_rate != 16000:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    data = resampler(waveform).squeeze(0).cpu().numpy()

                generate_kwargs = {"task": task}
                if language not in {"auto", "detect", "None"}:
                    generate_kwargs["language"] = language

                started_at = time.perf_counter()
                with torch.inference_mode():
                    output = asr_pipeline(
                        data,
                        generate_kwargs=generate_kwargs,
                    )
                runtime_sec = round(time.perf_counter() - started_at, 3)
                results.append(
                    {
                        "clip_id": clip_id,
                        "transcript": str(output.get("text", "")).strip(),
                        "model_id": model_id,
                        "task": task,
                        "language": language,
                        "precision": precision,
                        "device": "NVIDIA GPU",
                        "runtime_sec": runtime_sec,
                        "audio_duration_sec": round(len(data) / 16000, 3),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "clip_id": clip_id,
                        "transcript": "",
                        "model_id": model_id,
                        "task": task,
                        "language": language,
                        "precision": precision,
                        "device": "NVIDIA GPU",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results

    @app.function(
        image=image,
        gpu="T4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={"/cache/model": model_cache},
    )
    def transcribe_hinglish_batch_t4(
        audio_records: list[dict[str, Any]],
        language: str = "hi",
        task: str = "transcribe",
        precision: str = "float16",
        model_revision: str = MODEL_REVISION,
    ) -> list[dict[str, Any]]:
        """Batch inference on an NVIDIA T4."""
        return _run_whisper_batch(
            MODEL_ID, model_revision, audio_records, language, task, precision
        )

    @app.function(
        image=image,
        gpu="L4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={"/cache/model": model_cache},
    )
    def transcribe_hinglish_batch_l4(
        audio_records: list[dict[str, Any]],
        language: str = "hi",
        task: str = "transcribe",
        precision: str = "float16",
        model_revision: str = MODEL_REVISION,
    ) -> list[dict[str, Any]]:
        """Batch inference on an NVIDIA L4."""
        return _run_whisper_batch(
            MODEL_ID, model_revision, audio_records, language, task, precision
        )


def _estimate_cost(gpu: str, estimated_runtime_sec: int) -> str:
    # Published Modal GPU rates vary by workload; these are conservative planning
    # figures for approval before any billable run.
    # Source: Modal pricing page (verified October 16, 2026).
    rates = {
        "T4": 0.000164,
        "L4": 0.000222,
    }
    rate = rates.get(gpu, rates["T4"])
    return f"${estimated_runtime_sec * rate:.3f}"


def dry_run(audio_path: Path, gpu: str, clip_id: str, language: str, task: str) -> int:
    """Print a billable-run estimate without invoking a GPU."""
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 1

    size_bytes = audio_path.stat().st_size
    estimated_runtime_sec = max(45, int(size_bytes / 50000) + 30)
    print("MendSpeech Hinglish Modal dry run")
    print(f"audio_path: {audio_path}")
    print(f"audio_size_bytes: {size_bytes}")
    print(f"clip_id: {clip_id}")
    print(f"gpu: {gpu}")
    print(f"model_id: {MODEL_ID}")
    print(f"language: {language}")
    print(f"task: {task}")
    print(f"max_containers: {MAX_CONTAINERS}")
    print(f"scaledown_window_sec: {SCALEDOWN_WINDOW_SEC}")
    print(f"function_timeout_sec: {FUNCTION_TIMEOUT_SEC}")
    print(f"estimated_runtime_sec: {estimated_runtime_sec}")
    print(f"estimated_gpu_cost: {_estimate_cost(gpu, estimated_runtime_sec)}")
    print("status: no GPU launched; pass --execute to run billable inference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly execute Hinglish Whisper inference on Modal"
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        action="append",
        required=True,
        help="Audio file to transcribe. Repeat for a small approved batch.",
    )
    parser.add_argument("--clip-id", action="append", default=None)
    parser.add_argument("--gpu", choices=["T4", "L4"], default=DEFAULT_GPU)
    parser.add_argument("--language", choices=["hi", "auto"], default="hi")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    audio_paths = args.audio_path
    clip_ids = args.clip_id or []
    if len(clip_ids) not in (0, len(audio_paths)):
        print("--clip-id must be omitted or provided once per --audio-path")
        return 1

    if not args.execute:
        total_bytes = 0
        all_exist = True
        for path in audio_paths:
            if not path.exists():
                all_exist = False
                print(f"Audio file not found: {path}")
            else:
                total_bytes += path.stat().st_size

        if not all_exist:
            return 1
        estimated_runtime = max(60, min(300, 45 + 12 * len(audio_paths)))
        print("MendSpeech Hinglish Modal dry run")
        for path, clip_id in zip(audio_paths, clip_ids or audio_paths):
            print(f"  clip: {clip_id or path.stem}")
        print(f"audio_files: {len(audio_paths)}")
        print(f"total_audio_bytes: {total_bytes}")
        print(f"gpu: {args.gpu}")
        print(f"model_id: {MODEL_ID}")
        print(f"model_revision: {MODEL_REVISION}")
        print(f"language: {args.language}")
        print(f"task: {args.task}")
        print(f"max_containers: {MAX_CONTAINERS}")
        print(f"scaledown_window_sec: {SCALEDOWN_WINDOW_SEC}")
        print(f"function_timeout_sec: {FUNCTION_TIMEOUT_SEC}")
        print(f"estimated_gpu_runtime_sec: {estimated_runtime}")
        print(f"official_gpu_rate: $0.000164/sec for T4, $0.000222/sec for L4")
        print(f"estimated_gpu_cost: {_estimate_cost(args.gpu, estimated_runtime)}")
        print(
            "additional_cost_notes: CPU and memory are also billed; "
            "first run may add model-download and cold-start latency"
        )
        print("status: no GPU launched; pass --execute for approved inference")
        return 0

    if modal is None:
        print("Modal is not installed. Install with: pip install -e '.[hinglish]'")
        return 1

    missing_paths = [str(path) for path in audio_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"Audio file not found: {path}")
        return 1

    audio_records = [
        {
            "clip_id": clip_ids[index] if clip_ids else path.stem,
            "audio_bytes": path.read_bytes(),
        }
        for index, path in enumerate(audio_paths)
    ]

    print(f"Launching batched Modal inference on {args.gpu}...")
    function = (
        transcribe_hinglish_batch_t4
        if args.gpu == "T4"
        else transcribe_hinglish_batch_l4
    )
    try:
        with app.run():
            result = function.remote(
                audio_records,
                language=args.language,
                task=args.task,
            )
    except Exception as exc:
        print(f"Modal execution failed: {type(exc).__name__}: {exc}")
        return 1

    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
