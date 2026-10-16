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
APP_NAME = "mendspeech-hinglish-asr"
FUNCTION_TIMEOUT_SEC = 600
DEFAULT_GPU = "T4"
MAX_CONTAINERS = 1
# Modal requires a positive scaledown window even when near-idle shutdown is
# desired. One second is the code-level minimum.
SCALEDOWN_WINDOW_SEC = 1

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

    def _run_whisper(
        model_id: str,
        audio_bytes: bytes,
        clip_id: str,
        language: str,
        task: str,
        precision: str,
    ) -> dict[str, Any]:
        """Run Whisper inference inside a GPU container."""
        import io as io_module

        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        os.environ.setdefault("HF_HOME", "/cache/model")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/model/transformers")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache/model/huggingface")

        data, sample_rate = sf.read(io_module.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate != 16000:
            waveform = torch.from_numpy(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            data = resampler(waveform).squeeze(0).numpy()

        torch_dtype = torch.float16 if precision == "float16" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, cache_dir="/cache/model")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
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

        started_at = time.perf_counter()
        with torch.inference_mode():
            output = asr_pipeline(
                data,
                generate_kwargs={"task": task, "language": language},
            )
        runtime_sec = round(time.perf_counter() - started_at, 3)

        return {
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

    @app.function(
        image=image,
        gpu="T4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={"/cache/model": model_cache},
    )
    def transcribe_audio_t4(
        audio_bytes: bytes,
        clip_id: str = "clip",
        language: str = "hi",
        task: str = "transcribe",
        precision: str = "float16",
    ) -> dict[str, Any]:
        """Inference on an NVIDIA T4."""
        return _run_whisper(
            MODEL_ID,
            audio_bytes,
            clip_id,
            language,
            task,
            precision,
        )

    @app.function(
        image=image,
        gpu="L4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={"/cache/model": model_cache},
    )
    def transcribe_audio_l4(
        audio_bytes: bytes,
        clip_id: str = "clip",
        language: str = "hi",
        task: str = "transcribe",
        precision: str = "float16",
    ) -> dict[str, Any]:
        """Inference on an NVIDIA L4."""
        return _run_whisper(
            MODEL_ID,
            audio_bytes,
            clip_id,
            language,
            task,
            precision,
        )


def _estimate_cost(gpu: str, estimated_runtime_sec: int) -> str:
    # Published Modal GPU rates vary by workload; these are conservative planning
    # figures for approval before any billable run.
    rates = {
        "T4": 0.00043,
        "L4": 0.00070,
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
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--gpu", choices=["T4", "L4"], default=DEFAULT_GPU)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    audio_path = args.audio_path
    clip_id = args.clip_id or audio_path.stem
    if not args.execute:
        return dry_run(audio_path, args.gpu, clip_id, args.language, args.task)

    if modal is None:
        print("Modal is not installed. Install with: pip install -e '.[hinglish]'")
        return 1

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 1

    function = transcribe_audio_t4 if args.gpu == "T4" else transcribe_audio_l4
    print(f"Launching Modal inference on {args.gpu}...")
    try:
        result = function.remote(
            audio_path.read_bytes(),
            clip_id=clip_id,
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
