"""Modal entry point for the Phase 7.2 label-free pool scoring job (J2).

This is a ONE-TIME candidate-scoring run for the 300-clip MUCS pool under the
frozen Phase 7.1 C1 decode budget. It follows the proven Phase 7.1 runner
(``infra/modal_hinglish_decoder_probe.py``) for every mechanism it reuses:

* one Modal app, one GPU endpoint class bound to a single T4, one container,
  a 900 s function timeout, and no automatic retries;
* the Whisper model is loaded exactly once per container inside an
  ``@modal.enter()`` hook; the whole pool batch is decoded in one invocation
  of the pool method;
* decode configuration is frozen at ``config_generate_kwargs("C1")`` from the
  single-source diagnostics module (``max_new_tokens=160``), the provisional
  damage limiter from the probe. C1 bounds looping decode cost per clip and
  is NOT a hallucination cure — this runner only measures pool behavior
  under that bounded budget;
* instrumentation is identical to the probe: the diagnostics module source
  travels into the container as a launch argument, is hashed for provenance,
  and is imported by file path there. ``model.generate`` is wrapped so the
  call forces ``return_dict_in_generate=True`` and ``output_scores=True``
  (packaging only — the probe verified the instrumented text equals the
  archived baseline text) and records sequences plus per-step chosen token
  ids;
* EOS status is measured two ways, as in the probe: from generated token ids
  and from the recorded per-step chosen tokens (the argmax of recorded
  scores), because whisper treats the pad token as EOS and strips trailing
  EOS from returned sequences. EOS is never inferred from transcript text.

Label-free discipline: no ground truth is loaded anywhere in this job and no
word-error metric is computed. Each record exposes generated-text evidence
(decoded text plus repetition statistics) and model-derived confidence
evidence (chosen-token log-probabilities and entropy from the recorded score
tensors). Ground-truth-dependent repetition criteria are disabled by design
(see ``_repeat_flag``); the record still carries the raw repetition finding
for downstream review.

Data transport and privacy: only clip ids reach the function; pool manifest
and WAV files are read from a read-only Modal Volume mounted at ``/pool``
(designed layout: ``manifest.csv`` at the root, ``audio/<clip_id>.wav``
files, and ``audio_checksums.json``). Per-clip records — including the
``decoded_text`` field — are appended one JSON line at a time to
``scoring_checkpoint.jsonl`` on a scratch output Volume mounted at
``/outputs`` after every clip and committed there. The decoded text exists
only in that output Volume payload; it is never a module constant and never
printed. If the container dies, ``--resume`` (an explicit operator choice,
not an automatic second run) starts one fresh container that skips already
completed clips; an infra-level abort (container death, timeout) is visible
in the run log, never as a per-clip record.

Offline smoke: the pool method body factors into ``process_clip``, a pure
per-clip function callable without Modal, and the device/precision/cache can
be overridden with ``MENDSPEECH_LABELFREE_DEVICE`` (default ``cuda``),
``MENDSPEECH_LABELFREE_PRECISION`` (default ``float16``) and
``MENDSPEECH_LABELFREE_CACHE_DIR`` (default ``/cache/model``) so the full
instrumented path can be exercised on local CPU before the single paid
launch. ``--limit N`` runs that local CPU-first smoke through the exact same
batch code path without touching Modal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow ``python infra/...`` from repo root
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_ID = os.environ.get("MENDSPEECH_HINGLISH_MODEL", "openai/whisper-small")
MODEL_REVISION = os.environ.get(
    "MENDSPEECH_HINGLISH_MODEL_REVISION", "973afd24965f72e36ca33b3055d56a652f456b4d"
)
APP_NAME = "mendspeech-hinglish-label-free-score"
# One-time pool run: the 900 s ceiling is the maximum Modal function timeout.
FUNCTION_TIMEOUT_SEC = 900
DEFAULT_GPU = "T4"
GPU_NAME = "NVIDIA T4"
MAX_CONTAINERS = 1
# Modal requires a scaledown window between 2 and 3600 seconds. Two seconds is
# the smallest near-immediate scale-to-zero value accepted by the SDK.
SCALEDOWN_WINDOW_SEC = 2
LABELFREE_DEVICE = os.environ.get("MENDSPEECH_LABELFREE_DEVICE", "cuda")
LABELFREE_PRECISION = os.environ.get("MENDSPEECH_LABELFREE_PRECISION", "float16")
#: Model cache root. The Modal container mounts the model volume at /cache/model;
#: offline CPU smoke tests override it to a scratch directory.
LABELFREE_CACHE_DIR = os.environ.get("MENDSPEECH_LABELFREE_CACHE_DIR", "/cache/model")
LANGUAGE = "hi"
TASK = "transcribe"
#: The scored configuration is frozen at the probe's C1 (provisional damage
#: limiter): baseline generate kwargs plus ``max_new_tokens=160``.
CONFIG_NAME = "C1"
EXPECTED_C1_MAX_NEW_TOKENS = 160
CONFIG_NOTE = (
    "C1 provisional damage limiter (bounds generated tokens at 160 under the "
    "probe's frozen budget); not a hallucination cure"
)

#: Diagnostics module provenance (same single source as the Phase 7.1 probe).
PROBE_VERSION_MARKER = "mendspeech.hinglish_decoder_probe.v1"
SCORE_VERSION_MARKER = "mendspeech.hinglish_label_free_score.v1"
DIAGNOSTICS_MODULE_PATH = _REPO_ROOT / "src" / "hinglish" / "decoder_diagnostics.py"

DEFAULT_POOL_VOLUME_NAME = "mucs-j2-pool"
OUTPUT_VOLUME_NAME = "mucs-j2-labelfree-outputs"
MODEL_CACHE_VOLUME_NAME = "mendspeech-hinglish-model-cache"
POOL_MOUNT_PATH = "/pool"
OUTPUT_MOUNT_PATH = "/outputs"
MODEL_CACHE_MOUNT_PATH = "/cache/model"

MANIFEST_FILENAME = "manifest.csv"
CHECKSUMS_FILENAME = "audio_checksums.json"
AUDIO_DIR_NAME = "audio"
CHECKPOINT_FILENAME = "scoring_checkpoint.jsonl"
RESULTS_FILENAME = "scoring_results.json"
RESULTS_SHA_FILENAME = "scoring_results.sha256"
LOCAL_DEFAULT_OUTPUTS_DIRNAME = "_local_scoring_outputs"
EXPECTED_POOL_CLIP_COUNT = 300

# Cost/time planning constants calibrated from the measured Phase 7.1 T4 run:
# model load 60-90 s, container startup 30-60 s (upper band used here),
# C1-style decode roughly 1.5-2.0 s for budget-cap clips and 0.6-0.9 s for
# short clips; the linear per-clip estimate sits between those bands.
RATE_T4_USD_PER_SEC = 0.0001639  # T4 at 0.59 USD/hour
MODEL_LOAD_SEC = 90.0
CONTAINER_STARTUP_SEC = 60.0
OVERHEAD_SEC = MODEL_LOAD_SEC + CONTAINER_STARTUP_SEC
BASE_PER_CLIP_SEC = 0.55
PER_CLIP_SEC_PER_AUDIO_SEC = 0.016
PLANNING_BUDGET_USD = 0.25

try:
    import modal
except ImportError:  # pragma: no cover - Modal is an infra-only dependency
    modal = None  # type: ignore[assignment]

if modal is not None:
    pool_volume = modal.Volume.from_name(
        DEFAULT_POOL_VOLUME_NAME,
        create_if_missing=False,
    )
    outputs_volume = modal.Volume.from_name(
        OUTPUT_VOLUME_NAME,
        create_if_missing=True,
    )
    model_cache = modal.Volume.from_name(
        MODEL_CACHE_VOLUME_NAME,
        create_if_missing=False,
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


@dataclass(frozen=True)
class ScoringContext:
    """Bundle of everything one per-clip scoring call needs.

    Built once per invocation (locally or inside the container) from the
    loaded model assets plus the imported diagnostics module. ``capture``
    is the live instrumentation capture the wrapped ``model.generate`` appends
    to; it is reset before each clip.
    """

    model_id: str
    model_revision: str
    language: str
    task: str
    precision: str
    device: str
    config_name: str
    config_note: str
    generate_kwargs: Dict[str, Any]
    eos_token_ids: List[int]
    prefix_ids: List[int]
    generation_config_max_length: int
    diagnostics: Any
    capture: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate(text: str, limit: int = 200) -> str:
    """Collapse an error string to at most ``limit`` characters."""
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _load_diagnostics_source() -> str:
    """Read the diagnostics module source used inside the scoring container."""
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

    The scoring container has no repository checkout, so the module source is
    passed as a launch argument, written to a temporary file there, and
    imported by file path. This keeps the experiment semantics single-sourced
    (the same code runs locally and in the container) and records a source
    hash for provenance.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="decoder_diagnostics_"))
    module_path = temp_dir / "decoder_diagnostics.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "score_decoder_diagnostics", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not import score diagnostics module")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection can resolve __module__.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _chosen_token_ids(scores) -> Optional[List[int]]:
    """Extract the per-step chosen token ids from recorded transition scores.

    Greedy decoding selects the argmax of the post-processor logits, and the
    recorded ``scores`` hold exactly those logits, so the argmax reproduces
    the selected token for every step. Requires batch size one (scoring calls
    the pipeline with a single waveform).
    """
    if not scores:
        return None
    chosen: List[int] = []
    for step_scores in scores:
        if step_scores is None or step_scores.dim() != 2 or step_scores.shape[0] != 1:
            return None
        chosen.append(int(step_scores.argmax(dim=-1)[0].item()))
    return chosen


def _score_statistics(scores, chosen: Optional[List[int]]) -> Optional[Dict[str, Any]]:
    """Summarize chosen-token probabilities from the recorded score tensors.

    For each step ``t`` the chosen token is ``argmax(scores[t])``; the record
    carries the mean and minimum log-probability of the chosen token and the
    mean step entropy, all computed in float32 for numerical stability.
    Returns ``None`` when scores or chosen ids are absent (or shapes do not
    match a batch of one) so records stay JSON-safe.
    """
    if not scores or not chosen or len(scores) != len(chosen):
        return None
    import torch

    logprobs: List[float] = []
    entropies: List[float] = []
    for step_scores, step_chosen in zip(scores, chosen):
        if step_scores is None or step_scores.dim() != 2 or step_scores.shape[0] != 1:
            return None
        index = int(step_chosen)
        if not 0 <= index < step_scores.shape[1]:
            return None
        row = step_scores[0].to(torch.float32)
        log_probs = torch.log_softmax(row, dim=-1)
        probs = torch.softmax(row, dim=-1)
        logprobs.append(float(log_probs[index].item()))
        # Masked tokens carry probability zero and log-probability -inf; their
        # 0 * (-inf) term must not poison the mean entropy.
        entropy_terms = probs * torch.nan_to_num(log_probs, neginf=0.0)
        entropies.append(float(-entropy_terms.sum().item()))
    return {
        "mean_logprob": round(sum(logprobs) / len(logprobs), 6),
        "min_logprob": round(min(logprobs), 6),
        "entropy_mean": round(sum(entropies) / len(entropies), 6),
        "step_count": len(logprobs),
    }


def _wrap_generate(capture, original_generate):
    """Return a wrapped ``model.generate`` that keeps token behavior intact.

    ``return_dict_in_generate=True`` and ``output_scores=True`` change only
    how generation results are packaged; the sampled token sequence and the
    stopping behaviour are unchanged. The wrapper records the produced
    sequences, chosen tokens, and a small score summary, then returns the
    plain ``sequences`` tensor so the pipeline proceeds exactly as it does in
    the baseline path.
    """

    def wrapped_generate(*args, **kwargs):
        kwargs = dict(kwargs)
        if not kwargs.get("return_dict_in_generate"):
            kwargs["return_dict_in_generate"] = True
        if not kwargs.get("output_scores"):
            kwargs["output_scores"] = True
        output = original_generate(*args, **kwargs)
        record = capture.record_sequences(getattr(output, "sequences", None), kwargs)
        chosen = _chosen_token_ids(getattr(output, "scores", None))
        record["chosen_token_ids"] = chosen
        record["score_summary"] = _score_statistics(
            getattr(output, "scores", None), chosen
        )
        return output.sequences

    return wrapped_generate


def _install_capture_wrapper(model, diagnostics) -> Any:
    """Bind a fresh instrumentation capture to the model's generate call.

    Idempotent per process: the original ``generate`` is stashed once and every
    later call wraps that original, so repeated invocations on the same
    container never nest wrappers. Returns the capture the caller reads and
    resets per clip.
    """
    base = getattr(model, "_mendspeech_base_generate", None)
    if base is None:
        base = model.generate
        model._mendspeech_base_generate = base  # type: ignore[attr-defined]
    capture = diagnostics.GenerationCapture()
    model.generate = _wrap_generate(capture, base)
    return capture


def _collect_eos_token_ids(model) -> List[int]:
    """Collect generation-config EOS ids (never inferred from text)."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        return []
    values = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    return sorted({int(value) for value in values})


def _build_forced_prefix_ids(model, tokenizer) -> List[int]:
    """Best-effort decoder prefix ids: start + language/task/notimestamps."""
    ids: List[int] = []
    start_token = model.generation_config.decoder_start_token_id
    if start_token is not None:
        ids.append(int(start_token))
    lang_token = None
    lang_to_id = getattr(tokenizer, "lang_to_id", None)
    if isinstance(lang_to_id, dict):
        lang_token = lang_to_id.get(LANGUAGE)
    if lang_token is None:
        for name in (f"<|{LANGUAGE}|>", f"<|{LANGUAGE.casefold()}|>"):
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


def _effective_device_precision() -> Tuple[str, str]:
    """Resolve the (device, precision) pair with a CPU-first smoke guard."""
    device = LABELFREE_DEVICE
    precision = LABELFREE_PRECISION
    if not device.startswith("cuda") and precision == "float16":
        precision = "float32"
    return device, precision


def _load_runtime_assets() -> Dict[str, Any]:
    """Load whisper-small, build the pipeline, and collect decode anchors.

    Called once per container by the ``@modal.enter()`` hook (and once per
    local smoke invocation). The returned mapping holds the pipeline, the raw
    model (for capture binding), and the derive-time anchors used by every
    per-clip record.
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    os.environ.setdefault("HF_HOME", LABELFREE_CACHE_DIR)
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", f"{LABELFREE_CACHE_DIR}/transformers"
    )
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE", f"{LABELFREE_CACHE_DIR}/huggingface"
    )

    device, precision = _effective_device_precision()
    torch_dtype = torch.float16 if precision == "float16" else torch.float32
    pipeline_device = 0 if device.startswith("cuda") else -1
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=LABELFREE_CACHE_DIR,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=LABELFREE_CACHE_DIR,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
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
    return {
        "pipeline": asr_pipeline,
        "model": model,
        "processor": processor,
        "eos_token_ids": eos_token_ids,
        "prefix_ids": prefix_ids,
        "generation_config_max_length": generation_config_max_length,
        "device": device,
        "precision": precision,
    }


def _word_repetition_finding(decoded_words: List[str]) -> Dict[str, Any]:
    """Measure exact consecutive repetition in decoded text words.

    Mirrors ``src/hinglish/diagnostics.py::analyze_repetition`` semantics on
    the provided word sequence (whitespace split of the decoded text): the
    longest run of one identical word, the longest phrase ``p`` with two
    consecutive occurrences, and the number of consecutive-occurrence starts
    at that ``p``. Applied to model-generated text only, which is valid
    generated-output evidence in a label-free job. An empty word list yields
    the same values the mirrored version computes for empty input (a
    ``max_consecutive_repeat`` of 1 and no phrase), so consumers can compare
    records without special cases.
    """
    tokens = list(decoded_words)
    max_repeat = 1
    current_run = 1
    for previous, current in zip(tokens, tokens[1:]):
        if current == previous:
            current_run += 1
            max_repeat = max(max_repeat, current_run)
        else:
            current_run = 1
    longest_phrase = 0
    max_possible = len(tokens) // 2
    for k in range(1, max_possible + 1):
        for start in range(len(tokens) - (2 * k) + 1):
            if tokens[start : start + k] == tokens[start + k : start + 2 * k]:
                longest_phrase = max(longest_phrase, k)
    phrase_count = 0
    if longest_phrase:
        for start in range(len(tokens)):
            if (
                start + longest_phrase <= len(tokens)
                and tokens[start : start + longest_phrase]
                == tokens[start + longest_phrase : start + 2 * longest_phrase]
            ):
                phrase_count += 1
    return {
        "max_consecutive_repeat": max_repeat,
        "longest_repeated_phrase": longest_phrase,
        "repeated_phrase_count": phrase_count,
    }


def _repeat_flag(word_finding: Dict[str, Any], word_count: int) -> bool:
    """Word-level repeated-loop flag for the decoded text only.

    Mirrors ``src/hinglish/diagnostics.py::likely_repetitive_hallucination``
    with this job's label-free posture: the two criteria that compare against
    ground truth (expected word count and counted extra words) have no
    available input, so they take the neutral no-observation form and stay
    inactive. What remains active — and is enough to witness literal looping
    transcripts — is the consecutive-word trigger ``max_consecutive_repeat
    >= 3 with more than 3 decoded words``. Phrase-level findings are still
    recorded per clip for downstream review, and token-level periodic tails
    (``decode_diagnostics``) catch loops across non-identical words.
    """
    max_consecutive_repeat = int(word_finding.get("max_consecutive_repeat", 0) or 0)
    local_repeat = max_consecutive_repeat >= 3 and word_count > 3
    return bool(local_repeat)


def _clip_outcome(
    generated_token_count: Optional[int],
    instrumentation_ok: bool,
    error_text: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Classify a per-clip attempt into (status, failure_class).

    Args:
        generated_token_count: Generated tokens beyond the forced prefix
            (including 0), or ``None`` when no decode evidence exists.
        instrumentation_ok: True when exactly one instrumented generate call
            with batch-size-one sequences was captured.
        error_text: Set when the clip raised (load or inference failure).

    Returns:
        tuple: ``(status, failure_class)`` where status is ``"ok"`` or
        ``"failed"`` and failure_class is ``None``, ``"runtime_error"``
        (any exception or missing instrumentation), or ``"empty_status"``
        (zero generated tokens — the model emitted nothing). An infra-level
        abort such as container death or timeout never produces a record at
        all; it surfaces in the run log only.
    """
    if error_text or not instrumentation_ok or generated_token_count is None:
        return "failed", "runtime_error"
    if generated_token_count == 0:
        return "failed", "empty_status"
    return "ok", None


def process_clip(
    audio_path: Path,
    pipeline,
    cfg: ScoringContext,
    audio_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one pool clip and return its full record (network-free, pure).

    This is the single per-clip code path used by the container batch and by
    the offline CPU smoke. Args:
        audio_path: WAV path to load (mounted volume path in-container).
        pipeline: the ASR pipeline object (instrumented model already bound).
        cfg: ScoringContext with frozen kwargs, anchors, diagnostics, capture.
        audio_duration_sec: manifest duration (seconds). When ``None`` the
            decoded duration from the loaded waveform is used.

    Returns:
        dict: the frozen per-clip record schema. ``decoded_text`` (private)
        rides in the record and only leaves this function toward the output
        volume payload (checkpoint/result writes).
    """
    import soundfile as sf
    import torch

    clip_id = Path(audio_path).stem

    def _base_record(error: Optional[str], duration: Optional[float]) -> Dict[str, Any]:
        return {
            "clip_id": clip_id,
            "config": cfg.config_name,
            "config_note": cfg.config_note,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "language": cfg.language,
            "task": cfg.task,
            "precision": cfg.precision,
            "device": cfg.device,
            "status": "failed",
            "failure_class": "runtime_error",
            "error": error,
            "audio_duration_sec": duration,
            "inference_runtime_sec": None,
            "generated_token_count": None,
            "eos_emitted_effective": None,
            "eos_emitted_ids_witness": None,
            "eos_emitted_scores_witness": None,
            "hit_length_limit_effective": None,
            "max_consecutive_repeated_tokens": None,
            "repeated_token_fraction": None,
            "repetition_flag": None,
            "repetition_words": None,
            "logprob_stats": None,
            "chosen_token_ids": None,
            "instrumentation": None,
            "decode_diagnostics": None,
            "generate_kwargs": dict(cfg.generate_kwargs),
            "decoding_anchor": {
                "prefix_ids": list(cfg.prefix_ids),
                "eos_token_ids": list(cfg.eos_token_ids),
                "generation_config_max_length": cfg.generation_config_max_length,
            },
            "decoded_text": None,
        }

    # --- audio load (volume path or local smoke path) -------------------
    try:
        data, sample_rate = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate != 16000:
            data = resample_to_16k(data, sample_rate)
    except Exception as exc:  # noqa: BLE001 - per-clip failures are records
        record = _base_record(
            _truncate(f"audio_load: {type(exc).__name__}: {exc}"), None
        )
        if audio_duration_sec is not None:
            record["audio_duration_sec"] = float(audio_duration_sec)
        return record

    duration_sec = (
        float(audio_duration_sec)
        if audio_duration_sec is not None
        else round(len(data) / 16000, 3)
    )

    # --- instrumented generation ---------------------------------------
    cfg.capture.reset()
    try:
        started_at = time.perf_counter()
        with torch.inference_mode():
            output = pipeline(data, generate_kwargs=cfg.generate_kwargs)
        runtime_sec = round(time.perf_counter() - started_at, 3)
    except Exception as exc:  # noqa: BLE001
        record = _base_record(_truncate(f"{type(exc).__name__}: {exc}"), duration_sec)
        return record

    capture_records = list(cfg.capture.captures)
    instrumentation_ok = len(capture_records) == 1
    generated_ids: List[int] = []
    chosen: Optional[List[int]] = None
    prefix_match_count = 0
    decode_diagnostics: Optional[Dict[str, Any]] = None
    score_summary: Optional[Dict[str, Any]] = None
    if instrumentation_ok and capture_records:
        chosen = capture_records[0].get("chosen_token_ids")
        score_summary = capture_records[0].get("score_summary")
        sequences = capture_records[0].get("sequences") or []
        batch_rows = len(sequences)
        instrumentation_ok = batch_rows == 1 and chosen is not None
        generated_ids = [int(t) for t in sequences[0]] if batch_rows else []
        if generated_ids[: len(cfg.prefix_ids)] == list(cfg.prefix_ids):
            prefix_match_count = len(cfg.prefix_ids)
        probe = cfg.diagnostics.analyze_generation(
            generated_ids,
            decoder_prefix_tokens=prefix_match_count,
            configured_max_length=cfg.generation_config_max_length,
            configured_max_new_tokens=cfg.generate_kwargs.get("max_new_tokens"),
            eos_token_ids=cfg.eos_token_ids,
        )
        decode_diagnostics = asdict(probe)
    if not instrumentation_ok:
        record = _base_record(
            _truncate(
                "instrumentation missing: expected one generate capture with "
                "batch-size-one sequences and chosen ids"
            ),
            duration_sec,
        )
        record["generated_token_count"] = len(generated_ids) or None
        return record

    # --- two-way EOS / length-limit evidence (probe semantics) ----------
    last_chosen = chosen[-1] if chosen else None
    scores_witness = bool(last_chosen is not None and last_chosen in set(cfg.eos_token_ids))
    ids_witness = bool((decode_diagnostics or {}).get("eos_emitted", False))
    eos_emitted_effective = bool(ids_witness or scores_witness)
    hit_length_limit_effective = bool(
        (decode_diagnostics or {}).get("hit_length_limit", False) and not scores_witness
    )

    # --- token-level repetition (ids beyond prefix) ---------------------
    repetition_dict = (decode_diagnostics or {}).get("token_repetition") or {}
    generated_portion = generated_ids[prefix_match_count:]
    repeated_positions = sum(
        1 for previous, current in zip(generated_portion, generated_portion[1:])
        if previous == current
    )
    repeated_fraction = (
        round(repeated_positions / len(generated_portion), 6)
        if generated_portion
        else 0.0
    )

    # --- label-free repetition evidence from decoded text ---------------
    decoded_words = str(output.get("text", "")).strip().split()
    word_finding = _word_repetition_finding(decoded_words)
    repetition_flag = _repeat_flag(word_finding, len(decoded_words))

    outcome_status, outcome_class = _clip_outcome(
        len(generated_portion), instrumentation_ok, None
    )

    record = {
        "clip_id": clip_id,
        "config": cfg.config_name,
        "config_note": cfg.config_note,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "language": cfg.language,
        "task": cfg.task,
        "precision": cfg.precision,
        "device": cfg.device,
        "status": outcome_status,
        "failure_class": outcome_class,
        "error": None,
        "audio_duration_sec": duration_sec,
        "inference_runtime_sec": runtime_sec,
        "generated_token_count": len(generated_portion),
        "eos_emitted_effective": eos_emitted_effective,
        "eos_emitted_ids_witness": ids_witness,
        "eos_emitted_scores_witness": scores_witness,
        "hit_length_limit_effective": hit_length_limit_effective,
        "max_consecutive_repeated_tokens": repetition_dict.get(
            "max_consecutive_token_repeats"
        ),
        "repeated_token_fraction": repeated_fraction,
        "repetition_flag": repetition_flag,
        "repetition_words": word_finding,
        "logprob_stats": score_summary,
        "chosen_token_ids": list(chosen or []),
        "instrumentation": {
            "return_dict_in_generate": True,
            "output_scores": True,
            "generate_call_count": len(capture_records),
            "instrumentation_ok": instrumentation_ok,
            "chosen_step_count": len(chosen) if chosen else 0,
            "last_chosen_token_id": last_chosen,
        },
        "decode_diagnostics": decode_diagnostics,
        "generate_kwargs": capture_records[0]["kwargs"],
        "decoding_anchor": {
            "prefix_ids": list(cfg.prefix_ids),
            "eos_token_ids": list(cfg.eos_token_ids),
            "generation_config_max_length": cfg.generation_config_max_length,
            "prefix_matched": prefix_match_count == len(cfg.prefix_ids),
        },
        "decoded_text": str(output.get("text", "")).strip(),
    }
    return record


def resample_to_16k(data, sample_rate: int):
    """Resample a mono float32 waveform to 16 kHz via torchaudio."""
    import torch
    import torchaudio

    waveform = torch.from_numpy(data).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    return resampler(waveform).squeeze(0).cpu().numpy()


def _load_manifest(volume_root: Path) -> List[Dict[str, str]]:
    """Load the pool manifest (clip_id, audio_duration_sec, shard, ...).

    Args:
        volume_root: directory holding ``manifest.csv`` at its root (the
            Modal pool volume in-container, a local mirror for smoke).

    Returns:
        list[dict]: manifest rows with string values only.

    Raises:
        ValueError: when required columns, row ids, or durations are bad.
    """
    manifest_path = volume_root / MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = {"clip_id", "audio_duration_sec"} - set(fieldnames)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows = [row for row in reader if (row.get("clip_id") or "").strip()]
    ids = [row["clip_id"].strip() for row in rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted({clip for clip in ids if ids.count(clip) > 1})
        raise ValueError(f"duplicate manifest clip ids: {duplicates[:10]}")
    for row in rows:
        row["clip_id"] = row["clip_id"].strip()
        try:
            float(row["audio_duration_sec"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"manifest clip {row['clip_id']} has non-numeric "
                f"audio_duration_sec: {row['audio_duration_sec']!r}"
            ) from exc
    return rows


def _checksum_index(payload: Any) -> Dict[str, str]:
    """Normalize ``audio_checksums.json`` into a clip_id -> sha256 mapping.

    Accepted shapes (kept deliberately loose; the pool packaging stage owns
    the schema, and the contract requires per-clip verification):

    * ``{"<clip_id or path>": "<hex>"}`` — keys may be clip ids, file names,
      or ``audio/<clip_id>.wav`` style paths;
    * ``{"<key>": {"sha256": "<hex>"}}`` — nested digest dictionaries;
    * ``{"files": {...one of the above...}}`` — nested container mapping.
    """
    if not isinstance(payload, dict):
        raise ValueError("audio_checksums.json must be a JSON object")
    table = payload.get("files") if isinstance(payload.get("files"), dict) else payload
    index: Dict[str, str] = {}
    for key, value in table.items():
        if not isinstance(key, str):
            raise ValueError(f"checksum key is not a string: {key!r}")
        if isinstance(value, dict):
            value = value.get("sha256", value.get("digest"))
        if not isinstance(value, str) or not value:
            raise ValueError(f"checksum value missing for {key!r}")
        name = Path(key).name
        if name.endswith(".wav"):
            name = name[: -len(".wav")]
        index[name] = value.strip().casefold()
    if not index:
        raise ValueError("audio_checksums.json carries no per-file digests")
    return index


def _duration_histogram(durations: List[float]) -> Dict[str, int]:
    """Bucket audio durations (seconds) into a fixed histogram."""
    buckets = {"lt_2s": 0, "2_5s": 0, "5_10s": 0, "10_20s": 0, "gt_20s": 0}
    for duration in durations:
        if duration < 2.0:
            buckets["lt_2s"] += 1
        elif duration < 5.0:
            buckets["2_5s"] += 1
        elif duration < 10.0:
            buckets["5_10s"] += 1
        elif duration < 20.0:
            buckets["10_20s"] += 1
        else:
            buckets["gt_20s"] += 1
    return buckets


def _preflight_pool_dir(pool_dir: Path, *, require_full_pool: bool) -> Dict[str, Any]:
    """Verify the local pool mirror end to end (called for every mode).

    Checks: manifest load, exactly the manifest's WAV set under ``audio/``,
    the expected pool size (skipped for ``--limit`` smoke pools), and that
    every WAV matches its ``audio_checksums.json`` digest. Returns summary
    statistics (payload size, duration histogram) for the launch estimate.

    Raises:
        ValueError: on any violation; the caller prints and exits 2.
    """
    if not pool_dir.is_dir():
        raise ValueError(f"pool dir not found: {pool_dir}")
    rows = _load_manifest(pool_dir)
    clip_ids = [row["clip_id"] for row in rows]
    count = len(rows)
    if require_full_pool and count != EXPECTED_POOL_CLIP_COUNT:
        raise ValueError(
            f"manifest has {count} clips; expected {EXPECTED_POOL_CLIP_COUNT}"
        )
    if count == 0:
        raise ValueError("pool manifest has no rows")

    audio_dir = pool_dir / AUDIO_DIR_NAME
    if not audio_dir.is_dir():
        raise ValueError(f"pool audio dir not found: {audio_dir}")
    wav_files = sorted(audio_dir.glob("*.wav"))
    wav_ids = [path.stem for path in wav_files]
    if len(wav_ids) != len(clip_ids) or set(wav_ids) != set(clip_ids):
        missing = sorted(set(clip_ids) - set(wav_ids))[:10]
        extra = sorted(set(wav_ids) - set(clip_ids))[:10]
        raise ValueError(
            "manifest/audio mismatch: "
            f"missing {missing} extra {extra}"
        )
    # Align samples to the manifest order for per-clip verification.
    wav_by_id = {path.stem: path for path in wav_files}
    wav_files = [wav_by_id[clip_id] for clip_id in clip_ids]

    checksums_path = pool_dir / CHECKSUMS_FILENAME
    if not checksums_path.is_file():
        raise ValueError(f"checksum file not found: {checksums_path}")
    index = _checksum_index(json.loads(checksums_path.read_text(encoding="utf-8")))
    mismatches: List[str] = []
    for clip_id, wav_file in zip(clip_ids, wav_files):
        expected = index.get(clip_id)
        if expected is None:
            mismatches.append(f"{clip_id}:no_checksum_entry")
        elif _sha256_file(wav_file) != expected:
            mismatches.append(f"{clip_id}:sha_mismatch")
    if mismatches:
        raise ValueError(f"audio checksum problems: {mismatches[:10]}")

    payload_bytes = sum(
        path.stat().st_size
        for path in (pool_dir.rglob("*") if pool_dir.is_dir() else [])
        if path.is_file()
    )
    durations = [float(row["audio_duration_sec"]) for row in rows]
    return {
        "pool_dir": str(pool_dir),
        "clip_count": count,
        "clip_ids": clip_ids,
        "payload_bytes": payload_bytes,
        "payload_megabytes": round(payload_bytes / (1024 * 1024), 2),
        "duration_histogram": _duration_histogram(durations),
        "total_audio_sec": round(sum(durations), 3),
        "manifest_sha256": _sha256_file(pool_dir / MANIFEST_FILENAME),
    }


def _estimate_wall_and_cost(durations: List[float]) -> Dict[str, Any]:
    """Estimate wall time and cost from per-clip durations (T4 calibration).

    Formula: ``est_clip = 0.55 + 0.016 * audio_duration_sec`` seconds, so
    ``est_wall = 150 s + sum(est_clip)`` where the 150 s overhead is the
    conservative upper band of container startup (30-60 s) and model load
    (60-90 s) measured during the Phase 7.1 T4 probe. Cost uses
    ``RATE_T4_USD_PER_SEC``; the timeout worst case prices a full 900 s
    container.
    """
    per_clip = [round(BASE_PER_CLIP_SEC + PER_CLIP_SEC_PER_AUDIO_SEC * d, 3) for d in durations]
    inference_total = round(sum(per_clip), 3)
    wall = round(OVERHEAD_SEC + inference_total, 3)
    return {
        "formula": (
            f"est_clip={BASE_PER_CLIP_SEC}+{PER_CLIP_SEC_PER_AUDIO_SEC}*dur; "
            f"wall={OVERHEAD_SEC}+sum(est_clip)"
        ),
        "per_clip_mean_sec": round(sum(per_clip) / len(per_clip), 3) if per_clip else 0.0,
        "per_clip_max_sec": round(max(per_clip), 3) if per_clip else 0.0,
        "inference_total_sec": inference_total,
        "overhead_sec": round(OVERHEAD_SEC, 3),
        "wall_seconds": wall,
        "estimated_gpu_cost_usd": round(wall * RATE_T4_USD_PER_SEC, 5),
        "timeout_worst_case_seconds": FUNCTION_TIMEOUT_SEC,
        "timeout_worst_case_cost_usd": round(
            FUNCTION_TIMEOUT_SEC * RATE_T4_USD_PER_SEC, 5
        ),
        "planning_budget_usd": PLANNING_BUDGET_USD,
    }


def _append_checkpoint_line(checkpoint_path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON line per clip with flush+fsync for crash durability.

    The write fails loud: if the output volume cannot be written, the
    invocation aborts rather than continuing without per-clip durability.
    """
    line = json.dumps(record, ensure_ascii=False)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resume_completed_clip_ids(checkpoint_path: Path) -> List[str]:
    """Ids with an ``ok`` checkpoint record (resume support).

    Malformed or truncated trailing lines (an interrupted write) are counted
    and skipped, never trusted. Only ``ok`` records count as complete, so
    failed attempts are reprocessed under resume.
    """
    completed: List[str] = []
    bad_lines = 0
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            status = record.get("status")
            clip_id = record.get("clip_id")
            if status == "ok" and clip_id:
                completed.append(str(clip_id))
    print(
        f"resume scan: {len(completed)} completed clips, "
        f"{bad_lines} bad checkpoint lines skipped",
        flush=True,
    )
    return completed


def _final_records(checkpoint_path: Path, clip_ids: List[str]) -> List[Dict[str, Any]]:
    """Rebuild per-clip truth from the checkpoint log (last record wins).

    The checkpoint is the source of truth for the final results artifact:
    resumed runs append fresh lines, and the latest record for each manifest
    clip is used.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            clip_id = record.get("clip_id")
            if clip_id:
                latest[str(clip_id)] = record
    return [latest[clip_id] for clip_id in clip_ids if clip_id in latest]


def _results_payload(
    final_records: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the final results document (run meta, summary, per-clip records).

    The summary separates infra-level success from per-clip recognition
    success: ``infra_successful_clips`` counts records whose decode ran to a
    measured end (status ``ok``), ``recognition_failed_clips`` counts failed
    recognitions, and ``recognition_successful_clips`` counts ``ok`` records
    whose decode produced non-empty generated text. Counting uses statuses,
    flags, and durations only — no decoded-text length statistic, and no
    metrics against any held-out text (this job is label-free).
    """
    records = list(final_records)
    ok = sum(1 for record in records if record.get("status") == "ok")
    failed = sum(1 for record in records if record.get("status") == "failed")
    recognition_ok = sum(
        1
        for record in records
        if record.get("status") == "ok" and (record.get("decoded_text") or "").strip()
    )
    failure_classes: Dict[str, int] = {}
    for record in records:
        failure_class = record.get("failure_class")
        if failure_class:
            failure_classes[failure_class] = failure_classes.get(failure_class, 0) + 1
    runtimes = [
        float(record["inference_runtime_sec"])
        for record in records
        if record.get("inference_runtime_sec") is not None
    ]
    summary = {
        "clips_requested": run_meta.get("clips_requested"),
        "clip_records_final": len(records),
        "infra_successful_clips": ok,
        "recognition_failed_clips": failed,
        "recognition_successful_clips": recognition_ok,
        "failure_class_counts": failure_classes,
        "total_inference_runtime_sec": round(sum(runtimes), 3),
        "max_inference_runtime_sec": round(max(runtimes), 3) if runtimes else None,
        "records_with_logprob_stats": sum(
            1 for record in records if record.get("logprob_stats")
        ),
        "records_eos_emitted_effective": sum(
            1 for record in records if record.get("eos_emitted_effective")
        ),
        "records_hit_length_limit_effective": sum(
            1 for record in records if record.get("hit_length_limit_effective")
        ),
        "records_repetition_flagged": sum(
            1 for record in records if record.get("repetition_flag")
        ),
    }
    return {
        "score_version": SCORE_VERSION_MARKER,
        "run_meta": run_meta,
        "summary": summary,
        "records": records,
    }


def _write_results(outputs_root: Path, payload: Dict[str, Any]) -> None:
    """Write ``scoring_results.json`` plus its sha256 sidecar on the output."""
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    results_path = outputs_root / RESULTS_FILENAME
    results_path.write_text(raw + "\n", encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    (outputs_root / RESULTS_SHA_FILENAME).write_text(
        digest + "\n", encoding="utf-8"
    )


def _build_scoring_context(assets: Dict[str, Any], diagnostics) -> ScoringContext:
    """Freeze the C1 kwargs and bundle anchors into a ScoringContext."""
    generate_kwargs = diagnostics.config_generate_kwargs(CONFIG_NAME)
    measured = generate_kwargs.get("max_new_tokens")
    if measured != EXPECTED_C1_MAX_NEW_TOKENS:
        raise ValueError(
            f"diagnostics C1 kwargs drifted: max_new_tokens={measured!r} "
            f"(frozen budget is {EXPECTED_C1_MAX_NEW_TOKENS})"
        )
    device, precision = _effective_device_precision()
    return ScoringContext(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        language=LANGUAGE,
        task=TASK,
        precision=precision,
        device=device,
        config_name=CONFIG_NAME,
        config_note=CONFIG_NOTE,
        generate_kwargs=generate_kwargs,
        eos_token_ids=list(assets["eos_token_ids"]),
        prefix_ids=list(assets["prefix_ids"]),
        generation_config_max_length=int(assets["generation_config_max_length"]),
        diagnostics=diagnostics,
        capture=_install_capture_wrapper(assets["model"], diagnostics),
    )


def _run_pool_batch(
    assets: Dict[str, Any],
    diagnostics_source: str,
    *,
    resume: bool = False,
    pool_root: str = POOL_MOUNT_PATH,
    outputs_root: str = OUTPUT_MOUNT_PATH,
    commit_outputs: Optional[Callable[[], None]] = None,
    row_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode the whole pool in one pass and finalize the output artifacts.

    One container, one model load (the caller's warmed assets), one loop. Per
    clip: process -> append one checkpoint line -> commit the output volume.
    The network return carries only run meta plus summary counts; per-clip
    records (with decoded texts) persist only in the output volume payload.

    Args:
        assets: warmed runtime from the enter hook (or the local loader).
        diagnostics_source: diagnostics module text (hashed for provenance).
        resume: when True, completed clips are skipped, one fresh invocation
            processes the rest, and the existing checkpoint is required.
        pool_root: mounted pool volume path (read-only).
        outputs_root: mounted scratch output volume path (write).
        commit_outputs: optional volume commit callback; invoked after every
            checkpoint append and after the final artifact writes.
        row_window: optional cap on manifest rows processed (``--limit``
            local smoke); ``None`` in production runs.

    Returns:
        dict: ``{"score_version", "run_meta", "summary"}`` — no records.
    """
    diagnostics = _import_diagnostics_from_source(diagnostics_source)
    context = _build_scoring_context(assets, diagnostics)
    pipeline = assets["pipeline"]

    pool_dir = Path(pool_root)
    outputs_dir = Path(outputs_root)
    checkpoint_path = outputs_dir / CHECKPOINT_FILENAME
    if not resume and checkpoint_path.exists():
        raise RuntimeError(
            f"{CHECKPOINT_FILENAME} already exists on the output volume; "
            "a fresh scored run must not append to a previous run. Pass "
            "--resume (explicit operator choice) to continue that run."
        )
    completed: List[str] = []
    if resume:
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"--resume given but {CHECKPOINT_FILENAME} not found on the "
                "output volume; run without --resume instead."
            )
        completed = _resume_completed_clip_ids(checkpoint_path)

    rows = _load_manifest(pool_dir)
    if row_window is not None:
        rows = rows[:row_window]
    started_at = time.time()
    appended = 0
    skipped_resumed = 0
    for row in rows:
        clip_id = row["clip_id"]
        if clip_id in completed:
            skipped_resumed += 1
            continue
        audio_path = pool_dir / AUDIO_DIR_NAME / f"{clip_id}.wav"
        try:
            duration = float(row["audio_duration_sec"])
        except (KeyError, TypeError, ValueError):
            duration = None
        record = process_clip(audio_path, pipeline, context, audio_duration_sec=duration)
        _append_checkpoint_line(checkpoint_path, record)
        appended += 1
        if commit_outputs is not None:
            commit_outputs()
        print(
            f"scored {clip_id}: status={record['status']} "
            f"class={record['failure_class']} "
            f"tokens={record['generated_token_count']} "
            f"runtime={record['inference_runtime_sec']}",
            flush=True,
        )

    final_records = _final_records(checkpoint_path, [row["clip_id"] for row in rows])
    device, precision = _effective_device_precision()
    run_meta = {
        "score_version": SCORE_VERSION_MARKER,
        "diagnostics_version": PROBE_VERSION_MARKER,
        "app_name": APP_NAME,
        "platform_gpu": GPU_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "language": LANGUAGE,
        "task": TASK,
        "config_name": CONFIG_NAME,
        "config_note": CONFIG_NOTE,
        "expected_pool_clip_count": EXPECTED_POOL_CLIP_COUNT,
        "clips_requested": len(rows),
        "precision": precision,
        "device": device,
        "manifest_sha256": _sha256_file(pool_dir / MANIFEST_FILENAME),
        "diagnostics_source_sha256": hashlib.sha256(
            diagnostics_source.encode("utf-8")
        ).hexdigest(),
        "volumes": {
            "pool": {"name": DEFAULT_POOL_VOLUME_NAME, "mount": POOL_MOUNT_PATH},
            "outputs": {"name": OUTPUT_VOLUME_NAME, "mount": OUTPUT_MOUNT_PATH},
        },
        "resume_used": bool(resume),
        "limit_applied": row_window,
        "run_started_unix": round(started_at, 3),
        "run_finished_unix": round(time.time(), 3),
        "checkpoint_records_appended": appended,
        "clips_skipped_resumed": skipped_resumed,
    }
    payload = _results_payload(final_records, run_meta)
    _write_results(outputs_dir, payload)
    if commit_outputs is not None:
        commit_outputs()
    print(
        "pool scoring finished: "
        f"{payload['summary']['infra_successful_clips']} ok / "
        f"{payload['summary']['recognition_failed_clips']} failed; "
        "records persisted on the output volume",
        flush=True,
    )
    return {
        "score_version": SCORE_VERSION_MARKER,
        "run_meta": run_meta,
        "summary": payload["summary"],
    }


if modal is not None:

    @app.cls(
        image=image,
        gpu="T4",
        timeout=FUNCTION_TIMEOUT_SEC,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        volumes={
            POOL_MOUNT_PATH: pool_volume,
            OUTPUT_MOUNT_PATH: outputs_volume,
            MODEL_CACHE_MOUNT_PATH: model_cache,
        },
    )
    class PoolClipScorer:
        """One-container score worker; the model loads once at start."""

        @modal.enter()
        def build_runtime(self) -> None:
            """Load whisper-small once per container (single paid load)."""
            self._assets = _load_runtime_assets()
            device, precision = _effective_device_precision()
            print(
                "scorer runtime ready: "
                f"device={device} precision={precision} "
                f"prefix_tokens={self._assets['generation_config_max_length']}",
                flush=True,
            )

        @modal.method()
        def score_pool(
            self, diagnostics_source: str, resume: bool = False
        ) -> Dict[str, Any]:
            """Score the entire pool batch in one invocation."""
            return _run_pool_batch(
                self._assets,
                diagnostics_source,
                resume=resume,
                pool_root=POOL_MOUNT_PATH,
                outputs_root=OUTPUT_MOUNT_PATH,
                commit_outputs=lambda: outputs_volume.commit(),
            )


def _run_local_limit(args: argparse.Namespace, diagnostics_source: str) -> int:
    """Local CPU-first smoke through the exact batch code path (no Modal).

    ``--limit N`` processes the first N manifest rows from ``--pool-dir`` on
    this machine with the env-overridden device/precision/cache, writing the
    checkpoint and results under ``--local-outputs-dir`` (default:
    ``<pool-dir>/{LOCAL_DEFAULT_OUTPUTS_DIRNAME}``). Billable Modal runs are
    never touched here; pairing ``--limit`` with ``--run`` is refused by the
    CLI.
    """
    outputs_dir = args.local_outputs_dir or args.pool_dir / LOCAL_DEFAULT_OUTPUTS_DIRNAME
    assets = _load_runtime_assets()
    print(
        "local smoke runtime ready: "
        f"device={assets['device']} precision={assets['precision']}",
        flush=True,
    )
    try:
        result = _run_pool_batch(
            assets,
            diagnostics_source,
            resume=args.resume,
            pool_root=str(args.pool_dir),
            outputs_root=str(outputs_dir),
            commit_outputs=None,
            row_window=args.limit,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"LOCAL RUN FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("local smoke outputs:", outputs_dir)
    print("checkpoint:", outputs_dir / CHECKPOINT_FILENAME)
    print("results:", outputs_dir / RESULTS_FILENAME)
    print("decoded texts live only in those files (never in the run stdout)")
    return 0


def _launch_modal_run(args: argparse.Namespace, estimate: Dict[str, Any]) -> int:
    """The single approved billable launch (caller has passed preflight)."""
    if modal is None:
        print("Modal is not installed. Install with: pip install -e '.[hinglish]'")
        return 1
    if args.volume_name != DEFAULT_POOL_VOLUME_NAME:
        print(
            "VOLUME NAME MISMATCH: this one-time runner is bound to "
            f"{DEFAULT_POOL_VOLUME_NAME!r}; rerun with the default --volume-name."
        )
        return 2
    if estimate["wall_seconds"] > FUNCTION_TIMEOUT_SEC:
        print(
            "COST/TIME GATE FAILED: estimated wall exceeds the 900 s function "
            "timeout; refusing to launch."
        )
        return 2
    if estimate["estimated_gpu_cost_usd"] > estimate["planning_budget_usd"]:
        print(
            "COST/TIME GATE FAILED: estimate exceeds the planning budget "
            f"({estimate['planning_budget_usd']} USD); refusing to launch."
        )
        return 2

    diagnostics_source = _load_diagnostics_source()
    print(
        f"Launching label-free pool scoring on {GPU_NAME} "
        "(single approved operator run)..."
    )
    try:
        with app.run():
            result = PoolClipScorer().score_pool.remote(
                diagnostics_source, resume=args.resume
            )
    except Exception as exc:  # noqa: BLE001 - reported, never retried
        print(f"Modal execution failed: {type(exc).__name__}: {exc}")
        print(
            "No automatic retry is authorized. Completed clips persist on the "
            "checkpoint; a continuation is an explicit operator choice (--resume)."
        )
        return 1
    if isinstance(result, dict):
        print(json.dumps(
            {"run_meta": result.get("run_meta"), "summary": result.get("summary")},
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(f"unexpected return type: {type(result).__name__}")
    print("per-clip records live on the output volume:")
    print(f"  {OUTPUT_MOUNT_PATH}/{CHECKPOINT_FILENAME}")
    print(f"  {OUTPUT_MOUNT_PATH}/{RESULTS_FILENAME}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run (default), explicit single-run launcher (--run), or "
            "local CPU-first smoke (--limit N) for the Phase 7.2 label-free "
            "300-clip pool scoring job (frozen C1 budget) on one Modal T4 "
            "container"
        )
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        required=True,
        help=(
            "Local pool mirror for preflight (manifest.csv, audio/*.wav, "
            "audio_checksums.json). The Modal run reads the seeded pool "
            "volume, never this directory."
        ),
    )
    parser.add_argument(
        "--volume-name",
        default=DEFAULT_POOL_VOLUME_NAME,
        help=(
            "Pool Modal Volume name (read-only). This one-time runner is "
            f"bound to {DEFAULT_POOL_VOLUME_NAME!r}; other values are refused."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the single billable Modal launch (preflight must pass first).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a prior run: one NEW container invocation skips "
            "completed clips (checkpoint records with status ok) and scores "
            "the rest. Resume is an explicit operator choice, NOT an "
            "authorized automatic second GPU run; failed attempts are "
            "reprocessed and may append fresh records."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Local CPU-first smoke only: process the first N manifest rows "
            "on this machine via the offline device/precision/cache env "
            "overrides. Never contacts Modal; refused together with --run."
        ),
    )
    parser.add_argument(
        "--local-outputs-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the --limit local smoke (default: "
            f"<pool-dir>/{LOCAL_DEFAULT_OUTPUTS_DIRNAME})."
        ),
    )
    args = parser.parse_args()

    if args.run and args.limit is not None:
        print("REFUSED: --limit is a local smoke surface and cannot be combined with --run.")
        return 2
    if args.limit is not None and args.limit <= 0:
        print("REFUSED: --limit must be a positive integer.")
        return 2

    preflight: Dict[str, Any]
    try:
        preflight = _preflight_pool_dir(args.pool_dir, require_full_pool=args.limit is None)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}")
        return 2
    if args.limit is not None and args.limit > preflight["clip_count"]:
        print(
            f"REFUSED: --limit {args.limit} exceeds the pool manifest "
            f"({preflight['clip_count']} rows)."
        )
        return 2

    try:
        diagnostics_source = _load_diagnostics_source()
    except (OSError, ValueError) as exc:
        print(f"DIAGNOSTICS CHECK FAILED: {type(exc).__name__}: {exc}")
        return 2

    durations: List[float] = []
    manifest_path = args.pool_dir / MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("clip_id") or "").strip():
                durations.append(float(row["audio_duration_sec"]))
    estimate = _estimate_wall_and_cost(durations)

    print("MendSpeech Hinglish label-free pool scoring (J2) — plan summary")
    print(f"score_version: {SCORE_VERSION_MARKER}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"volume_name: {args.volume_name} (mount {POOL_MOUNT_PATH}, read-only)")
    print(f"output volume: {OUTPUT_VOLUME_NAME} (mount {OUTPUT_MOUNT_PATH})")
    print(f"model_id: {MODEL_ID}")
    print(f"model_revision: {MODEL_REVISION}")
    print(f"gpu: {DEFAULT_GPU} ({GPU_NAME})")
    print(f"max_containers: {MAX_CONTAINERS}")
    print(f"scaledown_window_sec: {SCALEDOWN_WINDOW_SEC}")
    print(f"function_timeout_sec: {FUNCTION_TIMEOUT_SEC}")
    print(f"config: {CONFIG_NAME} — {CONFIG_NOTE}")
    print(f"language: {LANGUAGE} task: {TASK}")
    print(f"precision: {LABELFREE_PRECISION}")
    print(f"pool_clips: {preflight['clip_count']}")
    print(f"payload_verified: {preflight['clip_count']} sha256 matches")
    print(f"payload_size_mb: {preflight['payload_megabytes']}")
    print(f"duration_histogram_sec: {preflight['duration_histogram']}")
    print(f"manifest_sha256: {preflight['manifest_sha256']}")
    print(f"estimate: {json.dumps(estimate, indent=2)}")
    print(
        "overhead note: 150 s is the conservative upper band (startup 30-60 s "
        "+ model load 60-90 s measured on the Phase 7.1 T4 run)"
    )
    # Only clip ids and estimates above; decoded texts never reach stdout.
    if args.limit is not None:
        print(f"mode: local smoke (--limit {args.limit}); no Modal contact")
        return _run_local_limit(args, diagnostics_source)
    if args.run:
        return _launch_modal_run(args, estimate)
    print("status: no GPU launched; pass --run for the approved single run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
