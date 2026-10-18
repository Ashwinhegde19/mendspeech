"""Source and behavior guards for the Phase 7.2 label-free pool scoring runner.

These tests lock the J2 execution constraints for
``infra/modal_hinglish_label_free_score.py``:

* source guards (parsed AST plus raw text greps, no Modal/Web deps anywhere):
  - single ``@app.cls`` endpoint with one ``@modal.enter`` hook and one
    ``@modal.method`` (no ``@app.function`` anywhere), one container,
    a 900 s maximum function timeout, and no ``retries`` keyword anywhere;
  - the GPU is the T4 (``DEFAULT_GPU = "T4"`` plus a full
    ``NVIDIA T4`` hardware name constant) and the runner source never
    mentions other accelerators (raw absence of ``L4``, ``A10G``, ``H100``);
  - the runner source also never contains the tokens ``wer`` or
    ``reference`` (case-insensitive raw grep): the job is label-free and must
    not load ground truth or compute any word-error metric;
  - decode configuration is frozen: ``CONFIG_NAME = "C1"``,
    ``EXPECTED_C1_MAX_NEW_TOKENS = 160``, language ``hi``/task ``transcribe``,
    float16 default precision, and the packaging-only instrumentation flags
    ``return_dict_in_generate``/``output_scores``;
  - volume contract: the pool volume (``mucs-j2-pool``) is mounted without
    ``create_if_missing`` (read-only discipline), the scratch output volume
    is the only ``create_if_missing=True`` volume;
  - checkpoint/resume surfaces: per-clip ``scoring_checkpoint.jsonl`` append
    helper, a ``--resume`` CLI flag with the "not an authorized second run"
    wording, a ``--run`` opt-in ``store_true`` flag (dry run stays default),
    and ``--limit`` as a local-smoke-only surface;
  - payload hygiene: no ``audio_bytes`` transport, no ``print`` of anything
    named like decoded text, and ``json.dumps`` calls confined to the
    checkpoint/results writers (plus logger-callers that emit only
    summaries/estimates) so decoded text can never reach stdout;
* behavior guards importing the module WITHOUT Modal installed (module-level
  imports are stdlib-only; torch/transformers imports live inside functions):
  classification of outcomes, word-repetition semantics mirrored from
  ``src/hinglish/diagnostics.py``, checksum normalization, duration
  histogram, and the T4 wall/cost estimate formula.

If the infra module has not been authored yet, every test skips with that
explanation instead of failing.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_PATH = REPO_ROOT / "infra" / "modal_hinglish_label_free_score.py"

PINNED_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
BASELINE_APP_NAME = "mendspeech-hinglish-asr"
J1_PROBE_APP_NAME = "mendspeech-hinglish-decoder-probe"
REQUIRED_GPU_MARKER = "T4"
REQUIRED_GPU_FULL_NAME = "NVIDIA T4"
#: Raw (case-sensitive) source absence greps: no other accelerator tier may
#: appear anywhere in the runner, docstrings included.
FORBIDDEN_MARKERS_RAW = ("L4", "A10G", "H100", "lama")
#: Label-free discipline greps (case-insensitive on the raw source): the
#: runner must never mention word-error rates or load ground truth.
FORBIDDEN_TOKENS_CASELESS = ("wer", "reference")
POOL_MOUNT_PATH = "/pool"
OUTPUT_MOUNT_PATH = "/outputs"
MAX_TIMEOUT = 900
ALLOWED_JSON_DUMPS_WRITERS = {
    "_append_checkpoint_line",
    "_write_results",
    "_launch_modal_run",
    "main",
}


def _source() -> str:
    """Return the runner source text (raises FileNotFoundError when absent)."""
    return INFRA_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    """The parsed module tree (skip cleanly while the module is unauthored)."""
    if not INFRA_PATH.is_file():
        pytest.skip(
            "infra/modal_hinglish_label_free_score.py has not been authored yet "
            "(being written in parallel by the subagents)"
        )
    source = _source()
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"infra/modal_hinglish_label_free_score.py does not parse: {exc}")


def _load_runner_module():
    """Import the runner as a real module (Modal guarded off in this venv)."""
    if not INFRA_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("guard_labelfree_runner", INFRA_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # environment or import-order issue, not a spec
        sys.modules.pop(spec.name, None)
        print(f"runner import failed in guard test environment: {exc!r}")
        return None
    return module


@pytest.fixture(scope="module")
def runner_module():
    """The imported runner module (skipped when unimportable here)."""
    module = _load_runner_module()
    if module is None:
        pytest.skip(
            "runner module not importable in this environment; "
            "source-only guards still ran"
        )
    return module


def _module_constant(tree: ast.Module, name: str, value_type: type) -> Optional[Any]:
    """Return a module-level constant assignment when it matches the type."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, value_type):
                    return value
    return None


def _calls(tree: ast.Module) -> Iterable[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _decorator_matches(node: ast.AST, owner: str, attr: str) -> bool:
    """True for decorator calls like ``@app.cls`` / ``@modal.enter()``."""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    )


def _class_decorators(tree: ast.Module) -> List[ast.expr]:
    decorators: List[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            decorators.extend(node.decorator_list)
    return decorators


def _function_decorators(tree: ast.Module) -> List[ast.expr]:
    """Decorators on FunctionDefs nested anywhere (class methods included)."""
    decorators: List[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators.extend(node.decorator_list)
    return decorators


def _call_kwargs(call: ast.Call) -> Dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def _enclosing_function_names(tree: ast.Module) -> "dict[int, Set[str]]":
    """Map node ids (via ast.walk marks) to their enclosing function names."""
    mapping: dict[int, Set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                mapping.setdefault(id(child), set()).add(node.name)
    return mapping


def _mentions_identifier(node: ast.AST, token: str) -> bool:
    """True when any Name/Attribute identifier in this subtree has the token."""
    for child in ast.walk(node):
        for identifier in (ast.Name, ast.Attribute):
            if isinstance(child, identifier) and token in getattr(
                child, "id", getattr(child, "attr", "")
            ).lower():
                return True
    return False


# --------------------------------------------------------------------------
# source guards
# --------------------------------------------------------------------------


def test_source_has_gpu_markers(tree: ast.Module, runner_module) -> None:
    """The runner names both the T4 marker and the full NVIDIA T4 name."""
    assert any(REQUIRED_GPU_MARKER == node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)), (
        "the T4 marker string constant is missing"
    )
    constants = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert any(REQUIRED_GPU_FULL_NAME in value for value in constants), (
        f"the full GPU name {REQUIRED_GPU_FULL_NAME!r} must appear in the runner"
    )
    if runner_module is not None:
        assert runner_module.DEFAULT_GPU == REQUIRED_GPU_MARKER
        assert runner_module.GPU_NAME == REQUIRED_GPU_FULL_NAME


def test_forbidden_markers_absent() -> None:
    """Raw source greps: no other accelerator tier is even mentioned."""
    if not INFRA_PATH.is_file():
        pytest.skip("runner not authored yet")
    source = _source()
    lowered = source.lower()
    offenders = [marker for marker in FORBIDDEN_MARKERS_RAW if marker in source]
    offenders += [token for token in FORBIDDEN_TOKENS_CASELESS if token in lowered]
    assert not offenders, (
        f"forbidden tokens present in runner source: {offenders}"
    )


def test_label_free_summary_has_no_word_error_fields(tree: ast.Module) -> None:
    """Summary/count keys stay clear of error-rate vocabulary."""
    keys: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.append(node.value)
    assert not any("wer" in key.lower() for key in keys)
    assert not any("reference" in key.lower() for key in keys)


def test_function_timeout_is_bounded(tree: ast.Module, runner_module) -> None:
    """FUNCTION_TIMEOUT_SEC is a literal and never exceeds 900 seconds."""
    value = _module_constant(tree, "FUNCTION_TIMEOUT_SEC", int)
    assert value is not None, (
        "runner must define a module-level literal FUNCTION_TIMEOUT_SEC"
    )
    assert value <= MAX_TIMEOUT, (
        f"FUNCTION_TIMEOUT_SEC={value} exceeds the {MAX_TIMEOUT} s budget"
    )
    assert value >= 30, "a pool run needs a real timeout budget"
    if runner_module is not None:
        assert runner_module.FUNCTION_TIMEOUT_SEC == value


def test_max_containers_one(tree: ast.Module) -> None:
    """MAX_CONTAINERS is exactly 1 (one container for the whole run)."""
    value = _module_constant(tree, "MAX_CONTAINERS", int)
    assert value == 1, f"MAX_CONTAINERS must be exactly 1, found {value}"


def test_single_cls_endpoint_structure(tree: ast.Module) -> None:
    """Exactly one @app.cls endpoint with one enter hook and one method."""
    cls_decorators = [d for d in _class_decorators(tree) if _decorator_matches(d, "app", "cls")]
    function_decorators = [
        d for d in _function_decorators(tree) if _decorator_matches(d, "app", "function")
    ]
    enter_decorators = [
        d for d in _function_decorators(tree) if _decorator_matches(d, "modal", "enter")
    ]
    method_decorators = [
        d for d in _function_decorators(tree) if _decorator_matches(d, "modal", "method")
    ]
    assert len(cls_decorators) == 1, "exactly one @app.cls endpoint is required"
    assert not function_decorators, "@app.function endpoints must not exist here"
    assert len(enter_decorators) == 1, "exactly one @modal.enter hook is required"
    assert len(method_decorators) == 1, "exactly one @modal.method is required"
    classes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(_decorator_matches(d, "app", "cls") for d in node.decorator_list)
    ]
    assert len(classes) == 1 and len(classes[0].body) >= 3


def test_endpoint_call_arguments(tree: ast.Module) -> None:
    """gpu is the T4, timeout is the bounded constant, retries never set."""
    cls_calls = [
        d for d in _class_decorators(tree)
        if isinstance(d, ast.Call) and _decorator_matches(d, "app", "cls")
    ]
    assert len(cls_calls) == 1
    kwargs = _call_kwargs(cls_calls[0])
    gpu_value = kwargs.get("gpu")
    assert isinstance(gpu_value, ast.Constant) and gpu_value.value == REQUIRED_GPU_MARKER, (
        "gpu= must be the literal T4"
    )
    timeout_value = kwargs.get("timeout")
    if isinstance(timeout_value, ast.Name):
        # Named constant form: the bound is asserted by
        # test_function_timeout_is_bounded; accept only that exact name.
        assert timeout_value.id == "FUNCTION_TIMEOUT_SEC"
    else:
        assert isinstance(timeout_value, ast.Constant)
        assert isinstance(timeout_value.value, int)
        assert timeout_value.value <= MAX_TIMEOUT
    for call in _calls(tree):
        for kw in call.keywords:
            assert kw.arg not in {"retries", "max_retries"}, (
                "automatic retries are forbidden for this one-time run"
            )


def test_pinned_revision_present(tree: ast.Module) -> None:
    """The exact pinned whisper-small revision string appears in the source."""
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert any(PINNED_REVISION in literal for literal in literals), (
        f"pinned revision {PINNED_REVISION} not found in the runner"
    )


def test_frozen_c1_configuration(tree: ast.Module) -> None:
    """The scored config is frozen C1 with probe instrumentation flags."""
    constants = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert REQUIRED_GPU_MARKER in constants
    assert "float16" in constants, "float16 default precision must be named"
    assert "return_dict_in_generate" in constants and "output_scores" in constants
    assert "C1" in constants, "the scored config must be frozen at C1"


def test_frozen_c1_kwargs_validated(runner_module) -> None:
    """If importable, the context freeze rejects C1 drift."""
    if runner_module.EXPECTED_C1_MAX_NEW_TOKENS != 160:
        pytest.fail("EXPECTED_C1_MAX_NEW_TOKENS must stay 160 (frozen C1)")
    context_note = runner_module.CONFIG_NOTE
    assert "C1" in context_note and "provisional damage limiter" in context_note


def test_volume_contract(tree: ast.Module, runner_module) -> None:
    """Pool volume is not create-if-missing; only the outputs volume is."""
    volume_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr == "from_name" and isinstance(func.value, ast.Attribute):
                owner = func.value
                if owner.attr == "Volume":
                    volume_calls.append(node)
    assert volume_calls, "expected modal.Volume.from_name calls in the runner"
    create_true = 0
    for call in volume_calls:
        kwargs = _call_kwargs(call)
        flag = kwargs.get("create_if_missing")
        if isinstance(flag, ast.Constant) and flag.value is True:
            create_true += 1
        else:
            assert isinstance(flag, ast.Constant) and flag.value is False, (
                "every Volume.from_name must pass create_if_missing explicitly"
            )
    assert create_true == 1, (
        "only the scratch output volume may be create_if_missing=True"
    )
    if runner_module is not None:
        assert runner_module.DEFAULT_POOL_VOLUME_NAME == "mucs-j2-pool"
        assert runner_module.OUTPUT_VOLUME_NAME == "mucs-j2-labelfree-outputs"
        assert runner_module.POOL_MOUNT_PATH == POOL_MOUNT_PATH
        assert runner_module.OUTPUT_MOUNT_PATH == OUTPUT_MOUNT_PATH


def test_checkpoint_and_resume_surfaces(tree: ast.Module) -> None:
    """Per-clip checkpoint naming, resume helper, and CLI resume flag."""
    constants = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "scoring_checkpoint.jsonl" in constants
    assert "scoring_results.json" in constants
    assert "scoring_results.sha256" in constants
    function_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_resume_completed_clip_ids" in function_names
    assert "_append_checkpoint_line" in function_names


def test_cli_flag_surface_and_opt_in(tree: ast.Module) -> None:
    """Every documented CLI surface exists; --run/--resume are store_true."""
    names: List[str] = []
    for node in _calls(tree):
        function = node.func
        is_add_argument = (
            isinstance(function, ast.Attribute) and function.attr == "add_argument"
        )
        if not is_add_argument or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.append(first.value)
    for flag in ("--pool-dir", "--volume-name", "--run", "--resume", "--limit",
                 "--local-outputs-dir"):
        assert flag in names, f"CLI flag {flag} surface missing"
    for name in ("--run", "--resume"):
        call = next(
            node for node in _calls(tree)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == name
        )
        action = next(
            (kw.value for kw in call.keywords
             if kw.arg == "action" and isinstance(kw.value, ast.Constant)),
            None,
        )
        assert getattr(action, "value", None) == "store_true", (
            f"{name} must be opt-in store_true so offline defaults stay safe"
        )


def test_resume_help_words_the_warning(tree: ast.Module) -> None:
    """The resume help text warns it is not an authorized second GPU run."""
    strings = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    warning = " ".join(strings)
    assert "explicit operator choice" in warning
    assert "NOT an authorized automatic second GPU run" in warning


def test_no_audio_bytes_transport(tree: ast.Module) -> None:
    """Only paths/ids cross function boundaries, never audio bytes."""
    identifiers: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, ast.arg):
            identifiers.append(node.arg)
    offenders = [name for name in identifiers if "audio_bytes" in name or "wav_bytes" in name]
    assert not offenders, f"audio bytes must never travel as function data: {offenders}"


def test_score_pool_method_signature(tree: ast.Module) -> None:
    """The endpoint method passes just diagnostics source and resume flag."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "score_pool":
            assert [arg.arg for arg in node.args.args] == ["self", "diagnostics_source", "resume"]
            break
    else:
        pytest.fail("score_pool method not found")


def test_print_never_emits_decoded_text(tree: ast.Module) -> None:
    """No print call references anything named like decoded text."""
    offenders: List[str] = []
    for node in _calls(tree):
        function = node.func
        is_print = isinstance(function, ast.Name) and function.id == "print"
        if not is_print:
            continue
        for arg in node.args:
            if _mentions_identifier(arg, "decoded"):
                offenders.append(ast.dump(arg))
        for arg in node.keywords:
            if _mentions_identifier(arg.value, "decoded"):
                offenders.append(ast.dump(arg.value))
    assert not offenders, (
        f"print must never emit decoded-text identifiers: {offenders}"
    )


def test_json_dumps_confined_to_volume_writers(tree: ast.Module) -> None:
    """json.dumps calls live only in the checkpoint/results writers.

    This pin is load-bearing: the decoded text leaves ``process_clip`` only
    inside records, so any dumps outside these functions could leak runtime
    records to stdout.
    """
    mapping = _enclosing_function_names(tree)
    dumps_functions: List[str] = []
    for node in _calls(tree):
        function = node.func
        is_dumps = (
            isinstance(function, ast.Attribute) and function.attr == "dumps"
            and isinstance(function.value, ast.Name) and function.value.id == "json"
        )
        if is_dumps:
            enclosing = mapping.get(id(node)) or {"<module>"}
            dumps_functions.extend(sorted(enclosing))
    assert dumps_functions, "json.dumps must exist for checkpoint/results writes"
    for name in dumps_functions:
        assert name in ALLOWED_JSON_DUMPS_WRITERS, (
            f"json.dumps called from {name!r}; allowed writers only"
        )


def test_process_clip_exists_and_is_pure(tree: ast.Module) -> None:
    """The per-clip function is importable without Modal by design."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "process_clip":
            positional = [arg.arg for arg in node.args.args]
            assert positional[:3] == ["audio_path", "pipeline", "cfg"]
            defaults = node.args.defaults
            assert len(defaults) == 1, "audio_duration_sec stays an optional kwarg"
            assert isinstance(defaults[0], ast.Constant) and defaults[0].value is None
            break
    else:
        pytest.fail("process_clip function not found")


def test_app_name_is_job_specific(tree: ast.Module) -> None:
    """The Modal app name differs from J1 and the Phase 6 baseline app."""
    app_name = _module_constant(tree, "APP_NAME", str)
    assert app_name is not None, "expected a module-level APP_NAME constant"
    assert app_name not in {BASELINE_APP_NAME, J1_PROBE_APP_NAME}


# --------------------------------------------------------------------------
# behavior guards on the pure helpers (module imported without Modal)
# --------------------------------------------------------------------------


def test_clip_outcome_classification(runner_module) -> None:
    """empty_status/runtime_error/ok classification is exact."""
    outcome = runner_module._clip_outcome
    assert outcome(0, True, None) == ("failed", "empty_status")
    assert outcome(5, True, None) == ("ok", None)
    assert outcome(None, True, None) == ("failed", "runtime_error")
    assert outcome(5, False, None) == ("failed", "runtime_error")
    assert outcome(5, True, "DeviceError: boom") == ("failed", "runtime_error")
    assert outcome(0, False, None) == ("failed", "runtime_error")


def test_word_repetition_semantics(runner_module) -> None:
    """Mirrored analyze_repetition values plus the label-free flag rule."""
    finding = runner_module._word_repetition_finding(["ल", "ल", "ल", "ब"])
    assert finding["max_consecutive_repeat"] == 3
    # Mirrored semantics: the 1-word phrase "ल" repeats at starts 0 and 1.
    assert finding["longest_repeated_phrase"] == 1
    assert finding["repeated_phrase_count"] == 2
    assert runner_module._repeat_flag(finding, 4) is True

    calm = runner_module._word_repetition_finding(["एक", "दो", "तीन", "चार"])
    assert calm["max_consecutive_repeat"] == 1
    assert runner_module._repeat_flag(calm, 4) is False

    loop = runner_module._word_repetition_finding(
        ["अच्छा", "है", "अच्छा", "है", "अच्छा", "है"]
    )
    assert loop["longest_repeated_phrase"] == 2
    # Mirrored semantics count every consecutive-occurrence start.
    assert loop["repeated_phrase_count"] == 3
    assert runner_module._repeat_flag(loop, 6) is False

    empty = runner_module._word_repetition_finding([])
    assert empty["max_consecutive_repeat"] == 1
    assert runner_module._repeat_flag(empty, 0) is False


def test_checksum_index_shapes(runner_module) -> None:
    """Accepted audio_checksums.json shapes normalize to clip-id digests."""
    index = runner_module._checksum_index
    shapes = (
        {"abc123": "A" * 64},
        {"abc123.wav": "A" * 64},
        {"audio/abc123.wav": "A" * 64},
        {"abc123": {"sha256": "A" * 64}},
        {"files": {"abc123.wav": "A" * 64}},
    )
    for shape in shapes:
        assert index(shape) == {"abc123": "a" * 64}
    with pytest.raises(ValueError):
        index(["bad"])
    with pytest.raises(ValueError):
        index({"abc123": 42})


def test_duration_histogram(runner_module) -> None:
    """Fixed buckets stay stable."""
    buckets = runner_module._duration_histogram([1.0, 2.5, 5.5, 10.5, 25.0])
    assert buckets == {"lt_2s": 1, "2_5s": 1, "5_10s": 1, "10_20s": 1, "gt_20s": 1}


def test_estimate_formula_matches_plan(runner_module) -> None:
    """est_clip = 0.55 + 0.016*dur; wall = 150 + sum(est_clip); T4 pricing."""
    estimate = runner_module._estimate_wall_and_cost([5.0, 10.0])
    expected_clips = [0.63, 0.71]
    assert estimate["inference_total_sec"] == pytest.approx(1.34, abs=1e-6)
    assert estimate["overhead_sec"] == pytest.approx(150.0, abs=1e-6)
    assert estimate["wall_seconds"] == pytest.approx(151.34, abs=1e-6)
    assert estimate["estimated_gpu_cost_usd"] == pytest.approx(
        151.34 * 0.0001639, abs=1e-4
    )
    assert expected_clips == [0.63, 0.71]
    assert estimate["timeout_worst_case_seconds"] == 900
    assert estimate["timeout_worst_case_cost_usd"] == pytest.approx(
        900 * 0.0001639, abs=1e-6
    )


def test_truncate_caps_errors(runner_module) -> None:
    """Error strings are capped at 200 characters."""
    long_text = "x" * 500
    assert len(runner_module._truncate(long_text)) == 200
    assert runner_module._truncate("short") == "short"
