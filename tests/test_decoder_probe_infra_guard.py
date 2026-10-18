"""AST source guards for the Phase 7.1 Modal decoder probe module.

These tests parse ``infra/modal_hinglish_decoder_probe.py`` (source only,
``ast.parse`` — never imported, so no Modal, torch, or transformers
dependency is needed) and lock the J1 execution constraints:

* single-container, single-function execution with a <= 420 s function
  timeout and exactly one container;
* the GPU is the T4 (a DEFAULT_GPU/"T4" constant exists) and the literal
  ``L4`` never appears as a functional string: docstrings may discuss the
  constraint ("never L4"), but no code literal may -- in particular the
  ``gpu=`` keyword and the DEFAULT_GPU constant must carry T4 and nothing
  that contains or equals "L4";
* the pinned model revision string appears verbatim;
* ``--execute`` is an argparse flag with ``action="store_true"``, which
  keeps the offline dry run as the default behavior;
* the Modal app name differs from the Phase 6 baseline app
  (``mendspeech-hinglish-asr``) so the baseline job is never redefined.

If the infra module has not been authored yet, every test skips with that
explanation instead of failing.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_PATH = REPO_ROOT / "infra" / "modal_hinglish_decoder_probe.py"

PINNED_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
BASELINE_APP_NAME = "mendspeech-hinglish-asr"
FORBIDDEN_GPU_MARKER = "L4"
REQUIRED_GPU_MARKER = "T4"
APP_NAME_SCOPE = "decoder_probe_infra_guard_module"


def _source() -> str:
    """Return the infra probe module source text.

    Returns:
        File source.

    Raises:
        FileNotFoundError: Propagated when the module is absent; callers
            handle the skip.
    """
    return INFRA_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    """The parsed module tree (skip cleanly while the module is unauthored)."""
    if not INFRA_PATH.is_file():
        pytest.skip(
            "infra/modal_hinglish_decoder_probe.py has not been authored yet "
            "(being written in parallel by the main agent)"
        )
    source = _source()
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(
            f"infra/modal_hinglish_decoder_probe.py does not parse: {exc}"
        )


def _docstring_node_ids(tree: ast.Module) -> Set[int]:
    """Return ids of string constants that only serve as docstrings.

    Args:
        tree: Parsed module.

    Returns:
        Set of ``ast.Constant`` node ids for module/class/function
        docstrings.
    """
    ids: Set[int] = set()
    nodes: List[ast.AST] = [tree, *tree.body]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(node)
            nodes.extend(node.body)
    for node in nodes:
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            ids.add(id(node.body[0].value))
    return ids


def _string_constants(tree: ast.Module) -> Iterable[ast.Constant]:
    """Yield every string constant node in the module.

    Args:
        tree: Parsed module.

    Yields:
        ``ast.Constant`` nodes whose value is a string.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def _module_constant(
    tree: ast.Module, name: str, value_type: type
) -> Optional[Any]:
    """Return a module-level constant assignment when it matches the type.

    Args:
        tree: Parsed module.
        name: Exact assignment target name.
        value_type: Required value type (str or int).

    Returns:
        The assigned constant value, or None when the assignment does not
        exist at module level.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, value_type):
                    return value
    return None


def test_function_timeout_is_bounded(tree: ast.Module) -> None:
    """FUNCTION_TIMEOUT_SEC is a literal and never exceeds 420 seconds."""
    value = _module_constant(tree, "FUNCTION_TIMEOUT_SEC", int)
    assert value is not None, (
        "infra module must define a module-level literal FUNCTION_TIMEOUT_SEC"
    )
    assert value <= 420, f"FUNCTION_TIMEOUT_SEC={value} exceeds the 420 s budget"


def test_single_container_bound(tree: ast.Module) -> None:
    """MAX_CONTAINERS is exactly 1 (one container for the whole probe)."""
    value = _module_constant(tree, "MAX_CONTAINERS", int)
    assert value == 1, f"MAX_CONTAINERS must be exactly 1, found {value}"


def test_default_gpu_is_t4_constant(tree: ast.Module) -> None:
    """A GPU constant (e.g. DEFAULT_GPU) carries exactly the T4 marker."""
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if not isinstance(value, str) or REQUIRED_GPU_MARKER not in value:
                continue
            names = [
                target.id.upper() for target in node.targets
                if isinstance(target, ast.Name)
            ]
            if any("GPU" in name for name in names):
                matches.append((names, value))
    assert matches, (
        "expected a module-level GPU constant such as DEFAULT_GPU = 'T4'"
    )
    for _names, value in matches:
        assert FORBIDDEN_GPU_MARKER not in value


def test_gpu_literal_never_selected(tree: ast.Module) -> None:
    """No code string literal selects, names, or contains the forbidden GPU.

    Docstrings are excluded: the authored module may state the constraint
    in prose ("never L4"). Any non-docstring string containing "L4", any
    string equal to "L4", or any ``gpu=`` keyword value containing "L4"
    fails the guard.
    """
    docstring_ids = _docstring_node_ids(tree)
    offenders: List[str] = []
    for node in _string_constants(tree):
        if id(node) in docstring_ids:
            continue
        if node.value == FORBIDDEN_GPU_MARKER or FORBIDDEN_GPU_MARKER in node.value:
            offenders.append(repr(node.value))
    assert not offenders, f"forbidden GPU literal found in code strings: {offenders}"

    # gpu=<value> keyword values must name the T4 and never the forbidden GPU.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "gpu" and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str):
                        assert REQUIRED_GPU_MARKER in value, f"gpu={value!r} is not a T4"
                        assert FORBIDDEN_GPU_MARKER not in value


def test_pinned_revision_present(tree: ast.Module) -> None:
    """The exact pinned whisper-small revision string appears in the source."""
    literals = [node.value for node in _string_constants(tree)]
    assert any(PINNED_REVISION in literal for literal in literals), (
        f"pinned revision {PINNED_REVISION} not found in the infra module"
    )


def test_execute_flag_is_opt_in_store_true(tree: ast.Module) -> None:
    """--execute exists with action='store_true' (dry run stays default)."""
    executables: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_add_argument = (
            (isinstance(function, ast.Attribute) and function.attr == "add_argument")
            or (isinstance(function, ast.Name) and function.id == "add_argument")
        )
        if not is_add_argument or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        flag_name = first.value
        if "--execute" not in flag_name.split(","):
            continue
        action = next(
            (
                kw.value
                for kw in node.keywords
                if kw.arg == "action" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        executables.append(str(getattr(action, "value", None)))
    assert executables == ["store_true"], (
        "--execute must use action='store_true' (and exactly once) so the "
        f"offline dry run remains the default; found actions {executables or None}"
    )


def test_app_name_is_probe_specific(tree: ast.Module) -> None:
    """The Modal app name differs from the Phase 6 baseline app name."""
    app_name = _module_constant(tree, "APP_NAME", str)
    assert app_name is not None, "expected a module-level APP_NAME constant"
    assert app_name != BASELINE_APP_NAME, (
        "the probe must not redefine the baseline Modal app "
        f"({BASELINE_APP_NAME}); choose a separate name"
    )
