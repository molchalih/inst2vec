"""CI guard: every `log(` callsite in `modules/` and `core/` uses the
structured 4-arg form (scope, verb, target, result). The legacy 2-arg
form is removed in Task 16; this test makes future regressions visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = [REPO / "modules", REPO / "core"]
ALLOWED_TWO_ARG_LOCATIONS: set[tuple[str, int]] = {
    # path relative to repo, line number — populate only if a justified exception arises.
}


def _iter_log_calls(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "log") or (
                isinstance(f, ast.Attribute) and f.attr == "log"
            ):
                yield node


@pytest.mark.parametrize(
    "py_file",
    sorted(
        p for d in SEARCH_DIRS for p in d.rglob("*.py") if "/__pycache__/" not in str(p)
    ),
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_log_calls_use_structured_form(py_file: Path) -> None:
    violations: list[tuple[int, int]] = []
    for call in _iter_log_calls(py_file):
        rel = py_file.relative_to(REPO).as_posix()
        if (rel, call.lineno) in ALLOWED_TWO_ARG_LOCATIONS:
            continue
        # Need at least 3 positional args: scope, verb, target.
        if len(call.args) < 3:
            violations.append((call.lineno, len(call.args)))
    assert not violations, (
        f"{py_file}: legacy 2-arg log() calls at lines {violations}; "
        "use log(scope, verb, target, result, *, stats=...) instead."
    )
