"""Pattern layer above core.console for uniform stage/scope/item/event logging.

Provides decorators (@stage, @scope), a per-item context manager (item), and
two one-shot helpers (event, warn). All five read the active scope from a
ContextVar set by the decorators. Calling item/event/warn outside any scope
raises RuntimeError — strict by design so drift surfaces in tests, not in logs.
"""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, cast, get_args

from core.console import log as _render
from core.log_types import Result, Verb

__all__ = [
    "ItemContext",
    "Result",
    "StageResult",
    "Verb",
    "event",
    "item",
    "scope",
    "stage",
    "warn",
]

_VERBS: frozenset[str] = frozenset(get_args(Verb))
_RESULTS: frozenset[str] = frozenset(get_args(Result))

# First segment: lowercase letters only. Subscopes (colon-separated) may
# additionally contain underscores; e.g. "embed:video", "ingest:audio_mir".
_SCOPE_RE = re.compile(r"^[a-z]+(?::[a-z_]+)*$")

_scope_var: ContextVar[str | None] = ContextVar("inst2vec_log_scope", default=None)


@dataclass
class StageResult:
    """Returned by `@stage`-decorated functions; fields become SEAL stats.

    Usage:
        return StageResult(done=42, failed=3, sources=2)
        # → SEAL <stage> ok [time=..., done=42, failed=3, sources=2]
    """

    stats: Mapping[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.stats = dict(kwargs)


class ItemContext:
    """Per-item context manager state, returned by `with item(...) as t:`."""

    __slots__ = ("_started", "_stats", "elapsed_s", "exc", "failed")

    def __init__(self) -> None:
        self._stats: dict[str, Any] = {}
        self.failed: bool = False
        self.exc: BaseException | None = None
        self.elapsed_s: float = 0.0
        self._started: float = 0.0

    def stats(self, **kwargs: Any) -> None:
        """Accumulate stat fields; multiple calls merge (later wins on conflict)."""
        self._stats.update(kwargs)


def _require_scope() -> str:
    s = _scope_var.get()
    if s is None:
        raise RuntimeError("no active scope; wrap the call in @stage or @scope")
    return s


def _check_verb(verb: str) -> None:
    if verb not in _VERBS:
        raise ValueError(f"unknown verb {verb!r}; allowed: {sorted(_VERBS)}")


def _check_result(result: str) -> None:
    if result not in _RESULTS:
        raise ValueError(f"unknown result {result!r}; allowed: {sorted(_RESULTS)}")


def _check_scope(s: str) -> None:
    if not _SCOPE_RE.fullmatch(s):
        raise ValueError(f"invalid scope {s!r}; must match {_SCOPE_RE.pattern}")


@contextmanager
def item(verb: Verb, target: str) -> Iterator[ItemContext]:
    _check_verb(verb)
    scope = _require_scope()
    ctx = ItemContext()
    ctx._started = time.perf_counter()
    try:
        yield ctx
    except Exception as exc:
        ctx.failed = True
        ctx.exc = exc
        ctx.elapsed_s = time.perf_counter() - ctx._started
        merged = dict(ctx._stats)
        merged["err"] = repr(exc)
        merged["time"] = ctx.elapsed_s
        _render(scope, verb, target, "ERR", stats=merged)
        return  # __exit__ True → exception suppressed
    ctx.elapsed_s = time.perf_counter() - ctx._started
    merged = dict(ctx._stats)
    merged["time"] = ctx.elapsed_s
    _render(scope, verb, target, "ok", stats=merged)


def event(
    verb: Verb,
    target: str,
    *,
    result: Result = "ok",
    stats: Mapping[str, Any] | None = None,
) -> None:
    _check_verb(verb)
    _check_result(result)
    scope = _require_scope()
    _render(scope, verb, target, result, stats=stats)


def warn(
    verb: Verb,
    target: str,
    *,
    err: BaseException | str | None = None,
    stats: Mapping[str, Any] | None = None,
) -> None:
    _check_verb(verb)
    scope = _require_scope()
    merged: dict[str, Any] = dict(stats or {})
    if err is not None:
        merged["err"] = repr(err) if isinstance(err, BaseException) else err
    _render(scope, verb, target, "WARN", stats=merged)


def scope(template: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Set scope contextvar to a formatted template for the function call.

    `template` may include `{name}` placeholders bound against the wrapped
    function's arguments. A literal template (no '{') is validated and used
    as-is. Unknown placeholder → KeyError at call time.
    """
    has_placeholders = "{" in template
    if not has_placeholders:
        _check_scope(template)

    def deco[F: Callable[..., Any]](fn: F) -> F:
        sig = inspect.signature(fn) if has_placeholders else None

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if sig is not None:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                try:
                    formatted = template.format(**bound.arguments)
                except KeyError as e:
                    missing = e.args[0]
                    raise KeyError(
                        f"scope template {template!r} for {fn.__name__!r} "
                        f"references unknown placeholder {missing!r}; "
                        f"available: {sorted(bound.arguments)}"
                    ) from None
                _check_scope(formatted)
            else:
                formatted = template
            token = _scope_var.set(formatted)
            try:
                return fn(*args, **kwargs)
            finally:
                _scope_var.reset(token)

        return cast(F, wrapper)

    return deco


def stage(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Time a pipeline-stage function, emit SEAL on exit, advance pipeline bar.

    - Sets scope contextvar to `name` for the call.
    - Advances the pipeline progress bar if a `pipeline()` Live is active
      (no-op otherwise — safe to use from scripts).
    - Clean exit: emits `SEAL <name> ok [time=..., **StageResult.stats]`.
    - Exception: emits `SEAL <name> ERR [err=..., time=...]` and re-raises.
    """
    _check_scope(name)

    def deco[F: Callable[..., Any]](fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Lazy import: core.console imports nothing from core.log so no
            # cycle today, but the lazy form survives future refactors.
            from core.console import _maybe_advance_phase

            token = _scope_var.set(name)
            try:
                _maybe_advance_phase()
                t_start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    _render(
                        name,
                        "SEAL",
                        name,
                        "ERR",
                        stats={
                            "err": repr(exc),
                            "time": time.perf_counter() - t_start,
                        },
                    )
                    raise
                elapsed = time.perf_counter() - t_start
                merged: dict[str, Any] = {"time": elapsed}
                if isinstance(result, StageResult):
                    merged.update(result.stats)
                _render(name, "SEAL", name, "ok", stats=merged)
                return result
            finally:
                _scope_var.reset(token)

        return cast(F, wrapper)

    return deco
