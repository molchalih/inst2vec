"""Tiny shared fingerprint layer for stage-level idempotence.

Each central pipeline stage computes a Fingerprint(data, config,
dependency) on entry and asks ``is_stale`` whether its stored
counterpart still matches. On mismatch the stage wipes its outputs for
the scope, recomputes, and calls ``mark_complete``. The stage commits
its own transaction; ``mark_complete`` only merges.

Three canonical patterns
------------------------

1. **Config-only + ``fp.gate``.** ``data=""``, ``dependency=""``,
   ``config=hash(payload)``. The stage uses row-level predicates to
   find unfinished work; config drift wipes outputs via the
   ``on_drift`` callback. Canonical: ``modules/music/classify.py``,
   ``modules/captions/__init__.py``, ``modules/speech/__init__.py``.

2. **Full triple + ``fp.is_stale``.** All three fields hashed. A
   staleness check either resets and reruns the entire scope or
   skips it. Canonical: ``modules/filter/__init__.py``,
   ``modules/ingest/audio.py``, ``modules/clustering/*``.

3. **Per-item source hash overlay.** Pattern (2) plus a per-row
   source hash persisted alongside the output, gating individual
   re-computation when a row's inputs change without invalidating
   the whole scope. Canonical: ``modules/embeddings/runner.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from core.database import StageState
from core.log import event


@dataclass(frozen=True)
class Fingerprint:
    data: str
    config: str
    dependency: str


# ── hashing utilities (stages decide what to feed in) ────────────────────────


def hash_rows(rows: Iterable[tuple[Any, ...]]) -> str:
    """Stable SHA-256 over an iterable of tuples.

    Caller is responsible for passing rows sorted on a stable key.
    Record separator 0x1E prevents (1,2),(3) from colliding with
    (1),(2,3).
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compose_hashes(*hashes: str) -> str:
    """Stable, order-preserving composition of N hash strings."""
    return hash_text("".join(hashes))


def stable_subset_payload(obj: object, fields: tuple[str, ...] | list[str]) -> str:
    """Stable JSON string over a fixed allowlist of fields.

    Sorts the field list, reads each value via attribute or item access,
    serializes with ``sort_keys=True`` and ``default=str`` so unknown
    types (Path, datetime, enum) hash deterministically.
    """
    if isinstance(obj, Mapping):
        m: Mapping[Any, object] = obj
        payload = {f: m[f] for f in sorted(fields)}
    else:
        payload = {f: getattr(obj, f) for f in sorted(fields)}
    return json.dumps(payload, sort_keys=True, default=str)


def file_stat_for_hash(path: str | os.PathLike[str]) -> tuple[int, int]:
    """(size_bytes, mtime_ns) tuple, or (-1, -1) when the file is missing.

    Stable per-file input for ``hash_rows``: a missing file produces a
    sentinel hash so the dependent stage re-runs when the file appears.
    """
    if not os.path.exists(path):
        return (-1, -1)
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def row_diff(desired: dict[int, str], stored: dict[int, str | None]) -> set[int]:
    """Ids in ``desired`` whose ``stored`` hash is missing or different.

    Treats a NULL stored hash as stale. Ids present only in ``stored``
    (orphans) are not returned — the caller is expected to handle them
    separately if removal is required.
    """
    return {key for key, want in desired.items() if stored.get(key) != want}


# ── compare + store ──────────────────────────────────────────────────────────


def is_stale(session: Session, stage: str, scope: str, current: Fingerprint) -> bool:
    row = session.get(StageState, (stage, scope))
    if row is None:
        return True
    return (
        row.data_hash != current.data
        or row.config_hash != current.config
        or row.dependency_hash != current.dependency
    )


def mark_complete(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> None:
    """Merge the stage_state row. Caller commits."""
    session.merge(
        StageState(
            stage_name=stage,
            scope_key=scope,
            data_hash=current.data,
            config_hash=current.config,
            dependency_hash=current.dependency,
        )
    )


def gate(
    session: Session,
    stage: str,
    scope: str,
    current: Fingerprint,
    on_drift: Callable[[Session], None],
    *,
    log_scope: str = "",
    drift_msg: str = "",
    check_dependency: bool = False,
    check_data: bool = False,
) -> None:
    """Reset stage outputs on config (or optionally dependency / data) drift;
    emit the standard 3-state log line.

    Replaces the per-stage ``stored = session.get(...)`` / compare /
    log-and-reset boilerplate. Caller commits.

    ``check_dependency=True`` extends the drift check to dependency_hash for
    stages whose outputs depend on an upstream stage state. Default False
    preserves the original config-only semantics for existing callers.

    ``check_data=True`` extends the drift check to data_hash for stages that
    fingerprint their input rows directly (e.g. labels hashing the per-clip
    video file stat) — without this opt-in, replacing the source data would
    leave stale outputs and then ``mark_complete`` would seal the new
    data_hash, hiding the drift forever.

    ``log_scope`` and ``drift_msg`` are accepted for backward compatibility
    but unused — log lines pick up the active scope from the caller's ContextVar.
    """
    del log_scope, drift_msg  # unused; scope comes from caller's ContextVar
    stored = session.get(StageState, (stage, scope))
    if stored is None:
        event("SCAN", "fingerprint")
        return
    config_changed = stored.config_hash != current.config
    dep_changed = check_dependency and stored.dependency_hash != current.dependency
    data_changed = check_data and stored.data_hash != current.data
    if config_changed or dep_changed or data_changed:
        diff = describe_diff(session, stage, scope, current)
        event("SCAN", "fingerprint", stats={"diff": diff})
        on_drift(session)
    else:
        event("SKIP", "fingerprint")


def describe_diff(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> str:
    """Human-readable note for log lines.

    Returns 'no prior state' on first run, '' when no fields changed, or
    a '+'-joined list like 'data+dependency'.
    """
    row = session.get(StageState, (stage, scope))
    if row is None:
        return "no prior state"
    parts = []
    if row.data_hash != current.data:
        parts.append("data")
    if row.config_hash != current.config:
        parts.append("config")
    if row.dependency_hash != current.dependency:
        parts.append("dependency")
    return "+".join(parts)


def stage_dependency_hash(session: Session, stage: str, scope: str) -> str:
    """Stable digest of an upstream stage's StageState row.

    Returns ``hash_text("")`` when no row exists, otherwise a SHA-256 of
    the row's three hash fields concatenated. Stages use this for their
    own ``Fingerprint.dependency`` field so dependency-hashing stays
    consistent across stages and never drifts via local string concat.
    """
    row = session.get(StageState, (stage, scope))
    if row is None:
        return hash_text("")
    return hash_text(row.data_hash + row.config_hash + row.dependency_hash)
