"""Tiny shared fingerprint layer for stage-level idempotence.

Each central pipeline stage computes a Fingerprint(data, config,
dependency) on entry and asks ``is_stale`` whether its stored
counterpart still matches. On mismatch the stage wipes its outputs for
the scope, recomputes, and calls ``mark_complete``. The stage commits
its own transaction; ``mark_complete`` only merges.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from modules.database import StageState


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
