"""Shared status/retry helpers for the clip and cluster passes.

Both passes write a typed ORM row keyed by their primary key with the
same status / validation / payload / warnings / attempts contract. Pure
ORM-typed helpers; caller owns the transaction (commits).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session


def upsert_success(
    session: Session,
    model_cls: type,
    *,
    key: Any,
    validation: str,
    payload: dict,
    warnings: list[str],
    extras: dict[str, Any] | None = None,
    generation_seed: int | None = None,
) -> None:
    extras = extras or {}
    existing = session.get(model_cls, key)
    if existing is None:
        pk_cols = [c.name for c in model_cls.__table__.primary_key.columns]  # ty: ignore[unresolved-attribute]
        pk_values = key if isinstance(key, tuple) else (key,)
        kwargs = dict(zip(pk_cols, pk_values, strict=True))
        kwargs.update(
            status="success",
            validation=validation,
            payload=payload,
            warnings=warnings,
            error=None,
            attempts=1,
            generation_seed=generation_seed,
            **extras,
        )
        session.add(model_cls(**kwargs))
        return
    existing.status = "success"
    existing.validation = validation
    existing.payload = payload
    existing.warnings = warnings
    existing.error = None
    existing.attempts = (existing.attempts or 0) + 1
    if generation_seed is not None:
        existing.generation_seed = generation_seed
    for k, v in extras.items():
        setattr(existing, k, v)


def bump_failure(
    session: Session,
    model_cls: type,
    *,
    key: Any,
    error: str,
    max_attempts: int,
    generation_seed: int | None = None,
) -> None:
    existing = session.get(model_cls, key)
    if existing is None:
        pk_cols = [c.name for c in model_cls.__table__.primary_key.columns]  # ty: ignore[unresolved-attribute]
        pk_values = key if isinstance(key, tuple) else (key,)
        kwargs = dict(zip(pk_cols, pk_values, strict=True))
        existing = model_cls(**kwargs, status="pending", attempts=0)
        session.add(existing)
        session.flush()
    existing.attempts = (existing.attempts or 0) + 1
    existing.error = error
    existing.payload = None
    existing.warnings = None
    existing.validation = None
    if generation_seed is not None:
        existing.generation_seed = generation_seed
    if existing.attempts >= max_attempts:
        existing.status = "failed"
    else:
        existing.status = "pending"


def upsert_terminal_failure(
    session: Session,
    model_cls: type,
    *,
    key: Any,
    error: str,
    attempts: int,
) -> None:
    """Write a permanently-failed row, bypassing the retry budget.

    Used for cases like "cluster has zero member clips" where retries
    cannot help and the row must read as terminally failed.
    """
    existing = session.get(model_cls, key)
    if existing is None:
        pk_cols = [c.name for c in model_cls.__table__.primary_key.columns]  # ty: ignore[unresolved-attribute]
        pk_values = key if isinstance(key, tuple) else (key,)
        kwargs = dict(zip(pk_cols, pk_values, strict=True))
        existing = model_cls(**kwargs, status="failed", error=error, attempts=attempts)
        session.add(existing)
        return
    existing.status = "failed"
    existing.error = error
    existing.attempts = attempts
    existing.payload = None
    existing.warnings = None
    existing.validation = None
