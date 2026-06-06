"""Declarative base for the swipe-anchor app store.

This is a **separate** base from the pipeline's ``core.database`` — the app store
is its own database and never shares a metadata/registry with the pipeline DBs
(plan §2.1, §2.2). The only coupling to the pipeline is the read-only export job.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.orm import DeclarativeBase


class UTCDateTime(TypeDecorator):
    """DateTime that always returns tz-aware UTC datetimes on read.

    SQLite stores datetimes as text and drops timezone info; this decorator
    re-attaches UTC on process_result_value so callers always get aware
    datetimes regardless of backend.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[override]
        if value is None:
            return None
        if value.tzinfo is None:
            return value  # assume naive UTC, store as-is
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[override]
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass
