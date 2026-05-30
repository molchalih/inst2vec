"""Engine and session lifecycle for both the main and identity databases.

Sole owner of engine globals — no other file in the package or codebase
holds engine handles directly.
"""

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy import event as _sqla_event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.database.models import Base
from core.log import event, scope

# Columns added after the table was first shipped. `create_all()` never
# alters existing tables, so for each (table, column, sql_type) entry we
# ADD COLUMN once on engines that already have the table but not the column.
_LATE_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("user_clusters", "centrality", "FLOAT"),
    ("visualization_users", "centrality", "FLOAT"),
)


def _ensure_late_added_columns(engine: Engine) -> None:
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    for table, column, sql_type in _LATE_ADDED_COLUMNS:
        if table not in table_names:
            continue
        existing = {c["name"] for c in insp.get_columns(table)}
        if column in existing:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


def _apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    """Enable WAL + a generous busy_timeout on SQLite connections.

    WAL lets the embeddings drain loop (single writer) commit while the
    producer reads; busy_timeout makes brief contention wait instead of
    raising ``database is locked``. No-op for non-SQLite backends.
    """
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


_main_engine: Engine | None = None
_identity_engine: Engine | None = None


def get_engine() -> Engine:
    assert _main_engine is not None, "Call init_db() before using the database"
    return _main_engine


def get_session() -> Session:
    return Session(get_engine())


def get_identity_engine() -> Engine:
    assert _identity_engine is not None, (
        "Call init_db() before using the identity database"
    )
    return _identity_engine


@contextmanager
def get_identity_session() -> Iterator[Session]:
    with Session(get_identity_engine()) as s:
        yield s


@scope("db")
def init_db(database_url: str, identity_db_url: str) -> None:
    """Initialize both engines and create all tables for both databases.

    identity_db_url can be a full SQLAlchemy URL (sqlite://…) or a bare file
    path (auto-wrapped with sqlite:///).
    """
    global _main_engine, _identity_engine

    t0 = time.perf_counter()
    _main_engine = create_engine(database_url)
    _sqla_event.listen(_main_engine, "connect", _apply_sqlite_pragmas)
    Base.metadata.create_all(_main_engine)
    _ensure_late_added_columns(_main_engine)
    event(
        "INIT",
        "main",
        stats={
            "tables": len(Base.metadata.tables),
            "time": time.perf_counter() - t0,
        },
    )

    if identity_db_url.startswith("sqlite://"):
        wrapped_url = identity_db_url
    else:
        wrapped_url = f"sqlite:///{identity_db_url}"

    t1 = time.perf_counter()
    _identity_engine = create_engine(wrapped_url)
    _sqla_event.listen(_identity_engine, "connect", _apply_sqlite_pragmas)

    from core.database.identity import IdentityBase

    IdentityBase.metadata.create_all(_identity_engine)
    event(
        "INIT",
        "identity",
        stats={
            "tables": len(IdentityBase.metadata.tables),
            "time": time.perf_counter() - t1,
        },
    )

    from core.database import identity as _identity

    swept = _identity.sweep_orphans()
    if swept["users_swept"] or swept["clips_swept"]:
        event("WRITE", "identity-orphans", stats=swept)
