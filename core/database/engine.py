"""Engine and session lifecycle for the main, identity and serving databases.

Sole owner of engine globals — no other file in the package or codebase
holds engine handles directly.

Alembic (``core/database/migrations/``) is the production (Postgres) schema
path. The
``create_all`` + ``_LATE_ADDED_COLUMNS`` fast-path here remains for SQLite and
tests so fresh dev still bootstraps instantly (see D5 in the migration spec);
a fresh Postgres DB is brought up with ``alembic -n <db> upgrade head``.
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
    ("clip_labels", "generation_seed", "INTEGER"),
    ("cluster_labels", "generation_seed", "INTEGER"),
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


def _wrap_sqlite_path(url: str) -> str:
    """Accept a full SQLAlchemy URL or a bare file path (→ ``sqlite:///``)."""
    if "://" in url:
        return url
    return f"sqlite:///{url}"


_main_engine: Engine | None = None
_identity_engine: Engine | None = None
_serving_engine: Engine | None = None


def get_engine() -> Engine:
    assert _main_engine is not None, "Call init_db() before using the database"
    return _main_engine


def get_session() -> Session:
    return Session(get_engine())


def get_serving_engine() -> Engine:
    assert _serving_engine is not None, (
        "Call init_serving_db() before using the serving database"
    )
    return _serving_engine


@contextmanager
def get_serving_session() -> Iterator[Session]:
    with Session(get_serving_engine()) as s:
        yield s


@scope("db")
def init_serving_db(serving_database_url: str) -> None:
    """Initialize the serving engine and create all serving tables.

    Separate from ``init_db`` so ordinary pipeline runs that never touch the
    serving store stay cheap; the offload script and atlas API call this
    explicitly. ``serving_database_url`` may be a full SQLAlchemy URL or a bare
    file path (auto-wrapped with ``sqlite:///``). Alembic owns the prod
    (Postgres) schema; this ``create_all`` fast-path covers SQLite/dev/test.
    """
    global _serving_engine

    from core.database.serving_models import ServingBase

    t0 = time.perf_counter()
    _serving_engine = create_engine(_wrap_sqlite_path(serving_database_url))
    _sqla_event.listen(_serving_engine, "connect", _apply_sqlite_pragmas)
    ServingBase.metadata.create_all(_serving_engine)
    event(
        "INIT",
        "serving",
        stats={
            "tables": len(ServingBase.metadata.tables),
            "time": time.perf_counter() - t0,
        },
    )


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

    wrapped_url = _wrap_sqlite_path(identity_db_url)

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
