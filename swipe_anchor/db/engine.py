"""Engine / session helpers for the app store (plan §2.1).

Postgres in production, SQLite for the local MVP. Mirrors the pipeline's
``core.database.engine`` conventions (bare SQLite paths are wrapped, pragmas are
applied on connect) but keeps an independent engine + metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy import event as _sqla_event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from swipe_anchor.db.base import Base


def _wrap_sqlite_path(url: str) -> str:
    """Allow bare filesystem paths as a convenience, like ``core.database``."""
    if "://" in url:
        return url
    return f"sqlite:///{url}"


def _apply_sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.close()


def _ensure_sqlite_parent_dir(engine: Engine) -> None:
    """Create the parent directory for a file-backed SQLite DB if missing.

    The documented default url is ``sqlite:///data/swipe_anchor.db`` and ``data/``
    is gitignored/absent on a fresh checkout — without this, opening the engine
    raises ``OperationalError: unable to open database file``.
    """
    if engine.dialect.name != "sqlite":
        return
    database = engine.url.database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def apply_additive_migrations(engine: Engine) -> None:
    """Idempotently add columns introduced after a table was first created.

    The app store uses ``create_all`` (not Alembic), and ``create_all`` never
    ALTERs an existing table — so a column added to a model later (e.g.
    ``responses.shown_clips``) must be reconciled here, or reads/writes that
    reference it fail with "no such column" on a pre-existing DB. Portable across
    SQLite and Postgres via the dialect-compiled column type.
    """
    from sqlalchemy import inspect as _inspect

    from swipe_anchor.db.models import Response

    insp = _inspect(engine)
    if "responses" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("responses")}
    col = Response.__table__.c.shown_clips
    if col.name in existing:
        return
    coltype = col.type.compile(dialect=engine.dialect)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"ALTER TABLE responses ADD COLUMN {col.name} {coltype}"
        )


def create_app_engine(url: str, *, migrate: bool = True) -> Engine:
    """Create the app-store engine.

    With ``migrate=True`` (the default) it also ensures every table exists and
    reconciles additive columns, so the running backend self-heals an older
    schema. Pass ``migrate=False`` for read-only inspection (e.g. a ``--dry-run``)
    where no schema mutation is wanted.
    """
    url = _wrap_sqlite_path(url)
    _is_memory = url == "sqlite:///:memory:"
    engine = create_engine(
        url,
        **({"connect_args": {"check_same_thread": False}, "poolclass": StaticPool} if _is_memory else {}),
    )
    if engine.dialect.name == "sqlite":
        _sqla_event.listen(engine, "connect", _apply_sqlite_pragmas)
        _ensure_sqlite_parent_dir(engine)
    # Import models for their side effect of registering on Base.metadata.
    from swipe_anchor.db import models  # noqa: F401

    if migrate:
        Base.metadata.create_all(engine)
        apply_additive_migrations(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
