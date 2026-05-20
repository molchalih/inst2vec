"""Engine and session lifecycle for both the main and identity databases.

Sole owner of engine globals — no other file in the package or codebase
holds engine handles directly.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.console import log
from core.database.models import Base

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


def init_db(database_url: str, identity_db_url: str) -> None:
    """Initialize both engines and create all tables for both databases.

    identity_db_url can be a full SQLAlchemy URL (sqlite://…) or a bare file
    path (auto-wrapped with sqlite:///).
    """
    global _main_engine, _identity_engine

    t0 = time.perf_counter()
    _main_engine = create_engine(database_url)
    Base.metadata.create_all(_main_engine)
    log(
        "db",
        "INIT",
        "main",
        "ok",
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

    from core.database.identity import IdentityBase

    IdentityBase.metadata.create_all(_identity_engine)
    log(
        "db",
        "INIT",
        "identity",
        "ok",
        stats={
            "tables": len(IdentityBase.metadata.tables),
            "time": time.perf_counter() - t1,
        },
    )

    from core.database import identity as _identity

    swept = _identity.sweep_orphans()
    if swept["users_swept"] or swept["clips_swept"]:
        log("db", "WRITE", "identity-orphans", "ok", stats=swept)
