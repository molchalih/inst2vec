"""Engine and session lifecycle for both the main and identity databases.

Sole owner of engine globals — no other file in the package or codebase
holds engine handles directly.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from modules.database.models import Base

_main_engine: Engine | None = None
_identity_engine: Engine | None = None


def get_engine() -> Engine:
    assert _main_engine is not None, "Call init_db() before using the database"
    return _main_engine


def get_session() -> Session:
    return Session(get_engine())


def get_identity_engine() -> Engine:
    assert _identity_engine is not None, "Call init_db() before using the identity database"
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

    _main_engine = create_engine(database_url)
    Base.metadata.create_all(_main_engine)

    if identity_db_url.startswith("sqlite://"):
        wrapped_url = identity_db_url
    else:
        wrapped_url = f"sqlite:///{identity_db_url}"
    _identity_engine = create_engine(wrapped_url)

    # Import lazily to avoid a top-level engine→identity→engine cycle once
    # the package's identity submodule arrives in Task 6. Today this still
    # resolves to the flat modules.identity module.
    from modules.identity import IdentityBase
    IdentityBase.metadata.create_all(_identity_engine)

    # Keep the legacy modules.identity._engine in sync until Task 6 deletes it.
    import modules.identity as _identity_mod
    _identity_mod._engine = _identity_engine
