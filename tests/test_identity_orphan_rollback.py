"""B7: allocate_clip_identity / allocate_user_identity must roll back the
identity row when the caller raises inside the with-block.
"""

from __future__ import annotations

import pytest

from core.database import Base, Clip, User, get_engine, get_session
from core.database.engine import get_identity_engine, get_identity_session
from core.database.identity import (
    ClipIdentity,
    IdentityBase,
    UserIdentity,
    allocate_clip_identity,
    allocate_user_identity,
)


def _ensure_schemas() -> None:
    """Idempotent — create_all is a no-op if tables already exist."""
    Base.metadata.create_all(get_engine())
    IdentityBase.metadata.create_all(get_identity_engine())


def _clear_tables() -> None:
    """Remove all rows from identity and main tables between tests."""
    with get_identity_session() as s:
        s.query(ClipIdentity).delete()
        s.query(UserIdentity).delete()
        s.commit()
    main = get_session()
    try:
        main.query(Clip).delete()
        main.query(User).delete()
        main.commit()
    finally:
        main.close()


def test_allocate_clip_identity_rolls_back_on_exception() -> None:
    _ensure_schemas()
    _clear_tables()

    api_pk = 12345
    with (
        pytest.raises(RuntimeError, match="boom"),
        allocate_clip_identity(api_pk) as cid,
    ):
        assert isinstance(cid, int) and cid > 0
        raise RuntimeError("boom")

    with get_identity_session() as s:
        row = s.query(ClipIdentity).filter_by(api_pk=api_pk).first()
        assert row is None, "orphan ClipIdentity row left behind"


def test_allocate_clip_identity_commits_on_clean_exit() -> None:
    _ensure_schemas()
    _clear_tables()

    api_pk = 67890
    with allocate_clip_identity(api_pk) as cid:
        first_id = cid

    with get_identity_session() as s:
        row = s.query(ClipIdentity).filter_by(api_pk=api_pk).first()
        assert row is not None
        assert row.id == first_id


def test_allocate_clip_identity_idempotent_on_existing() -> None:
    _ensure_schemas()
    _clear_tables()

    api_pk = 222
    with allocate_clip_identity(api_pk) as cid_a:
        pass
    with allocate_clip_identity(api_pk) as cid_b:
        pass
    assert cid_a == cid_b


def test_allocate_user_identity_rolls_back_on_exception() -> None:
    _ensure_schemas()
    _clear_tables()

    username = "rollback_user"
    with (
        pytest.raises(RuntimeError, match="nope"),
        allocate_user_identity(username) as uid,
    ):
        assert isinstance(uid, int) and uid > 0
        raise RuntimeError("nope")

    with get_identity_session() as s:
        row = s.query(UserIdentity).filter_by(username=username).first()
        assert row is None
