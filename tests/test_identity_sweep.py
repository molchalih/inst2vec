"""Tests for sweep_orphans() in core/database/identity.py."""

from __future__ import annotations

from core.database import Base, Clip, User, get_engine, get_session
from core.database.engine import get_identity_engine, get_identity_session
from core.database.identity import (
    ClipIdentity,
    IdentityBase,
    UserIdentity,
    sweep_orphans,
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


def test_sweep_orphans_clean_db_returns_zero_counts():
    _ensure_schemas()
    _clear_tables()
    result = sweep_orphans()
    assert result == {"users_swept": 0, "clips_swept": 0}


def test_sweep_orphans_deletes_orphan_user_identity():
    _ensure_schemas()
    _clear_tables()

    # Create a UserIdentity with no matching User in main DB.
    orphan_id = None
    with get_identity_session() as s:
        orphan = UserIdentity(username="orphan_user")
        s.add(orphan)
        s.flush()
        orphan_id = orphan.id
        s.commit()

    # Create a matched pair: UserIdentity + User with the same id.
    matched_id = None
    with get_identity_session() as s:
        matched = UserIdentity(username="matched_user")
        s.add(matched)
        s.flush()
        matched_id = matched.id
        s.commit()

    main = get_session()
    try:
        main.add(User(id=matched_id))
        main.commit()
    finally:
        main.close()

    result = sweep_orphans()

    assert result == {"users_swept": 1, "clips_swept": 0}

    # Orphan should be gone; matched should remain.
    with get_identity_session() as s:
        assert s.get(UserIdentity, orphan_id) is None
        assert s.get(UserIdentity, matched_id) is not None


def test_sweep_orphans_deletes_orphan_clip_identity():
    _ensure_schemas()
    _clear_tables()

    # Create a User so we can attach Clip FKs.
    main = get_session()
    try:
        user = User(id=1)
        main.add(user)
        main.commit()
    finally:
        main.close()

    # Orphan ClipIdentity: identity row with no matching Clip.
    orphan_clip_id = None
    with get_identity_session() as s:
        orphan = ClipIdentity(api_pk=9001)
        s.add(orphan)
        s.flush()
        orphan_clip_id = orphan.id
        s.commit()

    # Matched pair: ClipIdentity + Clip with the same id.
    matched_clip_id = None
    with get_identity_session() as s:
        matched = ClipIdentity(api_pk=9002)
        s.add(matched)
        s.flush()
        matched_clip_id = matched.id
        s.commit()

    main = get_session()
    try:
        main.add(Clip(id=matched_clip_id, user_id=1))
        main.commit()
    finally:
        main.close()

    result = sweep_orphans()

    assert result == {"users_swept": 0, "clips_swept": 1}

    with get_identity_session() as s:
        assert s.get(ClipIdentity, orphan_clip_id) is None
        assert s.get(ClipIdentity, matched_clip_id) is not None


def test_init_db_calls_sweep_orphans(monkeypatch, tmp_path) -> None:
    """init_db must auto-sweep orphans on every pipeline boot."""
    from core.database import identity, init_db

    calls: list[str] = []

    def fake_sweep() -> dict[str, int]:
        calls.append("called")
        return {"users_swept": 0, "clips_swept": 0}

    monkeypatch.setattr(identity, "sweep_orphans", fake_sweep)

    db_url = f"sqlite:///{tmp_path}/main.db"
    id_url = f"sqlite:///{tmp_path}/identity.db"
    init_db(db_url, id_url)

    assert calls == ["called"]
