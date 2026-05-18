"""Identity DB: PII + Instagram API PKs.

Owns the identity ORM models and CRUD helpers. Engine and session lifecycle
live in core/database/engine.py — CRUD helpers below import
get_identity_session lazily inside each function body to avoid an
engine→identity→engine top-level cycle.

Identity-first invariant
------------------------
Callers MUST create the identity row first, commit it, then create the
matching main-DB row. If the main-DB insert fails, the identity row
becomes an orphan; ``sweep_orphans()`` reclaims those rows.

The sweep is single-threaded and idempotent. Run it between pipeline
invocations, not concurrently with one — pipeline is single-process and
serial by design.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class IdentityBase(DeclarativeBase):
    pass


class UserIdentity(IdentityBase):
    __tablename__ = "user_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    api_pk: Mapped[int | None] = mapped_column(BigInteger, nullable=True, unique=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    full_name: Mapped[str | None] = mapped_column(String)
    city_name: Mapped[str | None] = mapped_column(String)
    profile_pic_url: Mapped[str | None] = mapped_column(String)
    profile_pic_url_hd: Mapped[str | None] = mapped_column(String)


class ClipIdentity(IdentityBase):
    __tablename__ = "clip_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    api_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)


def get_or_create_user_identity(username: str) -> int:
    """Return sequential user id for username, creating a new UserIdentity if needed."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ui = s.query(UserIdentity).filter_by(username=username).first()
        if ui is None:
            ui = UserIdentity(username=username)
            s.add(ui)
            s.flush()
            uid = ui.id
            s.commit()
        else:
            uid = ui.id
    return uid


def update_user_identity(
    user_id: int,
    *,
    api_pk: int,
    full_name: str | None,
    city_name: str | None,
    profile_pic_url: str | None,
    profile_pic_url_hd: str | None,
) -> None:
    """Store API PK and PII for an existing UserIdentity row."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            raise LookupError(f"UserIdentity id={user_id} not found")
        ui.api_pk = api_pk
        ui.full_name = full_name
        ui.city_name = city_name
        ui.profile_pic_url = profile_pic_url
        ui.profile_pic_url_hd = profile_pic_url_hd
        s.commit()


def get_username(user_id: int) -> str:
    """Return the Instagram username for a user by sequential id."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            raise LookupError(f"UserIdentity id={user_id} not found")
        return ui.username


def get_api_pk(user_id: int) -> int | None:
    """Return the Instagram API PK for a user, or None if not yet fetched."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            raise LookupError(f"UserIdentity id={user_id} not found")
        return ui.api_pk


def get_profile_pic_url(user_id: int) -> str | None:
    """Return profile_pic_url from identity DB, or None."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            return None
        return ui.profile_pic_url


def get_or_create_clip_identity(api_pk: int) -> int:
    """Return sequential clip id for Instagram clip PK, creating a new entry if needed."""
    from core.database.engine import get_identity_session

    with get_identity_session() as s:
        ci = s.query(ClipIdentity).filter_by(api_pk=api_pk).first()
        if ci is None:
            ci = ClipIdentity(api_pk=api_pk)
            s.add(ci)
            s.flush()
            cid = ci.id
            s.commit()
        else:
            cid = ci.id
    return cid


def sweep_orphans() -> dict[str, int]:
    """Delete identity rows with no matching main-DB row.

    Returns ``{"users_swept": int, "clips_swept": int}``.
    """
    from core.database.engine import get_identity_session, get_session
    from core.database.models import Clip, User

    main = get_session()
    try:
        live_user_ids = {row.id for row in main.query(User.id).all()}
        live_clip_ids = {row.id for row in main.query(Clip.id).all()}
    finally:
        main.close()

    users_swept = 0
    clips_swept = 0
    with get_identity_session() as s:
        for ui in s.query(UserIdentity).all():
            if ui.id not in live_user_ids:
                s.delete(ui)
                users_swept += 1
        for ci in s.query(ClipIdentity).all():
            if ci.id not in live_clip_ids:
                s.delete(ci)
                clips_swept += 1
        s.commit()

    return {"users_swept": users_swept, "clips_swept": clips_swept}
