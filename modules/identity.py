"""Identity DB: stores PII and Instagram API PKs, separate from main DB."""

from __future__ import annotations

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import BigInteger, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()

_IDENTITY_DB_URL = os.environ.get("IDENTITY_DB_URL", "sqlite:///data/identity_map.db")


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


def init_identity_db(db_path: str | None = None):
    """Create identity DB tables and return engine. Uses _IDENTITY_DB_URL if db_path omitted."""
    url = f"sqlite:///{db_path}" if db_path else _IDENTITY_DB_URL
    eng = create_engine(url)
    IdentityBase.metadata.create_all(eng)
    return eng


# Module-level engine (replaced in tests via monkeypatch on `_engine`)
_engine = init_identity_db()


@contextmanager
def get_identity_session():
    with Session(_engine) as s:
        yield s


# ── Public CRUD helpers ────────────────────────────────────────────────────


def get_or_create_user_identity(username: str) -> int:
    """Return sequential user id for username, creating a new UserIdentity if needed."""
    with Session(_engine) as s:
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
    with Session(_engine) as s:
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
    with Session(_engine) as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            raise LookupError(f"UserIdentity id={user_id} not found")
        return ui.username


def get_api_pk(user_id: int) -> int | None:
    """Return the Instagram API PK for a user, or None if not yet fetched."""
    with Session(_engine) as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            raise LookupError(f"UserIdentity id={user_id} not found")
        return ui.api_pk


def get_profile_pic_url(user_id: int) -> str | None:
    """Return profile_pic_url from identity DB, or None."""
    with Session(_engine) as s:
        ui = s.get(UserIdentity, user_id)
        if ui is None:
            return None
        return ui.profile_pic_url


def get_or_create_clip_identity(api_pk: int) -> int:
    """Return sequential clip id for Instagram clip PK, creating a new entry if needed."""
    with Session(_engine) as s:
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
