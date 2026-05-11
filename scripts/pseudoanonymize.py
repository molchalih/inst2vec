"""
Pseudoanonymization migration script.

Run ONCE after data extraction is complete:
    python scripts/pseudoanonymize.py

Creates:
    data/identity_map.db  — PII + API PKs
    data/inst2vec.db.bak  — backup of the main DB before migration

Replaces Instagram API PKs with sequential internal IDs in the main DB,
renames pk/user_pk/clip_pk/entity_pk columns to id/user_id/clip_id/entity_id,
strips PII fields (username, full_name, city_name, profile_pic_*) from main DB,
renames on-disk files, and recomputes cluster_runs.dataset_hash.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class IdentityBase(DeclarativeBase):
    pass


class UserIdentity(IdentityBase):
    __tablename__ = "user_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String)
    city_name: Mapped[str | None] = mapped_column(String)
    profile_pic_url: Mapped[str | None] = mapped_column(String)
    profile_pic_url_hd: Mapped[str | None] = mapped_column(String)


class ClipIdentity(IdentityBase):
    __tablename__ = "clip_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_pk: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)


def init_identity_db(db_path: str):
    """Create identity_map.db and return its engine."""
    engine = create_engine(f"sqlite:///{db_path}")
    IdentityBase.metadata.create_all(engine)
    return engine
