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

import sqlite3
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


def _assign_ids(main_db: str) -> tuple[dict[int, int], dict[int, int]]:
    """Return (user_id_map, clip_id_map): {old_api_pk → new_sequential_id}.

    Users are ordered alphabetically by username for determinism.
    Clips are ordered by their user's new_id then by clip api_pk.
    """
    con = sqlite3.connect(main_db)
    users = con.execute("SELECT pk, username FROM users ORDER BY username").fetchall()
    user_map = {row[0]: i + 1 for i, row in enumerate(users)}

    # Sort clips by (user's new id, clip pk) for stable ordering
    clips = con.execute("SELECT pk, user_pk FROM clips ORDER BY user_pk, pk").fetchall()
    sorted_clips = sorted(clips, key=lambda r: (user_map.get(r[1], 0), r[0]))
    clip_map = {row[0]: i + 1 for i, row in enumerate(sorted_clips)}
    con.close()
    return user_map, clip_map


def _write_identity_map(
    main_db: str,
    identity_engine,
    user_map: dict[int, int],
    clip_map: dict[int, int],
) -> None:
    """Write PII + original API PKs into identity_map.db."""
    con = sqlite3.connect(main_db)
    users = con.execute(
        "SELECT pk, username, full_name, city_name, profile_pic_url, profile_pic_url_hd FROM users"
    ).fetchall()
    clips = con.execute("SELECT pk FROM clips").fetchall()
    con.close()

    with Session(identity_engine) as session:
        for api_pk, username, full_name, city_name, pic_url, pic_url_hd in users:
            new_id = user_map[api_pk]
            session.add(
                UserIdentity(
                    id=new_id,
                    api_pk=api_pk,
                    username=username or "",
                    full_name=full_name,
                    city_name=city_name,
                    profile_pic_url=pic_url,
                    profile_pic_url_hd=pic_url_hd,
                )
            )
        for (api_pk,) in clips:
            session.add(ClipIdentity(id=clip_map[api_pk], api_pk=api_pk))
        session.commit()
