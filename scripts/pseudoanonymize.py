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

Recovery: if migration fails, restore from backup:
    mv data/inst2vec.db.bak data/inst2vec.db
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3

from sqlalchemy import BigInteger, Integer, String, create_engine
from sqlalchemy.engine import Engine
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

    # Validate FK integrity: all clips must reference a valid user
    orphaned = [r[0] for r in clips if r[1] not in user_map]
    if orphaned:
        raise ValueError(
            f"_assign_ids: {len(orphaned)} clip(s) reference unknown user_pk: {orphaned[:5]}"
        )

    sorted_clips = sorted(clips, key=lambda r: (user_map[r[1]], r[0]))
    clip_map = {row[0]: i + 1 for i, row in enumerate(sorted_clips)}
    con.close()
    return user_map, clip_map


def _write_identity_map(
    main_db: str,
    identity_engine: Engine,
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


def _migrate_main_db(
    main_db: str, user_map: dict[int, int], clip_map: dict[int, int]
) -> None:
    """Migrate main DB values: update PKs and FKs, null PII, rename columns."""
    con = sqlite3.connect(main_db)
    con.execute("PRAGMA foreign_keys = OFF")

    # Create temp mapping tables
    con.execute("CREATE TEMP TABLE _user_map (old_pk INTEGER, new_id INTEGER)")
    con.execute("CREATE TEMP TABLE _clip_map (old_pk INTEGER, new_id INTEGER)")

    # Populate temp tables
    con.executemany("INSERT INTO _user_map VALUES (?, ?)", user_map.items())
    con.executemany("INSERT INTO _clip_map VALUES (?, ?)", clip_map.items())

    # Update FK columns BEFORE PK columns (using old PKs for matching)
    con.execute("""
        UPDATE clips SET user_pk = (
            SELECT new_id FROM _user_map WHERE old_pk = clips.user_pk
        )
    """)

    con.execute("""
        UPDATE user_embeddings SET user_pk = (
            SELECT new_id FROM _user_map WHERE old_pk = user_embeddings.user_pk
        )
    """)

    con.execute("""
        UPDATE user_clusters SET user_pk = (
            SELECT new_id FROM _user_map WHERE old_pk = user_clusters.user_pk
        )
    """)

    con.execute("""
        UPDATE clip_embeddings SET clip_pk = (
            SELECT new_id FROM _clip_map WHERE old_pk = clip_embeddings.clip_pk
        )
    """)

    con.execute("""
        UPDATE downloads SET entity_pk = (
            SELECT new_id FROM _user_map WHERE old_pk = downloads.entity_pk
        )
        WHERE file_type = 'profile_pic'
    """)

    con.execute("""
        UPDATE downloads SET entity_pk = (
            SELECT new_id FROM _clip_map WHERE old_pk = downloads.entity_pk
        )
        WHERE file_type IN ('thumbnail', 'video')
    """)

    # Update PK columns AFTER FKs
    con.execute("""
        UPDATE clips SET pk = (
            SELECT new_id FROM _clip_map WHERE old_pk = clips.pk
        )
    """)

    con.execute("""
        UPDATE users SET pk = (
            SELECT new_id FROM _user_map WHERE old_pk = users.pk
        )
    """)

    # NULL out PII fields using table recreation
    # Get all non-PII columns and copy data to new table
    con.execute("""
        CREATE TABLE users_new AS
        SELECT pk, NULL as username, NULL as full_name, following_count, NULL as city_name,
               user_disqualified, parse_status, NULL as profile_pic_url, NULL as profile_pic_url_hd
        FROM users
    """)

    con.execute("DROP TABLE users")
    con.execute("ALTER TABLE users_new RENAME TO users")

    # Re-enable FK enforcement
    con.execute("PRAGMA foreign_keys = ON")
    con.commit()
    con.close()


def _rename_columns(main_db: str) -> None:
    """Rename pk/user_pk/clip_pk/entity_pk columns to id/user_id/clip_id/entity_id."""
    con = sqlite3.connect(main_db)

    # Execute ALTER TABLE RENAME COLUMN statements in order
    con.execute("ALTER TABLE users RENAME COLUMN pk TO id")
    con.execute("ALTER TABLE clips RENAME COLUMN pk TO id")
    con.execute("ALTER TABLE clips RENAME COLUMN user_pk TO user_id")
    con.execute("ALTER TABLE downloads RENAME COLUMN entity_pk TO entity_id")
    con.execute("ALTER TABLE clip_embeddings RENAME COLUMN clip_pk TO clip_id")
    con.execute("ALTER TABLE user_embeddings RENAME COLUMN user_pk TO user_id")
    con.execute("ALTER TABLE user_clusters RENAME COLUMN user_pk TO user_id")

    con.commit()
    con.close()


def _rename_files(
    data_dir: str, user_map: dict[int, int], clip_map: dict[int, int]
) -> None:
    """Rename on-disk files from old PKs to new sequential IDs."""
    # Map (directory, extension, id_map) tuples
    renames = [
        ("source/profile_pics", "jpg", user_map),
        ("source/thumbnails", "jpg", clip_map),
        ("source/videos", "mp4", clip_map),
    ]

    for subdir, ext, id_map in renames:
        dir_path = os.path.join(data_dir, subdir)

        # Skip if directory doesn't exist
        if not os.path.isdir(dir_path):
            continue

        for old_pk, new_id in id_map.items():
            old_file = os.path.join(dir_path, f"{old_pk}.{ext}")
            new_file = os.path.join(dir_path, f"{new_id}.{ext}")

            # Only rename if old file exists and new file doesn't
            if os.path.exists(old_file) and not os.path.exists(new_file):
                os.rename(old_file, new_file)


def _update_dataset_hash(main_db: str) -> None:
    """Recompute cluster_runs.dataset_hash based on new user IDs."""
    con = sqlite3.connect(main_db)

    # Get distinct embedding_case values
    embedding_cases = con.execute(
        "SELECT DISTINCT embedding_case FROM cluster_runs"
    ).fetchall()

    for (embedding_case,) in embedding_cases:
        # Query user_id from user_embeddings for this embedding_case
        user_ids = sorted(
            row[0]
            for row in con.execute(
                "SELECT user_id FROM user_embeddings WHERE embedding_case = ?",
                (embedding_case,),
            )
        )

        # Skip if no user embeddings
        if not user_ids:
            continue

        # Compute hash
        hash_str = ",".join(str(x) for x in user_ids)
        dataset_hash = hashlib.sha256(hash_str.encode()).hexdigest()

        # Update cluster_runs
        con.execute(
            "UPDATE cluster_runs SET dataset_hash = ? WHERE embedding_case = ?",
            (dataset_hash, embedding_case),
        )

    con.commit()
    con.close()


def pseudoanonymize(
    main_db: str = "data/inst2vec.db",
    identity_db: str = "data/identity_map.db",
    data_dir: str = "data",
) -> None:
    """Pseudoanonymize main DB: replace API PKs with sequential IDs, strip PII."""
    # Guard: detect if migration already ran
    with sqlite3.connect(main_db) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        if "id" in cols and "pk" not in cols:
            raise RuntimeError(
                f"{main_db!r} appears already migrated (users.id exists, users.pk absent). "
                "Delete identity_map.db and restore the .bak file to re-run."
            )

    print(f"[pseudoanonymize] backing up {main_db} → {main_db}.bak")
    shutil.copy2(main_db, main_db + ".bak")

    print("[pseudoanonymize] assigning sequential IDs …")
    user_map, clip_map = _assign_ids(main_db)
    print(f"  {len(user_map)} users, {len(clip_map)} clips")

    print(f"[pseudoanonymize] writing identity map → {identity_db}")
    identity_engine = init_identity_db(identity_db)
    _write_identity_map(main_db, identity_engine, user_map, clip_map)

    print("[pseudoanonymize] migrating main DB values …")
    _migrate_main_db(main_db, user_map, clip_map)

    print("[pseudoanonymize] renaming SQL columns …")
    _rename_columns(main_db)

    print("[pseudoanonymize] renaming on-disk files …")
    _rename_files(data_dir, user_map, clip_map)

    print("[pseudoanonymize] recomputing cluster_runs.dataset_hash …")
    _update_dataset_hash(main_db)

    print("[pseudoanonymize] done.")


if __name__ == "__main__":
    pseudoanonymize()
