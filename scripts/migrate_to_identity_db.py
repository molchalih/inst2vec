"""
One-time migration script for existing inst2vec databases.

Run ONCE before using the new identity-DB-first pipeline:
    python scripts/migrate_to_identity_db.py

What it does:
  1. Backs up data/inst2vec.db → data/inst2vec.db.premigration
  2. Assigns sequential integer IDs (users alphabetically, clips by user then api_pk)
  3. Writes PII + API PKs to identity_map.db
  4. Remaps all PK/FK values in main DB to sequential IDs
  5. NULLs out PII columns (username, full_name, city_name, profile_pic_url, profile_pic_url_hd)
  6. Renames columns: pk→id, user_pk→user_id, clip_pk→clip_id, entity_pk→entity_id
  7. Renames on-disk files to match new sequential IDs
  8. Recomputes cluster_runs.dataset_hash
  9. Adds and backfills parse_status

Recovery: if migration fails, restore from backup:
    mv data/inst2vec.db.premigration data/inst2vec.db
    rm -f data/identity_map.db
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.identity import ClipIdentity, IdentityBase, UserIdentity


def migrate(
    main_db: str = "data/inst2vec.db",
    identity_db: str = "data/identity_map.db",
    data_dir: str = "data",
) -> None:
    with sqlite3.connect(main_db) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        if "id" in cols and "pk" not in cols:
            raise RuntimeError(
                f"{main_db!r} appears already migrated (users.id exists, users.pk absent). "
                "Nothing to do."
            )

    print(f"[migrate] backing up {main_db} → {main_db}.premigration")
    shutil.copy2(main_db, main_db + ".premigration")

    print("[migrate] assigning sequential IDs …")
    user_map, clip_map = _assign_ids(main_db)
    print(f"  {len(user_map)} users, {len(clip_map)} clips")

    print(f"[migrate] writing identity map → {identity_db}")
    _write_identity_db(main_db, identity_db, user_map, clip_map)

    print("[migrate] remapping PK/FK values …")
    _remap_ids(main_db, user_map, clip_map)

    print("[migrate] nulling PII and renaming columns …")
    _null_pii_and_rename(main_db)

    print("[migrate] renaming on-disk files …")
    _rename_files(data_dir, user_map, clip_map)

    print("[migrate] recomputing cluster_runs.dataset_hash …")
    _update_dataset_hash(main_db)

    print("[migrate] backfilling parse_status …")
    _backfill_parse_status(main_db)

    print("[migrate] done.")


# ── Helpers ────────────────────────────────────────────────────────────────


def _assign_ids(main_db: str) -> tuple[dict[int, int], dict[int, int]]:
    """Return (user_map, clip_map): {old_api_pk → new_sequential_id}.

    Users ordered alphabetically by username for determinism.
    Clips ordered by user's new_id then clip api_pk.
    """
    con = sqlite3.connect(main_db)
    users = con.execute("SELECT pk, username FROM users ORDER BY username").fetchall()
    user_map = {row[0]: i + 1 for i, row in enumerate(users)}

    clips = con.execute("SELECT pk, user_pk FROM clips ORDER BY user_pk, pk").fetchall()
    orphaned = [r[0] for r in clips if r[1] not in user_map]
    if orphaned:
        raise ValueError(f"{len(orphaned)} clip(s) reference unknown user_pk: {orphaned[:5]}")

    sorted_clips = sorted(clips, key=lambda r: (user_map[r[1]], r[0]))
    clip_map = {row[0]: i + 1 for i, row in enumerate(sorted_clips)}
    con.close()
    return user_map, clip_map


def _write_identity_db(
    main_db: str,
    identity_db: str,
    user_map: dict[int, int],
    clip_map: dict[int, int],
) -> None:
    """Populate identity_map.db with PII and original API PKs."""
    eng = create_engine(f"sqlite:///{identity_db}")
    IdentityBase.metadata.create_all(eng)

    con = sqlite3.connect(main_db)
    users = con.execute(
        "SELECT pk, username, full_name, city_name, profile_pic_url, profile_pic_url_hd FROM users"
    ).fetchall()
    clips = con.execute("SELECT pk FROM clips").fetchall()
    con.close()

    with Session(eng) as s:
        for api_pk, username, full_name, city_name, pic_url, pic_url_hd in users:
            s.add(UserIdentity(
                id=user_map[api_pk],
                api_pk=api_pk,
                username=username or "",
                full_name=full_name,
                city_name=city_name,
                profile_pic_url=pic_url,
                profile_pic_url_hd=pic_url_hd,
            ))
        for (api_pk,) in clips:
            s.add(ClipIdentity(id=clip_map[api_pk], api_pk=api_pk))
        s.commit()


def _remap_ids(
    main_db: str, user_map: dict[int, int], clip_map: dict[int, int]
) -> None:
    """Remap all PK/FK integer values to sequential IDs (using old column names)."""
    con = sqlite3.connect(main_db)
    con.execute("PRAGMA foreign_keys = OFF")

    con.execute("CREATE TEMP TABLE _user_map (old_pk INTEGER, new_id INTEGER)")
    con.execute("CREATE TEMP TABLE _clip_map (old_pk INTEGER, new_id INTEGER)")
    con.executemany("INSERT INTO _user_map VALUES (?, ?)", user_map.items())
    con.executemany("INSERT INTO _clip_map VALUES (?, ?)", clip_map.items())

    con.execute("UPDATE clips SET user_pk = (SELECT new_id FROM _user_map WHERE old_pk = clips.user_pk)")
    con.execute("UPDATE user_embeddings SET user_pk = (SELECT new_id FROM _user_map WHERE old_pk = user_embeddings.user_pk)")
    con.execute("UPDATE user_clusters SET user_pk = (SELECT new_id FROM _user_map WHERE old_pk = user_clusters.user_pk)")
    con.execute("UPDATE clip_embeddings SET clip_pk = (SELECT new_id FROM _clip_map WHERE old_pk = clip_embeddings.clip_pk)")
    con.execute("UPDATE downloads SET entity_pk = (SELECT new_id FROM _user_map WHERE old_pk = downloads.entity_pk) WHERE file_type = 'profile_pic'")
    con.execute("UPDATE downloads SET entity_pk = (SELECT new_id FROM _clip_map WHERE old_pk = downloads.entity_pk) WHERE file_type IN ('thumbnail', 'video')")
    con.execute("UPDATE clips SET pk = (SELECT new_id FROM _clip_map WHERE old_pk = clips.pk)")
    con.execute("UPDATE users SET pk = (SELECT new_id FROM _user_map WHERE old_pk = users.pk)")

    con.execute("PRAGMA foreign_keys = ON")
    con.commit()
    con.close()


def _null_pii_and_rename(main_db: str) -> None:
    """NULL out PII columns, add parse_status, rename pk→id / user_pk→user_id / etc."""
    con = sqlite3.connect(main_db)

    # Recreate users table: NULL PII, add parse_status
    con.execute("""
        CREATE TABLE users_new AS
        SELECT pk,
               NULL AS username,
               NULL AS full_name,
               following_count,
               NULL AS city_name,
               user_disqualified,
               parse_status,
               NULL AS profile_pic_url,
               NULL AS profile_pic_url_hd
        FROM users
    """)
    con.execute("DROP TABLE users")
    con.execute("ALTER TABLE users_new RENAME TO users")

    # Rename columns
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
    for subdir, ext, id_map in [
        ("source/profile_pics", "jpg", user_map),
        ("source/thumbnails", "jpg", clip_map),
        ("source/videos", "mp4", clip_map),
    ]:
        dir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(dir_path):
            continue
        for old_pk, new_id in id_map.items():
            old_file = os.path.join(dir_path, f"{old_pk}.{ext}")
            new_file = os.path.join(dir_path, f"{new_id}.{ext}")
            if os.path.exists(old_file) and not os.path.exists(new_file):
                os.rename(old_file, new_file)


def _update_dataset_hash(main_db: str) -> None:
    con = sqlite3.connect(main_db)
    for (embedding_case,) in con.execute(
        "SELECT DISTINCT embedding_case FROM cluster_runs"
    ).fetchall():
        user_ids = sorted(
            r[0] for r in con.execute(
                "SELECT user_id FROM user_embeddings WHERE embedding_case = ?",
                (embedding_case,),
            )
        )
        if not user_ids:
            continue
        dataset_hash = hashlib.sha256(
            ",".join(str(x) for x in user_ids).encode()
        ).hexdigest()
        con.execute(
            "UPDATE cluster_runs SET dataset_hash = ? WHERE embedding_case = ?",
            (dataset_hash, embedding_case),
        )
    con.commit()
    con.close()


def _backfill_parse_status(main_db: str) -> None:
    with sqlite3.connect(main_db) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        if "parse_status" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN parse_status VARCHAR")

        con.execute("UPDATE users SET parse_status = 'pending' WHERE parse_status IS NULL")
        con.execute("""
            UPDATE users SET parse_status = 'success'
            WHERE parse_status = 'pending'
              AND (following_count IS NOT NULL OR id IN (SELECT user_id FROM clips))
        """)
        con.execute("""
            UPDATE users SET parse_status = 'failed'
            WHERE parse_status = 'pending'
              AND id IN (
                SELECT entity_id FROM downloads
                WHERE file_type = 'profile_pic' AND parse_available = 0
              )
        """)
        con.commit()


if __name__ == "__main__":
    migrate()
