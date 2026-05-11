"""Tests for scripts/migrate_to_identity_db.py."""

import hashlib
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from modules.identity import UserIdentity


def _seed_old_db(path: str):
    """Create a minimal DB with the old (pre-migration) schema."""
    con = sqlite3.connect(path)
    con.executescript("""
        PRAGMA foreign_keys = OFF;
        CREATE TABLE users (
            pk INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            profile_pic_url TEXT,
            profile_pic_url_hd TEXT,
            following_count INTEGER,
            city_name TEXT,
            user_disqualified INTEGER,
            parse_status TEXT
        );
        CREATE TABLE clips (
            pk INTEGER PRIMARY KEY,
            user_pk INTEGER NOT NULL,
            play_count INTEGER, disqualified INTEGER
        );
        CREATE TABLE downloads (
            entity_pk INTEGER, file_type TEXT, success INTEGER, parse_available INTEGER,
            PRIMARY KEY (entity_pk, file_type)
        );
        CREATE TABLE clip_embeddings (
            clip_pk INTEGER, embedding_case TEXT, embedding BLOB,
            PRIMARY KEY (clip_pk, embedding_case)
        );
        CREATE TABLE user_embeddings (
            user_pk INTEGER, embedding_case TEXT, embedding BLOB,
            PRIMARY KEY (user_pk, embedding_case)
        );
        CREATE TABLE user_clusters (
            user_pk INTEGER, embedding_case TEXT, cluster_id INTEGER,
            umap_x REAL, umap_y REAL,
            PRIMARY KEY (user_pk, embedding_case)
        );
        CREATE TABLE cluster_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding_case TEXT, dataset_hash TEXT,
            umap_n_components INTEGER DEFAULT 15,
            umap_n_neighbors INTEGER DEFAULT 15,
            umap_min_dist REAL DEFAULT 0.0,
            umap_metric TEXT DEFAULT 'cosine',
            umap2d_n_neighbors INTEGER DEFAULT 15,
            umap2d_min_dist REAL DEFAULT 0.0,
            umap2d_metric TEXT DEFAULT 'cosine',
            hdbscan_min_cluster_size INTEGER DEFAULT 5,
            hdbscan_min_samples INTEGER,
            hdbscan_cluster_selection_method TEXT DEFAULT 'eom',
            hdbscan_metric TEXT DEFAULT 'euclidean',
            random_state INTEGER DEFAULT 42,
            n_clusters INTEGER DEFAULT 1,
            noise_ratio REAL DEFAULT 0.0,
            min_size INTEGER DEFAULT 1,
            median_size INTEGER DEFAULT 1,
            max_size INTEGER DEFAULT 1
        );
        INSERT INTO users VALUES (111, 'alice', 'Alice A', 'http://p/a', 'http://p/a_hd', 5, 'Paris', 0, 'success');
        INSERT INTO users VALUES (222, 'bob',   'Bob B',   'http://p/b', 'http://p/b_hd', 8, 'Lyon',  0, 'success');
        INSERT INTO clips VALUES (901, 111, 500, 0);
        INSERT INTO clips VALUES (902, 111, 300, 0);
        INSERT INTO clips VALUES (903, 222, 100, 0);
        INSERT INTO downloads VALUES (111, 'profile_pic', 1, 1);
        INSERT INTO downloads VALUES (901, 'thumbnail', 1, 1);
        INSERT INTO downloads VALUES (901, 'video', 1, 1);
        INSERT INTO clip_embeddings VALUES (901, 'video', X'DEADBEEF');
        INSERT INTO user_embeddings VALUES (111, 'video', X'CAFEBABE');
        INSERT INTO user_embeddings VALUES (222, 'video', X'FEEDFACE');
        INSERT INTO user_clusters VALUES (111, 'video', 0, 1.0, 2.0);
        INSERT INTO cluster_runs (embedding_case, dataset_hash) VALUES ('video', 'stale_hash');
        PRAGMA foreign_keys = ON;
    """)
    con.commit()
    con.close()


def test_migrate_guard_already_migrated(tmp_path):
    """Migration must refuse if DB already has `id` not `pk`."""
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, following_count INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="already migrated"):
        migrate(
            main_db=db,
            identity_db=str(tmp_path / "identity_map.db"),
            data_dir=str(tmp_path),
        )


def test_migrate_creates_backup(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    assert os.path.exists(db + ".premigration")


def test_migrate_renames_columns(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    user_cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
    clip_cols = {r[1] for r in con.execute("PRAGMA table_info(clips)")}
    dl_cols = {r[1] for r in con.execute("PRAGMA table_info(downloads)")}
    ce_cols = {r[1] for r in con.execute("PRAGMA table_info(clip_embeddings)")}
    ue_cols = {r[1] for r in con.execute("PRAGMA table_info(user_embeddings)")}
    uc_cols = {r[1] for r in con.execute("PRAGMA table_info(user_clusters)")}
    assert "id" in user_cols and "pk" not in user_cols
    assert "user_id" in clip_cols and "user_pk" not in clip_cols
    assert "entity_id" in dl_cols and "entity_pk" not in dl_cols
    assert "clip_id" in ce_cols and "clip_pk" not in ce_cols
    assert "user_id" in ue_cols and "user_pk" not in ue_cols
    assert "user_id" in uc_cols and "user_pk" not in uc_cols
    con.close()


def test_migrate_nulls_pii(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT username, full_name, city_name, profile_pic_url, profile_pic_url_hd FROM users"
    ).fetchall()
    for row in rows:
        assert all(v is None for v in row)
    con.close()


def test_migrate_sequential_ids(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    user_ids = {r[0] for r in con.execute("SELECT id FROM users")}
    clip_ids = {r[0] for r in con.execute("SELECT id FROM clips")}
    assert user_ids == {1, 2}  # alice=1, bob=2 (alphabetical)
    assert clip_ids == {1, 2, 3}
    con.close()


def test_migrate_fk_references_correct(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    rows = {row for row in con.execute("SELECT id, user_id FROM clips")}
    assert (1, 1) in rows  # alice's clip 901 → id=1, user_id=1
    assert (2, 1) in rows  # alice's clip 902 → id=2, user_id=1
    assert (3, 2) in rows  # bob's clip 903 → id=3, user_id=2
    ue_ids = {r[0] for r in con.execute("SELECT user_id FROM user_embeddings")}
    assert ue_ids == {1, 2}
    ce_ids = {r[0] for r in con.execute("SELECT clip_id FROM clip_embeddings")}
    assert ce_ids == {1}
    uc_ids = {r[0] for r in con.execute("SELECT user_id FROM user_clusters")}
    assert uc_ids == {1}
    dl_rows = {
        (r[0], r[1]) for r in con.execute("SELECT entity_id, file_type FROM downloads")
    }
    assert (1, "profile_pic") in dl_rows
    assert (1, "thumbnail") in dl_rows
    assert (1, "video") in dl_rows
    con.close()


def test_migrate_populates_identity_db(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    idb = str(tmp_path / "identity_map.db")
    _seed_old_db(str(tmp_path / "inst2vec.db"))
    migrate(
        main_db=str(tmp_path / "inst2vec.db"), identity_db=idb, data_dir=str(tmp_path)
    )
    eng = create_engine(f"sqlite:///{idb}")
    assert "user_identities" in inspect(eng).get_table_names()
    with Session(eng) as s:
        alice = s.query(UserIdentity).filter_by(username="alice").first()
        assert alice is not None
        assert alice.api_pk == 111
        assert alice.full_name == "Alice A"
        assert alice.city_name == "Paris"
        assert s.query(UserIdentity).count() == 2


def test_migrate_backfills_parse_status(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    statuses = {r[0] for r in con.execute("SELECT parse_status FROM users")}
    assert statuses == {"success"}
    con.close()


def test_migrate_updates_dataset_hash(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    con = sqlite3.connect(db)
    new_user_ids = sorted(
        r[0]
        for r in con.execute(
            "SELECT user_id FROM user_embeddings WHERE embedding_case='video'"
        )
    )
    expected_hash = hashlib.sha256(
        ",".join(str(x) for x in new_user_ids).encode()
    ).hexdigest()
    stored_hash = con.execute(
        "SELECT dataset_hash FROM cluster_runs WHERE embedding_case='video'"
    ).fetchone()[0]
    assert stored_hash == expected_hash
    con.close()


def test_migrate_renames_disk_files(tmp_path):
    from scripts.migrate_to_identity_db import migrate

    db = str(tmp_path / "inst2vec.db")
    _seed_old_db(db)
    for subdir, ext, name in [
        ("profile_pics", "jpg", "111"),
        ("thumbnails", "jpg", "901"),
        ("videos", "mp4", "901"),
    ]:
        d = tmp_path / "source" / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.{ext}").write_bytes(b"fake")
    migrate(
        main_db=db,
        identity_db=str(tmp_path / "identity_map.db"),
        data_dir=str(tmp_path),
    )
    assert (tmp_path / "source" / "profile_pics" / "1.jpg").exists()
    assert not (tmp_path / "source" / "profile_pics" / "111.jpg").exists()
    assert (tmp_path / "source" / "thumbnails" / "1.jpg").exists()
    assert (tmp_path / "source" / "videos" / "1.mp4").exists()
