import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from scripts.pseudoanonymize import init_identity_db, IdentityBase, UserIdentity, ClipIdentity, _assign_ids, _write_identity_map


def test_init_identity_db_creates_tables(tmp_path):
    db_path = str(tmp_path / "identity_map.db")
    engine = init_identity_db(db_path)
    tables = inspect(engine).get_table_names()
    assert "user_identities" in tables
    assert "clip_identities" in tables


def test_user_identity_columns(tmp_path):
    engine = init_identity_db(str(tmp_path / "identity_map.db"))
    cols = {c["name"] for c in inspect(engine).get_columns("user_identities")}
    assert cols == {"id", "api_pk", "username", "full_name", "city_name",
                    "profile_pic_url", "profile_pic_url_hd"}


def test_clip_identity_columns(tmp_path):
    engine = init_identity_db(str(tmp_path / "identity_map.db"))
    cols = {c["name"] for c in inspect(engine).get_columns("clip_identities")}
    assert cols == {"id", "api_pk"}


def _seed_main_db(path: str):
    """Create a minimal main DB with old schema for testing."""
    con = sqlite3.connect(path)
    con.executescript("""
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
            user_pk INTEGER NOT NULL REFERENCES users(pk),
            play_count INTEGER,
            disqualified INTEGER
        );
        INSERT INTO users VALUES (111111111, 'alice', 'Alice A', 'http://pic/a', 'http://pic/a_hd', 10, 'Paris', 0, 'success');
        INSERT INTO users VALUES (222222222, 'bob',   'Bob B',   'http://pic/b', 'http://pic/b_hd', 20, 'Lyon',  0, 'success');
        INSERT INTO clips VALUES (9001, 111111111, 500, 0);
        INSERT INTO clips VALUES (9002, 111111111, 300, 0);
        INSERT INTO clips VALUES (9003, 222222222, 100, 0);
    """)
    con.commit()
    con.close()


def test_assign_ids_returns_deterministic_sequential_maps(tmp_path):
    db = str(tmp_path / "inst2vec.db")
    _seed_main_db(db)
    user_map, clip_map = _assign_ids(db)
    # Users sorted by username → alice=1, bob=2
    assert user_map == {111111111: 1, 222222222: 2}
    # Clips sorted by user new_id then clip pk → 9001=1, 9002=2, 9003=3
    assert clip_map == {9001: 1, 9002: 2, 9003: 3}


def test_write_identity_map_stores_pii(tmp_path):
    main_db = str(tmp_path / "inst2vec.db")
    identity_db = str(tmp_path / "identity_map.db")
    _seed_main_db(main_db)
    user_map, clip_map = _assign_ids(main_db)
    engine = init_identity_db(identity_db)
    _write_identity_map(main_db, engine, user_map, clip_map)

    with Session(engine) as s:
        alice = s.get(UserIdentity, 1)
        assert alice is not None
        assert alice.api_pk == 111111111
        assert alice.username == "alice"
        assert alice.full_name == "Alice A"
        assert alice.city_name == "Paris"

        c1 = s.get(ClipIdentity, 1)
        assert c1 is not None
        assert c1.api_pk == 9001
