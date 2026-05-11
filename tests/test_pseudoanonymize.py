import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import create_engine, inspect

from scripts.pseudoanonymize import init_identity_db, IdentityBase, UserIdentity, ClipIdentity


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
