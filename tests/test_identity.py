import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import modules.identity as identity_mod
from modules.identity import (
    ClipIdentity,
    IdentityBase,
    UserIdentity,
    get_or_create_clip_identity,
    get_or_create_user_identity,
    get_api_pk,
    get_profile_pic_url,
    get_username,
    init_identity_db,
    update_user_identity,
)


@pytest.fixture(autouse=True)
def isolated_identity_engine(tmp_path, monkeypatch):
    """Replace the module-level engine with an in-memory one for each test."""
    eng = create_engine("sqlite:///:memory:")
    IdentityBase.metadata.create_all(eng)
    monkeypatch.setattr(identity_mod, "_engine", eng)
    yield eng


def test_init_identity_db_creates_tables(tmp_path):
    db_path = str(tmp_path / "identity_map.db")
    engine = init_identity_db(db_path)
    from sqlalchemy import inspect
    tables = inspect(engine).get_table_names()
    assert "user_identities" in tables
    assert "clip_identities" in tables


def test_get_or_create_user_identity_returns_sequential_id():
    id1 = get_or_create_user_identity("alice")
    id2 = get_or_create_user_identity("bob")
    assert id1 != id2
    assert isinstance(id1, int)
    assert isinstance(id2, int)


def test_get_or_create_user_identity_is_idempotent():
    id1 = get_or_create_user_identity("carol")
    id2 = get_or_create_user_identity("carol")
    assert id1 == id2


def test_get_username_returns_stored_username():
    uid = get_or_create_user_identity("diana")
    assert get_username(uid) == "diana"


def test_update_user_identity_stores_pii():
    uid = get_or_create_user_identity("eve")
    update_user_identity(
        uid,
        api_pk=99999,
        full_name="Eve Smith",
        city_name="Berlin",
        profile_pic_url="https://example.com/eve.jpg",
        profile_pic_url_hd="https://example.com/eve-hd.jpg",
    )
    from modules.identity import get_identity_session
    with get_identity_session() as s:
        ui = s.get(UserIdentity, uid)
        assert ui.api_pk == 99999
        assert ui.full_name == "Eve Smith"
        assert ui.city_name == "Berlin"
        assert ui.profile_pic_url == "https://example.com/eve.jpg"
        assert ui.profile_pic_url_hd == "https://example.com/eve-hd.jpg"


def test_get_api_pk_returns_none_before_update():
    uid = get_or_create_user_identity("frank")
    assert get_api_pk(uid) is None


def test_get_api_pk_returns_value_after_update():
    uid = get_or_create_user_identity("grace")
    update_user_identity(uid, api_pk=12345, full_name=None, city_name=None,
                         profile_pic_url=None, profile_pic_url_hd=None)
    assert get_api_pk(uid) == 12345


def test_get_profile_pic_url_returns_none_before_update():
    uid = get_or_create_user_identity("hank")
    assert get_profile_pic_url(uid) is None


def test_get_profile_pic_url_returns_value_after_update():
    uid = get_or_create_user_identity("iris")
    update_user_identity(uid, api_pk=1, full_name=None, city_name=None,
                         profile_pic_url="https://example.com/iris.jpg",
                         profile_pic_url_hd=None)
    assert get_profile_pic_url(uid) == "https://example.com/iris.jpg"


def test_get_or_create_clip_identity_returns_sequential_id():
    cid1 = get_or_create_clip_identity(api_pk=9001)
    cid2 = get_or_create_clip_identity(api_pk=9002)
    assert cid1 != cid2
    assert isinstance(cid1, int)


def test_get_or_create_clip_identity_is_idempotent():
    cid1 = get_or_create_clip_identity(api_pk=9003)
    cid2 = get_or_create_clip_identity(api_pk=9003)
    assert cid1 == cid2


def test_clip_identity_api_pk_stored():
    cid = get_or_create_clip_identity(api_pk=9004)
    from modules.identity import get_identity_session
    with get_identity_session() as s:
        ci = s.get(ClipIdentity, cid)
        assert ci.api_pk == 9004
