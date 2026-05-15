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
    get_api_pk,
    get_or_create_clip_identity,
    get_or_create_user_identity,
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
    update_user_identity(
        uid,
        api_pk=12345,
        full_name=None,
        city_name=None,
        profile_pic_url=None,
        profile_pic_url_hd=None,
    )
    assert get_api_pk(uid) == 12345


def test_get_profile_pic_url_returns_none_before_update():
    uid = get_or_create_user_identity("hank")
    assert get_profile_pic_url(uid) is None


def test_get_profile_pic_url_returns_value_after_update():
    uid = get_or_create_user_identity("iris")
    update_user_identity(
        uid,
        api_pk=1,
        full_name=None,
        city_name=None,
        profile_pic_url="https://example.com/iris.jpg",
        profile_pic_url_hd=None,
    )
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


def test_load_usernames_from_csv_creates_user_with_sequential_id(tmp_path, monkeypatch):
    """CSV loading must write username to identity DB, not main DB."""
    import csv

    from sqlalchemy import create_engine

    from modules.database import Base, User

    # --- main DB: in-memory ---
    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)
    monkeypatch.setattr("modules.utils.get_session", lambda: Session(main_eng))

    # --- identity DB already isolated by autouse fixture ---
    # The autouse fixture (isolated_identity_engine) already replaced _engine with in-memory

    # --- write CSV ---
    csv_path = str(tmp_path / "data.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(
            [
                ["https://instagram.com/alice"],
                ["https://instagram.com/bob"],
            ]
        )

    from modules.utils import load_usernames_from_csv

    load_usernames_from_csv(csv_path=csv_path)

    # Main DB: user rows exist with sequential IDs, no username column
    with Session(main_eng) as s:
        users = s.query(User).all()
        assert len(users) == 2
        user_ids = {u.id for u in users}
        # IDs are small sequential ints (not large hash values)
        assert all(1 <= uid <= 1000 for uid in user_ids)

    # Identity DB: usernames stored there (via the monkeypatched _engine)
    from modules.identity import UserIdentity, get_identity_session

    with get_identity_session() as s:
        usernames = {ui.username for ui in s.query(UserIdentity).all()}
        assert usernames == {"alice", "bob"}


def test_load_usernames_from_csv_missing_file_is_noop(monkeypatch):
    """If the CSV does not exist the function returns without raising."""
    from modules.utils import load_usernames_from_csv

    load_usernames_from_csv(csv_path="/nonexistent/path/data.csv")

    from modules.identity import UserIdentity, get_identity_session

    with get_identity_session() as s:
        assert s.query(UserIdentity).count() == 0


def test_download_uses_entity_id_and_fetches_pic_url_from_identity_db(
    tmp_path, monkeypatch
):
    """_try_download must use entity_id (not entity_pk); profile_pic_url from identity DB."""
    import os

    from sqlalchemy import create_engine

    from modules.database import Base, Download, User

    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)

    # The autouse fixture already replaced _engine with in-memory; add a UserIdentity
    from modules.identity import UserIdentity, get_identity_session

    with get_identity_session() as s:
        s.add(
            UserIdentity(
                id=1,
                username="downloader",
                profile_pic_url="https://example.com/pic.jpg",
            )
        )
        s.commit()

    with Session(main_eng) as s:
        s.add(User(id=1, is_eligible=True))
        s.commit()

    monkeypatch.setattr("modules.download.get_session", lambda: Session(main_eng))

    import modules.download as dl_mod

    profile_pic_dir = str(tmp_path / "profile_pics")
    thumbnail_dir = str(tmp_path / "thumbnails")
    video_dir = str(tmp_path / "videos")
    for d in [profile_pic_dir, thumbnail_dir, video_dir]:
        os.makedirs(d, exist_ok=True)

    # Stub _download to succeed without actually fetching HTTP
    monkeypatch.setattr(
        dl_mod, "_download", lambda url, path, max_attempts, retry_delay: True
    )

    dl_mod.download_files(
        batch_size=10,
        max_clips=0,
        max_attempts=3,
        retry_delay=2,
        profile_pic_dir=profile_pic_dir,
        thumbnail_dir=thumbnail_dir,
        video_dir=video_dir,
    )

    with Session(main_eng) as s:
        row = s.query(Download).filter_by(entity_id=1, file_type="profile_pic").first()
        assert row is not None, (
            "Download row not created — likely entity_pk/entity_id bug"
        )
        assert row.success is True
