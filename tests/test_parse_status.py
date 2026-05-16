import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import (
    Base,
    Clip,
    IdentityBase,
    User,
    UserIdentity,
)
from modules.database import engine as engine_mod

# fetch_profiles tests updated in Task 5 (parse.py now reads username from identity DB)


@pytest.mark.skip(reason="modules.filter not yet implemented (Task 3+)")
def test_finalize_unresolved_for_non_success_parse_status(monkeypatch):  # type: ignore[misc]
    """Test postponed pending filter module implementation."""
    pass


# ── Task 5: fetch_profiles tests ──────────────────────────────────────────


def _make_identity_engine():
    from sqlalchemy import create_engine

    eng = create_engine("sqlite:///:memory:")
    IdentityBase.metadata.create_all(eng)
    return eng


def test_fetch_profiles_reads_username_from_identity_db(monkeypatch):
    """fetch_profiles must get username from identity DB, not User.username."""
    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)
    id_eng = _make_identity_engine()
    monkeypatch.setattr(engine_mod, "_identity_engine", id_eng)

    with Session(id_eng) as s:
        s.add(UserIdentity(id=1, username="fetchme"))
        s.commit()
    with Session(main_eng) as s:
        s.add(User(id=1, parse_status=None))
        s.commit()

    called_with_username = []
    called_with_pk = []

    class FakeClient:
        def user_by_username_v1(self, username):
            called_with_username.append(username)
            return {
                "user": {
                    "pk": 55555,
                    "full_name": "Fetch Me",
                    "profile_pic_url": "http://p/fm",
                    "profile_pic_url_hd": "http://p/fm_hd",
                    "following_count": 42,
                    "city_name": "NYC",
                }
            }

        def user_clips_v2(self, pk):
            called_with_pk.append(pk)
            return {"response": {"items": []}}

    monkeypatch.setattr("modules.parse.Client", lambda token: FakeClient())
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(main_eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)

    from modules.parse import fetch_profiles

    fetch_profiles(hiker_api_key="test_key")

    # Must have called the API with the correct username
    assert called_with_username == ["fetchme"]
    # Must have called clips API with the Instagram API PK (not sequential ID)
    assert called_with_pk == ["55555"]

    # Main DB: no PII, following_count set, parse_status success
    with Session(main_eng) as s:
        u = s.get(User, 1)
        assert u.parse_status == "success"
        assert u.following_count == 42

    # Identity DB: PII and API PK stored
    with Session(id_eng) as s:
        ui = s.get(UserIdentity, 1)
        assert ui.api_pk == 55555
        assert ui.full_name == "Fetch Me"
        assert ui.city_name == "NYC"


def test_fetch_profiles_stores_sequential_clip_ids(monkeypatch):
    """Clip rows must use sequential IDs from identity DB, not Instagram PKs."""
    from modules.database import ClipIdentity

    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)
    id_eng = _make_identity_engine()
    monkeypatch.setattr(engine_mod, "_identity_engine", id_eng)

    with Session(id_eng) as s:
        s.add(UserIdentity(id=1, username="clipper"))
        s.commit()
    with Session(main_eng) as s:
        s.add(User(id=1, parse_status=None))
        s.commit()

    LARGE_API_PK = 3770212309156376545

    class FakeClient:
        def user_by_username_v1(self, username):
            return {
                "user": {
                    "pk": 99,
                    "full_name": None,
                    "profile_pic_url": None,
                    "profile_pic_url_hd": None,
                    "following_count": None,
                    "city_name": None,
                }
            }

        def user_clips_v2(self, pk):
            return {
                "response": {
                    "items": [
                        {
                            "media": {
                                "pk": LARGE_API_PK,
                                "thumbnail_url": None,
                                "video_url": None,
                                "caption": None,
                                "comment_count": 0,
                                "reshare_count": 0,
                                "like_count": 0,
                                "play_count": 1000,
                            }
                        }
                    ]
                }
            }

    monkeypatch.setattr("modules.parse.Client", lambda token: FakeClient())
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(main_eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)

    from modules.parse import fetch_profiles

    fetch_profiles(hiker_api_key="test_key")

    with Session(main_eng) as s:
        clips = s.query(Clip).all()
        assert len(clips) == 1
        # Sequential ID, NOT the large Instagram PK
        assert clips[0].id != LARGE_API_PK
        assert clips[0].id < 10_000_000

    # Identity DB: clip identity exists with original api_pk
    with Session(id_eng) as s:
        ci = s.query(ClipIdentity).filter_by(api_pk=LARGE_API_PK).first()
        assert ci is not None
        assert ci.id == clips[0].id  # IDs must match


def test_fetch_profiles_retries_then_succeeds_third(monkeypatch):
    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)
    id_eng = _make_identity_engine()
    monkeypatch.setattr(engine_mod, "_identity_engine", id_eng)

    with Session(id_eng) as s:
        s.add(UserIdentity(id=100, username="retry_user"))
        s.commit()
    with Session(main_eng) as s:
        s.add(User(id=100, parse_status=None))
        s.commit()

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def user_by_username_v1(self, username):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary")
            return {
                "user": {
                    "pk": 200,
                    "full_name": "OK",
                    "profile_pic_url": None,
                    "profile_pic_url_hd": None,
                    "following_count": None,
                    "city_name": None,
                }
            }

        def user_clips_v2(self, pk):
            return {"response": {"items": []}}

    monkeypatch.setattr("modules.parse.Client", lambda token: FakeClient())
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(main_eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)

    from modules.parse import fetch_profiles

    fetch_profiles(hiker_api_key="test_key")

    with Session(main_eng) as s:
        u = s.get(User, 100)
        assert u.parse_status == "success"

    # API PK in identity DB
    with Session(id_eng) as s:
        ui = s.get(UserIdentity, 100)
        assert ui.api_pk == 200


def test_fetch_profiles_all_attempts_fail(monkeypatch):
    main_eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(main_eng)
    id_eng = _make_identity_engine()
    monkeypatch.setattr(engine_mod, "_identity_engine", id_eng)

    with Session(id_eng) as s:
        s.add(UserIdentity(id=101, username="fail_user"))
        s.commit()
    with Session(main_eng) as s:
        s.add(User(id=101, parse_status=None))
        s.commit()

    class FakeClient:
        def user_by_username_v1(self, username):
            raise RuntimeError("always")

        def user_clips_v2(self, pk):
            return {"response": {"items": []}}

    monkeypatch.setattr("modules.parse.Client", lambda token: FakeClient())
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(main_eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)

    from modules.parse import fetch_profiles

    fetch_profiles(hiker_api_key="test_key")

    with Session(main_eng) as s:
        u = s.get(User, 101)
        assert u.parse_status == "failed"
