import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from modules.database import Base, User, Clip, Download, _migrate_users_table, _backfill_parse_status


def test_migrate_users_table_adds_parse_status_column():
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE users (pk BIGINT PRIMARY KEY, username VARCHAR NOT NULL UNIQUE)"
            )
        )
        conn.commit()
    assert "parse_status" not in {c["name"] for c in inspect(eng).get_columns("users")}
    _migrate_users_table(eng)
    names = {c["name"] for c in inspect(eng).get_columns("users")}
    assert "parse_status" in names


def test_backfill_success_failed_pending_and_precedence():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(pk=1, username="a", full_name="Parsed", parse_status=None))
        s.add(User(pk=2, username="b", parse_status=None))
        s.add(
            Download(
                entity_pk=2,
                file_type="profile_pic",
                success=False,
                parse_available=False,
            )
        )
        s.add(User(pk=3, username="c", parse_status=None))
        s.add(User(pk=4, username="d", full_name="Both", parse_status=None))
        s.add(
            Download(
                entity_pk=4,
                file_type="profile_pic",
                success=False,
                parse_available=False,
            )
        )
        s.commit()
        _backfill_parse_status(s)
        s.commit()

    with Session(eng) as s:
        assert s.get(User, 1).parse_status == "success"
        assert s.get(User, 2).parse_status == "failed"
        assert s.get(User, 3).parse_status == "pending"
        assert s.get(User, 4).parse_status == "success"


def test_fetch_profiles_retries_then_succeeds_fourth(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(pk=100, username="retry_user", parse_status="pending"))
        s.commit()

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def user_by_username_v1(self, username):
            self.calls += 1
            if self.calls < 4:
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
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)
    monkeypatch.setenv("BATCH_SIZE", "5")

    from modules.parse import fetch_profiles

    fetch_profiles()

    with Session(eng) as s:
        u = s.query(User).filter_by(username="retry_user").one()
        assert u.parse_status == "success"
        assert u.pk == 200


def test_fetch_profiles_all_attempts_fail(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(pk=101, username="fail_user", parse_status="pending"))
        s.commit()

    class FakeClient:
        def user_by_username_v1(self, username):
            raise RuntimeError("always")

        def user_clips_v2(self, pk):
            return {"response": {"items": []}}

    monkeypatch.setattr("modules.parse.Client", lambda token: FakeClient())
    monkeypatch.setattr("modules.parse.get_session", lambda: Session(eng))
    monkeypatch.setattr("modules.parse.time.sleep", lambda _: None)

    from modules.parse import fetch_profiles

    fetch_profiles()

    with Session(eng) as s:
        u = s.query(User).filter_by(username="fail_user").one()
        assert u.parse_status == "failed"


def test_finalize_unresolved_for_non_success_parse_status(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(pk=1, username="u1", parse_status="pending"))
        s.add(User(pk=2, username="u2", parse_status="failed"))
        s.add(User(pk=3, username="u3", full_name="Ok", parse_status="success"))
        for i in range(4):
            s.add(
                Clip(
                    pk=3000 + i,
                    user_pk=3,
                    play_count=100000,
                    disqualified=0,
                )
            )
        s.commit()

    monkeypatch.setattr("modules.finalize.get_session", lambda: Session(eng))
    monkeypatch.setenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "0")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS", "0")
    monkeypatch.setenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-99")

    import modules.finalize as fin_mod

    monkeypatch.setattr(fin_mod, "TARGET_CLIPS_PER_USER", 4)

    from modules.finalize import finalize_user_dataset

    finalize_user_dataset("A")

    with Session(eng) as s:
        assert s.get(User, 1).user_disqualified is None
        assert s.get(User, 2).user_disqualified is None
        assert s.get(User, 3).user_disqualified in (0, 1)
