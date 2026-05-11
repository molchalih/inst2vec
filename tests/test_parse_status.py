import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from modules.database import (
    Base,
    Clip,
    Download,
    User,
    _backfill_parse_status,
    _migrate_users_table,
)


def test_migrate_users_table_adds_parse_status_column():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE users (pk BIGINT PRIMARY KEY, username VARCHAR NOT NULL UNIQUE)"
            )
        )
    assert "parse_status" not in {c["name"] for c in inspect(eng).get_columns("users")}
    _migrate_users_table(eng)
    names = {c["name"] for c in inspect(eng).get_columns("users")}
    assert "parse_status" in names


def test_backfill_success_failed_pending_and_precedence():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, following_count=100, parse_status=None))  # has following_count → success
        s.add(User(id=2, parse_status=None))                        # has failed download → failed
        s.add(
            Download(
                entity_id=2,
                file_type="profile_pic",
                success=False,
                parse_available=False,
            )
        )
        s.add(User(id=3, parse_status=None))                        # nothing → pending
        s.add(User(id=4, following_count=200, parse_status=None))   # both → success wins
        s.add(
            Download(
                entity_id=4,
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


# fetch_profiles tests updated in Task 5 (parse.py now reads username from identity DB)


def test_finalize_unresolved_for_non_success_parse_status(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="pending"))
        s.add(User(id=2, parse_status="failed"))
        s.add(User(id=3, parse_status="success"))
        for i in range(4):
            s.add(
                Clip(
                    id=3000 + i,
                    user_id=3,
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
        u1 = s.get(User, 1)
        assert u1 is not None
        assert u1.user_disqualified is None
        u2 = s.get(User, 2)
        assert u2 is not None
        assert u2.user_disqualified is None
        u3 = s.get(User, 3)
        assert u3 is not None
        assert u3.user_disqualified in (0, 1)
