"""Tests for scripts/retry_failed_mir.py."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _engine():
    from core.database import Base

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed(session):
    from core.database import AudioMIR, Clip, User

    session.add(User(id=1))
    for cid in (1, 2, 3, 4):
        session.add(Clip(id=cid, user_id=1, is_selected=True))
    session.flush()
    session.add(AudioMIR(clip_id=1, is_mir_extracted=True))  # OK row, untouched
    session.add(AudioMIR(clip_id=2, is_mir_extracted=False, mir_error="maest"))
    session.add(AudioMIR(clip_id=3, is_mir_extracted=False, mir_error="effnet"))
    session.add(AudioMIR(clip_id=4, is_mir_extracted=False, mir_error="audio_load"))
    session.commit()


def test_retry_all_failed_rows():
    from core.database import AudioMIR
    from scripts.retry_failed_mir import retry_failed

    eng = _engine()
    with Session(eng) as s:
        _seed(s)

    with Session(eng) as s:
        n = retry_failed(s, error=None)
        assert n == 3
        s.commit()

    with Session(eng) as s:
        rows = {row.clip_id: row for row in s.query(AudioMIR).all()}
        assert rows[1].is_mir_extracted is True  # untouched
        assert rows[2].is_mir_extracted is None
        assert rows[2].mir_error is None
        assert rows[3].is_mir_extracted is None
        assert rows[4].is_mir_extracted is None


def test_retry_filters_by_error_kind():
    from core.database import AudioMIR
    from scripts.retry_failed_mir import retry_failed

    eng = _engine()
    with Session(eng) as s:
        _seed(s)

    with Session(eng) as s:
        n = retry_failed(s, error="maest")
        assert n == 1
        s.commit()

    with Session(eng) as s:
        rows = {row.clip_id: row for row in s.query(AudioMIR).all()}
        assert rows[2].is_mir_extracted is None  # reset
        assert rows[3].is_mir_extracted is False  # left alone
        assert rows[3].mir_error == "effnet"
        assert rows[4].is_mir_extracted is False  # left alone


def test_retry_rejects_unknown_error_kind():
    import pytest

    from scripts.retry_failed_mir import retry_failed

    eng = _engine()
    with Session(eng) as s, pytest.raises(ValueError):
        retry_failed(s, error="typo")
