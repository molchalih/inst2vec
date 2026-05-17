"""Schema-level tests for the Clip speech-detection flag."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, User


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_clip_has_is_speech_detected_column():
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(Clip(id=10, user_id=1, is_speech_detected=True))
        s.commit()
        row = s.query(Clip).filter_by(id=10).one()
        assert row.is_speech_detected is True


def test_clip_is_speech_detected_defaults_to_null():
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(Clip(id=10, user_id=1))
        s.commit()
        row = s.query(Clip).filter_by(id=10).one()
        assert row.is_speech_detected is None
