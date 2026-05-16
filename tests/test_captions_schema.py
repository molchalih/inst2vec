"""Schema-level tests for Clip.caption_clean."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, Clip, User


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_clip_has_caption_clean_column():
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(Clip(id=10, user_id=1, caption_text="raw", caption_clean="raw"))
        s.commit()
        row = s.query(Clip).filter_by(id=10).one()
        assert row.caption_clean == "raw"


def test_clip_caption_clean_defaults_to_null():
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(Clip(id=10, user_id=1))
        s.commit()
        row = s.query(Clip).filter_by(id=10).one()
        assert row.caption_clean is None
