"""Schema-level tests for the Music model state flags."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Music


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_music_has_is_audio_features_extracted_column():
    eng = _engine()
    with Session(eng) as s:
        s.add(Music(artist="a", track="t", is_audio_features_extracted=True))
        s.commit()
        row = s.query(Music).filter_by(artist="a").one()
        assert row.is_audio_features_extracted is True


def test_music_is_audio_features_extracted_defaults_to_null():
    eng = _engine()
    with Session(eng) as s:
        s.add(Music(artist="a", track="t"))
        s.commit()
        row = s.query(Music).filter_by(artist="a").one()
        assert row.is_audio_features_extracted is None


def test_music_has_is_reccobeats_resolved_column():
    eng = _engine()
    with Session(eng) as s:
        s.add(Music(artist="a", track="t", is_reccobeats_resolved=True))
        s.commit()
        row = s.query(Music).filter_by(artist="a").one()
        assert row.is_reccobeats_resolved is True


def test_music_is_reccobeats_resolved_defaults_to_null():
    eng = _engine()
    with Session(eng) as s:
        s.add(Music(artist="b", track="u"))
        s.commit()
        row = s.query(Music).filter_by(artist="b").one()
        assert row.is_reccobeats_resolved is None
