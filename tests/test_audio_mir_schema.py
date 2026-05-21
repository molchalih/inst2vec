"""Schema-level tests for the AudioMIR model."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.database import AudioMIR, Base, Clip, User


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed_clip(s: Session, clip_id: int = 1) -> None:
    s.add(User(id=1))
    s.add(Clip(id=clip_id, user_id=1))
    s.flush()


def test_audio_mir_round_trip_full_row():
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s)
        s.add(
            AudioMIR(
                clip_id=1,
                is_mir_extracted=True,
                mir_error=None,
                approachability=0.42,
                engagement=0.31,
                danceability=0.77,
                is_aggressive=False,
                is_happy=True,
                is_party=False,
                is_relaxed=False,
                is_sad=False,
                is_acoustic=True,
                is_electronic=False,
                is_instrumental=False,
                is_female_voice=True,
                is_bright_timbre=True,
                is_tonal=True,
                genre_labels="Rock---Indie Rock, Electronic---House",
                genre_scores="0.50, 0.40",
                moodtheme_labels="happy",
                moodtheme_scores="0.30",
                instrument_labels="guitar",
                instrument_scores="0.60",
                audio_duration_s=12.3,
                inference_time_ms=420.5,
            )
        )
        s.commit()
        row = s.query(AudioMIR).filter_by(clip_id=1).one()
        assert row.is_mir_extracted is True
        assert row.danceability == 0.77
        assert row.genre_labels.startswith("Rock---Indie Rock")
        assert row.created_at is not None
        assert row.updated_at is not None


def test_audio_mir_unique_per_clip():
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s)
        s.add(AudioMIR(clip_id=1))
        s.commit()
        s.add(AudioMIR(clip_id=1))
        with pytest.raises(IntegrityError):
            s.commit()


def test_audio_mir_nullable_defaults():
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s)
        s.add(AudioMIR(clip_id=1))
        s.commit()
        row = s.query(AudioMIR).filter_by(clip_id=1).one()
        assert row.is_mir_extracted is None
        assert row.danceability is None
        assert row.is_happy is None
        assert row.genre_labels is None
