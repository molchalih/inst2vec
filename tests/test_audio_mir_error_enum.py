"""Schema-level guard: AudioMIR.mir_error is a constrained enum."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session


def _fresh_engine():
    from core.database import Base

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_valid_error_values_accepted():
    from core.database import AudioMIR, Clip, User

    eng = _fresh_engine()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1, is_selected=True))
        s.flush()
        for value in ("maest", "effnet", "audio_load", "no_audio_file"):
            row = AudioMIR(clip_id=1, is_mir_extracted=False, mir_error=value)
            s.merge(row)
            s.flush()


def test_unknown_error_value_rejected():
    from core.database import AudioMIR, Clip, User

    eng = _fresh_engine()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1, is_selected=True))
        s.flush()
        with pytest.raises(StatementError):
            s.add(AudioMIR(clip_id=1, is_mir_extracted=False, mir_error="typo"))
            s.flush()


def test_null_error_value_accepted():
    """NULL must remain valid (success rows leave mir_error blank)."""
    from core.database import AudioMIR, Clip, User

    eng = _fresh_engine()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1, is_selected=True))
        s.flush()
        s.add(AudioMIR(clip_id=1, is_mir_extracted=True, mir_error=None))
        s.flush()
