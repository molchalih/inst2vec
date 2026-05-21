"""Tests for the MIR fingerprint payload + reset helper."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _engine():
    from core.database import Base

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _mir_settings(**overrides):
    from core.config import MirSettings

    return MirSettings(**overrides)


def test_payload_stable_for_identical_settings():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings())
    b = mir_config_payload(_mir_settings())
    assert a == b


def test_payload_changes_with_binary_threshold():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings(binary_threshold=0.5))
    b = mir_config_payload(_mir_settings(binary_threshold=0.6))
    assert a != b


def test_payload_excludes_operational_knobs():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings(download_concurrency=2, commit_every=10))
    b = mir_config_payload(_mir_settings(download_concurrency=8, commit_every=50))
    assert a == b


def test_payload_changes_with_topk_or_checkpoint():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings())
    b = mir_config_payload(_mir_settings(topk_genre=5))
    c = mir_config_payload(_mir_settings(maest_checkpoint="other.pb"))
    assert a != b
    assert a != c
    assert b != c


def test_reset_audio_mir_nulls_descriptor_columns_but_preserves_row():
    from core.database import AudioMIR, Clip, User
    from modules.mir.state import reset_audio_mir

    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1))
        s.add(
            AudioMIR(
                clip_id=1,
                is_mir_extracted=True,
                danceability=0.7,
                is_happy=True,
                genre_labels="x",
                genre_scores="0.5",
                audio_duration_s=10.0,
                inference_time_ms=5.0,
            )
        )
        s.commit()
        original = s.query(AudioMIR).filter_by(clip_id=1).one().created_at

        reset_audio_mir(s)

        row = s.query(AudioMIR).filter_by(clip_id=1).one()
        assert row.is_mir_extracted is None
        assert row.danceability is None
        assert row.is_happy is None
        assert row.genre_labels is None
        assert row.genre_scores is None
        assert row.audio_duration_s is None
        assert row.inference_time_ms is None
        assert row.clip_id == 1
        assert row.created_at == original


def test_reset_audio_mir_on_empty_db_is_noop():
    from modules.mir.state import reset_audio_mir

    eng = _engine()
    with Session(eng) as s:
        reset_audio_mir(s)


def test_stage_and_scope_constants():
    from core.pipeline import Stage
    from modules.mir.state import POS, SCOPE_MIR, STAGE_MIR

    assert STAGE_MIR is Stage.MIR
    assert SCOPE_MIR == "all"
    assert POS == 0


def test_reset_columns_covers_audio_mir_schema():
    """_RESET_COLUMNS must cover every AudioMIR column except the PK and timestamps."""
    from core.database import AudioMIR
    from modules.mir.state import _RESET_COLUMNS

    all_cols = {c.name for c in AudioMIR.__table__.columns}
    excluded = {"clip_id", "created_at", "updated_at"}
    assert set(_RESET_COLUMNS) == all_cols - excluded
