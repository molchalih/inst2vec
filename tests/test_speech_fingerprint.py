"""Config-drift fingerprint gate for the speech pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from modules.database import (
    Base,
    Clip,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.speech import process_speech
from modules.speech.state import SCOPE_SPEECH, STAGE_SPEECH, reset_speech_outputs


@dataclass
class _SpeechCfg:
    whisper_model: str = "tiny"
    commit_every: int = 1
    translate_model: str = "google/gemma-2b-it"
    translate_target_lang: str = "en"
    translation_max_chars: int = 4096
    translate_max_new_tokens: int = 256
    logprob_threshold: float = -1.0
    compression_threshold: float = 2.4
    min_meaningful_chars: int = 3
    vad_enabled: bool = False
    vad_sampling_rate: int = 16000
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 250
    vad_speech_pad_ms: int = 30
    vad_min_total_speech_s: float = 0.5


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _seed(session, *, is_speech_detected: bool | None = True) -> None:
    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=is_speech_detected,
            speech_transcription="hello world",
            speech_language="en",
            speech_translation="hello world",
        )
    )
    session.commit()


def test_reset_speech_outputs_nulls_the_four_fields(db_session):
    _seed(db_session)
    reset_speech_outputs(db_session)

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None
    assert clip.speech_transcription is None
    assert clip.speech_language is None
    assert clip.speech_translation is None


def test_first_run_seals_stage(monkeypatch, db_session):
    """process_speech computes fingerprint, runs row-level work (mocked
    no-ops here), and seals StageState."""
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: None)

    _seed(db_session)
    process_speech(_SpeechCfg(), video_dir="/tmp", speech_audio_dir="/tmp")

    state = db_session.get(StageState, (STAGE_SPEECH, SCOPE_SPEECH))
    assert state is not None


def test_unchanged_config_does_not_reset(monkeypatch, db_session):
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: None)

    _seed(db_session)
    cfg = _SpeechCfg()
    process_speech(cfg, video_dir="/tmp", speech_audio_dir="/tmp")
    process_speech(cfg, video_dir="/tmp", speech_audio_dir="/tmp")

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is True, "unchanged config must not reset clips"
    assert clip.speech_transcription == "hello world"


def test_config_change_resets_speech_columns(monkeypatch, db_session):
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: None)

    _seed(db_session)
    process_speech(_SpeechCfg(), video_dir="/tmp", speech_audio_dir="/tmp")
    process_speech(
        _SpeechCfg(whisper_model="medium"),
        video_dir="/tmp",
        speech_audio_dir="/tmp",
    )

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None, "config drift must NULL is_speech_detected"
    assert clip.speech_transcription is None
    assert clip.speech_language is None
    assert clip.speech_translation is None
