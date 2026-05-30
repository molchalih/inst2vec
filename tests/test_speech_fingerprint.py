"""Config-drift fingerprint gate for the speech pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.database import (
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
    translate_batch_size: int = 16
    logprob_threshold: float = -1.0
    compression_threshold: float = 2.4
    min_meaningful_chars: int = 3
    dirty_min_chars: int = 5
    dirty_min_letter_ratio: float = 0.3
    vad_enabled: bool = False
    vad_sampling_rate: int = 16000
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 250
    vad_speech_pad_ms: int = 30
    vad_min_total_speech_s: float = 0.5
    vad_ffmpeg_timeout_s: int = 60


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


def test_reset_speech_outputs_also_clears_ineligible_clips(db_session):
    """Stale speech outputs on a currently-ineligible clip must be cleared
    too so re-selection in a later run does not skip re-processing.

    Verifies all seven NULL'd columns: the four content columns plus the
    three Whisper metric columns (speech_confidence, speech_avg_logprob,
    speech_compression_ratio). Drift between this and
    ``reset_speech_outputs`` itself silently leaks stale metrics across
    config-drift resets — guard the full surface here.
    """
    from core.database import User

    db_session.merge(User(id=2, is_selected=True, is_eligible=True))
    db_session.merge(
        Clip(
            id=20,
            user_id=2,
            is_selected=False,
            is_downloaded=True,
            is_speech_detected=True,
            speech_transcription="old",
            speech_language="en",
            speech_translation="old",
            speech_confidence=0.9,
            speech_avg_logprob=-0.3,
            speech_compression_ratio=1.4,
        )
    )
    db_session.commit()

    reset_speech_outputs(db_session)
    clip = db_session.query(Clip).filter_by(id=20).one()
    assert clip.is_speech_detected is None
    assert clip.speech_transcription is None
    assert clip.speech_language is None
    assert clip.speech_translation is None
    assert clip.speech_confidence is None
    assert clip.speech_avg_logprob is None
    assert clip.speech_compression_ratio is None


def test_first_run_seals_stage(monkeypatch, db_session):
    """process_speech computes fingerprint, runs row-level work (mocked
    no-ops here), and seals StageState."""
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: (0, 0))
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: 0)

    _seed(db_session)
    process_speech(_SpeechCfg(), video_dir="/tmp", speech_audio_dir="/tmp")

    state = db_session.get(StageState, (STAGE_SPEECH, SCOPE_SPEECH))
    assert state is not None


def test_unchanged_config_does_not_reset(monkeypatch, db_session):
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: (0, 0))
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: 0)

    _seed(db_session)
    cfg = _SpeechCfg()
    process_speech(cfg, video_dir="/tmp", speech_audio_dir="/tmp")
    process_speech(cfg, video_dir="/tmp", speech_audio_dir="/tmp")

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is True, "unchanged config must not reset clips"
    assert clip.speech_transcription == "hello world"


def test_config_change_resets_speech_columns(monkeypatch, db_session):
    from modules import speech as speech_pkg

    monkeypatch.setattr(speech_pkg, "classify_speech", lambda **kw: (0, 0))
    monkeypatch.setattr(speech_pkg, "translate_speech", lambda **kw: None)
    monkeypatch.setattr(speech_pkg, "clean_speech", lambda: 0)

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


def test_speech_config_payload_ignores_operational_knobs():
    """commit_every, vad_ffmpeg_timeout_s and translate_batch_size must not
    invalidate the speech fingerprint — they're purely operational."""
    from modules.speech.state import speech_config_payload

    base = _SpeechCfg()
    bumped_commit = _SpeechCfg(commit_every=base.commit_every + 5)
    bumped_vad_to = _SpeechCfg(vad_ffmpeg_timeout_s=base.vad_ffmpeg_timeout_s + 30)
    bumped_batch = _SpeechCfg(translate_batch_size=base.translate_batch_size * 4)
    assert speech_config_payload(base) == speech_config_payload(bumped_commit)
    assert speech_config_payload(base) == speech_config_payload(bumped_vad_to)
    assert speech_config_payload(base) == speech_config_payload(bumped_batch)


def test_speech_config_payload_flips_on_output_affecting_knob():
    from modules.speech.state import speech_config_payload

    base = _SpeechCfg()
    bumped_model = _SpeechCfg(whisper_model="small")
    bumped_vad_on = _SpeechCfg(vad_enabled=True)
    assert speech_config_payload(base) != speech_config_payload(bumped_model)
    assert speech_config_payload(base) != speech_config_payload(bumped_vad_on)
