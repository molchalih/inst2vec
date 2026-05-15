"""Tests for scripts/retry_failed_speech_detection.py"""

from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, Clip, User
from modules.speech.vad import VadConfig


def _make_db(tmp_path: Path):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=None,
        )
    )
    s.commit()
    (tmp_path / "10.mp4").write_bytes(b"fake")
    return s


def test_retry_invokes_classify_speech_with_kwargs(tmp_path, monkeypatch):
    s = _make_db(tmp_path)
    monkeypatch.setattr("scripts.retry_failed_speech_detection.get_session", lambda: s)

    captured = {}

    def _fake_classify(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "scripts.retry_failed_speech_detection.classify_speech", _fake_classify
    )

    from scripts.retry_failed_speech_detection import retry_failed_speech_detection

    retry_failed_speech_detection(
        video_dir=str(tmp_path),
        speech_audio_dir=str(tmp_path / "audio"),
        whisper_model="tiny",
        commit_every=10,
        logprob_threshold=-0.8,
        compression_threshold=2.4,
        min_meaningful_chars=8,
        vad_config=VadConfig(enabled=False),
    )

    assert captured["video_dir"] == str(tmp_path)
    assert captured["speech_audio_dir"] == str(tmp_path / "audio")
    assert captured["whisper_model"] == "tiny"
    assert captured["commit_every"] == 10
    assert captured["logprob_threshold"] == -0.8
    assert captured["compression_threshold"] == 2.4
    assert captured["min_meaningful_chars"] == 8
    assert captured["vad_config"] == VadConfig(enabled=False)


def test_retry_short_circuits_when_no_unresolved_clips(tmp_path, monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    monkeypatch.setattr("scripts.retry_failed_speech_detection.get_session", lambda: s)

    fake = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr("scripts.retry_failed_speech_detection.classify_speech", fake)

    from scripts.retry_failed_speech_detection import retry_failed_speech_detection

    retry_failed_speech_detection(
        video_dir=str(tmp_path),
        speech_audio_dir=str(tmp_path / "audio"),
        whisper_model="tiny",
        commit_every=10,
        logprob_threshold=-0.8,
        compression_threshold=2.4,
        min_meaningful_chars=8,
        vad_config=VadConfig(enabled=False),
    )
    fake.assert_not_called()
