"""Behavior tests for classify_speech / clean_speech."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, User
from modules.speech.classify import classify_speech, clean_speech
from modules.speech.vad import VadConfig, VadResult


@pytest.fixture
def db_session(monkeypatch, tmp_path):
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

    monkeypatch.setattr("modules.speech.classify.get_session", lambda: s)
    yield s, tmp_path
    s.close()


@pytest.fixture(autouse=True)
def _vad_passthrough(monkeypatch):
    """Default: VAD returns is_speech_detected=True with the original path so
    pre-VAD tests still exercise the Whisper branch unchanged."""

    def fake(media_path, _out_dir, _config):
        return VadResult(is_speech_detected=True, speech_audio_path=media_path)

    monkeypatch.setattr("modules.speech.classify.prepare_for_whisper", fake)


def _kwargs(video_dir: Path):
    return dict(
        video_dir=str(video_dir),
        whisper_model="tiny",
        commit_every=50,
        logprob_threshold=-0.8,
        compression_threshold=2.4,
        min_meaningful_chars=4,
    )


def _stub_whisper(monkeypatch, transcribe_result):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = transcribe_result
    fake_whisper = SimpleNamespace(load_model=lambda _name: fake_model, Whisper=object)
    monkeypatch.setattr("modules.speech.classify.whisper", fake_whisper)
    return fake_model


def test_classify_meaningful_text_sets_is_speech_detected_true(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "hello there friend",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
            ],
        },
    )
    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is True
    assert clip.speech_transcription == "hello there friend"
    assert clip.speech_language == "en"


def test_classify_low_logprob_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "uncertain words here",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.5, "avg_logprob": -1.5, "compression_ratio": 1.0}
            ],
        },
    )
    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False


def test_classify_high_compression_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "Thank you. Thank you. Thank you.",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 3.0}
            ],
        },
    )
    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False


def test_classify_repeated_token_loop_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "thanks thanks thanks thanks thanks",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
            ],
        },
    )
    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False


def test_classify_hallucination_marker_in_text_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "subtitles by DimaTorzok",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
            ],
        },
    )
    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False


def test_classify_missing_file_leaves_null(db_session, monkeypatch):
    s, tmp_path = db_session
    (tmp_path / "10.mp4").unlink()
    fake = _stub_whisper(monkeypatch, {"text": "x", "language": "en", "segments": []})

    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None
    assert clip.speech_transcription is None
    fake.transcribe.assert_not_called()


def test_classify_whisper_exception_leaves_null(db_session, monkeypatch):
    s, tmp_path = db_session
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("OOM")
    fake_whisper = SimpleNamespace(load_model=lambda _: fake_model, Whisper=object)
    monkeypatch.setattr("modules.speech.classify.whisper", fake_whisper)

    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None
    assert clip.speech_transcription is None


def test_classify_skips_already_resolved_clips(db_session, monkeypatch):
    s, tmp_path = db_session
    s.query(Clip).filter_by(id=10).update({"is_speech_detected": True})
    s.commit()

    fake_model = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(
        "modules.speech.classify.whisper",
        SimpleNamespace(load_model=lambda _: fake_model, Whisper=object),
    )

    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )
    # No exception means no clip was queried for transcription.


def test_classify_passes_hallucination_control_kwargs(db_session, monkeypatch):
    """Whisper must be called with condition_on_previous_text=False, beam_size=1, temperature=0."""
    _, tmp_path = db_session
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {
        "text": "hello there",
        "language": "en",
        "segments": [
            {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
        ],
    }
    monkeypatch.setattr(
        "modules.speech.classify.whisper",
        SimpleNamespace(load_model=lambda _: fake_model, Whisper=object),
    )

    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=VadConfig(enabled=False),
        **_kwargs(tmp_path),
    )

    fake_model.transcribe.assert_called_once()
    kwargs = fake_model.transcribe.call_args.kwargs
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["beam_size"] == 1
    assert kwargs["temperature"] == 0


def test_clean_speech_resets_marker_translations(db_session):
    s, _ = db_session
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_speech_detected = True
    clip.speech_translation = "subtitles by DimaTorzok"
    s.commit()

    clean_speech()

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False
    assert clip.speech_translation is None


def test_clean_speech_nulls_poisoned_text(db_session):
    """clean_speech must NULL the seven speech columns when a
    hallucination marker is detected, so downstream text builders and
    the dependency hash see the poisoned row as empty."""
    s, _ = db_session
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_speech_detected = True
    clip.speech_transcription = "real text DimaTorzok"
    clip.speech_language = "en"
    clip.speech_translation = "real text DimaTorzok"
    clip.speech_confidence = 0.9
    clip.speech_avg_logprob = -0.3
    clip.speech_compression_ratio = 1.4
    s.commit()

    clean_speech()

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False
    assert clip.speech_transcription is None
    assert clip.speech_language is None
    assert clip.speech_translation is None
    assert clip.speech_confidence is None
    assert clip.speech_avg_logprob is None
    assert clip.speech_compression_ratio is None


def _vad_disabled():
    return VadConfig(enabled=False)


def _vad_enabled():
    return VadConfig(
        enabled=True,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=150,
        min_total_speech_s=0.5,
    )


def _stub_vad(monkeypatch, result):
    calls = []

    def fake(media_path, out_dir, config):
        calls.append((media_path, out_dir, config))
        return result

    monkeypatch.setattr("modules.speech.classify.prepare_for_whisper", fake)
    return calls


def test_vad_no_speech_skips_whisper_and_sets_false(db_session, monkeypatch):
    s, tmp_path = db_session
    _stub_vad(
        monkeypatch,
        VadResult(is_speech_detected=False, speech_audio_path=None),
    )
    fake_whisper = SimpleNamespace(
        load_model=MagicMock(side_effect=AssertionError("Whisper must not load")),
        Whisper=object,
    )
    monkeypatch.setattr("modules.speech.classify.whisper", fake_whisper)

    classify_speech(
        speech_audio_dir=str(tmp_path / "audio"),
        vad_config=_vad_enabled(),
        **_kwargs(tmp_path),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False
    assert clip.speech_transcription is None


def test_vad_speech_path_is_passed_to_whisper(db_session, monkeypatch):
    s, tmp_path = db_session
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    speech_wav = audio_dir / "10.wav"
    speech_wav.write_bytes(b"speech-only")
    _stub_vad(
        monkeypatch,
        VadResult(
            is_speech_detected=True,
            speech_audio_path=speech_wav,
            total_speech_seconds=1.2,
        ),
    )
    model = _stub_whisper(
        monkeypatch,
        {
            "text": "hello there friend",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
            ],
        },
    )

    classify_speech(
        speech_audio_dir=str(audio_dir),
        vad_config=_vad_enabled(),
        **_kwargs(tmp_path),
    )

    model.transcribe.assert_called_once()
    assert model.transcribe.call_args.args[0] == str(speech_wav)
    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is True


def test_reset_speech_outputs_nulls_all_seven_columns(db_session):
    """Reset must NULL every speech column, including the three metric
    columns, so derived stats can't carry stale prior-config values."""
    from modules.speech.state import reset_speech_outputs

    s, _ = db_session
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_speech_detected = True
    clip.speech_transcription = "hello"
    clip.speech_language = "en"
    clip.speech_translation = "hello"
    clip.speech_confidence = 0.9
    clip.speech_avg_logprob = -0.3
    clip.speech_compression_ratio = 1.4
    s.commit()

    reset_speech_outputs(s)

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None
    assert clip.speech_transcription is None
    assert clip.speech_language is None
    assert clip.speech_translation is None
    assert clip.speech_confidence is None
    assert clip.speech_avg_logprob is None
    assert clip.speech_compression_ratio is None
