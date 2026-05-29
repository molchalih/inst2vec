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


@pytest.fixture(autouse=True)
def _probe_audio_stream_true(monkeypatch):
    """Default: treat every video file as having an audio stream so existing
    tests are not affected by the new pre-gate."""
    monkeypatch.setattr(
        "modules.speech.classify.probe_audio_stream", lambda _path: True
    )


def _kwargs(video_dir: Path):
    return dict(
        video_dir=str(video_dir),
        whisper_model="tiny",
        commit_every=50,
        logprob_threshold=-0.8,
        compression_threshold=2.4,
        min_meaningful_chars=4,
        dirty_min_chars=5,
        dirty_min_letter_ratio=0.3,
    )


def _fake_transcribe_result(transcribe_result):
    """Translate the legacy openai-whisper dict shape used in these tests
    into faster-whisper's ``(segments_iter, info)`` pair so the existing
    fixtures keep documenting their original intent (same text + segment
    stats), and only the seam shape changes."""
    text = transcribe_result.get("text") or ""
    language = transcribe_result.get("language") or ""
    seg_dicts = transcribe_result.get("segments") or []
    fw_segments = [
        SimpleNamespace(
            text=text if i == 0 else "",
            no_speech_prob=sd.get("no_speech_prob", 0.0),
            avg_logprob=sd.get("avg_logprob", 0.0),
            compression_ratio=sd.get("compression_ratio", 0.0),
        )
        for i, sd in enumerate(seg_dicts)
    ]
    info = SimpleNamespace(language=language, language_probability=1.0, duration=1.0)
    return iter(fw_segments), info


def _stub_whisper(monkeypatch, transcribe_result):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = _fake_transcribe_result(transcribe_result)
    monkeypatch.setattr(
        "modules.speech.classify.WhisperModel",
        lambda *_a, **_k: fake_model,
    )
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
    monkeypatch.setattr(
        "modules.speech.classify.WhisperModel", lambda *_a, **_k: fake_model
    )

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

    factory = MagicMock(side_effect=AssertionError("WhisperModel must not load"))
    monkeypatch.setattr("modules.speech.classify.WhisperModel", factory)

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
    fake_model.transcribe.return_value = _fake_transcribe_result(
        {
            "text": "hello there",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.5}
            ],
        }
    )
    monkeypatch.setattr(
        "modules.speech.classify.WhisperModel", lambda *_a, **_k: fake_model
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


def test_clean_speech_keeps_translation_containing_short_corpus_phrase(db_session):
    """A real translation that merely begins with a short common corpus
    entry like ``"Yeah."`` must NOT be nulled — the SQL substring filter
    only fires on substring-safe phrases (long generics + extras)."""
    s, _ = db_session
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_speech_detected = True
    clip.speech_translation = "Yeah. I agree with everything you just said."
    s.commit()

    clean_speech()

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is True
    assert clip.speech_translation == "Yeah. I agree with everything you just said."


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


def test_classify_too_short_transcript_flagged_but_kept(db_session, monkeypatch):
    """Whisper returns ``"you"`` (3 chars). The new dirty-min-chars rule flags
    the clip as no-speech AND keeps the transcript stored for analysis."""
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "you",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.0}
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
    assert clip.speech_transcription == "you"


def test_classify_low_letter_ratio_flagged_but_kept(db_session, monkeypatch):
    """Digit-heavy noise trips has_low_letter_ratio; transcript preserved."""
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "1, 2, 3, 4, 5.",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.0}
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
    assert clip.speech_transcription == "1, 2, 3, 4, 5."


def test_classify_file_hallucination_flagged_but_kept(db_session, monkeypatch):
    """A Vexa-corpus hallucination ("Thanks for watching!") is detected by the
    new file-backed has_hallucination_marker, flag flips, transcript preserved."""
    s, tmp_path = db_session
    _stub_whisper(
        monkeypatch,
        {
            "text": "Thanks for watching!",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.0}
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
    assert clip.speech_transcription == "Thanks for watching!"


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
    factory = MagicMock(side_effect=AssertionError("WhisperModel must not load"))
    monkeypatch.setattr("modules.speech.classify.WhisperModel", factory)

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


def test_classify_speech_marks_no_audio_stream_terminal(db_session, monkeypatch):
    """A video present on disk with no audio stream must be marked
    is_speech_detected=False without invoking VAD/ffmpeg, so the row is
    not retried on the next pipeline run."""
    from modules.speech import classify as classify_mod

    s, tmp_path = db_session

    monkeypatch.setattr(classify_mod, "probe_audio_stream", lambda _path: False)

    def _vad_must_not_run(*_args, **_kwargs):
        raise AssertionError("VAD must not run when audio stream is missing")

    monkeypatch.setattr(classify_mod, "prepare_for_whisper", _vad_must_not_run)

    classify_speech(
        video_dir=str(tmp_path),
        speech_audio_dir=str(tmp_path / "audio"),
        whisper_model="tiny",
        commit_every=1,
        logprob_threshold=-1.0,
        compression_threshold=2.4,
        min_meaningful_chars=1,
        dirty_min_chars=5,
        dirty_min_letter_ratio=0.3,
        vad_config=VadConfig(enabled=True),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is False


def test_classify_speech_leaves_probe_failure_retryable(db_session, monkeypatch):
    """An ffprobe failure (None) must NOT be sealed as is_speech_detected=False;
    the row stays NULL so the next run can retry once ffprobe recovers."""
    from modules.speech import classify as classify_mod

    s, tmp_path = db_session

    monkeypatch.setattr(classify_mod, "probe_audio_stream", lambda _path: None)

    def _vad_must_not_run(*_args, **_kwargs):
        raise AssertionError("VAD must not run when ffprobe fails")

    monkeypatch.setattr(classify_mod, "prepare_for_whisper", _vad_must_not_run)

    classify_speech(
        video_dir=str(tmp_path),
        speech_audio_dir=str(tmp_path / "audio"),
        whisper_model="tiny",
        commit_every=1,
        logprob_threshold=-1.0,
        compression_threshold=2.4,
        min_meaningful_chars=1,
        dirty_min_chars=5,
        dirty_min_letter_ratio=0.3,
        vad_config=VadConfig(enabled=True),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_speech_detected is None


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
