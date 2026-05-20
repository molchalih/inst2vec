"""Behavior tests for translate_speech."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, User
from modules.speech.translate import translate_speech


@pytest.fixture
def db_session(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.commit()
    monkeypatch.setattr("modules.speech.translate.get_session", lambda: s)
    yield s
    s.close()


def _kwargs():
    return dict(
        commit_every=50,
        translate_model="dummy/model",
        translate_target_lang="en",
        translation_max_chars=1000,
        translate_max_new_tokens=200,
    )


def _stub_translator(monkeypatch, translate_fn):
    fake = MagicMock()
    fake.model_id = "dummy/model"
    fake.device = "cpu"
    fake.translate_text.side_effect = translate_fn
    monkeypatch.setattr("core.translate.GemmaTranslator", lambda model_id: fake)
    return fake


def test_translate_writes_translation_for_detected_non_english(db_session, monkeypatch):
    s = db_session
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=True,
            speech_transcription="bonjour le monde",
            speech_language="fr",
        )
    )
    s.commit()
    _stub_translator(monkeypatch, lambda **kw: "hello world")

    translate_speech(**_kwargs())

    assert s.query(Clip).filter_by(id=10).one().speech_translation == "hello world"


def test_translate_skips_when_is_speech_detected_not_true(db_session, monkeypatch):
    s = db_session
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=False,
            speech_transcription="bonjour",
            speech_language="fr",
        )
    )
    s.commit()
    fake = _stub_translator(monkeypatch, lambda **kw: "hello")

    translate_speech(**_kwargs())

    assert s.query(Clip).filter_by(id=10).one().speech_translation is None
    fake.translate_text.assert_not_called()


def test_translate_skips_english(db_session, monkeypatch):
    s = db_session
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=True,
            speech_transcription="hello",
            speech_language="en",
        )
    )
    s.commit()
    fake = _stub_translator(monkeypatch, lambda **kw: "should not be used")

    translate_speech(**_kwargs())

    assert s.query(Clip).filter_by(id=10).one().speech_translation is None
    fake.translate_text.assert_not_called()


def test_translate_empty_result_leaves_translation_null(db_session, monkeypatch):
    s = db_session
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=True,
            speech_transcription="bonjour",
            speech_language="fr",
        )
    )
    s.commit()
    _stub_translator(monkeypatch, lambda **kw: "")

    translate_speech(**_kwargs())

    assert s.query(Clip).filter_by(id=10).one().speech_translation is None


def test_translate_exception_leaves_translation_null(db_session, monkeypatch):
    s = db_session
    s.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_speech_detected=True,
            speech_transcription="bonjour",
            speech_language="fr",
        )
    )
    s.commit()

    def _raise(**kw):
        raise RuntimeError("boom")

    _stub_translator(monkeypatch, _raise)

    translate_speech(**_kwargs())

    assert s.query(Clip).filter_by(id=10).one().speech_translation is None
