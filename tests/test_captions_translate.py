"""Behavior tests for translate_captions."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.captions.translate import translate_captions
from modules.config import CaptionsSettings
from modules.database import Base, Clip, User


def _cfg():
    return CaptionsSettings(
        commit_every=2,
        translate_model="dummy",
        translate_target_lang="en",
        translation_max_chars=1000,
        translate_max_new_tokens=200,
    )


@pytest.fixture
def session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.commit()
        yield eng, s


def _add(s, clip_id, clean, lang, translation=None):
    s.add(
        Clip(
            id=clip_id,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text=clean,
            caption_clean=clean,
            caption_language=lang,
            caption_translation=translation,
        )
    )
    s.commit()


def _stub_translator(monkeypatch, *, translate_fn):
    class _T:
        model_id = "dummy"
        device = "cpu"

        def translate_text(
            self, *, text, source_lang_code, target_lang_code, max_new_tokens
        ):
            return translate_fn(text, source_lang_code, target_lang_code)

    monkeypatch.setattr(
        "modules.captions.translate.GemmaTranslator", lambda model_id: _T()
    )


def test_translate_writes_translation_for_non_english(session, monkeypatch):
    eng, s = session
    _add(s, 1, "hola", "es")
    _stub_translator(monkeypatch, translate_fn=lambda t, src, dst: "hello")
    translate_captions(_cfg(), engine=eng)
    assert s.query(Clip).filter_by(id=1).one().caption_translation == "hello"


def test_translate_skips_english_clip(session, monkeypatch):
    eng, s = session
    _add(s, 1, "hi", "en")
    _stub_translator(monkeypatch, translate_fn=lambda *_: "BAD")
    translate_captions(_cfg(), engine=eng)
    assert s.query(Clip).filter_by(id=1).one().caption_translation is None


def test_translate_skips_already_translated(session, monkeypatch):
    eng, s = session
    _add(s, 1, "hola", "es", translation="prefilled")
    _stub_translator(monkeypatch, translate_fn=lambda *_: "should not run")
    translate_captions(_cfg(), engine=eng)
    assert s.query(Clip).filter_by(id=1).one().caption_translation == "prefilled"


def test_translate_logs_failure_and_continues(session, monkeypatch):
    eng, s = session
    _add(s, 1, "hola", "es")
    _add(s, 2, "bonjour", "fr")

    def fail_first(text, *_):
        if text == "hola":
            raise RuntimeError("boom")
        return "hello"

    _stub_translator(monkeypatch, translate_fn=fail_first)
    logged = []
    monkeypatch.setattr(
        "modules.captions.translate.log",
        lambda scope, msg, level="info": logged.append((scope, msg, level)),
    )
    translate_captions(_cfg(), engine=eng)
    row1 = s.query(Clip).filter_by(id=1).one()
    row2 = s.query(Clip).filter_by(id=2).one()
    assert row1.caption_translation is None
    assert row2.caption_translation == "hello"
    assert any("1" in msg and "boom" in msg for _, msg, _ in logged), logged
