"""Behavior tests for translate_captions."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.database import Base, Clip, User
from modules.captions.translate import translate_captions


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

        def translate_batch(self, items, *, max_new_tokens, batch_size):
            return [translate_fn(text, src, dst) for (text, src, dst) in items]

    monkeypatch.setattr("core.translate.GemmaTranslator", lambda model_id: _T())


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
    logged: list[tuple] = []
    monkeypatch.setattr(
        "core.translate.log",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    translate_captions(_cfg(), engine=eng)
    row1 = s.query(Clip).filter_by(id=1).one()
    row2 = s.query(Clip).filter_by(id=2).one()
    assert row1.caption_translation is None
    assert row2.caption_translation == "hello"
    assert any(
        len(args) >= 4
        and args[1] == "MT"
        and args[2] == "cap_1"
        and args[3] == "ERR"
        and "boom" in str(kwargs.get("stats", {}).get("err", ""))
        for args, kwargs in logged
    ), logged
