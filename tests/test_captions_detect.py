"""Behavior tests for detect_caption_language."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.captions.detect import detect_caption_language
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


def _add_clip(s, clip_id, caption_clean, caption_language=None):
    s.add(
        Clip(
            id=clip_id,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text=caption_clean,
            caption_clean=caption_clean,
            caption_language=caption_language,
        )
    )
    s.commit()


def _stub_detector(monkeypatch, iso_by_text: dict[str, str | None]):
    class _Lang:
        def __init__(self, code):
            self.iso_code_639_1 = SimpleNamespace(name=code.upper()) if code else None

    class _Detector:
        def detect_language_of(self, text):
            return _Lang(iso_by_text.get(text, "EN"))

    class _Builder:
        def build(self):
            return _Detector()

    fake_lingua = SimpleNamespace(
        LanguageDetectorBuilder=SimpleNamespace(from_all_languages=lambda: _Builder())
    )
    monkeypatch.setattr(
        "modules.captions.detect.LanguageDetectorBuilder",
        fake_lingua.LanguageDetectorBuilder,
    )


def test_detect_writes_language_from_caption_clean(session, monkeypatch):
    eng, s = session
    _add_clip(s, 1, "hola amigo")
    _stub_detector(monkeypatch, {"hola amigo": "es"})
    detect_caption_language(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_language == "es"


def test_detect_skips_clip_with_existing_language(session, monkeypatch):
    eng, s = session
    _add_clip(s, 1, "hola amigo", caption_language="pl")
    _stub_detector(monkeypatch, {"hola amigo": "es"})
    detect_caption_language(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_language == "pl"


def test_detect_skips_clip_without_caption_clean(session, monkeypatch):
    eng, s = session
    s.add(
        Clip(
            id=1,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="raw",
            caption_clean=None,
        )
    )
    s.commit()
    _stub_detector(monkeypatch, {"raw": "es"})
    detect_caption_language(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_language is None
