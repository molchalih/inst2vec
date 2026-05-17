"""Behavior tests for clean_captions."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.database import Base, Clip, User
from modules.captions.clean import clean_captions


def _cfg() -> CaptionsSettings:
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


def _add_clip(s, clip_id, caption_text, is_selected=True, is_downloaded=True):
    s.add(
        Clip(
            id=clip_id,
            user_id=1,
            is_selected=is_selected,
            is_downloaded=is_downloaded,
            caption_text=caption_text,
        )
    )
    s.commit()


def test_clean_captions_writes_caption_clean_without_touching_caption_text(session):
    eng, s = session
    _add_clip(s, 1, "hello @bob   world\n")
    clean_captions(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_text == "hello @bob   world\n"
    assert row.caption_clean == "hello world"


def test_clean_captions_stores_empty_string_for_mentions_only_result(session):
    eng, s = session
    _add_clip(s, 1, "@a @b @c")
    clean_captions(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_clean == ""


def test_clean_captions_idempotent_skips_filled_rows(session):
    eng, s = session
    _add_clip(s, 1, "hello world")
    s.query(Clip).filter_by(id=1).update({Clip.caption_clean: "PREFILLED"})
    s.commit()
    clean_captions(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_clean == "PREFILLED"


def test_clean_captions_skips_unselected_clips(session):
    eng, s = session
    _add_clip(s, 1, "hello world", is_selected=False)
    clean_captions(_cfg(), engine=eng)
    row = s.query(Clip).filter_by(id=1).one()
    assert row.caption_clean is None
