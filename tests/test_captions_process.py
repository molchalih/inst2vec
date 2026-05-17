"""Tests for the process_captions orchestrator."""

import pytest

from modules import captions as captions_pkg
from modules.captions import process_captions
from modules.config import CaptionsSettings
from modules.database import (
    Base,
    Clip,
    StageState,
    User,
    get_engine,
    get_session,
)


def _cfg():
    return CaptionsSettings(
        commit_every=2,
        translate_model="dummy",
        translate_target_lang="en",
        translation_max_chars=1000,
        translate_max_new_tokens=200,
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, User):
        session.query(model).delete()
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        for model in (StageState, Clip, User):
            session.query(model).delete()
        session.commit()
        session.close()


def test_process_captions_calls_each_stage_in_order(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        captions_pkg,
        "clean_captions",
        lambda cfg, *, engine=None: calls.append("clean"),
    )
    monkeypatch.setattr(
        captions_pkg,
        "detect_caption_language",
        lambda cfg, *, engine=None: calls.append("detect"),
    )
    monkeypatch.setattr(
        captions_pkg,
        "translate_captions",
        lambda cfg, *, engine=None: calls.append("translate"),
    )
    process_captions(_cfg())
    assert calls == ["clean", "detect", "translate"]


def test_process_captions_propagates_engine(monkeypatch, db_session):
    eng = get_engine()
    seen: list = []
    monkeypatch.setattr(
        captions_pkg,
        "clean_captions",
        lambda cfg, *, engine=None: seen.append(("clean", engine)),
    )
    monkeypatch.setattr(
        captions_pkg,
        "detect_caption_language",
        lambda cfg, *, engine=None: seen.append(("detect", engine)),
    )
    monkeypatch.setattr(
        captions_pkg,
        "translate_captions",
        lambda cfg, *, engine=None: seen.append(("translate", engine)),
    )
    process_captions(_cfg(), engine=eng)
    assert all(e is eng for _, e in seen)


def test_reset_caption_outputs_nulls_the_three_fields(db_session):
    from modules.captions.state import reset_caption_outputs
    from modules.database import Clip, User

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="raw",
            caption_clean="cleaned",
            caption_language="en",
            caption_translation="translated",
        )
    )
    db_session.commit()

    reset_caption_outputs(db_session)
    c = db_session.query(Clip).filter_by(id=10).one()
    assert c.caption_text == "raw", "caption_text is upstream input, must be preserved"
    assert c.caption_clean is None
    assert c.caption_language is None
    assert c.caption_translation is None


def test_process_captions_config_change_triggers_reset(monkeypatch, db_session):
    import modules.captions as captions_pkg
    from modules.captions import process_captions
    from modules.config import CaptionsSettings
    from modules.database import Clip, User

    # Stub the three row-level stages so only the gate matters.
    monkeypatch.setattr(
        captions_pkg, "clean_captions", lambda cfg, *, engine=None: None
    )
    monkeypatch.setattr(
        captions_pkg, "detect_caption_language", lambda cfg, *, engine=None: None
    )
    monkeypatch.setattr(
        captions_pkg, "translate_captions", lambda cfg, *, engine=None: None
    )

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="raw",
            caption_clean="cleaned",
            caption_language="en",
            caption_translation="translated",
        )
    )
    db_session.commit()

    base = CaptionsSettings(
        commit_every=1,
        translate_model="m",
        translate_target_lang="en",
        translation_max_chars=4096,
        translate_max_new_tokens=256,
    )
    process_captions(base)
    db_session.expire_all()
    assert db_session.query(Clip).filter_by(id=10).one().caption_clean == "cleaned"

    process_captions(base.model_copy(update={"translate_model": "m2"}))
    db_session.expire_all()
    c = db_session.query(Clip).filter_by(id=10).one()
    assert c.caption_clean is None
    assert c.caption_language is None
    assert c.caption_translation is None


def test_process_captions_unchanged_config_does_not_reset(monkeypatch, db_session):
    """Two consecutive calls with the same config must preserve seeded
    caption data."""
    import modules.captions as captions_pkg
    from modules.captions import process_captions
    from modules.config import CaptionsSettings
    from modules.database import Clip, User

    monkeypatch.setattr(
        captions_pkg, "clean_captions", lambda cfg, *, engine=None: None
    )
    monkeypatch.setattr(
        captions_pkg, "detect_caption_language", lambda cfg, *, engine=None: None
    )
    monkeypatch.setattr(
        captions_pkg, "translate_captions", lambda cfg, *, engine=None: None
    )

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="raw",
            caption_clean="cleaned",
            caption_language="en",
            caption_translation="translated",
        )
    )
    db_session.commit()

    cfg = CaptionsSettings(
        commit_every=1,
        translate_model="m",
        translate_target_lang="en",
        translation_max_chars=4096,
        translate_max_new_tokens=256,
    )
    process_captions(cfg)
    process_captions(cfg)

    db_session.expire_all()
    c = db_session.query(Clip).filter_by(id=10).one()
    assert c.caption_clean == "cleaned", "unchanged config must not reset"
    assert c.caption_language == "en"
    assert c.caption_translation == "translated"
