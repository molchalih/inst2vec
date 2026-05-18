"""Schema tests for Music.recognition_status."""

import pytest

from core.database import (
    Base,
    Clip,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        for model in (StageState, Clip, Music, User):
            session.query(model).delete()
        session.commit()
        session.close()


def test_music_recognition_status_default_is_pending(db_session):
    from core.database import Music

    m = Music(artist="x", track="y")
    db_session.add(m)
    db_session.flush()
    assert m.recognition_status == "pending"


def test_no_match_constant_is_gone():
    import modules.music.state as state

    assert not hasattr(state, "_NO_MATCH")
