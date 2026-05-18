from __future__ import annotations

from core.database import Base, Clip, User, get_engine, get_session
from core.database.models import ClipFilterScratch


def test_clip_filter_scratch_round_trip():
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        user = User(id=1)
        clip = Clip(id=1, user_id=1)
        session.add_all([user, clip])
        session.commit()

        scratch = ClipFilterScratch(
            clip_id=1,
            log_plays=4.5,
            creator_relative_robust_z=-1.2,
            is_creator_low_outlier=True,
        )
        session.add(scratch)
        session.commit()

        fetched = session.get(ClipFilterScratch, 1)
        assert fetched is not None
        assert fetched.log_plays == 4.5
        assert fetched.creator_relative_robust_z == -1.2
        assert fetched.is_creator_low_outlier is True
    finally:
        session.close()


def test_clip_drops_scratch_columns():
    cols = {c.name for c in Clip.__table__.columns}
    assert "log_plays" not in cols
    assert "creator_relative_robust_z" not in cols
    assert "is_creator_low_outlier" not in cols
