from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, Clip, User


def test_clip_has_filter_columns():
    clip_cols = {c.key for c in Clip.__table__.columns}
    for col in [
        "is_garbage",
        "is_low_play_count",
        "is_too_short",
        "is_too_long",
        "is_too_old",
        "is_low_percentile",
        "is_high_percentile",
        "is_creator_low_outlier",
        "log_plays",
        "creator_relative_robust_z",
        "is_eligible",
        "is_selected",
    ]:
        assert col in clip_cols, f"Clip missing column: {col}"


def test_user_has_filter_columns():
    user_cols = {c.key for c in User.__table__.columns}
    for col in [
        "is_low_plays_median",
        "is_not_enough_clips",
        "is_selected",
        "log_plays_median",
        "log_plays_mad",
    ]:
        assert col in user_cols, f"User missing column: {col}"


def test_clip_filter_columns_default_null():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=100, user_id=1))
        s.commit()
        clip = s.get(Clip, 100)
        assert clip.is_garbage is None
        assert clip.is_eligible is None
        assert clip.is_selected is None
        assert clip.log_plays is None


def test_user_filter_columns_default_null():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=2))
        s.commit()
        user = s.get(User, 2)
        assert user.is_low_plays_median is None
        assert user.is_not_enough_clips is None
        assert user.is_selected is None
        assert user.log_plays_median is None
        assert user.log_plays_mad is None


def test_filter_settings_load():
    from modules.config import FilterSettings
    cfg = FilterSettings(
        min_play_count=1000,
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1640995200,
        creator_min_median_views=10000,
        min_eligible_clips_per_user=10,
        global_low_percentile=5,
        global_high_percentile=99,
        creator_low_z_threshold=-3.5,
        selection_pool_percent=0.20,
        selected_clips_per_user=10,
        selection_random_seed=42,
    )
    assert cfg.min_play_count == 1000
    assert cfg.selected_clips_per_user == 10


from modules.filter import (
    _is_garbage,
    _is_low_play_count,
    _is_too_short,
    _is_too_long,
    _is_too_old,
)

def _make_clip(**kwargs):
    """Build a minimal Clip-like object via SimpleNamespace."""
    from types import SimpleNamespace
    defaults = dict(
        video_duration=10.0,
        taken_at=1700000000,
        play_count=5000,
        download_url="http://example.com/v.mp4",
        like_count=100,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)

def test_is_garbage_missing_video_duration():
    c = _make_clip(video_duration=None)
    assert _is_garbage(c) is True

def test_is_garbage_zero_video_duration():
    c = _make_clip(video_duration=0.0)
    assert _is_garbage(c) is True

def test_is_garbage_missing_taken_at():
    c = _make_clip(taken_at=None)
    assert _is_garbage(c) is True

def test_is_garbage_zero_taken_at():
    c = _make_clip(taken_at=0)
    assert _is_garbage(c) is True

def test_is_garbage_missing_play_count():
    c = _make_clip(play_count=None)
    assert _is_garbage(c) is True

def test_is_garbage_zero_play_count():
    c = _make_clip(play_count=0)
    assert _is_garbage(c) is True

def test_is_garbage_missing_download_url():
    c = _make_clip(download_url=None)
    assert _is_garbage(c) is True

def test_is_garbage_empty_download_url():
    c = _make_clip(download_url="")
    assert _is_garbage(c) is True

def test_is_garbage_missing_like_count():
    c = _make_clip(like_count=None)
    assert _is_garbage(c) is True

def test_is_garbage_valid_clip():
    c = _make_clip()
    assert _is_garbage(c) is False

def test_is_low_play_count_below():
    c = _make_clip(play_count=999)
    assert _is_low_play_count(c, min_play_count=1000) is True

def test_is_low_play_count_at():
    c = _make_clip(play_count=1000)
    assert _is_low_play_count(c, min_play_count=1000) is False

def test_is_too_short_below():
    c = _make_clip(video_duration=2.9)
    assert _is_too_short(c, min_video_duration=3) is True

def test_is_too_short_at():
    c = _make_clip(video_duration=3.0)
    assert _is_too_short(c, min_video_duration=3) is False

def test_is_too_long_above():
    c = _make_clip(video_duration=80.1)
    assert _is_too_long(c, max_video_duration=80) is True

def test_is_too_long_at():
    c = _make_clip(video_duration=80.0)
    assert _is_too_long(c, max_video_duration=80) is False

def test_is_too_old_below():
    c = _make_clip(taken_at=1640995199)
    assert _is_too_old(c, min_taken_at=1640995200) is True

def test_is_too_old_at():
    c = _make_clip(taken_at=1640995200)
    assert _is_too_old(c, min_taken_at=1640995200) is False


def _make_db():
    from modules.database import Base
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_flag_garbage_clips_sets_is_garbage():
    from modules.filter import _flag_garbage_clips
    from modules.database import Clip, User
    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        # garbage: missing play_count
        s.add(Clip(id=1, user_id=1, video_duration=10.0, taken_at=1700000000,
                   play_count=None, video_url="http://x.com/v.mp4", like_count=5))
        # valid clip
        s.add(Clip(id=2, user_id=1, video_duration=10.0, taken_at=1700000000,
                   play_count=5000, video_url="http://x.com/v.mp4", like_count=5))
        s.commit()
        _flag_garbage_clips(s)
        s.commit()
        c1 = s.get(Clip, 1)
        c2 = s.get(Clip, 2)
        assert c1.is_garbage is True
        assert c2.is_garbage is False
