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
